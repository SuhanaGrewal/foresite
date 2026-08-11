"""
Trace Agent

Runs a minimal agent through a local Ollama server (Model = Qwen2.5-1.5B-Instruct),
using fake tools, and logs every step to a trace file.

- intentionally simple; not meant to be a "good" agent
- produces real log of agent behavior, fed into the cache simulator
- talks to Ollama's *native* /api/chat (via the `ollama` python package), not its
  OpenAI-compatible shim, because only the native API returns real per-call
  timing fields (prompt_eval_count/duration, eval_count/duration, load_duration).
  Those are logged alongside each model_call as an engine-reported proxy for
  cache reuse -- Ollama doesn't expose a labeled prefix-cache-hit flag the way
  vLLM's Prometheus endpoint does, so this is honestly a timing proxy, not a
  hit/miss classification.

Requires: pip install ollama
Requires: `ollama serve` running locally, and `ollama pull qwen2.5:1.5b`
"""

import json
import os
import re
import threading
import time
from datetime import datetime, timezone

import ollama
import requests

MODEL = "qwen2.5:1.5b"

# set to "false" to fall back to the old hardcoded fake tool outputs, for
# quick offline/API-cost-free iteration (see fake_weather_tool/fake_search_tool)
USE_REAL_WEATHER = os.environ.get("USE_REAL_WEATHER", "true").lower() != "false"
USE_REAL_SEARCH = os.environ.get("USE_REAL_SEARCH", "true").lower() != "false"


def call_model_with_retry(messages, tools, max_retries=4):
    """
    - retries the API call with backoff if the local Ollama server errors out
      (e.g. still loading the model, transient connection issue)
    """
    for attempt in range(max_retries):
        try:
            response = ollama.chat(model=MODEL, messages=messages, tools=tools)
            if response.message is not None:
                return response
        except (ollama.ResponseError, ConnectionError) as e:
            wait_seconds = 5 * (attempt + 1)
            print(f"  [ollama error ({e}), retrying in {wait_seconds}s...]")
            time.sleep(wait_seconds)
            continue
        wait_seconds = 5 * (attempt + 1)
        print(f"  [empty response, retrying in {wait_seconds}s...]")
        time.sleep(wait_seconds)
    raise RuntimeError("Model call failed after multiple retries against local Ollama server.")


def cache_metrics(response) -> dict:
    """
    pulls Ollama's real per-call timing fields off a chat response, for logging
    alongside each model_call -- see module docstring for why these are a timing
    proxy, not a real prefix-cache-hit signal.
    """
    return {
        "load_duration_ns": response.load_duration,
        "prompt_eval_count": response.prompt_eval_count,
        "prompt_eval_duration_ns": response.prompt_eval_duration,
        "eval_count": response.eval_count,
        "eval_duration_ns": response.eval_duration,
        "total_duration_ns": response.total_duration,
    }


TRACE_LOG_PATH = "traces.jsonl"

# sub-agents dispatched by the orchestrator run in real OS threads
# (via asyncio.to_thread), so multiple agents may call log_step at the
# same time. this guards the shared trace file's append.
_log_lock = threading.Lock()


# returns pretend data

def fake_weather_tool(city: str) -> str:
    fake_data = {
        "delhi": "38C, hazy, hot and dry",
        "los angeles": "22C, sunny, light breeze",
        "tokyo": "27C, humid, partly cloudy",
        "santa monica": "20C, sunny, ocean breeze",
        "runyon canyon": "24C, sunny, dry",
        "griffith park": "23C, sunny, mild breeze",
        "topanga state park": "21C, sunny, cool morning",
        "london": "16C, overcast, light rain",
    }
    key = city.strip().lower()
    return f"Weather in {city}: {fake_data.get(key, '22C, sunny, light breeze')}"


def fake_search_tool(query: str) -> str:
    return f"Top result for '{query}': Trail is open, moderate difficulty, dogs allowed."


# WMO weather interpretation codes, per Open-Meteo's published code table
# (https://open-meteo.com/en/docs) -- decoding a real API's numeric response
# into text, not fabricated data.
_WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow fall", 73: "moderate snow fall", 75: "heavy snow fall", 77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


def real_weather_tool(city: str) -> str:
    """
    real current conditions via Open-Meteo (free, no API key):
    geocode the city name to lat/lon, then fetch current weather for that point
    """
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()
        results = geo.get("results")
        if not results:
            return f"Weather in {city}: city not found."

        place = results[0]
        forecast = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
            },
            timeout=10,
        ).json()
        current = forecast["current"]
    except (requests.RequestException, KeyError) as e:
        return f"Weather in {city}: lookup failed ({e})."

    condition = _WMO_CODES.get(current["weather_code"], f"weather code {current['weather_code']}")
    return (
        f"Weather in {place['name']}, {place.get('country', '')}: "
        f"{current['temperature_2m']}C, {condition}, "
        f"{current['relative_humidity_2m']}% humidity, "
        f"wind {current['wind_speed_10m']} km/h"
    )


# real_search_tool: grounded in a FRAMES question's gold Wikipedia articles.
#
# a real "web_search" implementation needs a search index or a paid search
# API; what's built here instead is FRAMES-shaped: run_frames_batch.py
# pre-fetches one FRAMES question's real gold Wikipedia articles (via
# frames_data.py) and activates them here, then this tool returns whichever
# gold article best lexically overlaps the model's query. this is a
# deliberately simplified stand-in for real retrieval -- no embeddings, no
# distractor documents, no forced minimum tool-call counts, unlike
# IntentKV's full FRAMES adaptation (which built those specifically to
# stress-test a cache-pruning algorithm we aren't testing here).
#
# tasks with no active FRAMES corpus (e.g. the hiking/trail tasks in
# run_batch.py, which have no free real backing data source) fall back to
# fake_search_tool rather than returning nothing.
_active_search_corpus = None


def set_search_corpus(corpus: list[dict]) -> None:
    global _active_search_corpus
    _active_search_corpus = corpus


def clear_search_corpus() -> None:
    global _active_search_corpus
    _active_search_corpus = None


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "for", "to",
    "and", "or", "what", "who", "which", "that", "this", "its", "did", "do", "does",
}


def _tokenize(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def real_search_tool(query: str) -> str:
    if not _active_search_corpus:
        return fake_search_tool(query)

    query_tokens = _tokenize(query)
    scored = [
        (len(query_tokens & _tokenize(doc["title"] + " " + doc["text"][:500])), doc)
        for doc in _active_search_corpus
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_doc = scored[0][1]

    snippet = best_doc["text"][:1200]
    return f"Wikipedia: {best_doc['title']} ({best_doc['url']})\n{snippet}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information relevant to the query",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


def run_tool(name: str, tool_input: dict) -> str:
    if name == "get_weather":
        if USE_REAL_WEATHER:
            return real_weather_tool(tool_input["city"])
        return fake_weather_tool(tool_input["city"])
    if name == "web_search":
        if USE_REAL_SEARCH:
            return real_search_tool(tool_input["query"])
        return fake_search_tool(tool_input["query"])
    return "Unknown tool"


# logging piece

def log_step(
    step_index: int,
    event_type: str,
    content_snapshot: str,
    agent_id: str = "main",
    parent_id: str = None,
    extra: dict = None,
    trace_log_path: str = None,
):
    """
    appends one line to the trace file, recording:
    - step_index: which step in the loop this is
    - event_type: model_call, tool_call, or final_answer
    - content_snapshot: the full text context sent/used at this step, used later to check prefix overlap
    - timestamp: real wall clock time between each step
    - agent_id: which agent produced this event, defaults to "main" for single-agent traces
    - parent_id: agent_id of whoever spawned this agent, used later to compute graph depth and fan_out
    """
    record = {
        "step_index": step_index,
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "content_snapshot": content_snapshot,
        "content_length_chars": len(content_snapshot),
        "agent_id": agent_id,
        "parent_id": parent_id,
    }
    if extra:
        record.update(extra)

    path = trace_log_path or TRACE_LOG_PATH
    with _log_lock:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


# agent loop

def run_agent(
    user_task: str,
    max_steps: int = 6,
    agent_id: str = "main",
    parent_id: str = None,
    trace_log_path: str = None,
    upstream_context: str = None,
):
    """
    runs the tool-calling loop for one agent, returns its final answer text
    (or None if it hits max_steps without producing one)

    - agent_id, parent_id, trace_log_path let this loop be reused for
      sub-agents spawned by the orchestrator, not just the single
      top-level "main" agent
    - upstream_context, if given, is prepended to the task so a sub-agent
      can see the results of the sub-tasks it depends on
    """
    task_text = f"{upstream_context}\n\nNow: {user_task}" if upstream_context else user_task
    messages = [{"role": "user", "content": task_text}]
    step = 0

    while step < max_steps:
        context_snapshot = json.dumps(messages)
        response = call_model_with_retry(messages, TOOLS)
        # logged after the call (not before, like the old OpenRouter version) so the
        # real per-call cache-timing fields can ride along on the same trace row;
        # context_snapshot itself is still the prompt state as it was *before* this call
        log_step(
            step,
            "model_call",
            context_snapshot,
            agent_id=agent_id,
            parent_id=parent_id,
            trace_log_path=trace_log_path,
            extra=cache_metrics(response),
        )

        message = response.message
        tool_calls = message.tool_calls

        if not tool_calls:
            final_text = message.content or ""
            log_step(step, "final_answer", final_text, agent_id=agent_id, parent_id=parent_id, trace_log_path=trace_log_path)
            print(f"\n[{agent_id}] Final answer:\n{final_text}")
            return final_text

        # add the model's response (including its tool request) to history
        messages.append(message.model_dump())

        # run each requested tool and log it, then feed results back in
        for tc in tool_calls:
            tool_input = tc.function.arguments  # ollama gives already-parsed dict args
            log_step(
                step,
                "tool_call",
                json.dumps(tool_input),
                agent_id=agent_id,
                parent_id=parent_id,
                trace_log_path=trace_log_path,
                extra={"tool_name": tc.function.name},
            )
            result = run_tool(tc.function.name, tool_input)
            messages.append(
                {
                    "role": "tool",
                    "tool_name": tc.function.name,
                    "content": result,
                }
            )

        step += 1

    print(f"[{agent_id}] Hit max_steps without a final answer.")
    return None