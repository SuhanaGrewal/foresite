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
import threading
import time
from datetime import datetime, timezone

import ollama

MODEL = "qwen2.5:1.5b"


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
            "description": "Search the web for trail/location info",
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
        return fake_weather_tool(tool_input["city"])
    if name == "web_search":
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