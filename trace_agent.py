"""
Step 3: A minimal toy agent that calls a model through OpenRouter, uses a
couple of fake tools, and logs every step to a trace file.

This is intentionally simple -- the point isn't a "good" agent, it's a
REAL log of agent behavior we can later feed into the cache simulator
instead of made-up random data.

Requires: pip install openai python-dotenv
Requires: OPENROUTER_API_KEY set in a .env file in this folder
"""

import json
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()  # reads variables from a .env file in this folder, if present

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

MODEL = "gemma-4-26b-a4b-it:free"


def call_model_with_retry(messages, tools, max_retries=4):
    """Retry the API call with backoff if rate limited or response is empty."""
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )
        if response.choices is not None:
            return response
        wait_seconds = 5 * (attempt + 1)
        print(f"  [rate limited or empty response, retrying in {wait_seconds}s...]")
        time.sleep(wait_seconds)
    raise RuntimeError("Model call failed after multiple retries -- likely rate limited.")

TRACE_LOG_PATH = "traces.jsonl"


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
):
    """
    Append one line to the trace file. Each line records:
    - step_index: which step in the loop this is
    - event_type: 'model_call', 'tool_call', 'final_answer', etc
    - content_snapshot: the FULL text context that was sent/used at this
      step -- this is what we'll later check for prefix overlap
    - timestamp: real wall-clock time, so we can measure gaps between steps
    - agent_id: which agent produced this event (e.g. "planner",
      "subagent_1"). Defaults to "main" for single-agent traces.
    - parent_id: the agent_id of whoever spawned this agent. None for the
      root/planner agent. Together with agent_id, this lets us later
      compute graph depth (hops back to a None parent) and fan_out
      (how many events list this agent_id as their parent).
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

    with open(TRACE_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


# agent loop

def run_agent(user_task: str, max_steps: int = 6):
    messages = [{"role": "user", "content": user_task}]
    step = 0

    while step < max_steps:
        # snapshot exactly what we're about to send -- this full messages
        # history is the "context" that a real KV cache would need to
        # process. Logging it lets us later check how much of it repeats
        # from one step to the next.
        context_snapshot = json.dumps(messages)
        log_step(step, "model_call", context_snapshot)

        response = call_model_with_retry(messages, TOOLS)

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls

        if not tool_calls:
            final_text = choice.message.content or ""
            log_step(step, "final_answer", final_text)
            print(f"\nFinal answer:\n{final_text}")
            return

        # add the model's response (including its tool request) to history
        messages.append(choice.message.model_dump())

        # run each requested tool and log it, then feed results back in
        for tc in tool_calls:
            tool_input = json.loads(tc.function.arguments)
            log_step(
                step,
                "tool_call",
                json.dumps(tool_input),
                extra={"tool_name": tc.function.name},
            )
            result = run_tool(tc.function.name, tool_input)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

        step += 1

    print("Hit max_steps without a final answer.")


BATCH_TASKS = [
    "Compare the weather in Delhi, Los Angeles, and Tokyo right now, and recommend which city is best for outdoor sightseeing today.",
    "Check trail conditions and weather for both Runyon Canyon and Griffith Park, then tell me which is the better hike this afternoon.",
    "I'm deciding between a beach day and a hiking day this weekend. Check the weather for both Santa Monica and Runyon Canyon and recommend one.",
    "Plan a 2-day trip: check the weather for Saturday and Sunday, search for one indoor and one outdoor activity, and write a short itinerary.",
    "Compare hiking conditions across Runyon Canyon, Griffith Park, and Topanga State Park, and rank them best to worst for today.",
]

if __name__ == "__main__":
    os.makedirs("traces", exist_ok=True)

    for i, task in enumerate(BATCH_TASKS, start=1):
        TRACE_LOG_PATH = f"traces/run_{i}.jsonl"
        if os.path.exists(TRACE_LOG_PATH):
            os.remove(TRACE_LOG_PATH)

        print(f"\n=== Running task {i}/{len(BATCH_TASKS)}: {task}")
        run_agent(task)
        print(f"Trace written to {TRACE_LOG_PATH}")
        if i < len(BATCH_TASKS):
            time.sleep(10)  # brief pause between tasks to respect rate limits
