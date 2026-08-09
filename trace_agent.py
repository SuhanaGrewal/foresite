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
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()  # reads variables from a .env file in this folder, if present

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

MODEL = "google/gemma-4-26b-a4b-it:free" 

TRACE_LOG_PATH = "traces.jsonl" 


# return pretend data

def fake_weather_tool(city: str) -> str:
    return f"Weather in {city}: 22C, sunny, light breeze."


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

def log_step(step_index: int, event_type: str, content_snapshot: str, extra: dict = None):
    """
    Append one line to the trace file. Each line records:
    - step_index: which step in the loop this is
    - event_type: 'model_call', 'tool_call', 'final_answer', etc
    - content_snapshot: the FULL text context that was sent/used at this
      step -- this is what we'll later check for prefix overlap
    - timestamp: real wall-clock time, so we can measure gaps between steps
    """
    record = {
        "step_index": step_index,
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "content_snapshot": content_snapshot,
        "content_length_chars": len(content_snapshot),
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

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

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


if __name__ == "__main__":
    # clear old trace file so each run starts fresh
    if os.path.exists(TRACE_LOG_PATH):
        os.remove(TRACE_LOG_PATH)

    run_agent("Find out if it's a good day to hike Runyon Canyon and tell me what to bring.")

    print(f"\nTrace written to {TRACE_LOG_PATH} -- open it to see the raw log.")
