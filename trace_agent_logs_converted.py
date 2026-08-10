"""
Trace to Events Converter

Turns a real trace file (traces/run_N.jsonl) into a flat, ordered list of
"item touches", the format both the cache simulator and the
feature-extraction code expect.

- item touch: an entry marking a point when the agent accessed a piece of content
- the full sequence, read in order, shows what got accessed and when
"""

import hashlib
import json
import sys


def item_id(content: str) -> str:
    """
    - hashes text into a short, stable ID
    - identical text always hashes to the same ID
    - this is how the simulator recognizes reused content, not new content
    """
    return hashlib.md5(content.encode()).hexdigest()[:10]


def trace_to_events(trace_path: str) -> list:
    """
    - reads a trace file, returns an ordered list of item IDs
    - one entry per content chunk touched
    - model_call steps carry the full, growing message history, so each
      one gets broken into individual messages
    - if an individual message reappears later it gets the same ID both times
    """
    events = []

    with open(trace_path) as f:
        for line in f:
            record = json.loads(line)
            event_type = record["event_type"]

            if event_type == "model_call":
                try:
                    messages = json.loads(record["content_snapshot"])
                except json.JSONDecodeError:
                    messages = [record["content_snapshot"]]

                for msg in messages:
                    msg_text = json.dumps(msg, sort_keys=True)
                    events.append(item_id(msg_text))

            else:
                # tool_call / final_answer, treat as one item each
                events.append(item_id(record["content_snapshot"]))

    return events


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 trace_to_events.py traces/run_1.jsonl")
    else:
        events = trace_to_events(sys.argv[1])
        print(f"Total events: {len(events)}")
        print(f"Distinct items: {len(set(events))}")
        print(events)
