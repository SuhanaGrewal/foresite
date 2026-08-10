"""
Converts the real traces (run_N.jsonl file) into a flat event sequence (a list
of "item touches*"), which is the format both the cache simulator and feature-extraction code need.

*item touches: an ordered list showing which piece of content the agent accessed at a specific point.
"""

import hashlib
import json
import sys


def item_id(content: str) -> str:
    """
    Turn a piece of text into a short ID; 2 identical pieces of text produce the same ID to achieve prefix overlap.
    """
    return hashlib.md5(content.encode()).hexdigest()[:10]


def trace_to_events(trace_path: str) -> list:
    """
    - Read trace file and produce a flat, ordered list of item IDs; 1 entry per distinct piece of content touched.
    - model_call steps: break apart growing message history into individual messages; repeated messages show up as repeated IDs.
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
                # tool_call / final_answer -- treat as one item each
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
