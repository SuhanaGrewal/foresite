"""
Feature Extraction

Converts traces/run_N.jsonl files into a labeled, per-item-touch feature
table for training a cache-eviction predictor.

Row = one item_id's occurrence at one point in one trace. This mirrors the
granularity trace_to_events() already established (one touch per message
inside a growing model_call snapshot, one touch per tool_call/final_answer,
spawn events excluded entirely since they're orchestrator bookkeeping, not
real content a model ever processed). item_id() and trace_to_events() are
imported from trace_agent_logs_converted.py so hashing stays consistent
with the cache simulator, and so this file's own row sequence can be
cross-checked against the canonical one. Raw records are still read
directly, since trace_to_events() drops per-record metadata (timestamp,
agent_id, parent_id) this needs.
"""

import json
from datetime import datetime

from trace_agent_logs_converted import item_id, trace_to_events

# will_be_reused label: "does this item_id appear again within the next N
# touches, trace-wide". Trace-wide (not per-agent) because a real KV cache
# is shared context, not scoped to one agent.
REUSE_WINDOW_EVENTS = 5


def _parse_ts(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str)


def _load_records(trace_path: str) -> list[dict]:
    records = []
    with open(trace_path) as f:
        for line in f:
            records.append(json.loads(line))
    # Append order should already be chronological (writes are lock-guarded
    # in log_step), but each record's timestamp is computed just before that
    # lock is acquired, so under a rare thread-scheduling race, file order
    # could diverge slightly from true timestamp order. Sort defensively --
    # row order must be trustworthy since every "since last seen" and label
    # calculation depends on it.
    records.sort(key=lambda r: r["timestamp"])
    return records


def _explode_touches(records: list[dict]) -> list[dict]:
    """One entry per item_id occurrence, in trace order, carrying the
    per-record metadata trace_to_events() throws away."""
    touches = []
    for record in records:
        event_type = record["event_type"]
        if event_type == "spawn":
            continue

        if event_type == "model_call":
            try:
                messages = json.loads(record["content_snapshot"])
            except json.JSONDecodeError:
                messages = [record["content_snapshot"]]
            item_ids = [item_id(json.dumps(m, sort_keys=True)) for m in messages]
        else:
            item_ids = [item_id(record["content_snapshot"])]

        for iid in item_ids:
            touches.append(
                {
                    "item_id": iid,
                    "event_type": event_type,
                    "timestamp": record["timestamp"],
                    "agent_id": record["agent_id"],
                    "parent_id": record.get("parent_id"),
                    "content_length_chars": record["content_length_chars"],
                }
            )
    return touches


def _compute_labels(touches: list[dict], window: int = REUSE_WINDOW_EVENTS) -> list[int]:
    """will_be_reused[i] = 1 if touches[i]'s item_id appears again anywhere
    in touches[i+1 : i+1+window]. This is the ONLY place future positions
    are read -- kept isolated from every feature-computing function below."""
    labels = [0] * len(touches)
    for i, t in enumerate(touches):
        future_ids = {touches[j]["item_id"] for j in range(i + 1, min(i + 1 + window, len(touches)))}
        labels[i] = 1 if t["item_id"] in future_ids else 0
    return labels


def build_rows_for_trace(trace_path: str, window: int = REUSE_WINDOW_EVENTS) -> list[dict]:
    records = _load_records(trace_path)
    touches = _explode_touches(records)
    labels = _compute_labels(touches, window)

    trace_id = trace_path.split("/")[-1].rsplit(".", 1)[0]

    rows = []
    for i, t in enumerate(touches):
        rows.append(
            {
                "trace_id": trace_id,
                "row_index": i,
                "item_id": t["item_id"],
                "agent_id": t["agent_id"],
                "parent_id": t["parent_id"],
                "event_type": t["event_type"],
                "timestamp": t["timestamp"],
                "content_length_chars": t["content_length_chars"],
                "will_be_reused": labels[i],
            }
        )

    # Correctness cross-check: our row sequence's item_ids, in order, must
    # exactly match the canonical trace_to_events() output for this file.
    expected = trace_to_events(trace_path)
    actual = [r["item_id"] for r in rows]
    assert actual == expected, (
        f"{trace_path}: row item_id sequence diverges from trace_to_events() "
        f"at position {next(i for i in range(min(len(actual), len(expected))) if actual[i] != expected[i]) if actual != expected else 'length mismatch'}"
    )

    return rows


if __name__ == "__main__":
    import sys

    trace_path = sys.argv[1] if len(sys.argv) > 1 else "traces/run_6.jsonl"
    rows = build_rows_for_trace(trace_path)

    print(f"Trace: {trace_path}")
    print(f"Rows generated: {len(rows)}")
    print(f"Cross-check against trace_to_events(): PASSED")
    reused = sum(r["will_be_reused"] for r in rows)
    print(f"will_be_reused=1: {reused} ({reused / len(rows):.1%})")
    print(f"will_be_reused=0: {len(rows) - reused} ({(len(rows) - reused) / len(rows):.1%})")
    print("\nFirst 8 rows:")
    for r in rows[:8]:
        print(r)
