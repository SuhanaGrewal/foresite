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

import tiktoken

from trace_agent_logs_converted import item_id, trace_to_events

# will_be_reused label: "does this item_id appear again within the next N
# touches, trace-wide". Trace-wide (not per-agent) because a real KV cache
# is shared context, not scoped to one agent.
REUSE_WINDOW_EVENTS = 5

# No official tiktoken encoding exists for Gemma; cl100k_base is used as a
# comparable general-purpose tokenizer for a real (non-character-proxy)
# token count, per the "tiktoken or a comparable tokenizer" requirement.
_ENCODING = tiktoken.get_encoding("cl100k_base")


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


def _annotate_records(records: list[dict]) -> list[dict]:
    """Compute the record-level fields we derive ourselves: token_count and
    measured_latency_seconds. Both are computed once per record and reused
    across every touch exploded out of that record -- same scope as the
    already-logged content_length_chars, since all of these describe "how
    big/slow was the context at this step", not any one message's own size.

    measured_latency_seconds is backward-looking: the gap between this
    model_call and this SAME agent_id's previous logged record (any event
    type). Only computed for model_call records, since that's what "this
    model_call's timestamp" refers to; None for an agent's first record,
    since nothing meaningful precedes it, and None for non-model_call
    records. A forward-looking (next-event) definition was ruled out because
    it would read future information into a feature.
    """
    last_ts_by_agent: dict[str, str] = {}
    for record in records:
        if record["event_type"] == "spawn":
            continue

        record["token_count"] = len(_ENCODING.encode(record["content_snapshot"]))

        agent = record["agent_id"]
        if record["event_type"] == "model_call" and agent in last_ts_by_agent:
            record["measured_latency_seconds"] = (
                _parse_ts(record["timestamp"]) - _parse_ts(last_ts_by_agent[agent])
            ).total_seconds()
        else:
            record["measured_latency_seconds"] = None

        last_ts_by_agent[agent] = record["timestamp"]

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
                    "token_count": record["token_count"],
                    "measured_latency_seconds": record["measured_latency_seconds"],
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


_EVENT_TYPES = ("model_call", "tool_call", "final_answer")


def build_rows_for_trace(trace_path: str, window: int = REUSE_WINDOW_EVENTS) -> list[dict]:
    records = _load_records(trace_path)
    records = _annotate_records(records)
    touches = _explode_touches(records)
    labels = _compute_labels(touches, window)

    trace_id = trace_path.split("/")[-1].rsplit(".", 1)[0]

    # backward-only running state: every dict here is read for row i then
    # updated with row i's own values, so row i can never see itself or
    # anything after it.
    last_seen_index: dict[str, int] = {}
    last_seen_ts: dict[str, str] = {}
    agent_touch_counts = {}
    for t in touches:
        agent_touch_counts[t["agent_id"]] = agent_touch_counts.get(t["agent_id"], 0) + 1
    agent_seen_so_far: dict[str, int] = {}

    rows = []
    for i, t in enumerate(touches):
        iid = t["item_id"]
        agent = t["agent_id"]

        if iid in last_seen_index:
            steps_since_last_seen = i - last_seen_index[iid]
            seconds_since_last_seen = (
                _parse_ts(t["timestamp"]) - _parse_ts(last_seen_ts[iid])
            ).total_seconds()
        else:
            steps_since_last_seen = -1
            seconds_since_last_seen = -1.0

        agent_seen_so_far[agent] = agent_seen_so_far.get(agent, 0) + 1
        # this agent's own progress through its own run, not the whole trace
        fraction_of_trace_completed = agent_seen_so_far[agent] / agent_touch_counts[agent]

        row = {
            "trace_id": trace_id,
            "row_index": i,
            "item_id": iid,
            "agent_id": agent,
            "parent_id": t["parent_id"],
            "event_type": t["event_type"],
            "timestamp": t["timestamp"],
            "content_length_chars": t["content_length_chars"],
            "token_count": t["token_count"],
            "steps_since_last_seen": steps_since_last_seen,
            "seconds_since_last_seen": seconds_since_last_seen,
            "fraction_of_trace_completed": fraction_of_trace_completed,
            "measured_latency_seconds": t["measured_latency_seconds"],
            "will_be_reused": labels[i],
        }
        for et in _EVENT_TYPES:
            row[f"event_type_{et}"] = 1 if t["event_type"] == et else 0
        rows.append(row)

        # update AFTER computing this row, so this occurrence never feeds
        # its own "since last seen" calculation
        last_seen_index[iid] = i
        last_seen_ts[iid] = t["timestamp"]

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
