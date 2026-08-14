"""
Feature Extraction

Converts traces/run_N.jsonl files into a labeled, per-item-touch feature
table for training a cache-eviction predictor.

- row = one item_id's occurrence at one point in one trace
- mirrors the granularity trace_to_events() already established: one
  touch per message inside a growing model_call snapshot, one touch per
  tool_call/final_answer, spawn events excluded entirely (bookkeeping,
  not real content a model ever processed)
- item_id() and trace_to_events() are imported from
  trace_agent_logs_converted.py so hashing stays consistent with the
  cache simulator, and this file's own row sequence can be cross-checked
  against the canonical one
- raw records are still read directly, since trace_to_events() drops
  per-record metadata (timestamp, agent_id, parent_id) this needs

Limitations, deliberately out of scope for now:

- within-agent execution-progress features (an "agent_step_fraction" of
  steps/max_steps, or raw counts like tool_calls_so_far,
  elapsed_seconds_since_agent_start, distinct_tools_used_so_far) are not
  built here, and shouldn't be added as standalone features without
  revisiting this first
- a leak-free version of "how far into its own execution is this
  specific sub-agent" needs a denominator to normalize against, and
  there are only two ways to get one: a future value from this same
  trace (leakage, not allowed), or a historical average from prior
  traces of the same task type (needs more collected trace volume than
  we have right now to be meaningful)
- this was tried before as fraction_of_trace_completed and removed for
  exactly this reason, see the comment in build_rows_for_trace()
- revisit once enough historical trace data exists to compute reliable
  per-task-type averages using only traces OTHER than the one being
  featurized

dag_completion_fraction (completed_subtasks / total_subtasks_in_plan) is
unaffected by this limitation and IS built: the DAG's total sub-task
count is known upfront from the plan itself, not from this trace's
actual future execution, so it needs no future value and no historical
average to compute.
"""

import glob
import json
from datetime import datetime

import pandas as pd
import tiktoken

from trace_agent_logs_converted import item_id, trace_to_events

# will_be_reused label: "does this item_id appear again within the next N
# touches, trace-wide". Trace-wide (not per-agent) because a real KV cache
# is shared context, not scoped to one agent.
REUSE_WINDOW_EVENTS = 5

# No official tiktoken encoding exists for the models this project runs
# (formerly Gemma via OpenRouter, now Qwen2.5 via local Ollama); cl100k_base
# is used as a comparable general-purpose tokenizer for a real
# (non-character-proxy) token count, per the "tiktoken or a comparable
# tokenizer" requirement.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _parse_ts(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str)


def _load_records(trace_path: str) -> list[dict]:
    """
    - reads every record from a trace file
    - sorts by timestamp defensively: append order should already be
      chronological (writes are lock-guarded in log_step), but each
      record's timestamp is computed just before that lock is acquired,
      so under a rare thread-scheduling race, file order could diverge
      slightly from true timestamp order
    - row order must be trustworthy, since every "since last seen" and
      label calculation depends on it
    """
    records = []
    with open(trace_path) as f:
        for line in f:
            records.append(json.loads(line))
    records.sort(key=lambda r: r["timestamp"])
    return records


def _annotate_records(records: list[dict], depends_on_map: dict) -> list[dict]:
    """
    - computes the record-level fields we derive ourselves: token_count,
      measured_latency_seconds, and dag_completion_fraction
    - all three computed once per record and reused across every touch
      exploded out of that record -- same scope as the already-logged
      content_length_chars, since all of these describe "how big/slow/
      far-along was the run at this step", not any one message's own size

    measured_latency_seconds:
    - backward-looking: the gap between this model_call and this SAME
      agent_id's previous logged record (any event type)
    - only computed for model_call records, since that's what "this
      model_call's timestamp" refers to
    - None for an agent's first record (nothing meaningful precedes it),
      and None for non-model_call records
    - a forward-looking (next-event) definition was ruled out, since it
      would read future information into a feature

    dag_completion_fraction:
    - completed_subtasks / total_subtasks_in_plan
    - only counts sub-agents that produced a final_answer in a STRICTLY
      EARLIER record than this one -- a record's own final_answer never
      counts toward its own fraction, same "never let a row see its own
      event" rule as everything else in this file
    - total_subtasks_in_plan excludes the synthesizer, since it isn't
      part of the planner's own DAG
    - legacy single-agent traces have no DAG at all
      (total_subtasks_in_plan == 0), so this is None there, not a
      divide by zero
    """
    plan_subtask_ids = {agent for agent in depends_on_map if agent != "synthesizer"}
    total_subtasks = len(plan_subtask_ids)
    completed_subtasks: set[str] = set()

    last_ts_by_agent: dict[str, str] = {}
    for record in records:
        if record["event_type"] == "spawn":
            continue

        record["token_count"] = len(_ENCODING.encode(record["content_snapshot"]))

        record["dag_completion_fraction"] = (
            len(completed_subtasks) / total_subtasks if total_subtasks > 0 else None
        )

        agent = record["agent_id"]
        if record["event_type"] == "model_call" and agent in last_ts_by_agent:
            record["measured_latency_seconds"] = (
                _parse_ts(record["timestamp"]) - _parse_ts(last_ts_by_agent[agent])
            ).total_seconds()
        else:
            record["measured_latency_seconds"] = None

        last_ts_by_agent[agent] = record["timestamp"]

        # mark completion AFTER computing this record's own fraction above,
        # so it only affects records that come after it
        if record["event_type"] == "final_answer" and agent in plan_subtask_ids:
            completed_subtasks.add(agent)

    return records


def _explode_touches(records: list[dict]) -> list[dict]:
    """
    - one entry per item_id occurrence, in trace order
    - carries the per-record metadata trace_to_events() throws away
    """
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
                    "dag_completion_fraction": record["dag_completion_fraction"],
                }
            )
    return touches


def _compute_labels(touches: list[dict], window: int = REUSE_WINDOW_EVENTS) -> list[int]:
    """
    - will_be_reused[i] = 1 if touches[i]'s item_id appears again anywhere
      in touches[i+1 : i+1+window]
    - this is the ONLY place future positions are read, kept isolated
      from every feature-computing function below
    """
    labels = [0] * len(touches)
    for i, t in enumerate(touches):
        future_ids = {touches[j]["item_id"] for j in range(i + 1, min(i + 1 + window, len(touches)))}
        labels[i] = 1 if t["item_id"] in future_ids else 0
    return labels


def _strip_agent_prefix(agent_id: str) -> str:
    """
    - depends_on lists use raw sub-task ids (e.g. "weather_tokyo")
    - sub-agents' own agent_id carries a "subagent_" prefix
    - strips it so the two vocabularies can be compared directly
    """
    prefix = "subagent_"
    return agent_id[len(prefix):] if agent_id.startswith(prefix) else agent_id


def _build_graph_maps(records: list[dict]) -> tuple[dict, dict]:
    """
    - reads from the RAW records (spawn events included, read directly
      per the requirement, not through the item-touch pipeline)
    - parent_map: agent_id -> parent_id (same for every record of that agent)
    - depends_on_map: agent_id -> raw dependency ids, from that agent's
      own spawn record only (legacy single-agent traces have none)
    """
    parent_map: dict[str, str] = {}
    depends_on_map: dict[str, list] = {}
    for record in records:
        agent = record["agent_id"]
        if agent not in parent_map:
            parent_map[agent] = record.get("parent_id")
        if record["event_type"] == "spawn":
            depends_on_map[agent] = record.get("depends_on", [])
    return parent_map, depends_on_map


def _compute_agent_depth(agent_id: str, parent_map: dict) -> int:
    """
    - hops back to a root, walking parent_id generically rather than
      hardcoding the string "planner"
    - the planner's own decomposition call is never itself logged as a
      row today, so any parent_id that isn't itself a logged agent_id is
      treated as one implicit hop to an external root
    - gives depth 0 to legacy "main" traces (parent_id=None) and depth 1
      to today's sub-agents/synthesizer (parent_id="planner",
      unresolvable), while still supporting real multi-level chains if
      the schema ever grows one
    """
    depth = 0
    current = agent_id
    seen = set()
    while True:
        parent = parent_map.get(current)
        if parent is None:
            return depth
        if parent in seen or parent not in parent_map:
            return depth + 1
        seen.add(current)
        depth += 1
        current = parent


def _compute_sibling_count(parent_map: dict) -> dict:
    """
    - sibling_count[agent_id] = how many OTHER agents share this agent's
      parent_id -- a direct, upfront-known signal for how wide a
      concurrent fan-out this agent was spawned into. Many siblings
      (10+) means a sweep-shaped burst of mostly one-off sub-agents;
      a handful means an ordinary 2-5-subtask plan; zero means a legacy
      single-agent trace with no concurrent structure at all.
    - added specifically because event_type_model_call (a structural
      artifact -- see trace_agent.py's module docstring on
      trace_to_events()) was found to dominate the trained model's
      feature importance without transferring to sweep-shaped traces,
      where sub-agents are short-lived and don't accumulate much
      growing context. sibling_count gives the model a DIRECT signal
      for "this looks like a sweep touch" instead of making it infer
      that indirectly through a feature that doesn't actually carry
      that meaning.
    - known entirely from the plan's own spawn structure (parent_map,
      built from every record including spawn events, before any
      touches are exploded out), not from anything about how the trace
      unfolds afterward -- same "known upfront from the plan itself"
      reasoning agent_depth/fan_out already rely on, so like those two,
      this needs no backward-looking leakage check (assert_no_leakage
      already exempts agent_depth/fan_out/is_dependency_of_sink on
      exactly this basis).
    """
    by_parent: dict[str, list] = {}
    for agent, parent in parent_map.items():
        by_parent.setdefault(parent, []).append(agent)

    return {agent: len(by_parent[parent]) - 1 for agent, parent in parent_map.items()}


def _compute_fan_out_and_sink_deps(depends_on_map: dict) -> tuple[dict, set]:
    """
    - fan_out[raw_id] = how many other agents' depends_on lists name raw_id
    - sink ids = the synthesizer's own depends_on, literally the DAG's
      terminal sub-tasks, whatever the plan's shape
    - dependency_of_sink = union of the sink agents' own depends_on
      lists, i.e. "is this row a prerequisite of something that feeds
      the final answer"
    """
    fan_out: dict[str, int] = {}
    for deps in depends_on_map.values():
        for d in deps:
            fan_out[d] = fan_out.get(d, 0) + 1

    sink_ids = set(depends_on_map.get("synthesizer", []))
    dependency_of_sink = set()
    for agent, deps in depends_on_map.items():
        if _strip_agent_prefix(agent) in sink_ids:
            dependency_of_sink.update(deps)

    return fan_out, dependency_of_sink


_EVENT_TYPES = ("model_call", "tool_call", "final_answer")


def build_rows_for_trace(trace_path: str, window: int = REUSE_WINDOW_EVENTS) -> list[dict]:
    records = _load_records(trace_path)

    # graph structure comes from the raw records directly, spawn events
    # included, before those get filtered out of the item-touch pipeline
    parent_map, depends_on_map = _build_graph_maps(records)
    fan_out_map, dependency_of_sink = _compute_fan_out_and_sink_deps(depends_on_map)
    sibling_count_map = _compute_sibling_count(parent_map)

    records = _annotate_records(records, depends_on_map)
    touches = _explode_touches(records)
    labels = _compute_labels(touches, window)

    trace_id = trace_path.split("/")[-1].rsplit(".", 1)[0]

    # backward-only running state: every dict here is read for row i then
    # updated with row i's own values, so row i can never see itself or
    # anything after it.
    last_seen_index: dict[str, int] = {}
    last_seen_ts: dict[str, str] = {}

    # a within-agent execution-progress feature (fraction_of_trace_completed
    # = agent_seen_so_far / agent's eventual total touch count) used to be
    # computed here and was deliberately removed. its denominator was that
    # agent's FINAL touch count, only knowable once the agent's run had
    # actually finished, i.e. a future value from this same trace. the only
    # leak-free way to normalize "how far along is this sub-agent" is a
    # historical average from other traces of the same task type, which
    # needs more collected trace volume than we have right now to be
    # meaningful. see the module docstring's limitations section.

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

        raw_agent = _strip_agent_prefix(agent)

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
            "measured_latency_seconds": t["measured_latency_seconds"],
            "dag_completion_fraction": t["dag_completion_fraction"],
            "agent_depth": _compute_agent_depth(agent, parent_map),
            "fan_out": fan_out_map.get(raw_agent, 0),
            "sibling_count": sibling_count_map.get(agent, 0),
            "is_dependency_of_sink": raw_agent in dependency_of_sink,
            "will_be_reused": labels[i],
        }
        for et in _EVENT_TYPES:
            row[f"event_type_{et}"] = 1 if t["event_type"] == et else 0
        rows.append(row)

        # update AFTER computing this row, so this occurrence never feeds
        # its own "since last seen" calculation
        last_seen_index[iid] = i
        last_seen_ts[iid] = t["timestamp"]

    # correctness cross-check: our row sequence's item_ids, in order, must
    # exactly match the canonical trace_to_events() output for this file
    expected = trace_to_events(trace_path)
    actual = [r["item_id"] for r in rows]
    assert actual == expected, (
        f"{trace_path}: row item_id sequence diverges from trace_to_events() "
        f"at position {next(i for i in range(min(len(actual), len(expected))) if actual[i] != expected[i]) if actual != expected else 'length mismatch'}"
    )

    return rows, touches, depends_on_map


# leakage check

def _independent_recompute(touches: list[dict], i: int, depends_on_map: dict) -> dict:
    """
    - recomputes row i's backward-looking features completely
      independently, using ONLY touches[:i]
    - a hard slice cutoff that makes it structurally impossible for this
      recomputation to see row i or anything later
    - used to verify the running-state implementation in
      build_rows_for_trace against a second, independently-written path,
      not just re-derive the same code
    """
    t = touches[i]
    earlier = touches[:i]

    same_item_positions = [j for j, e in enumerate(earlier) if e["item_id"] == t["item_id"]]
    if same_item_positions:
        last_j = same_item_positions[-1]
        steps = i - last_j
        seconds = (_parse_ts(t["timestamp"]) - _parse_ts(earlier[last_j]["timestamp"])).total_seconds()
    else:
        steps, seconds = -1, -1.0

    plan_subtask_ids = {agent for agent in depends_on_map if agent != "synthesizer"}
    total_subtasks = len(plan_subtask_ids)
    if total_subtasks > 0:
        completed = {
            e["agent_id"]
            for e in earlier
            if e["event_type"] == "final_answer" and e["agent_id"] in plan_subtask_ids
        }
        dag_completion_fraction = len(completed) / total_subtasks
    else:
        dag_completion_fraction = None

    return {
        "steps_since_last_seen": steps,
        "seconds_since_last_seen": seconds,
        "dag_completion_fraction": dag_completion_fraction,
    }


def assert_no_leakage(rows: list[dict], touches: list[dict], depends_on_map: dict) -> None:
    """
    - fails loudly if any backward-looking feature doesn't match an
      independent recomputation that structurally cannot see the future
      (touches[:i])
    - also fails if measured_latency_seconds is ever negative, which
      could only happen if it were measured against a later event
    """
    for i, row in enumerate(rows):
        expected = _independent_recompute(touches, i, depends_on_map)
        for key, expected_value in expected.items():
            actual_value = row[key]
            if expected_value is None:
                assert actual_value is None, (
                    f"leakage check failed at row {i} ({key}): expected None, got {actual_value}"
                )
            elif isinstance(expected_value, float):
                assert abs(actual_value - expected_value) < 1e-6, (
                    f"leakage check failed at row {i} ({key}): expected {expected_value}, got {actual_value}"
                )
            else:
                assert actual_value == expected_value, (
                    f"leakage check failed at row {i} ({key}): expected {expected_value}, got {actual_value}"
                )

        if row["measured_latency_seconds"] is not None:
            assert row["measured_latency_seconds"] >= 0, (
                f"leakage check failed at row {i}: negative measured_latency_seconds "
                f"means it was computed against a LATER event, not an earlier one"
            )


# multi-trace assembly

def build_feature_table(trace_paths: list[str], window: int = REUSE_WINDOW_EVENTS) -> pd.DataFrame:
    all_rows = []
    for trace_path in trace_paths:
        rows, touches, depends_on_map = build_rows_for_trace(trace_path, window)
        assert_no_leakage(rows, touches, depends_on_map)
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    # excludes traces/run_sweep_*.jsonl and traces/run_sweep_holdout_*.jsonl:
    # both match the naive "run_*.jsonl" glob too, and folding either into
    # the general features.csv would contaminate it -- the sweep traces are
    # a deliberately separate stress-test dataset (see build_sweep_features.py
    # and build_sweep_holdout_features.py), kept out of the general dataset
    # on purpose. this glob previously WAS naive ("run_*.jsonl") and got
    # caught only because build_sweep_features.py's identical bug was
    # caught first (a rebuild there reported 20 traces instead of 10) --
    # fixed here defensively before it could do the same silently to
    # features.csv, which has no similar row-count sanity check today.
    trace_paths = sorted(p for p in glob.glob("traces/run_*.jsonl") if "sweep" not in p)

    df = build_feature_table(trace_paths)
    df.to_csv("features.csv", index=False)

    reused_pct = df["will_be_reused"].mean() * 100
    print(f"Traces processed: {len(trace_paths)}")
    print(f"Distinct traces in table: {df['trace_id'].nunique()}")
    print(f"Row count: {len(df)}")
    print(f"Class balance -- will_be_reused=1: {reused_pct:.1f}%  will_be_reused=0: {100 - reused_pct:.1f}%")
    print(f"Leakage check: PASSED for all {len(trace_paths)} traces")
    print("Saved to features.csv")