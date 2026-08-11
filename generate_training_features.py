"""
Generate Fake Features

Synthesizes a CSV with the exact schema feature_extraction.py produces,
so train_predictor.py can be built and exercised end-to-end before real
trace data is ready. This is NOT a data model of anything -- it exists
purely to prove the training pipeline runs correctly. Swap in a real
features.csv later; train_predictor.py's code does not change, only
which file is passed via --input.

Writes to features_fake.csv (never touches the real features.csv).
"""

import numpy as np
import pandas as pd

SEED = 42
OUTPUT_PATH = "features_fake.csv"

# trace_id -> (n_rows, is_orchestrator)
# Mirrors the real data's shape: mostly legacy single-agent traces plus one
# orchestrator-style trace with populated DAG/graph features.
TRACES = [
    ("fake_run_1", 9, False),
    ("fake_run_2", 7, False),
    ("fake_run_3", 6, False),
    ("fake_run_4", 10, False),
    ("fake_run_5", 5, False),
    ("fake_run_6", 20, True),
]

AGENT_IDS_LEGACY = ["main"]
AGENT_IDS_ORCH = ["main", "subagent_a", "subagent_b", "synthesizer"]
EVENT_TYPES = ["model_call", "tool_call", "final_answer"]
EVENT_TYPE_WEIGHTS = [0.55, 0.35, 0.10]


def make_trace_rows(rng, trace_id, n_rows, is_orchestrator):
    agent_ids = AGENT_IDS_ORCH if is_orchestrator else AGENT_IDS_LEGACY
    rows = []
    start = pd.Timestamp("2026-08-10T06:00:00Z")
    for row_index in range(n_rows):
        event_type = rng.choice(EVENT_TYPES, p=EVENT_TYPE_WEIGHTS)
        agent_id = rng.choice(agent_ids)
        parent_id = "" if agent_id == "main" else "main"

        steps_since_last_seen = -1 if rng.random() < 0.6 else int(rng.integers(1, 6))
        seconds_since_last_seen = -1.0 if steps_since_last_seen == -1 else float(rng.uniform(0.5, 30))
        measured_latency_seconds = (
            float(rng.uniform(0.0001, 2.0)) if event_type == "model_call" and row_index > 0 else None
        )
        dag_completion_fraction = (
            round(float(rng.uniform(0.0, 1.0)), 3) if is_orchestrator and row_index > 2 else None
        )
        agent_depth = int(rng.integers(0, 3)) if is_orchestrator else 0
        fan_out = int(rng.integers(0, 3)) if is_orchestrator else 0
        is_dependency_of_sink = bool(rng.random() < 0.2) if is_orchestrator else False

        # imbalanced positive label, ~15% base rate
        will_be_reused = int(rng.random() < 0.15)

        rows.append(
            {
                "trace_id": trace_id,
                "row_index": row_index,
                "item_id": f"{trace_id}_item{row_index}_{rng.integers(0, 1_000_000):06d}",
                "agent_id": agent_id,
                "parent_id": parent_id,
                "event_type": event_type,
                "timestamp": (start + pd.Timedelta(seconds=row_index * 12)).isoformat(),
                "content_length_chars": int(rng.integers(20, 2000)),
                "token_count": int(rng.integers(5, 500)),
                "steps_since_last_seen": steps_since_last_seen,
                "seconds_since_last_seen": seconds_since_last_seen,
                "measured_latency_seconds": measured_latency_seconds,
                "dag_completion_fraction": dag_completion_fraction,
                "agent_depth": agent_depth,
                "fan_out": fan_out,
                "is_dependency_of_sink": is_dependency_of_sink,
                "will_be_reused": will_be_reused,
                "event_type_model_call": int(event_type == "model_call"),
                "event_type_tool_call": int(event_type == "tool_call"),
                "event_type_final_answer": int(event_type == "final_answer"),
            }
        )
    return rows


def main():
    rng = np.random.default_rng(SEED)
    all_rows = []
    for trace_id, n_rows, is_orchestrator in TRACES:
        all_rows.extend(make_trace_rows(rng, trace_id, n_rows, is_orchestrator))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"wrote {len(df)} fake rows across {df['trace_id'].nunique()} fake traces to {OUTPUT_PATH}")
    print(df.groupby("trace_id").agg(rows=("row_index", "count"), positives=("will_be_reused", "sum")))
    print(f"overall positive rate: {df['will_be_reused'].mean():.3f}")


if __name__ == "__main__":
    main()
