"""
Run Eviction Benchmark

Final evaluation (Step 6): benchmarks the trained predictor as a real
cache-eviction policy (run_predictor, kv-cache-simulator.py) against
LRU, LFU, and Belady's optimal, on real held-out trace data -- not a
fake generated event sequence.

Step 2 of this build: reuses train_predictor.py's own trace-level
train/test split (same TRACES_CSV_PATH, same RANDOM_SEED -- so this is
exactly the 52 traces the trained model in model.joblib never saw
during training or cross-validation, a genuine held-out set) and
converts each test trace to its event sequence via trace_to_events()
(trace_agent_logs_converted.py), then concatenates them into one
combined sequence.

Concatenating (rather than evaluating each trace separately) simulates
shared-cache contention: multiple different tasks' traces competing for
the same fixed-size cache over time, the way a real serving system would
see many concurrent/sequential requests sharing one KV cache -- not each
task getting its own isolated cache.
"""

from train_predictor import load_features, split_by_trace
from trace_agent_logs_converted import trace_to_events

FEATURES_CSV_PATH = "features.csv"
TRACES_DIR = "traces"


def build_combined_test_sequence(features_csv_path: str = FEATURES_CSV_PATH):
    """
    returns (combined_events, test_ids): the concatenated event sequence
    across every held-out test trace (in sorted trace_id order, for
    reproducibility), and the sorted list of test trace_ids used.
    """
    df = load_features(features_csv_path)
    _, _, _, test_ids = split_by_trace(df)

    combined_events = []
    for trace_id in test_ids:
        trace_path = f"{TRACES_DIR}/{trace_id}.jsonl"
        combined_events.extend(trace_to_events(trace_path))

    return combined_events, test_ids


if __name__ == "__main__":
    combined_events, test_ids = build_combined_test_sequence()
    print(f"held-out test traces: {len(test_ids)}")
    print(f"combined event sequence length: {len(combined_events)}")
    print(f"distinct items in combined sequence: {len(set(combined_events))}")
    print(f"test trace_ids: {test_ids}")
