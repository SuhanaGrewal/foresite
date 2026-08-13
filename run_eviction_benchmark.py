"""
Run Eviction Benchmark

Final evaluation (Step 6): benchmarks the trained predictor as a real
cache-eviction policy (run_predictor_dynamic, kv-cache-simulator.py)
against LRU, LFU, and Belady's optimal, on real held-out trace data --
not a fake generated event sequence.

Reuses train_predictor.py's own trace-level train/test split (same
features.csv, same RANDOM_SEED -- so this is exactly the 52 traces the
trained model in model.joblib never saw during training or
cross-validation, a genuine held-out set), and builds one combined,
per-touch sequence across all of them, in sorted trace_id then row_index
order -- built directly from features.csv (not trace_to_events()), so
every eviction policy benchmarked sees the exact identical sequence by
construction. (feature_extraction.py asserts, every time it builds
features.csv, that each trace's row order already matches
trace_to_events() exactly, so this is a safe simplification, not an
assumption taken on faith.)

Concatenating (rather than evaluating each trace separately) simulates
shared-cache contention: multiple different tasks' traces competing for
the same fixed-size cache over time, the way a real serving system would
see many concurrent/sequential requests sharing one KV cache -- not each
task getting its own isolated cache.

The predictor policy re-scores every item's predicted reuse probability
on every touch, using live-recomputed temporal features (row_index,
steps_since_last_seen, seconds_since_last_seen) against the combined
sequence's own timeline -- see run_predictor_dynamic's docstring in
kv-cache-simulator.py for exactly what's recomputed live vs. kept as
each touch's own real per-occurrence value, and why.

Runs LRU, LFU, Belady's optimal, and the predictor-driven policy on that
combined sequence at a few cache sizes -- small/medium/large as a
fraction of the sequence's distinct item count, not fixed absolute
sizes -- and reports hit rate for each.
"""

import importlib.util

import joblib
import pandas as pd

from train_predictor import clean_feature_values, load_features, split_by_trace

# kv-cache-simulator.py's filename has a hyphen, so it isn't a valid Python
# module name and can't be reached with a normal `import` statement --
# loaded directly from its file path instead.
_kv_sim_spec = importlib.util.spec_from_file_location("kv_cache_simulator", "kv-cache-simulator.py")
_kv_cache_simulator = importlib.util.module_from_spec(_kv_sim_spec)
_kv_sim_spec.loader.exec_module(_kv_cache_simulator)
run_lru = _kv_cache_simulator.run_lru
run_lfu = _kv_cache_simulator.run_lfu
run_belady = _kv_cache_simulator.run_belady
run_predictor_dynamic = _kv_cache_simulator.run_predictor_dynamic

FEATURES_CSV_PATH = "features.csv"
MODEL_PATH = "model.joblib"

# cache sizes to benchmark, as a fraction of the combined sequence's
# distinct item count -- scales sensibly as the trace dataset grows,
# rather than hardcoding absolute sizes that would drift out of relevance.
CACHE_SIZE_FRACTIONS = {"small": 0.05, "medium": 0.20, "large": 0.50}


def build_combined_test_touches(feature_names: list, features_csv_path: str = FEATURES_CSV_PATH):
    """
    returns (touches, test_ids): touches is a list of per-touch dicts
    ({"item_id": ..., "static_features": {...}}), one per real touch across
    every held-out test trace, concatenated in sorted trace_id then
    row_index order. static_features carries each touch's own real values
    for every column in feature_names (cleaned identically to how
    load_features() cleans training data, via the shared
    clean_feature_values helper, so the model sees the same value
    distribution at eviction time it saw during training) -- including
    placeholder values for row_index/steps_since_last_seen/
    seconds_since_last_seen, which run_predictor_dynamic always overwrites
    with values recomputed live against the combined sequence.
    """
    df_for_split = load_features(features_csv_path)
    _, _, _, test_ids = split_by_trace(df_for_split)

    raw = pd.read_csv(features_csv_path)
    raw = raw[raw["trace_id"].isin(test_ids)].copy()

    # sorted trace_id order (matching test_ids), then row order within a
    # trace -- this is what "concatenated in order" means for the combined
    # sequence
    raw["trace_id"] = pd.Categorical(raw["trace_id"], categories=sorted(test_ids), ordered=True)
    raw = raw.sort_values(["trace_id", "row_index"])
    raw = clean_feature_values(raw)

    touches = [
        {"item_id": record["item_id"], "static_features": {name: record[name] for name in feature_names}}
        for record in raw[["item_id"] + feature_names].to_dict("records")
    ]
    return touches, test_ids


def run_benchmark():
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]

    touches, test_ids = build_combined_test_touches(feature_names)
    combined_events = [t["item_id"] for t in touches]
    n_distinct = len(set(combined_events))

    print(f"held-out test traces: {len(test_ids)}")
    print(f"combined event sequence: {len(combined_events)} events, {n_distinct} distinct items")
    print()

    header = f"{'cache size':<24}{'LRU':>8}{'LFU':>8}{'Belady':>8}{'Predictor':>11}"
    print(header)
    print("-" * len(header))

    for size_name, fraction in CACHE_SIZE_FRACTIONS.items():
        cache_size = max(1, round(fraction * n_distinct))

        lru_hit_rate = run_lru(combined_events, cache_size)
        lfu_hit_rate = run_lfu(combined_events, cache_size)
        belady_hit_rate = run_belady(combined_events, cache_size)
        predictor_hit_rate = run_predictor_dynamic(touches, cache_size, model, feature_names)

        label = f"{size_name} ({cache_size}, {fraction:.0%})"
        print(f"{label:<24}{lru_hit_rate:>7.1%} {lfu_hit_rate:>7.1%} {belady_hit_rate:>7.1%} {predictor_hit_rate:>10.1%}")

    print()
    print(
        "NOTE: predictor scores are now recomputed on every touch (row_index,\n"
        "steps_since_last_seen, seconds_since_last_seen live against the combined\n"
        "sequence), not frozen at first occurrence -- see run_predictor_dynamic's\n"
        "docstring for exactly what's recomputed live vs. kept as each touch's own\n"
        "real per-occurrence value, and why (some features, e.g. dag_completion_fraction,\n"
        "are inherently scoped to one trace's own DAG and have no coherent cross-trace\n"
        "redefinition). seconds_since_last_seen uses a simulated 1-second-per-touch\n"
        "clock, matching the abstract discrete-time-step model LRU/LFU/Belady already use."
    )


if __name__ == "__main__":
    run_benchmark()
