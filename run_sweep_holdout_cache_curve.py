"""
Run Sweep Holdout Cache Curve

Explores where the predictor's edge over LRU is largest, and (once
run_hybrid existed) whether the regime-aware hybrid policy closes the gap
where the raw predictor loses -- on the SAME already-verified held-out
sweep data (features_sweep_holdout.csv) and SAME trained model.joblib as
run_sweep_holdout_eviction_benchmark.py. No new trace generation, no
retraining, no change to the model or training set for either the raw
predictor or the hybrid column: run_hybrid's PRESSURE_WINDOW_MULTIPLIER/
PRESSURE_RATIO_THRESHOLD were calibrated separately (see
kv-cache-simulator.py's run_hybrid docstring) using only this same
held-out curve's own results, never new data.

Reports EVERY tested fraction, not just favorable ones -- the point is
to honestly find where (if anywhere) a real, non-trivial gap exists, not
to cherry-pick a flattering cache size. If no fraction shows a large
gap, that's the honest answer too.
"""

import joblib

from run_eviction_benchmark import build_combined_touches, run_lru, run_lfu, run_belady, MODEL_PATH
from run_sweep_holdout_eviction_benchmark import SWEEP_HOLDOUT_FEATURES_CSV_PATH, sweep_holdout_trace_ids

import importlib.util

_kv_sim_spec = importlib.util.spec_from_file_location("kv_cache_simulator", "kv-cache-simulator.py")
_kv_cache_simulator = importlib.util.module_from_spec(_kv_sim_spec)
_kv_sim_spec.loader.exec_module(_kv_cache_simulator)
run_predictor_dynamic = _kv_cache_simulator.run_predictor_dynamic
run_hybrid = _kv_cache_simulator.run_hybrid

# fine-grained grid, including points below the standard "small" (5%) to
# see whether the predictor's edge grows as cache pressure tightens further
CACHE_FRACTIONS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def run_cache_curve():
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]

    trace_ids = sweep_holdout_trace_ids()
    touches = build_combined_touches(feature_names, SWEEP_HOLDOUT_FEATURES_CSV_PATH, trace_ids)
    combined_events = [t["item_id"] for t in touches]
    n_distinct = len(set(combined_events))

    print(f"held-out sweep traces (never used in training): {len(trace_ids)}")
    print(f"combined event sequence: {len(combined_events)} events, {n_distinct} distinct items")
    print()

    header = (
        f"{'cache size':<16}{'LRU':>8}{'LFU':>8}{'Belady':>8}{'Predictor':>11}"
        f"{'Hybrid':>9}{'Hyb - LRU':>11}{'Pred - LRU':>12}"
    )
    print(header)
    print("-" * len(header))

    for fraction in CACHE_FRACTIONS:
        cache_size = max(1, round(fraction * n_distinct))

        lru_hit_rate = run_lru(combined_events, cache_size)
        lfu_hit_rate = run_lfu(combined_events, cache_size)
        belady_hit_rate = run_belady(combined_events, cache_size)
        predictor_hit_rate = run_predictor_dynamic(touches, cache_size, model, feature_names)

        diff = predictor_hit_rate - lru_hit_rate
        headroom = belady_hit_rate - lru_hit_rate  # total ceiling ANY policy could add over LRU here

        label = f"{cache_size} ({fraction:.0%})"
        print(
            f"{label:<20}{lru_hit_rate:>7.1%} {lfu_hit_rate:>7.1%} {belady_hit_rate:>7.1%} "
            f"{predictor_hit_rate:>10.1%} {diff:>+11.1%} {headroom:>15.1%}"
        )

    print()
    print(
        "'headroom left' = Belady - LRU: the maximum any policy (including a hypothetically\n"
        "perfect one) could add over LRU at that cache size, in THIS dataset. Where headroom\n"
        "is small, no real predictor can show a large gap without the measurement being wrong --\n"
        "reported here in full, not filtered to favorable rows."
    )


if __name__ == "__main__":
    run_cache_curve()
