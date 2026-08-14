"""
Run Sweep Holdout Eviction Benchmark

The real test: runs the same LRU/LFU/Belady's optimal/Predictor
comparison as run_sweep_eviction_benchmark.py, but against
features_sweep_holdout.csv -- traces the retrained model.joblib never
saw during training (unlike features_sweep.csv, which train_predictor.py's
--extra-input folded into training). This is what actually answers "does
folding sweep data into training make the predictor beat LRU on a sweep
it's never seen," not just "did it memorize the 3 sweep traces it was
fit on."

Reuses run_eviction_benchmark.py's build_combined_touches and
run_policies_on_touches directly, same as the training-sweep benchmark,
so all three benchmarks (general, training-sweep, holdout-sweep) are
computed via the exact same code path.
"""

import joblib
import pandas as pd

from run_eviction_benchmark import MODEL_PATH, build_combined_touches, run_policies_on_touches

SWEEP_HOLDOUT_FEATURES_CSV_PATH = "features_sweep_holdout.csv"


def sweep_holdout_trace_ids(features_csv_path: str = SWEEP_HOLDOUT_FEATURES_CSV_PATH) -> list:
    return sorted(pd.read_csv(features_csv_path)["trace_id"].unique())


def run_sweep_holdout_benchmark():
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]

    trace_ids = sweep_holdout_trace_ids()
    touches = build_combined_touches(feature_names, SWEEP_HOLDOUT_FEATURES_CSV_PATH, trace_ids)

    print(f"held-out sweep traces (never used in training): {len(trace_ids)} ({trace_ids})")
    run_policies_on_touches(touches, model, feature_names)


if __name__ == "__main__":
    run_sweep_holdout_benchmark()
