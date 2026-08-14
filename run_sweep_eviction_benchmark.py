"""
Run Sweep Eviction Benchmark

Step 4: re-runs the Step 6 eviction benchmark (same LRU/LFU/Belady's
optimal/Predictor comparison, same cache-size fractions) but built
specifically from the sweep traces (features_sweep.csv,
traces/run_sweep_*.jsonl) instead of the general held-out test set, so
the sweep-heavy access pattern is isolated rather than mixed into the
general benchmark's numbers.

Uses model.joblib -- the SAME trained model as the general benchmark, not
retrained on sweep data. This intentionally tests whether a model trained
on the general task distribution generalizes to (or degrades under) a
structurally different, sweep-heavy access pattern -- not whether a model
trained specifically on sweeps would do better at predicting sweeps.

Uses ALL 3 sweep traces (not a train/test split): the point of a train/
test split is to hold out data the model might have trained on. None of
the sweep traces were in model.joblib's training set either way (they
didn't exist yet when it was trained, and even if they had,
build_sweep_features.py deliberately keeps them out of features.csv --
see its docstring), so there's nothing to hold out here; the whole sweep
condition is unseen data.

Reuses run_eviction_benchmark.py's build_combined_touches and
run_policies_on_touches directly, so both benchmarks are computed by the
exact same code path -- the only difference between this script's numbers
and the general benchmark's is which traces went in.
"""

import joblib
import pandas as pd

from run_eviction_benchmark import MODEL_PATH, build_combined_touches, run_policies_on_touches

SWEEP_FEATURES_CSV_PATH = "features_sweep.csv"


def sweep_trace_ids(sweep_features_csv_path: str = SWEEP_FEATURES_CSV_PATH) -> list:
    return sorted(pd.read_csv(sweep_features_csv_path)["trace_id"].unique())


def run_sweep_benchmark():
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]

    trace_ids = sweep_trace_ids()
    touches = build_combined_touches(feature_names, SWEEP_FEATURES_CSV_PATH, trace_ids)

    print(f"sweep traces: {len(trace_ids)} ({trace_ids})")
    run_policies_on_touches(touches, model, feature_names)


if __name__ == "__main__":
    run_sweep_benchmark()
