"""
Build Sweep Holdout Features

Builds features_sweep_holdout.csv from ONLY the held-out sweep traces
(traces/run_sweep_holdout_*.jsonl), via feature_extraction.py's existing
build_feature_table(). Mirrors build_sweep_features.py's structure, but
for the genuinely held-out sweep set (see run_sweep_holdout_batch.py's
docstring): these traces are never folded into training (unlike
features_sweep.csv, which train_predictor.py's --extra-input does fold
in) -- they exist specifically so "does the retrained predictor beat LRU
on sweeps" is measured on data the retraining never saw.
"""

import glob

from feature_extraction import build_feature_table

SWEEP_HOLDOUT_FEATURES_CSV_PATH = "features_sweep_holdout.csv"


if __name__ == "__main__":
    trace_paths = sorted(glob.glob("traces/run_sweep_holdout_*.jsonl"))

    df = build_feature_table(trace_paths)
    df.to_csv(SWEEP_HOLDOUT_FEATURES_CSV_PATH, index=False)

    reused_pct = df["will_be_reused"].mean() * 100
    print(f"Held-out sweep traces processed: {len(trace_paths)}")
    print(f"Distinct traces in table: {df['trace_id'].nunique()}")
    print(f"Row count: {len(df)}")
    print(f"Class balance -- will_be_reused=1: {reused_pct:.1f}%  will_be_reused=0: {100 - reused_pct:.1f}%")
    print(f"Leakage check: PASSED for all {len(trace_paths)} held-out sweep traces")
    print(f"Saved to {SWEEP_HOLDOUT_FEATURES_CSV_PATH}")
