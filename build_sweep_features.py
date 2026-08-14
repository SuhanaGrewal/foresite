"""
Build Sweep Features

Builds features_sweep.csv from ONLY the sweep traces (traces/run_sweep_*.jsonl),
via feature_extraction.py's existing build_feature_table() -- no changes to
feature_extraction.py needed, it already takes an explicit trace-path list.

Kept as a separate CSV from the main features.csv, not folded in, for two
reasons:
1. train_predictor.py's train/test split is a random shuffle over
   features.csv's trace_ids, seeded but still order/set-dependent. Adding
   3 new trace_ids would shift that split, silently changing which traces
   the already-reported general-benchmark results (features.csv, 156
   traces) were computed against -- invalidating the "original results
   table" comparison this whole sweep test exists to produce.
2. These traces are a deliberately artificial stress pattern (wide
   concurrent fan-out, explicit non-LLM-planned subtask lists, no
   dependencies) -- mixing them into the general training set would skew
   the general model's learned distribution toward this one access
   pattern, which isn't representative of the general task suite it's
   meant to model.

The eviction benchmark (run_eviction_benchmark.py) is parameterized by
which features CSV + which trace_ids it evaluates, so pointing it at
features_sweep.csv's traces for the sweep-condition run doesn't require
touching the original general-benchmark path at all.
"""

import glob

from feature_extraction import build_feature_table

SWEEP_FEATURES_CSV_PATH = "features_sweep.csv"


if __name__ == "__main__":
    trace_paths = sorted(glob.glob("traces/run_sweep_*.jsonl"))

    df = build_feature_table(trace_paths)
    df.to_csv(SWEEP_FEATURES_CSV_PATH, index=False)

    reused_pct = df["will_be_reused"].mean() * 100
    print(f"Sweep traces processed: {len(trace_paths)}")
    print(f"Distinct traces in table: {df['trace_id'].nunique()}")
    print(f"Row count: {len(df)}")
    print(f"Class balance -- will_be_reused=1: {reused_pct:.1f}%  will_be_reused=0: {100 - reused_pct:.1f}%")
    print(f"Leakage check: PASSED for all {len(trace_paths)} sweep traces")
    print(f"Saved to {SWEEP_FEATURES_CSV_PATH}")
