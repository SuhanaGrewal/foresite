"""
Build Sweep Features

Builds features_sweep.csv from ONLY the TRAINING sweep traces
(traces/run_sweep_N.jsonl), via feature_extraction.py's existing
build_feature_table() -- no changes to feature_extraction.py needed, it
already takes an explicit trace-path list.

The glob pattern is traces/run_sweep_[0-9]*.jsonl, NOT traces/run_sweep_*.jsonl:
the naive `*` pattern also matches traces/run_sweep_holdout_*.jsonl (the
genuinely held-out evaluation set -- see run_sweep_holdout_batch.py),
which would silently fold held-out traces into training data. Caught
this by hand when a rebuild reported "20 traces" instead of the expected
10 -- worth stating plainly since it's exactly the kind of train/test
contamination this whole sweep methodology exists to avoid, and it
happened once already before being caught.

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
    trace_paths = sorted(glob.glob("traces/run_sweep_[0-9]*.jsonl"))
    assert not any("holdout" in p for p in trace_paths), (
        f"glob pattern accidentally matched a held-out trace: {[p for p in trace_paths if 'holdout' in p]}"
    )

    df = build_feature_table(trace_paths)
    df.to_csv(SWEEP_FEATURES_CSV_PATH, index=False)

    reused_pct = df["will_be_reused"].mean() * 100
    print(f"Sweep traces processed: {len(trace_paths)}")
    print(f"Distinct traces in table: {df['trace_id'].nunique()}")
    print(f"Row count: {len(df)}")
    print(f"Class balance -- will_be_reused=1: {reused_pct:.1f}%  will_be_reused=0: {100 - reused_pct:.1f}%")
    print(f"Leakage check: PASSED for all {len(trace_paths)} sweep traces")
    print(f"Saved to {SWEEP_FEATURES_CSV_PATH}")
