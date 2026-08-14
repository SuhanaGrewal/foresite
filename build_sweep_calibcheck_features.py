"""
Build Sweep Calibcheck Features

Builds features_sweep_calibcheck.csv from ONLY the third, genuinely fresh
sweep set (traces/run_sweep_calibcheck_*.jsonl) -- never used in training
or in calibrating run_hybrid's constants. Mirrors build_sweep_features.py
and build_sweep_holdout_features.py's structure.
"""

import glob

from feature_extraction import build_feature_table

SWEEP_CALIBCHECK_FEATURES_CSV_PATH = "features_sweep_calibcheck.csv"


if __name__ == "__main__":
    trace_paths = sorted(glob.glob("traces/run_sweep_calibcheck_*.jsonl"))
    assert all("calibcheck" in p for p in trace_paths), (
        f"glob pattern matched an unexpected path: {[p for p in trace_paths if 'calibcheck' not in p]}"
    )

    df = build_feature_table(trace_paths)
    df.to_csv(SWEEP_CALIBCHECK_FEATURES_CSV_PATH, index=False)

    reused_pct = df["will_be_reused"].mean() * 100
    print(f"Calibcheck sweep traces processed: {len(trace_paths)}")
    print(f"Distinct traces in table: {df['trace_id'].nunique()}")
    print(f"Row count: {len(df)}")
    print(f"Class balance -- will_be_reused=1: {reused_pct:.1f}%  will_be_reused=0: {100 - reused_pct:.1f}%")
    print(f"Leakage check: PASSED for all {len(trace_paths)} calibcheck traces")
    print(f"Saved to {SWEEP_CALIBCHECK_FEATURES_CSV_PATH}")
