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
