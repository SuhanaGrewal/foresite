"""
Run Sweep Calibcheck Cache Curve

The real generalization test: runs LRU/LFU/Belady/raw predictor/hybrid on
features_sweep_calibcheck.csv -- a third sweep set never used in training
(features_sweep.csv) or in calibrating run_hybrid's PRESSURE_WINDOW_MULTIPLIER/
PRESSURE_RATIO_THRESHOLD (features_sweep_holdout.csv). Uses model.joblib
and run_hybrid's constants completely UNCHANGED -- no retraining, no
re-calibration -- so this answers whether the calibration tuned against
one held-out set actually transfers, or was fit to that set's particular
characteristics. Every row reported, favorable or not.
"""

import joblib
import pandas as pd

from run_eviction_benchmark import build_combined_touches, run_lru, run_lfu, run_belady, MODEL_PATH

import importlib.util

_kv_sim_spec = importlib.util.spec_from_file_location("kv_cache_simulator", "kv-cache-simulator.py")
_kv_cache_simulator = importlib.util.module_from_spec(_kv_sim_spec)
_kv_sim_spec.loader.exec_module(_kv_cache_simulator)
run_predictor_dynamic = _kv_cache_simulator.run_predictor_dynamic
run_hybrid = _kv_cache_simulator.run_hybrid

SWEEP_CALIBCHECK_FEATURES_CSV_PATH = "features_sweep_calibcheck.csv"
CACHE_FRACTIONS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
