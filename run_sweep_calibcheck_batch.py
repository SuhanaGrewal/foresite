"""
Run Sweep Calibcheck Batch

Generates a THIRD, genuinely fresh sweep trace set (traces/run_sweep_calibcheck_N.jsonl)
-- never used in training (features_sweep.csv) or in calibrating run_hybrid's
PRESSURE_WINDOW_MULTIPLIER/PRESSURE_RATIO_THRESHOLD (features_sweep_holdout.csv).
Exists to answer one honest question: does the hybrid's calibration, tuned
entirely against the first held-out set's own 12-point curve, actually
generalize to sweep data it has never seen in any capacity -- or was that
calibration fit to that one dataset's particular characteristics?
"""

import argparse
import asyncio

from run_sweep_batch import (
    ERROR_LOG_PATH,
    TRACES_DIR,
    _independent_subtasks,
    _log_failure,
    _trace_is_complete,
    _wiki_descriptions,
    _weather_descriptions,
    run_one_sweep_task,
)

# uses product ids 7/11/13/15, the only ones not already queried across
# run_sweep_batch.py's and run_sweep_holdout_batch.py's catalog tasks
CATALOG_QUERIES_CALIBCHECK = [
    "Get product details for product id 7.",
    "Get product details for product id 11.",
    "Get product details for product id 13.",
    "Get product details for product id 15.",
    "Find products under $18 in the beauty category.",
    "Find products under $65 in the electronics category.",
]
