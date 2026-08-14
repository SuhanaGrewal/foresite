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
import os

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
    "Find products with a rating above 4.6.",
    "Find products with fewer than 20 units in stock.",
    "List every product category in the catalog.",
    "Find in-stock products under $30 in the kitchenware category.",
]

WIKIPEDIA_TOPICS_CALIBCHECK = [
    "the history of the Manhattan Project",
    "the discovery of tectonic plates",
    "the causes of the Nika riots",
    "the history of the Millau Viaduct",
    "the invention of the stethoscope",
    "the life of Rosalind Franklin",
    "the history of the Berlin Wall's checkpoints",
    "the discovery of quasars",
    "the causes of the Second Congo War",
    "the history of the Krakatoa eruption",
]

MIXED_WEATHER_CALIBCHECK = ["Vladivostok", "Ulaanbaatar", "Yerevan", "Tallinn"]
MIXED_CATALOG_CALIBCHECK = [
    "Find products under $50 in the toys category.",
    "Find products under $12 in the books category.",
]
MIXED_WIKI_CALIBCHECK = [
    "the history of the Vasa warship",
    "the discovery of the Dead Sea Scrolls",
]
MIXED_PYTHON_CALIBCHECK = [
    "Use the code execution tool to calculate the sum of the first 20 even numbers.",
    "Use the code execution tool to calculate 18 percent of 640.",
]

SWEEP_CALIBCHECK_TASKS = [
    {
        "name": "catalog_sweep_calibcheck",
        "user_task": "Look up 10 different category/price/rating/id queries in the product catalog and report a summary.",
        "descriptions": CATALOG_QUERIES_CALIBCHECK,
    },
    {
        "name": "wikipedia_sweep_calibcheck",
        "user_task": "Research 10 different, unrelated general-knowledge topics and report a brief summary of each.",
        "descriptions": _wiki_descriptions(WIKIPEDIA_TOPICS_CALIBCHECK),
    },
    {
        "name": "mixed_sweep_calibcheck",
        "user_task": "Complete 10 independent tasks spanning weather checks, product catalog lookups, topic research, and quick calculations, and report a summary.",
        "descriptions": (
            _weather_descriptions(MIXED_WEATHER_CALIBCHECK)
            + MIXED_CATALOG_CALIBCHECK
            + _wiki_descriptions(MIXED_WIKI_CALIBCHECK)
            + MIXED_PYTHON_CALIBCHECK
        ),
    },
]


async def run_sweep_calibcheck_batch(max_new: int = None):
    """idempotent + resumable, same pattern as run_sweep_holdout_batch.py -- skips
    any trace file that already completed, accepts --max-new for small batches."""
    os.makedirs(TRACES_DIR, exist_ok=True)
    succeeded = 0
    failed = 0
    newly_run = 0
