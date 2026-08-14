"""
Run Sweep Holdout Batch

Generates a SECOND, small set of sweep traces (traces/run_sweep_holdout_N.jsonl)
-- genuinely separate from the 3 traces in features_sweep.csv (which are
being folded into training, see train_predictor.py's --extra-input) --
so "does the retrained predictor actually beat LRU on sweeps" is measured
on data the retraining never saw, not on the traces it was fit to.

Same wide-fanout, no-dependencies structural pattern and same qwen2.5:1.5b
model override as run_sweep_batch.py (reuses its _independent_subtasks and
run_one_sweep_task directly, rather than duplicating them), but entirely
different content (different catalog queries, different topics) so this
tests generalization to the sweep PATTERN, not memorization of the
specific items in the training sweep traces.

Scoped small (2 tasks, ~30 sub-agents total) given the earlier resource
constraints on this laptop -- large enough to be a real held-out sample,
not so large it repeats the earlier strain.
"""

import asyncio
import os

from run_sweep_batch import ERROR_LOG_PATH, TRACES_DIR, _log_failure, run_one_sweep_task

HOLDOUT_CATALOG_QUERIES = [
    "Search the product catalog for 'sleeping bag' and report price and stock.",
    "Search the product catalog for 'tent' and report price and stock.",
    "Search the product catalog for 'backpack' and report price and stock.",
    "Find products under $30 in the beauty category.",
    "Find products under $50 in the electronics category.",
    "Find the highest-rated product in the kitchenware category.",
    "Find the highest-rated product in the toys category.",
    "Search the product catalog for 'trekking' and report price and stock.",
    "Search the product catalog for 'boot' and report price and stock.",
    "Search the product catalog for 'lotion' and report price and stock.",
    "List every product category in the catalog.",
    "Find products under $100 in the office_supplies category.",
    "Search the product catalog for 'notebook' and report price and stock.",
    "Search the product catalog for 'lamp' and report price and stock.",
]

HOLDOUT_WIKIPEDIA_TOPICS = [
    "the history of the Wright brothers and powered flight",
    "the discovery of X-rays",
    "the causes of the fall of the Berlin Wall",
    "the history of the Trans-Siberian Railway",
    "the invention of the internet",
    "the life of Nikola Tesla",
    "the history of the Rosetta Stone",
    "the causes of the Cuban Missile Crisis",
    "the discovery of insulin",
    "the history of the Hoover Dam",
    "the invention of the microscope",
    "the causes of the Spanish Civil War",
    "the history of the Statue of Liberty",
    "the discovery of Neptune",
    "the history of Machu Picchu",
    "the invention of the compass",
]

SWEEP_HOLDOUT_TASKS = [
    {
        "name": "catalog_sweep_holdout",
        "user_task": "Look up 14 different items and categories in the product catalog and report a summary of prices, stock, and ratings.",
        "descriptions": HOLDOUT_CATALOG_QUERIES,
    },
    {
        "name": "wikipedia_sweep_holdout",
        "user_task": "Research 16 different, unrelated general-knowledge topics and report a brief summary of each.",
        "descriptions": [f"Search the web for {topic} and report a brief summary." for topic in HOLDOUT_WIKIPEDIA_TOPICS],
    },
]


async def run_sweep_holdout_batch():
    os.makedirs(TRACES_DIR, exist_ok=True)
    succeeded = 0
    failed = 0

    for i, sweep in enumerate(SWEEP_HOLDOUT_TASKS, start=1):
        trace_path = f"{TRACES_DIR}/run_sweep_holdout_{i}.jsonl"
        if os.path.exists(trace_path):
            os.remove(trace_path)

        print(f"\n=== held-out sweep task {i}/{len(SWEEP_HOLDOUT_TASKS)}: {sweep['name']} ({len(sweep['descriptions'])} independent sub-tasks)")

        try:
            final_answer = await run_one_sweep_task(sweep, trace_path)
        except Exception as e:
            failed += 1
            _log_failure(sweep["name"], e)
            if os.path.exists(trace_path):
                os.remove(trace_path)
            print(f"    FAILED: {type(e).__name__}: {e}")
            continue

        succeeded += 1
        print(f"    synthesized answer: {final_answer[:200]}")
        print(f"    trace written to {trace_path}")

    print(f"\n=== held-out sweep batch complete: {succeeded} succeeded, {failed} failed ===")
    if failed:
        print(f"    see {ERROR_LOG_PATH} for failure details")


if __name__ == "__main__":
    asyncio.run(run_sweep_holdout_batch())
