"""
Run Sweep Holdout Batch

Generates sweep traces (traces/run_sweep_holdout_N.jsonl) held out from
training -- genuinely separate from the 3 traces in features_sweep.csv
(folded into training via train_predictor.py's --extra-input) -- so
"does the retrained predictor actually beat LRU on sweeps" is measured
on data the retraining never saw, not on the traces it was fit to.

Same wide-fanout, no-dependencies structural pattern and same qwen2.5:1.5b
model override as run_sweep_batch.py (reuses its _independent_subtasks and
run_one_sweep_task directly, rather than duplicating them). Every task's
content (catalog queries, cities, topics) is checked against everything
already used in training (run_sweep_batch.py) and earlier holdout tasks,
so this tests generalization to the sweep PATTERN, not memorization of
specific items already seen.

10 tasks total (2 originally run + 8 more added here, ~155 sub-agents
total) -- run via --max-new to process a few tasks per invocation rather
than all at once: the earlier 110-subagent single run proved too heavy
for this laptop, so this script is idempotent (skips any trace file that
already exists and completed successfully) and resumable, meant to be
invoked several times with a small --max-new each time rather than as
one large background job.
"""

import argparse
import asyncio
import json
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

# category/price/rating/id-based queries -- not more product-name searches,
# since the catalog only has 17 products and most distinct search terms
# were already used across training + the first holdout task
CATALOG_QUERIES_2 = [
    "Find products under $15 in the electronics category.",
    "Find products under $250 in the electronics category.",
    "Find products under $100 in the kitchenware category.",
    "Find products under $200 in the office_supplies category.",
    "Find the highest-rated product in the books category.",
    "Find the highest-rated product in the beauty category.",
    "Find products under $50 in the toys category.",
    "Find products under $20 in the books category.",
    "List every product category in the catalog.",
    "Find products under $30 in the electronics category.",
    "Get product details for product id 1.",
    "Get product details for product id 5.",
    "Get product details for product id 10.",
    "Get product details for product id 17.",
]

WIKIPEDIA_TOPICS_2 = [
    "the history of the Panama hat",
    "the causes of the Boer War",
    "the history of the Erie Canal",
    "the invention of the sewing machine",
    "the life of Ada Lovelace",
    "the history of the Colossus of Rhodes",
    "the causes of the Vietnam War",
    "the discovery of radioactivity",
    "the history of the Taj Mahal",
    "the invention of the cotton gin",
    "the causes of the Russian Revolution",
    "the discovery of Uranus",
    "the history of Angkor Wat",
    "the invention of the phonograph",
    "the causes of the Thirty Years' War",
    "the history of the Golden Gate Bridge",
]

WIKIPEDIA_TOPICS_3 = [
    "the history of the Berlin Airlift",
    "the discovery of oxygen",
    "the causes of the Opium Wars",
    "the history of the Channel Tunnel",
    "the invention of the smallpox vaccine",
    "the life of Charles Darwin",
    "the history of the Parthenon",
    "the causes of the Korean War",
    "the discovery of Antarctica",
    "the history of the Sydney Opera House",
    "the invention of the jet engine",
    "the causes of the Mexican Revolution",
    "the discovery of gravitational waves",
    "the history of Stonehenge",
    "the history of the Zeppelin airship",
    "the causes of the Hundred Years' War",
]

WIKIPEDIA_TOPICS_4 = [
    "the history of the Suez Crisis",
    "the discovery of the dwarf planet Ceres",
    "the causes of the Balkan Wars",
    "the history of the Golden Temple of Amritsar",
    "the invention of Braille",
    "the life of Isaac Newton",
    "the history of the Sphinx of Giza",
    "the causes of the Falklands War",
    "the discovery of Pluto's moons",
    "the history of Petra",
    "the invention of the elevator",
    "the causes of the Six-Day War",
    "the discovery of the structure of the atom",
    "the history of the Leaning Tower of Pisa",
    "the invention of the assembly line",
    "the causes of the War of 1812",
]

WEATHER_CITIES_1 = [
    "Nairobi", "Cairo", "Johannesburg", "Buenos Aires", "Santiago", "Lima",
    "Mexico City", "Toronto", "Vancouver", "Auckland", "Wellington",
    "Kuala Lumpur", "Hanoi", "Colombo", "Kathmandu", "Amman", "Beirut", "Tunis",
]

WEATHER_CITIES_2 = [
    "Montreal", "Chicago", "Miami", "Seattle", "Zurich", "Vienna", "Prague",
    "Budapest", "Stockholm", "Oslo", "Copenhagen", "Brussels", "Amsterdam",
    "Madrid", "Rome", "Milan", "Barcelona", "Dakar",
]

MIXED_WEATHER_1 = ["Kigali", "Doha", "Muscat", "Manama"]
MIXED_CATALOG_1 = [
    "Find products under $40 in the kitchenware category.",
    "Find the highest-rated product in the electronics category.",
    "Find products under $25 in the office_supplies category.",
    "List every product category in the catalog.",
]
MIXED_WIKI_1 = [
    "the history of the Great Sphinx",
    "the discovery of black holes",
    "the invention of the automobile",
]
MIXED_PYTHON_1 = [
    "Use the code execution tool to calculate the average of 10, 20, 30, 40, and 50.",
    "Use the code execution tool to calculate 2 to the power of 10.",
]

MIXED_WEATHER_2 = ["Tashkent", "Almaty", "Baku", "Tbilisi"]
MIXED_CATALOG_2 = [
    "Find products under $60 in the kitchenware category.",
    "Find the highest-rated product in the office_supplies category.",
    "Get product details for product id 3.",
    "Get product details for product id 12.",
]
MIXED_WIKI_2 = [
    "the history of the Great Fire of London",
    "the discovery of the Higgs boson",
    "the invention of the radio",
    "the causes of the Crimean War",
]
MIXED_PYTHON_2 = [
    "Use the code execution tool to calculate the factorial of 6.",
    "Use the code execution tool to calculate 30 percent of 850.",
]


def _wiki_descriptions(topics: list) -> list:
    return [f"Search the web for {topic} and report a brief summary." for topic in topics]


def _weather_descriptions(cities: list) -> list:
    return [f"Get the current weather for {city}." for city in cities]


SWEEP_HOLDOUT_TASKS = [
    {
        "name": "catalog_sweep_holdout",
        "user_task": "Look up 14 different items and categories in the product catalog and report a summary of prices, stock, and ratings.",
        "descriptions": HOLDOUT_CATALOG_QUERIES,
    },
    {
        "name": "wikipedia_sweep_holdout",
        "user_task": "Research 16 different, unrelated general-knowledge topics and report a brief summary of each.",
        "descriptions": _wiki_descriptions(HOLDOUT_WIKIPEDIA_TOPICS),
    },
    {
        "name": "catalog_sweep_holdout_2",
        "user_task": "Look up 14 different category/price/rating queries in the product catalog and report a summary.",
        "descriptions": CATALOG_QUERIES_2,
    },
    {
        "name": "wikipedia_sweep_holdout_2",
        "user_task": "Research 16 different, unrelated general-knowledge topics and report a brief summary of each.",
        "descriptions": _wiki_descriptions(WIKIPEDIA_TOPICS_2),
    },
    {
        "name": "weather_sweep_holdout_1",
        "user_task": "Check the current weather across 18 different cities around the world and report a summary.",
        "descriptions": _weather_descriptions(WEATHER_CITIES_1),
    },
    {
        "name": "wikipedia_sweep_holdout_3",
        "user_task": "Research 16 different, unrelated general-knowledge topics and report a brief summary of each.",
        "descriptions": _wiki_descriptions(WIKIPEDIA_TOPICS_3),
    },
    {
        "name": "mixed_sweep_holdout_1",
        "user_task": "Complete 13 independent tasks spanning weather checks, product catalog lookups, topic research, and quick calculations, and report a summary.",
        "descriptions": (
            _weather_descriptions(MIXED_WEATHER_1) + MIXED_CATALOG_1 + _wiki_descriptions(MIXED_WIKI_1) + MIXED_PYTHON_1
        ),
    },
    {
        "name": "mixed_sweep_holdout_2",
        "user_task": "Complete 14 independent tasks spanning weather checks, product catalog lookups, topic research, and quick calculations, and report a summary.",
        "descriptions": (
            _weather_descriptions(MIXED_WEATHER_2) + MIXED_CATALOG_2 + _wiki_descriptions(MIXED_WIKI_2) + MIXED_PYTHON_2
        ),
    },
    {
        "name": "wikipedia_sweep_holdout_4",
        "user_task": "Research 16 different, unrelated general-knowledge topics and report a brief summary of each.",
        "descriptions": _wiki_descriptions(WIKIPEDIA_TOPICS_4),
    },
    {
        "name": "weather_sweep_holdout_2",
        "user_task": "Check the current weather across 18 different cities around the world and report a summary.",
        "descriptions": _weather_descriptions(WEATHER_CITIES_2),
    },
]


def _trace_is_complete(trace_path: str) -> bool:
    if not os.path.exists(trace_path):
        return False
    try:
        with open(trace_path) as f:
            lines = [json.loads(l) for l in f]
    except (json.JSONDecodeError, OSError):
        return False
    return any(l.get("agent_id") == "synthesizer" and l.get("event_type") == "final_answer" for l in lines)


async def run_sweep_holdout_batch(max_new: int = None):
    """
    idempotent + resumable: skips any trace file that already exists and
    completed successfully, so this can be invoked repeatedly with a small
    max_new each time (smaller batches, verified one at a time) rather than
    processing all 10 tasks in a single run.
    """
    os.makedirs(TRACES_DIR, exist_ok=True)
    succeeded = 0
    failed = 0
    newly_run = 0

    for i, sweep in enumerate(SWEEP_HOLDOUT_TASKS, start=1):
        trace_path = f"{TRACES_DIR}/run_sweep_holdout_{i}.jsonl"

        if _trace_is_complete(trace_path):
            print(f"\n=== held-out sweep task {i}/{len(SWEEP_HOLDOUT_TASKS)}: {sweep['name']} -- already complete, skipping")
            continue

        if max_new is not None and newly_run >= max_new:
            print(f"\n=== reached --max-new {max_new}, stopping (rerun this script to continue with the rest)")
            break

        if os.path.exists(trace_path):
            os.remove(trace_path)  # partial/failed leftover

        print(f"\n=== held-out sweep task {i}/{len(SWEEP_HOLDOUT_TASKS)}: {sweep['name']} ({len(sweep['descriptions'])} independent sub-tasks)")
        newly_run += 1

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

    print(f"\n=== held-out sweep batch invocation complete: {succeeded} succeeded, {failed} failed this run ===")
    if failed:
        print(f"    see {ERROR_LOG_PATH} for failure details")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new", type=int, default=None, help="run at most this many NOT-yet-completed tasks, then stop")
    args = parser.parse_args()
    asyncio.run(run_sweep_holdout_batch(max_new=args.max_new))
