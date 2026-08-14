"""
Run Sweep Batch

Builds and runs SWEEP tasks: each forces the orchestrator into a wide,
concurrent fan-out (15-20+ independent sub-agents, no dependencies between
any of them), structurally different from the existing 2-5-subtask tasks
in run_batch.py/run_frames_batch.py.

Why wide fan-out, specifically: trace_to_events() decomposes a model_call
step into its full, growing message history -- so within ONE agent's own
sequential loop, every earlier message gets re-touched at every later step
by construction. A single long agent is inherently reuse-heavy, the
OPPOSITE of a sweep. The only place this architecture can produce "many
one-off touches in a short span" is many concurrent, mostly-independent
sub-agents, each with its own short-lived context that's active for a few
steps and then never touched again, all interleaved into one shared trace
log -- many small working sets colliding in a shared cache, not a literal
linear scan.

Sub-task plans here are constructed EXPLICITLY, not via orchestrator.py's
plan_task() (i.e. NOT LLM-generated): asking the planner model to
freeform-plan a 15-20-subtask DAG reliably -- when it already produces
occasional malformed JSON on ordinary 2-5-subtask plans (see
frames_batch_errors.log) -- would gate this deliberately-designed stress
test on an unrelated, already-fragile step. dispatch_subtasks()/
synthesize() are called directly instead: same real machinery (real
concurrent sub-agents, real tool calls, real local model, real trace
logging), minus the LLM planning step.

Originally scoped down to 3 sweep tasks (54 sub-agent runs total), not
the first design's 6 (110 runs): 110 real sub-agent runs against
qwen2.5:7b proved too heavy for this laptop. Also runs on qwen2.5:1.5b
instead of trace_agent.py's committed 7b default: this test is about the
ACCESS PATTERN (many one-off touches vs. a few reused ones), not model
reasoning quality, so the smaller/faster model doesn't compromise what's
being measured here (7b was chosen in trace_agent.py for reliable
multi-step reasoning, which isn't what sweep sub-tasks -- single simple
lookups each -- need).

Extended from 3 to 10 tasks after the first 3-trace training set proved
too small: with only 3 traces (342 rows) folded into training (even
up-weighted), the model could only learn whatever those 3 specific
traces happened to look like -- the exact same small-sample problem that
made the first 2-trace HELD-OUT set misleading (see
run_sweep_holdout_batch.py's docstring for that story). 7 more tasks
added here, all fresh content checked against every catalog query/city/
topic already used across the first 3 training tasks AND all 10 held-out
tasks, so training data stays genuinely distinct from what's used to
evaluate it.

Traces are written to traces/run_sweep_N.jsonl, kept separate from the
general batch/FRAMES traces. Like run_sweep_holdout_batch.py, this
runner is idempotent and resumable (--max-new), meant to be invoked
several times with a small batch each time rather than as one large job.
"""

import argparse
import asyncio
import json
import os
import traceback
from datetime import datetime, timezone

import trace_agent
from orchestrator import dispatch_subtasks, synthesize

TRACES_DIR = "traces"
ERROR_LOG_PATH = "sweep_batch_errors.log"

trace_agent.MODEL = "qwen2.5:1.5b"


def _independent_subtasks(descriptions: list) -> list:
    """turns a flat list of descriptions into an all-independent subtask plan (no depends_on -- maximizes concurrency/fan-out)"""
    return [{"id": f"lookup_{i}", "description": desc, "depends_on": []} for i, desc in enumerate(descriptions)]


CATALOG_QUERIES = [
    "Search the product catalog for 'headphones' and report price and stock.",
    "Search the product catalog for 'monitor' and report price and stock.",
    "Search the product catalog for 'keyboard' and report price and stock.",
    "Search the product catalog for 'espresso' and report price and stock.",
    "Search the product catalog for 'skillet' and report price and stock.",
    "Search the product catalog for 'knife set' and report price and stock.",
    "Search the product catalog for 'chair' and report price and stock.",
    "Search the product catalog for 'pencil' and report price and stock.",
    "Search the product catalog for 'organizer' and report price and stock.",
    "Search the product catalog for 'blocks' and report price and stock.",
    "Search the product catalog for 'remote control car' and report price and stock.",
    "Search the product catalog for 'sunscreen' and report price and stock.",
    "Search the product catalog for 'serum' and report price and stock.",
    "Search the product catalog for 'library' and report the price of any matching book.",
    "Search the product catalog for 'habits' and report the price of any matching book.",
    "Search the product catalog for 'brief history' and report the price of any matching book.",
    "List every product category in the catalog.",
    "Find the highest-rated product in the electronics category.",
]

WIKIPEDIA_TOPICS = [
    "the history of the printing press",
    "the discovery of penicillin",
    "the causes of the French Revolution",
    "how photosynthesis works",
    "the Great Barrier Reef",
    "the life of Marie Curie",
    "the history of the Suez Canal",
    "the invention of the telephone",
    "the Apollo 11 moon landing",
    "the fall of the Roman Empire",
    "the history of jazz music",
    "the discovery of DNA's double helix structure",
    "the Industrial Revolution",
    "the history of the Great Wall of China",
    "the invention of the telegraph",
    "the causes of World War I",
    "the history of the Panama Canal",
    "the theory of continental drift",
]

MIXED_WEATHER_CITIES = ["Lagos", "Casablanca", "Istanbul", "Bangkok", "Seoul", "Dublin"]
MIXED_CATALOG_QUERIES = [
    "Search the product catalog for 'speaker' and report price and stock.",
    "Search the product catalog for 'espresso maker' and report price and stock.",
    "List every product category in the catalog.",
    "Find products under $20 in the office_supplies category.",
    "Search the product catalog for 'pencil set' and report price and stock.",
    "Find the highest-rated product in the toys category.",
]
MIXED_WIKIPEDIA_TOPICS = [
    "the history of the Eiffel Tower",
    "the discovery of gravity",
    "the origins of the Olympic Games",
    "the history of the Silk Road",
]
MIXED_PYTHON_TASKS = [
    "Use the code execution tool to calculate the sum of the first 50 positive integers.",
    "Use the code execution tool to calculate 15 percent of 240.",
]


# category/price/rating/stock/id-based queries, not more product-name
# searches -- the catalog only has 17 products and most reasonable
# distinct search terms were already used across the first catalog_sweep
# and the held-out catalog sweeps
CATALOG_QUERIES_2 = [
    "Find products under $45 in the kitchenware category.",
    "Find products under $85 in the electronics category.",
    "Find products under $10 in the office_supplies category.",
    "Find products under $35 in the beauty category.",
    "Find products with a rating above 4.5.",
    "Find products with more than 50 units in stock.",
    "Get product details for product id 2.",
    "Get product details for product id 6.",
    "Get product details for product id 9.",
    "Get product details for product id 14.",
    "Get product details for product id 16.",
    "List every product category in the catalog.",
    "Find products under $35 in the toys category.",
    "Find products under $70 in the books category.",
]

WIKIPEDIA_TOPICS_4 = [
    "the history of the Domesday Book",
    "the discovery of Saturn's rings",
    "the causes of the Punic Wars",
    "the history of the Brooklyn Bridge",
    "the invention of the barometer",
    "the life of Leonardo da Vinci",
    "the history of the Alhambra",
    "the causes of the Boxer Rebellion",
    "the discovery of Halley's Comet",
    "the history of the Colosseum's construction",
    "the invention of the parachute",
    "the causes of the Algerian War",
    "the discovery of the electron",
    "the history of Chichen Itza",
    "the invention of the printing telegraph cable",
    "the causes of the Peloponnesian War",
]

WIKIPEDIA_TOPICS_5 = [
    "the history of the 1936 Berlin Olympics",
    "the discovery of Neptune's moons",
    "the causes of the Anglo-Zulu War",
    "the history of the Trans-Amazonian Highway",
    "the invention of the bicycle",
    "the life of Galileo Galilei",
    "the history of the Great Zimbabwe ruins",
    "the causes of the Iran-Iraq War",
    "the discovery of the asteroid belt",
    "the history of the Hagia Sophia",
    "the invention of the television",
    "the causes of the Winter War",
    "the discovery of DNA fingerprinting",
    "the history of the Palace of Versailles",
    "the invention of the seismograph",
    "the causes of the Sino-Japanese War",
]

WIKIPEDIA_TOPICS_6 = [
    "the history of the Berlin Zoo",
    "the discovery of Titan, the moon of Saturn",
    "the causes of the Yugoslav Wars",
    "the history of the Kariba Dam",
    "the invention of the microwave oven",
    "the life of Albert Einstein",
    "the history of the Palenque ruins",
    "the causes of the Rwandan genocide",
    "the discovery of Troy by Heinrich Schliemann",
    "the history of the Ellis Island immigration station",
    "the invention of the sextant",
    "the causes of the Nigerian Civil War",
    "the discovery of Neptune's rings",
    "the history of the Blue Mosque",
    "the invention of the barcode",
    "the causes of the Iran hostage crisis",
]

WEATHER_CITIES_1 = [
    "Singapore", "Hong Kong", "Taipei", "Shanghai", "Beijing", "Sydney",
    "Melbourne", "Brisbane", "Honolulu", "Anchorage", "Panama City",
    "San Jose", "Quito", "La Paz", "Montevideo", "Asuncion", "Caracas", "Georgetown",
]

WEATHER_CITIES_2 = [
    "Algiers", "Rabat", "Tripoli", "Khartoum", "Addis Ababa", "Kampala",
    "Dar es Salaam", "Lusaka", "Harare", "Gaborone", "Windhoek", "Maputo",
    "Antananarivo", "Port Louis", "Male", "Thimphu", "Dhaka", "Islamabad",
]

MIXED_WEATHER_2 = ["Suva", "Apia", "Nuku'alofa", "Port Moresby"]
MIXED_CATALOG_2 = [
    "Find products under $20 in the beauty category.",
    "Get product details for product id 4.",
    "Get product details for product id 8.",
    "Find products with more than 80 units in stock.",
]
MIXED_WIKI_2 = [
    "the history of the Berlin Philharmonic",
    "the discovery of exoplanets",
    "the invention of the camera",
    "the causes of the Franco-Prussian War",
]
MIXED_PYTHON_2 = [
    "Use the code execution tool to calculate the square root of 144.",
    "Use the code execution tool to calculate 12 times 12 times 12.",
]


def _wiki_descriptions(topics: list) -> list:
    return [f"Search the web for {topic} and report a brief summary." for topic in topics]


def _weather_descriptions(cities: list) -> list:
    return [f"Get the current weather for {city}." for city in cities]


SWEEP_TASKS = [
    {
        "name": "catalog_sweep",
        "user_task": "Look up 18 different items and categories in the product catalog and report a summary of prices, stock, and ratings.",
        "descriptions": CATALOG_QUERIES,
    },
    {
        "name": "wikipedia_sweep",
        "user_task": "Research 18 different, unrelated general-knowledge topics and report a brief summary of each.",
        "descriptions": _wiki_descriptions(WIKIPEDIA_TOPICS),
    },
    {
        "name": "mixed_sweep",
        "user_task": "Complete 18 independent tasks spanning weather checks, product catalog lookups, topic research, and quick calculations, and report a summary.",
        "descriptions": (
            _weather_descriptions(MIXED_WEATHER_CITIES)
            + MIXED_CATALOG_QUERIES
            + _wiki_descriptions(MIXED_WIKIPEDIA_TOPICS)
            + MIXED_PYTHON_TASKS
        ),
    },
    {
        "name": "catalog_sweep_2",
        "user_task": "Look up 14 different category/price/rating queries in the product catalog and report a summary.",
        "descriptions": CATALOG_QUERIES_2,
    },
    {
        "name": "wikipedia_sweep_4",
        "user_task": "Research 16 different, unrelated general-knowledge topics and report a brief summary of each.",
        "descriptions": _wiki_descriptions(WIKIPEDIA_TOPICS_4),
    },
    {
        "name": "wikipedia_sweep_5",
        "user_task": "Research 16 different, unrelated general-knowledge topics and report a brief summary of each.",
        "descriptions": _wiki_descriptions(WIKIPEDIA_TOPICS_5),
    },
    {
        "name": "weather_sweep_1",
        "user_task": "Check the current weather across 18 different cities around the world and report a summary.",
        "descriptions": _weather_descriptions(WEATHER_CITIES_1),
    },
    {
        "name": "weather_sweep_2",
        "user_task": "Check the current weather across 18 different cities around the world and report a summary.",
        "descriptions": _weather_descriptions(WEATHER_CITIES_2),
    },
    {
        "name": "mixed_sweep_2",
        "user_task": "Complete 14 independent tasks spanning weather checks, product catalog lookups, topic research, and quick calculations, and report a summary.",
        "descriptions": (
            _weather_descriptions(MIXED_WEATHER_2) + MIXED_CATALOG_2 + _wiki_descriptions(MIXED_WIKI_2) + MIXED_PYTHON_2
        ),
    },
    {
        "name": "wikipedia_sweep_6",
        "user_task": "Research 16 different, unrelated general-knowledge topics and report a brief summary of each.",
        "descriptions": _wiki_descriptions(WIKIPEDIA_TOPICS_6),
    },
]


def _log_failure(name: str, error: Exception) -> None:
    with open(ERROR_LOG_PATH, "a") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat()}] sweep task {name!r}\n")
        f.write(f"{traceback.format_exc()}\n")
        f.write("-" * 70 + "\n")


async def run_one_sweep_task(sweep: dict, trace_path: str):
    subtasks = _independent_subtasks(sweep["descriptions"])
    results = await dispatch_subtasks(subtasks, trace_log_path=trace_path)
    return synthesize(sweep["user_task"], subtasks, results, trace_log_path=trace_path)


def _trace_is_complete(trace_path: str) -> bool:
    if not os.path.exists(trace_path):
        return False
    try:
        with open(trace_path) as f:
            lines = [json.loads(l) for l in f]
    except (json.JSONDecodeError, OSError):
        return False
    return any(l.get("agent_id") == "synthesizer" and l.get("event_type") == "final_answer" for l in lines)


async def run_sweep_batch(max_new: int = None):
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

    for i, sweep in enumerate(SWEEP_TASKS, start=1):
        trace_path = f"{TRACES_DIR}/run_sweep_{i}.jsonl"

        if _trace_is_complete(trace_path):
            print(f"\n=== sweep task {i}/{len(SWEEP_TASKS)}: {sweep['name']} -- already complete, skipping")
            continue

        if max_new is not None and newly_run >= max_new:
            print(f"\n=== reached --max-new {max_new}, stopping (rerun this script to continue with the rest)")
            break

        if os.path.exists(trace_path):
            os.remove(trace_path)  # partial/failed leftover

        print(f"\n=== sweep task {i}/{len(SWEEP_TASKS)}: {sweep['name']} ({len(sweep['descriptions'])} independent sub-tasks)")
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

    print(f"\n=== sweep batch invocation complete: {succeeded} succeeded, {failed} failed this run ===")
    if failed:
        print(f"    see {ERROR_LOG_PATH} for failure details")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new", type=int, default=None, help="run at most this many NOT-yet-completed tasks, then stop")
    args = parser.parse_args()
    asyncio.run(run_sweep_batch(max_new=args.max_new))
