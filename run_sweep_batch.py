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

Traces are written to traces/run_sweep_N.jsonl, kept separate from the
general batch/FRAMES traces. See build_sweep_features.py for why they're
folded into a separate features_sweep.csv rather than the main features.csv.
"""

import asyncio
import os
import traceback
from datetime import datetime, timezone

from frames_data import build_corpus
from orchestrator import dispatch_subtasks, synthesize
from trace_agent import clear_search_corpus, set_search_corpus

TRACES_DIR = "traces"
ERROR_LOG_PATH = "sweep_batch_errors.log"


def _independent_subtasks(descriptions: list) -> list:
    """turns a flat list of descriptions into an all-independent subtask plan (no depends_on -- maximizes concurrency/fan-out)"""
    return [{"id": f"lookup_{i}", "description": desc, "depends_on": []} for i, desc in enumerate(descriptions)]


WEATHER_CITIES = [
    "Accra", "Casablanca", "Istanbul", "Bangkok", "Jakarta", "Manila",
    "Seoul", "Osaka", "Mumbai", "Karachi", "Tehran", "Riyadh", "Dubai",
    "Athens", "Lisbon", "Warsaw", "Bucharest", "Helsinki", "Bogota", "Perth",
]

CATALOG_QUERIES_1 = [
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

WIKIPEDIA_TOPICS_1 = [
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

CATALOG_QUERIES_2 = [
    "Search the product catalog for 'wireless' and report price and stock.",
    "Search the product catalog for '4K' and report price and stock.",
    "Search the product catalog for 'mechanical' and report price and stock.",
    "Find products under $15 in the beauty category.",
    "Find products under $20 in the kitchenware category.",
    "Find the highest-rated product in the books category.",
    "Find the highest-rated product in the office_supplies category.",
    "Search the product catalog for 'bluetooth' and report price and stock.",
    "Search the product catalog for 'iron' and report price and stock.",
    "Search the product catalog for 'vitamin' and report price and stock.",
]

WIKIPEDIA_TOPICS_2 = [
    "the history of chess",
    "the invention of the light bulb",
    "the causes of the Cold War",
    "the history of the Berlin Wall",
    "the discovery of Pluto",
    "the history of the Ottoman Empire",
    "the invention of the steam engine",
    "the causes of the American Revolution",
    "the history of the Colosseum",
    "the discovery of the periodic table",
]

# a real FRAMES question (google/frames-benchmark) -- the highest gold-document
# count found across the dataset (11 real Wikipedia sources; one malformed
# entry in the raw dataset combining two URLs was split and its fragment
# anchor dropped). Hardcoded rather than re-selected at runtime, so this
# sweep task is reproducible regardless of the dataset's own shuffling.
FRAMES_SWEEP_QUESTION = {
    "prompt": "Who had the best career batting average out of every player to hit a home run in the 2002 World Series matchup between the Anaheim Angeles and San Francisco Giants?",
    "answer": "Barry Bonds with a .298 lifetime batting average.",
    "wiki_links": [
        "https://en.wikipedia.org/wiki/2002_World_Series",
        "https://en.wikipedia.org/wiki/Barry_Bonds",
        "https://en.wikipedia.org/wiki/Darin_Erstad",
        "https://en.wikipedia.org/wiki/David_Bell_(baseball)",
        "https://en.wikipedia.org/wiki/Jeff_Kent",
        "https://en.wikipedia.org/wiki/J._T._Snow",
        "https://en.wikipedia.org/wiki/Reggie_Sanders",
        "https://en.wikipedia.org/wiki/Rich_Aurilia",
        "https://en.wikipedia.org/wiki/Scott_Spiezio",
        "https://en.wikipedia.org/wiki/Shawon_Dunston",
        "https://en.wikipedia.org/wiki/Tim_Salmon",
        "https://en.wikipedia.org/wiki/Troy_Glaus",
    ],
}

FRAMES_SWEEP_TOPICS = [
    "Look up information about the 2002 World Series and report a brief summary.",
    "Look up information about Barry Bonds and report his career batting average.",
    "Look up information about Darin Erstad and report his career batting average.",
    "Look up information about David Bell (baseball) and report his career batting average.",
    "Look up information about Jeff Kent and report his career batting average.",
    "Look up information about J. T. Snow and report his career batting average.",
    "Look up information about Reggie Sanders and report his career batting average.",
    "Look up information about Rich Aurilia and report his career batting average.",
    "Look up information about Scott Spiezio and report his career batting average.",
    "Look up information about Shawon Dunston and report his career batting average.",
    "Look up information about Tim Salmon and report his career batting average.",
    "Look up information about Troy Glaus and report his career batting average.",
    "Look up information about the history of the World Series and report a brief summary.",
    "Look up information about the Anaheim Angels franchise history.",
    "Look up information about the San Francisco Giants franchise history.",
    "Look up information about career batting average as a baseball statistic.",
]


SWEEP_TASKS = [
    {
        "name": "weather_sweep",
        "user_task": "Check the current weather across 20 different cities around the world and report a summary.",
        "descriptions": [f"Get the current weather for {city}." for city in WEATHER_CITIES],
        "frames_question": None,
    },
    {
        "name": "catalog_sweep",
        "user_task": "Look up 18 different items and categories in the product catalog and report a summary of prices, stock, and ratings.",
        "descriptions": CATALOG_QUERIES_1,
        "frames_question": None,
    },
    {
        "name": "wikipedia_sweep",
        "user_task": "Research 18 different, unrelated general-knowledge topics and report a brief summary of each.",
        "descriptions": [f"Search the web for {topic} and report a brief summary." for topic in WIKIPEDIA_TOPICS_1],
        "frames_question": None,
    },
    {
        "name": "mixed_sweep",
        "user_task": "Complete 18 independent tasks spanning weather checks, product catalog lookups, topic research, and quick calculations, and report a summary.",
        "descriptions": (
            [f"Get the current weather for {city}." for city in MIXED_WEATHER_CITIES]
            + MIXED_CATALOG_QUERIES
            + [f"Search the web for {topic} and report a brief summary." for topic in MIXED_WIKIPEDIA_TOPICS]
            + MIXED_PYTHON_TASKS
        ),
        "frames_question": None,
    },
    {
        "name": "frames_multihop_sweep",
        "user_task": (
            "Research every player who hit a home run in the 2002 World Series matchup between the "
            "Angels and Giants, plus related background, to determine who had the best career batting average."
        ),
        "descriptions": FRAMES_SWEEP_TOPICS,
        "frames_question": FRAMES_SWEEP_QUESTION,
    },
    {
        "name": "large_catalog_and_wiki_sweep",
        "user_task": "Look up 10 different product catalog queries and research 10 different, unrelated general-knowledge topics, and report a summary.",
        "descriptions": (
            CATALOG_QUERIES_2 + [f"Search the web for {topic} and report a brief summary." for topic in WIKIPEDIA_TOPICS_2]
        ),
        "frames_question": None,
    },
]


def _log_failure(name: str, error: Exception) -> None:
    with open(ERROR_LOG_PATH, "a") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat()}] sweep task {name!r}\n")
        f.write(f"{traceback.format_exc()}\n")
        f.write("-" * 70 + "\n")


async def run_one_sweep_task(sweep: dict, trace_path: str):
    subtasks = _independent_subtasks(sweep["descriptions"])

    if sweep["frames_question"] is not None:
        corpus = build_corpus(sweep["frames_question"])
        print(f"    fetched {len(corpus)}/{len(sweep['frames_question']['wiki_links'])} gold documents")
        set_search_corpus(corpus)

    try:
        results = await dispatch_subtasks(subtasks, trace_log_path=trace_path)
        final_answer = synthesize(sweep["user_task"], subtasks, results, trace_log_path=trace_path)
    finally:
        if sweep["frames_question"] is not None:
            clear_search_corpus()

    return final_answer


async def run_sweep_batch():
    os.makedirs(TRACES_DIR, exist_ok=True)
    succeeded = 0
    failed = 0

    for i, sweep in enumerate(SWEEP_TASKS, start=1):
        trace_path = f"{TRACES_DIR}/run_sweep_{i}.jsonl"
        if os.path.exists(trace_path):
            os.remove(trace_path)

        print(f"\n=== sweep task {i}/{len(SWEEP_TASKS)}: {sweep['name']} ({len(sweep['descriptions'])} independent sub-tasks)")

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

    print(f"\n=== sweep batch complete: {succeeded} succeeded, {failed} failed ===")
    if failed:
        print(f"    see {ERROR_LOG_PATH} for failure details")


if __name__ == "__main__":
    asyncio.run(run_sweep_batch())
