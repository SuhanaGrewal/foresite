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

Scoped down to 3 sweep tasks (54 sub-agent runs total), not the original
6 (110 runs): 110 real sub-agent runs against qwen2.5:7b proved too heavy
for this laptop. Kept the three tasks that best isolate different sweep
flavors -- catalog_sweep (one tool, many targets), wikipedia_sweep (one
tool, many unrelated topics), mixed_sweep (all four tools) -- still
genuinely wide-fanout and structurally distinct from each other, just
less total load. Also runs on qwen2.5:1.5b instead of trace_agent.py's
committed 7b default: this test is about the ACCESS PATTERN (many
one-off touches vs. a few reused ones), not model reasoning quality, so
the smaller/faster model doesn't compromise what's being measured here
(7b was chosen in trace_agent.py for reliable multi-step reasoning,
which isn't what sweep sub-tasks -- single simple lookups each -- need).

Traces are written to traces/run_sweep_N.jsonl, kept separate from the
general batch/FRAMES traces.
"""

import asyncio
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


SWEEP_TASKS = [
    {
        "name": "catalog_sweep",
        "user_task": "Look up 18 different items and categories in the product catalog and report a summary of prices, stock, and ratings.",
        "descriptions": CATALOG_QUERIES,
    },
    {
        "name": "wikipedia_sweep",
        "user_task": "Research 18 different, unrelated general-knowledge topics and report a brief summary of each.",
        "descriptions": [f"Search the web for {topic} and report a brief summary." for topic in WIKIPEDIA_TOPICS],
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
