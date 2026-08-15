"""
Demo

A live, runnable, end-to-end walkthrough of the whole Foresite pipeline on
one fresh task: real orchestrator run -> real trace -> real leakage-checked
features -> LRU vs the raw predictor vs the Hybrid policy, on that trace's
own real event sequence. Meant to be run in front of someone, not batched.

python3 demo.py                    # runs the built-in example task
python3 demo.py "your task here"   # runs a custom task

Everything downstream of trace generation reuses the project's existing,
already-verified functions directly (feature_extraction.build_rows_for_trace
and assert_no_leakage, kv-cache-simulator's run_lru/run_predictor_dynamic/
run_hybrid) rather than reimplementing any of that logic here.

Only the LRU-vs-Hybrid step-by-step comparison is specific to this file:
LRU and Hybrid are each simulated independently over the identical fresh
event sequence, and a line is only printed for a trace position where BOTH
policies evicted something at that exact step AND their victims differ --
steps where they agreed, or where one side's cache had already diverged
into a hit while the other missed, are skipped, since there is nothing to
usefully contrast there. This keeps the output focused on genuine
disagreements instead of a wall of restated agreement.
"""

import asyncio
import glob
import importlib.util
import os
import sys

import joblib
import pandas as pd

from feature_extraction import assert_no_leakage, build_rows_for_trace
from orchestrator import orchestrate
from train_predictor import clean_feature_values

# kv-cache-simulator.py's filename has a hyphen, so it isn't a valid Python
# module name and can't be reached with a normal `import` statement.
_kv_sim_spec = importlib.util.spec_from_file_location("kv_cache_simulator", "kv-cache-simulator.py")
_kv_cache_simulator = importlib.util.module_from_spec(_kv_sim_spec)
_kv_sim_spec.loader.exec_module(_kv_cache_simulator)
run_lru = _kv_cache_simulator.run_lru
run_predictor_dynamic = _kv_cache_simulator.run_predictor_dynamic
run_hybrid = _kv_cache_simulator.run_hybrid

DEFAULT_TASK = (
    "Search for who invented the telephone, check the current weather in Boston, "
    "and find a book under $15 in the product catalog with a rating above 4.5."
)
DEMO_TRACE_PATH = "traces/demo_run.jsonl"
MODEL_PATH = "model.joblib"

# cap on how many per-step disagreements to print -- a real trace can have
# dozens; printing all of them defeats the point of "only show what
# differed" (still readable, not an overwhelming wall of output).
MAX_COMPARISONS_SHOWN = 8

# 3% of this run's distinct items: inside the 1-5% band the README's
# calibration table verifies a real, consistent hybrid-over-lru margin at
# (+2.6pp / +2.1pp on the two independently tested datasets, specifically
# at 3%) -- picked over 1-2% so a single small demo trace still produces
# enough real eviction events for the comparison section to have content.
DEMO_CACHE_FRACTION = 0.03

# the 1-5% calibration was measured against combined sequences of 700-1700+
# distinct items (run_eviction_benchmark.py concatenates dozens of held-out
# traces); a single fresh demo trace has maybe 15-40. applying 3% literally
# there rounds to cache_size=1, which is degenerate -- with only one slot,
# every policy is forced to evict the sole cached item every time, so there
# is never a real CHOICE among multiple candidates for them to disagree
# about. floored so the demo can show an actual eviction decision, not just
# a trivially forced one; the demo prints both the fraction and the floored
# size so this is visible, not hidden.
MIN_DEMO_CACHE_SIZE = 3


def _run_trace(task: str) -> str:
    if os.path.exists(DEMO_TRACE_PATH):
        os.remove(DEMO_TRACE_PATH)
    os.makedirs("traces", exist_ok=True)
    print(f"Task: {task}\n")
    print("Running the real orchestrator (planner -> concurrent sub-agents -> synthesizer)...")
    final_answer = asyncio.run(orchestrate(task, trace_log_path=DEMO_TRACE_PATH))
    print(f"\n[final answer]\n{final_answer}\n")
    return DEMO_TRACE_PATH


def _build_touches(rows: list, feature_names: list) -> list:
    """
    adapts build_rows_for_trace()'s real per-row feature values (already
    leakage-checked) into the {"item_id", "static_features"} shape
    run_predictor_dynamic/run_hybrid expect. can't reuse
    run_eviction_benchmark.build_combined_touches directly -- that reads
    from features.csv by trace_id, and this fresh trace was never written
    into that file. cleaning goes through clean_feature_values, the same
    shared helper training used, so the model sees the same value
    distribution here it saw during training (bool -> int, NaN -> sentinel).
    """
    cleaned_rows = clean_feature_values(pd.DataFrame(rows)).to_dict("records")
    return [
        {"item_id": r["item_id"], "static_features": {name: r[name] for name in feature_names}}
        for r in cleaned_rows
    ]


def _describe_fate(events: list, step: int, item_id: str) -> str:
    for j in range(step + 1, len(events)):
        if events[j] == item_id:
            return f"reused {j - step} step(s) later"
    return "never reused again"


def _diverged_steps(lru_by_step: dict, other_by_step: dict) -> tuple:
    shared_steps = sorted(set(lru_by_step) & set(other_by_step))
    diverged = [s for s in shared_steps if lru_by_step[s] != other_by_step[s]["evicted"]]
    return shared_steps, diverged


def _print_one_comparison(events: list, label: str, lru_by_step: dict, other_by_step: dict, diverged: list) -> None:
    shown = diverged[:MAX_COMPARISONS_SHOWN]
    for step in shown:
        lru_victim = lru_by_step[step]
        other_decision = other_by_step[step]
        other_victim = other_decision["evicted"]

        lru_fate = _describe_fate(events, step, lru_victim)
        other_fate = _describe_fate(events, step, other_victim)
        extra = f" ({other_decision['note']})" if other_decision.get("note") else ""

        print(f"Step {step}:")
        print(f"  LRU evicts        {lru_victim}  -- {lru_fate}")
        print(
            f"  {label} evicts {other_victim}  -- {other_fate}  "
            f"(predicted reuse probability: {other_decision['predicted_score']:.2f}{extra})"
        )
        print()

    remaining = len(diverged) - len(shown)
    if remaining > 0:
        print(f"...and {remaining} more disagreement(s) not shown.\n")


def _print_comparisons(events: list, lru_decisions: list, hybrid_decisions: list, predictor_decisions: list) -> None:
    lru_by_step = {d["step"]: d["evicted"] for d in lru_decisions}
    hybrid_by_step = {
        d["step"]: {
            "evicted": d["evicted"],
            "predicted_score": d["predicted_score"],
            "note": "predictor-driven" if d["mode"] == "predictor" else "LRU-fallback, low pressure",
        }
        for d in hybrid_decisions
    }

    shared_steps, diverged = _diverged_steps(lru_by_step, hybrid_by_step)
    print(
        f"Eviction steps where both LRU and Hybrid evicted something: {len(shared_steps)} "
        f"({len(diverged)} of those disagreed on which item)\n"
    )

    if diverged:
        _print_one_comparison(events, "Hybrid", lru_by_step, hybrid_by_step, diverged)
        return

    # Hybrid only overrides LRU when its live working-set pressure signal
    # (see run_hybrid's docstring) dips low enough -- that signal was
    # calibrated against combined sequences of hundreds-to-thousands of
    # events, and structurally can't reach a triggering value within one
    # short single-task demo trace (its window only grows to
    # working_set_fraction * position, so a ~50-touch trace never gives it
    # enough history). Hybrid degenerating to exact LRU parity here is
    # therefore an honest, expected outcome, not a bug -- but it leaves
    # nothing to show, so fall back to showing where the underlying raw
    # predictor (the signal Hybrid gates behind that pressure check) would
    # have differed from LRU, so the demo still has real content to explain.
    print(
        "(Hybrid never overrode LRU on this run -- its pressure signal needs more trace history than a single\n"
        " short task provides to trigger predictor mode; see run_hybrid's docstring in kv-cache-simulator.py.\n"
        " Showing where the underlying predictor, which Hybrid gates behind that safety check, would have\n"
        " differed from LRU instead:)\n"
    )
    predictor_by_step = {
        d["step"]: {"evicted": d["evicted"], "predicted_score": d["predicted_score"], "note": None}
        for d in predictor_decisions
    }
    _, predictor_diverged = _diverged_steps(lru_by_step, predictor_by_step)
    if not predictor_diverged:
        print("(No disagreements from the raw predictor either on this run.)\n")
        return
    _print_one_comparison(events, "Predictor", lru_by_step, predictor_by_step, predictor_diverged)


def run_demo(task: str) -> None:
    # checked before running the (real, 1-3 minute) orchestrator below --
    # model.joblib is gitignored (a generated artifact, not checked into
    # the repo), so a fresh clone won't have it yet. failing here instead
    # of after the orchestrator call saves that wait, and this specific
    # message is deliberately distinct from the generic try/except in
    # main() below: a missing file is a one-time setup step, not the local
    # model having an off moment, and should read as one.
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run `python3 train_predictor.py` first "
            "(uses the features.csv already in the repo, takes under a minute) "
            "to generate it, then re-run demo.py."
        )

    trace_path = _run_trace(task)

    rows, touches, depends_on_map = build_rows_for_trace(trace_path)
    assert_no_leakage(rows, touches, depends_on_map)
    print(f"Extracted {len(rows)} real, leakage-checked feature rows from this trace.\n")

    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]

    sim_touches = _build_touches(rows, feature_names)
    events = [t["item_id"] for t in sim_touches]
    n_distinct = len(set(events))
    raw_cache_size = max(1, round(DEMO_CACHE_FRACTION * n_distinct))
    cache_size = max(MIN_DEMO_CACHE_SIZE, raw_cache_size)
    floor_note = f" (floored from {raw_cache_size} -- see MIN_DEMO_CACHE_SIZE)" if cache_size != raw_cache_size else ""

    print(f"Simulating eviction at cache_size={cache_size}{floor_note} ({DEMO_CACHE_FRACTION:.0%} of {n_distinct} distinct items)\n")

    lru_hit_rate, lru_decisions = run_lru(events, cache_size, record_decisions=True)
    predictor_hit_rate, predictor_decisions = run_predictor_dynamic(
        sim_touches, cache_size, model, feature_names, record_decisions=True
    )
    hybrid_hit_rate, hybrid_decisions = run_hybrid(sim_touches, cache_size, model, feature_names, record_decisions=True)

    header = f"{'policy':<12}{'hit rate':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'LRU':<12}{lru_hit_rate:>9.1%}")
    print(f"{'Predictor':<12}{predictor_hit_rate:>9.1%}")
    print(f"{'Hybrid':<12}{hybrid_hit_rate:>9.1%}")
    print()

    print("=== Where LRU and Hybrid disagreed ===\n")
    _print_comparisons(events, lru_decisions, hybrid_decisions, predictor_decisions)

    diff_pp = (hybrid_hit_rate - lru_hit_rate) * 100
    print("=== This one demo run ===")
    print(f"LRU hit rate:    {lru_hit_rate:.1%}")
    print(f"Hybrid hit rate: {hybrid_hit_rate:.1%}  ({diff_pp:+.1f}pp vs LRU)")
    print(
        "\nThis is a single fresh run on a small trace -- illustrative, not a "
        "statistically meaningful result. See README.md for the full evaluation "
        "across 150+ real traces and multiple held-out datasets."
    )


def main() -> None:
    task = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK
    try:
        run_demo(task)
    except FileNotFoundError as e:
        # one-time setup gap (e.g. model.joblib not generated yet), not a
        # flaky-run issue -- kept out of the generic retry message below so
        # it doesn't read as "just try again" when it won't help.
        print(f"\n[setup incomplete: {e}]")
    except Exception as e:
        if os.path.exists(DEMO_TRACE_PATH):
            os.remove(DEMO_TRACE_PATH)
        print(f"\n[demo failed: {type(e).__name__}: {e}]")
        print("This task didn't complete cleanly (the small local model can have an off moment).")
        print("Try again, or run with no arguments to use the default example task.")


if __name__ == "__main__":
    main()
