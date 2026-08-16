"""
Demo

A live, runnable, end-to-end walkthrough of the whole Foresite pipeline on
one fresh task: real orchestrator run -> real trace -> real leakage-checked
features -> LRU vs the raw predictor vs the Hybrid policy. Meant to be run
in front of someone, not batched.

python3 demo.py                    # runs the built-in example task
python3 demo.py "your task here"   # runs a custom task

Everything downstream of trace generation reuses the project's existing,
already-verified functions directly (feature_extraction.build_rows_for_trace
and assert_no_leakage, kv-cache-simulator's run_lru/run_predictor_dynamic/
run_hybrid, run_eviction_benchmark's build_combined_touches, and
run_sweep_holdout_eviction_benchmark's sweep_holdout_trace_ids) rather than
reimplementing any of that logic here.

The eviction sim does NOT run on the fresh trace alone. run_hybrid's live
pressure signal (see its docstring in kv-cache-simulator.py) only ever
switches into predictor mode once its working-set window has accumulated
enough distinct items -- calibrated against combined sequences of
hundreds-to-thousands of events, the same scale run_eviction_benchmark.py
uses to verify the README's numbers. A single ~20-50 touch demo trace never
reaches that scale, so Hybrid would be structurally stuck at exact LRU
parity every time -- not a bug, just the wrong regime to evaluate it in.
Instead, the fresh trace is appended AFTER the same held-out sweep-heavy
background traffic (features_sweep_holdout.csv -- 10 traces, 746 distinct
items, never used in training or calibration) that README.md's calibration
table is actually measured against -- this mirrors how a real KV cache
operates (continuously warm from prior traffic, never cold-started per
task) and reaches the specific scale AND workload regime (sweep-heavy,
many one-off lookups) where the hybrid's advantage is real and verified.
The general run_eviction_benchmark.py held-out split is NOT used here --
README.md is explicit that normal, non-sweep traces are a regime where
"LRU already performs close to Belady's theoretical optimum," so background
traffic drawn from there would (correctly) show little to no hybrid
advantage; it isn't the regime the hybrid was built to help with.

The step-by-step comparison only looks at steps inside the fresh task's OWN
region (after the background traffic, so the pressure signal has already
warmed up) -- this keeps the printed disagreements about the task that was
actually just run live, not about the background traffic. A line is only
printed where BOTH policies evicted something at that exact step AND their
victims differ -- steps where they agreed are skipped, since there's
nothing to usefully contrast there.
"""

import asyncio
import importlib.util
import os
import sys

import joblib
import pandas as pd

from feature_extraction import assert_no_leakage, build_rows_for_trace
from orchestrator import orchestrate
from run_eviction_benchmark import build_combined_touches
from run_sweep_holdout_eviction_benchmark import SWEEP_HOLDOUT_FEATURES_CSV_PATH, sweep_holdout_trace_ids
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

# 3% of the COMBINED (background + live task) sequence's distinct items:
# inside the 1-5% band the README's calibration table verifies a real,
# consistent hybrid-over-lru margin (+2.6pp / +2.1pp on the two
# independently tested datasets, specifically at 3%). Meaningful here
# because, unlike a bare fresh trace, the combined sequence is now at the
# same hundreds-to-thousands-of-events scale that calibration was measured
# against -- see the module docstring.
DEMO_CACHE_FRACTION = 0.03

# safety-net floor in case the held-out background set ever shrinks well
# below its current ~1700 events -- shouldn't trigger in practice at that
# scale, but keeps cache_size from ever degenerating to 1 (which would
# force every policy to evict the sole cached item every time, leaving no
# real choice for them to disagree about).
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


def _diverged_steps(lru_by_step: dict, other_by_step: dict, min_step: int = 0) -> tuple:
    shared_steps = sorted(s for s in set(lru_by_step) & set(other_by_step) if s >= min_step)
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


def _print_comparisons(
    events: list, lru_decisions: list, hybrid_decisions: list, predictor_decisions: list, live_region_start: int
) -> None:
    lru_by_step = {d["step"]: d["evicted"] for d in lru_decisions}
    hybrid_by_step = {
        d["step"]: {
            "evicted": d["evicted"],
            "predicted_score": d["predicted_score"],
            "note": "predictor-driven" if d["mode"] == "predictor" else "LRU-fallback, low pressure",
        }
        for d in hybrid_decisions
    }

    # restricted to live_region_start onward: the fresh task's own steps,
    # evicted after the background traffic has already warmed up the
    # pressure signal -- see module docstring for why the background is
    # there at all. Excludes the background traffic's own steps so the
    # printed disagreements are about the task that was just run live.
    shared_steps, diverged = _diverged_steps(lru_by_step, hybrid_by_step, min_step=live_region_start)
    print(
        f"Eviction steps (within your task's own run) where both LRU and Hybrid evicted something: "
        f"{len(shared_steps)} ({len(diverged)} of those disagreed on which item)\n"
    )

    if diverged:
        _print_one_comparison(events, "Hybrid", lru_by_step, hybrid_by_step, diverged)
        return

    # Falls back to showing the raw predictor's own disagreements (still
    # restricted to the live region) if Hybrid genuinely stayed at parity
    # with LRU for every eviction in this particular run -- possible if,
    # e.g., the task was too short to produce any evictions in its own
    # region at all, or the predictor and LRU happened to agree throughout.
    print(
        "(Hybrid didn't override LRU on any eviction within your task's own run this time.\n"
        " Showing where the underlying predictor, which Hybrid gates behind a live pressure\n"
        " check, would have differed from LRU instead:)\n"
    )
    predictor_by_step = {
        d["step"]: {"evicted": d["evicted"], "predicted_score": d["predicted_score"], "note": None}
        for d in predictor_decisions
    }
    _, predictor_diverged = _diverged_steps(lru_by_step, predictor_by_step, min_step=live_region_start)
    if not predictor_diverged:
        print("(No disagreements from the raw predictor either, within your task's own run.)\n")
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
            f"{MODEL_PATH} not found. Run `python3 train_predictor.py --extra-input features_sweep.csv` "
            "first (uses feature CSVs already in the repo, takes under a minute) to generate it, "
            "then re-run demo.py. The --extra-input flag matters: without it, the model won't have "
            "learned the sweep-heavy reuse pattern the predictor/hybrid need to show a real advantage."
        )

    trace_path = _run_trace(task)

    rows, touches, depends_on_map = build_rows_for_trace(trace_path)
    assert_no_leakage(rows, touches, depends_on_map)
    print(f"Extracted {len(rows)} real, leakage-checked feature rows from this trace.\n")

    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]

    live_touches = _build_touches(rows, feature_names)

    # background traffic: the same sweep-heavy held-out set (never seen
    # during training or calibration) README.md's calibration table is
    # measured against -- see module docstring for why the fresh trace
    # alone isn't enough, and why this specific dataset (not the general
    # held-out split) is the one that matters here.
    print("Loading held-out sweep-heavy background traffic (same dataset README.md's calibration table verifies against)...")
    background_trace_ids = sweep_holdout_trace_ids()
    background_touches = build_combined_touches(feature_names, SWEEP_HOLDOUT_FEATURES_CSV_PATH, background_trace_ids)
    live_region_start = len(background_touches)
    sim_touches = background_touches + live_touches

    events = [t["item_id"] for t in sim_touches]
    n_distinct = len(set(events))
    raw_cache_size = max(1, round(DEMO_CACHE_FRACTION * n_distinct))
    cache_size = max(MIN_DEMO_CACHE_SIZE, raw_cache_size)
    floor_note = f" (floored from {raw_cache_size} -- see MIN_DEMO_CACHE_SIZE)" if cache_size != raw_cache_size else ""

    print(
        f"Combined sequence: {len(background_touches)} background events "
        f"({len(background_trace_ids)} held-out traces) + {len(live_touches)} events from your task just now "
        f"= {len(events)} total, {n_distinct} distinct items\n"
    )
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
    print("(hit rates are over the whole combined sequence, matching how README.md's table was measured)\n")

    print("=== Where LRU and Hybrid disagreed on your task's own eviction decisions ===\n")
    _print_comparisons(events, lru_decisions, hybrid_decisions, predictor_decisions, live_region_start)

    diff_pp = (hybrid_hit_rate - lru_hit_rate) * 100
    print("=== This one demo run ===")
    print(f"LRU hit rate:    {lru_hit_rate:.1%}")
    print(f"Hybrid hit rate: {hybrid_hit_rate:.1%}  ({diff_pp:+.1f}pp vs LRU)")
    print(
        "\nThis is one fresh run of the pipeline against a fixed held-out background -- illustrative of the "
        "mechanism, not a re-run of the full statistical evaluation. See README.md for that evaluation across "
        "150+ real traces and multiple independent held-out datasets."
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
