"""
Run Batch

Runs every task in BATCH_TASKS through the full orchestrator pipeline
(plan, concurrent dispatch, synthesis), saving each one to its own trace
file in traces/.

Tasks are deliberately spread across unrelated domains -- geography/weather,
history/science trivia, finance/math, general retail lookup, and multi-tool
combinations of those -- rather than one recurring theme, so the resulting
traces (and any features later extracted from them) reflect varied agent
behavior instead of one narrow, repetitive task shape.
"""

import asyncio
import os

from orchestrator import orchestrate

BATCH_TASKS = [
    "Compare the current weather in Reykjavik, Nairobi, and Singapore, and recommend which city has the most comfortable conditions for outdoor sightseeing today.",
    "Search for when the first successful human heart transplant took place, then calculate how many years ago that was from 2026.",
    "Find an in-stock pair of wireless headphones under $100 in the product catalog, and calculate the total price including 8.5% sales tax.",
    "Search for the boiling point of water at sea level in Celsius, then calculate what that is in Fahrenheit.",
    "List the product categories available in the catalog, then find the highest-rated item in the kitchenware category.",
    "Check the current weather in Tokyo. If it's raining, search the product catalog for an indoor-friendly item to recommend instead of going outside.",
    "Search for the history and impact of the printing press, then summarize it in two sentences.",
    "Calculate the compound interest on $2,000 invested at 5% annually for 8 years, compounded monthly.",
]


async def run_batch():
    os.makedirs("traces", exist_ok=True)

    for i, task in enumerate(BATCH_TASKS, start=1):
        trace_path = f"traces/run_{i}.jsonl"
        if os.path.exists(trace_path):
            os.remove(trace_path)

        print(f"\n=== Running task {i}/{len(BATCH_TASKS)}: {task}")
        await orchestrate(task, trace_log_path=trace_path)
        print(f"Trace written to {trace_path}")


if __name__ == "__main__":
    asyncio.run(run_batch())
