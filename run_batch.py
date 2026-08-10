"""
Run Batch

Runs every task in BATCH_TASKS through the full orchestrator pipeline
(plan, concurrent dispatch, synthesis), saving each one to its own trace
file in traces/.
"""

import asyncio
import os
import time

from orchestrator import orchestrate

BATCH_TASKS = [
    "Compare the weather in Delhi, Los Angeles, and Tokyo right now, and recommend which city is best for outdoor sightseeing today.",
    "Check trail conditions and weather for both Runyon Canyon and Griffith Park, then tell me which is the better hike this afternoon.",
    "I'm deciding between a beach day and a hiking day this weekend. Check the weather for both Santa Monica and Runyon Canyon and recommend one.",
    "Plan a 2-day trip to Los Angeles: check the weather for Saturday and Sunday, search for one indoor and one outdoor activity, and write a short itinerary.",
    "Compare hiking conditions across Runyon Canyon, Griffith Park, and Topanga State Park, and rank them best to worst for today.",
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

        if i < len(BATCH_TASKS):
            time.sleep(10)  # brief pause between tasks to respect rate limits


if __name__ == "__main__":
    asyncio.run(run_batch())
