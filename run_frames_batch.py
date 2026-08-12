"""
Run FRAMES Batch

Runs real FRAMES benchmark questions (Krishna et al.) through the full
orchestrator pipeline, with web_search grounded in that question's real
gold Wikipedia articles (fetched by frames_data.py). See
trace_agent.py's real_search_tool docstring for how this adapts FRAMES
(a single-shot QA benchmark) into a multi-step agentic task, and what's
deliberately simplified relative to IntentKV's full adaptation.

Prints the model's final answer next to FRAMES' gold answer for each
question -- this is NOT an automated grader (that needs a real exact-match
scorer or LLM-judge, out of scope here), just something to eyeball.
"""

import asyncio
import os

from frames_data import build_corpus, load_frames_questions
from orchestrator import orchestrate
from trace_agent import clear_search_corpus, set_search_corpus

N_QUESTIONS = 10


async def run_frames_batch():
    os.makedirs("traces", exist_ok=True)
    questions = load_frames_questions(n=N_QUESTIONS)

    for i, q in enumerate(questions, start=1):
        trace_path = f"traces/run_frames_{i}.jsonl"
        if os.path.exists(trace_path):
            os.remove(trace_path)

        print(f"\n=== FRAMES question {i}/{len(questions)}: {q['prompt']}")
        print(f"    gold answer: {q['answer']}")

        corpus = build_corpus(q)
        print(f"    fetched {len(corpus)}/{len(q['wiki_links'])} gold documents")

        set_search_corpus(corpus)
        try:
            final_answer = await orchestrate(q["prompt"], trace_log_path=trace_path)
        finally:
            clear_search_corpus()

        print(f"    model answer: {final_answer}")
        print(f"    trace written to {trace_path}")


if __name__ == "__main__":
    asyncio.run(run_frames_batch())
