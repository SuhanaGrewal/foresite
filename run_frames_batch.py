"""
Run FRAMES Batch

- runs real FRAMES benchmark questions through the full orchestrator pipeline, with web_search grounded in that question's real
  Wikipedia articles (fetched by frames_data.py).
- see trace_agent.py's real_search_tool docstring for how this adapts FRAMES (a QA benchmark) into a multi-step agentic task.
- prints the model's final answer next to FRAMES' gold answer for each question.
- a single question failing (e.g. the planner producing malformed JSON, which
  raises ValueError -- the small local model does this occasionally) does NOT
  abort the batch: the failure is logged to ERROR_LOG_PATH with the question
  and error, that question's partial trace (if any) is discarded, and the
  batch moves on to the next question.
"""

import asyncio
import os
import traceback
from datetime import datetime, timezone

from frames_data import build_corpus, load_frames_questions
from orchestrator import orchestrate
from trace_agent import clear_search_corpus, set_search_corpus

N_QUESTIONS = 70
ERROR_LOG_PATH = "frames_batch_errors.log"


def _log_failure(index: int, question: dict, error: Exception) -> None:
    with open(ERROR_LOG_PATH, "a") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat()}] question {index}: {question['prompt']!r}\n")
        f.write(f"{traceback.format_exc()}\n")
        f.write("-" * 70 + "\n")


async def run_frames_batch():
    os.makedirs("traces", exist_ok=True)
    questions = load_frames_questions(n=N_QUESTIONS)
    succeeded = 0
    failed = 0

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
        except Exception as e:
            failed += 1
            _log_failure(i, q, e)
            if os.path.exists(trace_path):
                os.remove(trace_path)
            print(f"    FAILED: {type(e).__name__}: {e}")
            continue
        finally:
            clear_search_corpus()

        succeeded += 1
        print(f"    model answer: {final_answer}")
        print(f"    trace written to {trace_path}")

    print(f"\n=== FRAMES batch complete: {succeeded} succeeded, {failed} failed ===")
    if failed:
        print(f"    see {ERROR_LOG_PATH} for failure details")


if __name__ == "__main__":
    asyncio.run(run_frames_batch())
