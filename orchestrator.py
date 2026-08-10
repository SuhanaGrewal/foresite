"""
Phase A: multi-agent orchestrator for Foresite.

This module adds a planner that decomposes a user task into a dependency
graph of sub-tasks, and a scheduler that dispatches those sub-tasks as real
concurrent agents. The synthesis step that combines their results is added
separately on top of this.
"""

import asyncio
import json

from trace_agent import call_model_with_retry, log_step, run_agent


PLANNER_SYSTEM_PROMPT = """You are a planning agent. Given a user task, break it into a small \
number of independent sub-tasks that can be handed off to separate worker agents.

Rules:
- Each sub-task must be small enough for one worker agent to complete on its own \
(e.g. "get the weather for Tokyo", not "plan my whole trip").
- If two sub-tasks don't need each other's output, they must NOT depend on each other, \
so they can run in parallel.
- Only add a dependency when a sub-task genuinely needs another sub-task's result as input \
(e.g. a "compare and recommend" step needs the individual lookups to finish first).
- Give every sub-task a short, unique, lowercase snake_case id.

Respond with ONLY a JSON object, no prose, no markdown code fences, matching exactly this shape:
{
  "subtasks": [
    {"id": "some_id", "description": "what this sub-agent should do", "depends_on": []},
    {"id": "another_id", "description": "...", "depends_on": ["some_id"]}
  ]
}
"""


def _strip_code_fence(text: str) -> str:
    """Models sometimes wrap JSON in ```json ... ``` even when told not to."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # drop opening fence (with optional language tag)
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _validate_plan(subtasks: list[dict]) -> None:
    """Basic sanity checks so a malformed plan fails loudly instead of breaking
    the scheduler later with a confusing KeyError or infinite wait."""
    if not subtasks:
        raise ValueError("Plan has zero sub-tasks.")

    ids = [st["id"] for st in subtasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Plan has duplicate sub-task ids: {ids}")

    id_set = set(ids)
    for st in subtasks:
        for dep in st.get("depends_on", []):
            if dep not in id_set:
                raise ValueError(
                    f"Sub-task '{st['id']}' depends on unknown id '{dep}'"
                )
            if dep == st["id"]:
                raise ValueError(f"Sub-task '{st['id']}' depends on itself")

    # cycle check via topological sort (Kahn's algorithm)
    remaining = {st["id"]: set(st.get("depends_on", [])) for st in subtasks}
    resolved = set()
    while remaining:
        ready = [sid for sid, deps in remaining.items() if deps <= resolved]
        if not ready:
            raise ValueError(f"Plan has a dependency cycle among: {list(remaining)}")
        for sid in ready:
            resolved.add(sid)
            del remaining[sid]


def plan_task(user_task: str) -> list[dict]:
    """Ask the planner model to decompose user_task into a list of sub-task
    dicts: {"id": str, "description": str, "depends_on": [str, ...]}."""
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_task},
    ]

    response = call_model_with_retry(messages, tools=None)
    raw_text = response.choices[0].message.content or ""
    cleaned = _strip_code_fence(raw_text)

    try:
        parsed = json.loads(cleaned)
        subtasks = parsed["subtasks"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(
            f"Planner did not return valid plan JSON. Raw response:\n{raw_text}"
        ) from e

    _validate_plan(subtasks)
    return subtasks


# --- sub-agent dispatch -----------------------------------------------

def _format_upstream_context(dep_results: dict) -> str:
    """Turn {sub_task_id: result_text} into a plain-text block a downstream
    sub-agent's prompt can read."""
    lines = ["Results from prerequisite sub-tasks:"]
    for dep_id, text in dep_results.items():
        lines.append(f"- {dep_id}: {text}")
    return "\n".join(lines)


async def _run_subtask(subtask: dict, tasks_by_id: dict, planner_id: str, trace_log_path: str):
    """Wait for this sub-task's dependencies (if any), then run it.

    tasks_by_id maps sub-task id -> its asyncio.Task. Sub-tasks with no
    dependency in common never await each other, so the event loop is free
    to run them at the same time.
    """
    dep_ids = subtask.get("depends_on", [])
    dep_results = {}
    if dep_ids:
        results = await asyncio.gather(*(tasks_by_id[d] for d in dep_ids))
        dep_results = dict(zip(dep_ids, results))

    agent_id = f"subagent_{subtask['id']}"
    upstream_context = _format_upstream_context(dep_results) if dep_results else None

    # parent_id always points at the planner (keeps depth = 1 hop from
    # planner); the real fan-in/fan-out graph is preserved here via
    # depends_on, logged once per sub-agent.
    log_step(
        0,
        "spawn",
        json.dumps({"depends_on": dep_ids}),
        agent_id=agent_id,
        parent_id=planner_id,
        trace_log_path=trace_log_path,
        extra={"depends_on": dep_ids},
    )

    # run_agent's OpenAI call is blocking (sync), so route it through a real
    # OS thread -- otherwise independent sub-agents would just take turns on
    # the single asyncio event loop thread instead of truly overlapping.
    result_text = await asyncio.to_thread(
        run_agent,
        subtask["description"],
        agent_id=agent_id,
        parent_id=planner_id,
        trace_log_path=trace_log_path,
        upstream_context=upstream_context,
    )
    return result_text


async def dispatch_subtasks(subtasks: list[dict], planner_id: str = "planner", trace_log_path: str = None) -> dict:
    """Run every sub-task, respecting depends_on, and return
    {sub_task_id: result_text} once everything has completed."""
    tasks_by_id = {}
    for st in subtasks:
        tasks_by_id[st["id"]] = asyncio.create_task(
            _run_subtask(st, tasks_by_id, planner_id, trace_log_path)
        )
    results = await asyncio.gather(*tasks_by_id.values())
    return dict(zip(tasks_by_id.keys(), results))


if __name__ == "__main__":
    task = "Compare the weather in Delhi, Los Angeles, and Tokyo right now, and recommend which city is best for outdoor sightseeing today."
    plan = plan_task(task)
    print(json.dumps(plan, indent=2))

    results = asyncio.run(dispatch_subtasks(plan, trace_log_path="traces/run_orchestrator_test.jsonl"))
    print("\n=== sub-task results ===")
    print(json.dumps(results, indent=2))
