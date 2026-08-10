"""
Phase A: multi-agent orchestrator for Foresite.

This module adds a planner that decomposes a user task into a dependency
graph of sub-tasks. Later pieces (sub-agent dispatch, synthesis) build on
top of this file. For now this only covers planning, so it can be tested
and confirmed in isolation before anything else depends on it.
"""

import json

from trace_agent import client, MODEL, call_model_with_retry


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


if __name__ == "__main__":
    task = "Compare the weather in Delhi, Los Angeles, and Tokyo right now, and recommend which city is best for outdoor sightseeing today."
    plan = plan_task(task)
    print(json.dumps(plan, indent=2))
