"""
Orchestrator

A planning agent breaks a task into a dependency graph of
sub-tasks (DAG); a scheduler runs them as real concurrent agents; and a
synthesizer combines their results into one final answer.
"""

import asyncio
import json

from trace_agent import cache_metrics, call_model_with_retry, log_step, run_agent


PLANNER_SYSTEM_PROMPT = """You are a planning agent. Given a user task, break it into a small \
number of independent sub-tasks that can be handed off to separate worker agents.

Each worker agent has access to these four tools -- plan sub-tasks that a worker can \
actually complete with them, and pick whichever tools genuinely fit the task rather than \
defaulting to the same one or two every time:
- get_weather(city): real current conditions for a real city.
- web_search(query): real web/reference lookup for facts, trivia, or general information.
- execute_python(code): runs a short Python snippet (calculations, data processing, logic \
checks) and returns its output.
- query_database(query_type, ...): looks up products in a local general retail catalog \
(electronics, books, kitchenware, office supplies, toys, beauty) -- search by name/category/price, \
fetch by id, or list categories.

Rules:
- Each sub-task must be small enough for one worker agent to complete on its own \
(ex: "get the weather for Tokyo", not "plan my whole trip").
- If two sub-tasks don't need each other's output, they must NOT depend on each other, \
so they can run in parallel.
- Only add a dependency when a sub-task genuinely needs another sub-task's result as input \
(ex:. a "compare and recommend" step needs the individual lookups to finish first).
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
    """
    - models sometimes wrap JSON in code fences even when told not to
    - strips those fences off before parsing
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _validate_plan(subtasks: list[dict]) -> None:
    """
    - checks the plan isn't empty
    - checks every sub-task id is unique
    - checks every depends_on entry points at a real, different sub-task
    - checks for dependency cycles via topological sort (kahn's algorithm)
    """
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
    """
    - asks the planner model to break user_task into sub-tasks
    - returns a list of dicts: id, description, depends_on
    """
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_task},
    ]

    response = call_model_with_retry(messages, tools=None)
    raw_text = response.message.content or ""
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


# sub-agent dispatch

def _format_upstream_context(dep_results: dict) -> str:
    """
    - turns {sub_task_id: result_text} into a plain-text block
    - a downstream sub-agent's prompt reads this to see its prerequisites' results
    """
    lines = ["Results from prerequisite sub-tasks:"]
    for dep_id, text in dep_results.items():
        lines.append(f"- {dep_id}: {text}")
    return "\n".join(lines)


async def _run_subtask(subtask: dict, tasks_by_id: dict, planner_id: str, trace_log_path: str):
    """
    - waits for this sub-task's dependencies, if any, before starting
    - tasks_by_id maps sub-task id to its asyncio task
    - sub-tasks with no dependency in common never await each other, so
      the event loop is free to run them at the same time
    - parent_id always points at the planner, keeping depth a simple
      one-hop calculation; the real fan-in/fan-out graph is preserved
      separately via depends_on, logged once per sub-agent
    - run_agent's API call is blocking, so it's routed through a real OS
      thread via asyncio.to_thread; otherwise independent sub-agents
      would just take turns instead of truly overlapping
    """
    dep_ids = subtask.get("depends_on", [])
    dep_results = {}
    if dep_ids:
        results = await asyncio.gather(*(tasks_by_id[d] for d in dep_ids))
        dep_results = dict(zip(dep_ids, results))

    agent_id = f"subagent_{subtask['id']}"
    upstream_context = _format_upstream_context(dep_results) if dep_results else None

    log_step(
        0,
        "spawn",
        json.dumps({"depends_on": dep_ids}),
        agent_id=agent_id,
        parent_id=planner_id,
        trace_log_path=trace_log_path,
        extra={"depends_on": dep_ids},
    )

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
    """
    - runs every sub-task, respecting depends_on
    - returns {sub_task_id: result_text} once everything has completed
    """
    tasks_by_id = {}
    for st in subtasks:
        tasks_by_id[st["id"]] = asyncio.create_task(
            _run_subtask(st, tasks_by_id, planner_id, trace_log_path)
        )
    results = await asyncio.gather(*tasks_by_id.values())
    return dict(zip(tasks_by_id.keys(), results))


# synthesis

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a synthesis agent. You are given the original user task and the "
    "results of the sub-tasks that were run to address it. Combine them into "
    "a single, direct final answer to the original task. Integrate the "
    "sub-task results into one coherent answer rather than repeating them "
    "verbatim."
)


def _sink_subtask_ids(subtasks: list[dict]) -> list[str]:
    """
    - finds sub-tasks that nothing else in the plan depends on
    - these are the DAG's terminal outputs, what synthesis actually needs
      to see, regardless of how many dependency layers came before them
    """
    depended_on = set()
    for st in subtasks:
        depended_on.update(st.get("depends_on", []))
    return [st["id"] for st in subtasks if st["id"] not in depended_on]


def synthesize(
    user_task: str,
    subtasks: list[dict],
    results: dict,
    planner_id: str = "planner",
    trace_log_path: str = None,
) -> str:
    """
    - runs one final agent that combines every sink sub-task's result
    - produces the answer to the original user task
    """
    sink_ids = _sink_subtask_ids(subtasks)
    sink_results = {sid: results[sid] for sid in sink_ids}

    synthesis_prompt = (
        f"Original task: {user_task}\n\n"
        "Sub-task results:\n"
        + "\n".join(f"- {sid}: {text}" for sid, text in sink_results.items())
    )

    log_step(
        0,
        "spawn",
        json.dumps({"depends_on": sink_ids}),
        agent_id="synthesizer",
        parent_id=planner_id,
        trace_log_path=trace_log_path,
        extra={"depends_on": sink_ids},
    )
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": synthesis_prompt},
    ]
    response = call_model_with_retry(messages, tools=None)
    # logged after the call, like trace_agent.run_agent's loop, so the real
    # per-call cache-timing fields land on the same trace row
    log_step(
        0,
        "model_call",
        synthesis_prompt,
        agent_id="synthesizer",
        parent_id=planner_id,
        trace_log_path=trace_log_path,
        extra=cache_metrics(response),
    )
    final_text = response.message.content or ""

    log_step(0, "final_answer", final_text, agent_id="synthesizer", parent_id=planner_id, trace_log_path=trace_log_path)
    print(f"\n[synthesizer] Final answer:\n{final_text}")
    return final_text


async def orchestrate(user_task: str, trace_log_path: str = None) -> str:
    """
    - full Phase A pipeline: plan, then concurrent dispatch, then synthesis
    """
    plan = plan_task(user_task)
    results = await dispatch_subtasks(plan, trace_log_path=trace_log_path)
    return synthesize(user_task, plan, results, trace_log_path=trace_log_path)


if __name__ == "__main__":
    task = "Search for who invented the telephone, check the current weather in Boston, and find a book under $15 in the product catalog with a rating above 4.5."
    final_answer = asyncio.run(orchestrate(task, trace_log_path="traces/run_orchestrator_test.jsonl"))
    print("\n=== FINAL ANSWER ===")
    print(final_answer)