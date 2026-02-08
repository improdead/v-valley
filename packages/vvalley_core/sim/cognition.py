"""Modern cognition adapters for simulation tasks."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Callable

from packages.vvalley_core.llm.task_runner import PolicyTaskRunner


PolicyLookup = Callable[[str], Any]
LogSink = Callable[[dict[str, Any]], None]


def _hash_int(value: str) -> int:
    return int(sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _heuristic_move(context: dict[str, Any]) -> dict[str, Any]:
    agent_id = str(context.get("agent_id") or "")
    step = int(context.get("step") or 0)
    directions = (
        {"dx": 0, "dy": 0, "reason": "pause_and_observe"},
        {"dx": 0, "dy": -1, "reason": "explore_north"},
        {"dx": 1, "dy": 0, "reason": "explore_east"},
        {"dx": 0, "dy": 1, "reason": "explore_south"},
        {"dx": -1, "dy": 0, "reason": "explore_west"},
    )
    idx = _hash_int(f"{agent_id}:{step}") % len(directions)
    return directions[idx]


class CognitionPlanner:
    """Planner facade with situation-dependent policy routing."""

    def __init__(self, *, policy_lookup: PolicyLookup, log_sink: LogSink | None = None) -> None:
        self._runner = PolicyTaskRunner(policy_lookup=policy_lookup, log_sink=log_sink)

    @staticmethod
    def _task_for_scope(scope: str) -> str:
        normalized = str(scope or "").strip().lower()
        if normalized in {"long", "long_term", "long-term", "long_term_plan"}:
            return "long_term_plan"
        if normalized in {"day", "daily", "daily_plan", "medium"}:
            return "daily_plan"
        if normalized in {"short", "short_action", "action"}:
            return "short_action"
        return "plan_move"

    def plan_move(self, context: dict[str, Any]) -> dict[str, Any]:
        task_name = self._task_for_scope(str(context.get("planning_scope") or "short_action"))
        result = self._runner.run(
            task_name=task_name,
            town_id=str(context.get("town_id") or "") or None,
            agent_id=str(context.get("agent_id") or "") or None,
            context=context,
            heuristic_fn=_heuristic_move,
        )
        output = dict(result.output)
        output["policy_task"] = task_name
        output["route"] = result.route
        output["used_tier"] = result.used_tier
        output["context_trimmed"] = result.context_trimmed
        return output
