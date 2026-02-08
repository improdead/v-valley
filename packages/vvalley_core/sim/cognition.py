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


def _heuristic_poignancy(context: dict[str, Any]) -> dict[str, Any]:
    description = str(context.get("description") or "").lower()
    if "idle" in description or "sleeping" in description:
        return {"score": 1}
    if any(kw in description for kw in ("chat", "talk", "convers")):
        return {"score": 4}
    if any(kw in description for kw in ("reflect", "plan", "goal")):
        return {"score": 5}
    return {"score": 3}


def _heuristic_conversation(context: dict[str, Any]) -> dict[str, Any]:
    a_name = str(context.get("agent_a_name") or "Agent A")
    b_name = str(context.get("agent_b_name") or "Agent B")
    a_activity = str(context.get("agent_a_activity") or "their routine")
    b_activity = str(context.get("agent_b_activity") or "their routine")
    step = int(context.get("step") or 0)
    templates = (
        [
            {"speaker": a_name, "text": f"Hey {b_name}, how is {b_activity} going?"},
            {"speaker": b_name, "text": f"Going well! What about your {a_activity}?"},
            {"speaker": a_name, "text": "Making progress. Let's catch up later."},
        ],
        [
            {"speaker": a_name, "text": f"Need any help with {b_activity}?"},
            {"speaker": b_name, "text": f"I'm alright for now. How about {a_activity}?"},
            {"speaker": a_name, "text": "All good. See you around."},
        ],
        [
            {"speaker": a_name, "text": "Town feels lively today."},
            {"speaker": b_name, "text": f"Yeah, I'm still working on {b_activity}."},
            {"speaker": a_name, "text": "Good luck with that!"},
        ],
    )
    idx = _hash_int(f"convo:{a_name}:{b_name}:{step}") % len(templates)
    return {
        "utterances": templates[idx],
        "summary": f"{a_name} and {b_name} exchanged thoughts about their activities.",
    }


def _heuristic_focal_points(context: dict[str, Any]) -> dict[str, Any]:
    statements = str(context.get("recent_statements") or "")
    words = [w.strip().lower() for w in statements.replace("\n", " ").split() if len(w.strip()) > 3]
    seen: set[str] = set()
    questions: list[str] = []
    for word in words:
        if word not in seen and len(questions) < 3:
            seen.add(word)
            questions.append(f"What is most important about {word}?")
    if not questions:
        questions = ["What should I focus on next?"]
    return {"questions": questions}


def _heuristic_reflection_insights(context: dict[str, Any]) -> dict[str, Any]:
    focal_point = str(context.get("focal_point") or "recent events")
    return {
        "insights": [
            {
                "insight": f"Reflecting on {focal_point} suggests maintaining current priorities.",
                "evidence_indices": [0],
            },
        ],
    }


def _heuristic_schedule_decomp(context: dict[str, Any]) -> dict[str, Any]:
    task_desc = str(context.get("task_description") or "activity")
    total_mins = int(context.get("total_duration_mins") or 60)
    if total_mins >= 90:
        third = total_mins // 3
        remainder = total_mins - third * 2
        return {
            "subtasks": [
                {"description": f"prepare for {task_desc}", "duration_mins": third},
                {"description": f"doing {task_desc}", "duration_mins": third},
                {"description": f"wrap up {task_desc}", "duration_mins": remainder},
            ],
        }
    return {
        "subtasks": [
            {"description": task_desc, "duration_mins": total_mins},
        ],
    }


def _heuristic_conversation_summary(context: dict[str, Any]) -> dict[str, Any]:
    agent_name = str(context.get("agent_name") or "Agent")
    partner_name = str(context.get("partner_name") or "someone")
    return {
        "memo": f"{agent_name} had a brief exchange with {partner_name} about daily activities.",
        "planning_thought": f"{agent_name} should follow up with {partner_name} when an opportunity arises.",
    }


def _heuristic_conversation_turn(context: dict[str, Any]) -> dict[str, Any]:
    agent_name = str(context.get("speaking_agent_name") or "Agent")
    partner_name = str(context.get("partner_name") or "someone")
    turn_number = int(context.get("turn_number") or 0)
    activity = str(context.get("speaking_agent_activity") or "their routine")
    if turn_number == 0:
        return {"utterance": f"Hey {partner_name}, how are things going?", "end_conversation": False}
    if turn_number >= 4:
        return {"utterance": f"Good talking with you, {partner_name}. See you around!", "end_conversation": True}
    templates = (
        f"I've been busy with {activity}. What about you?",
        f"That's interesting. Things in town have been lively lately.",
        f"Sounds good. Let's catch up again soon!",
    )
    idx = _hash_int(f"turn:{agent_name}:{turn_number}") % len(templates)
    end = turn_number >= 3
    return {"utterance": templates[idx], "end_conversation": end}


def _heuristic_daily_schedule_gen(context: dict[str, Any]) -> dict[str, Any]:
    agent_name = str(context.get("agent_name") or "Agent")
    return {
        "schedule": [
            {"description": "sleep", "duration_mins": 360},
            {"description": "morning routine", "duration_mins": 60},
            {"description": "work or task", "duration_mins": 300},
            {"description": "lunch", "duration_mins": 60},
            {"description": "socialize", "duration_mins": 120},
            {"description": "work follow-up", "duration_mins": 240},
            {"description": "leisure", "duration_mins": 180},
            {"description": "dinner", "duration_mins": 60},
            {"description": "night wind-down", "duration_mins": 60},
        ],
        "currently": f"{agent_name} is maintaining a stable daily routine in town.",
    }


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

    def _run_task(
        self,
        *,
        task_name: str,
        context: dict[str, Any],
        heuristic_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        result = self._runner.run(
            task_name=task_name,
            town_id=str(context.get("town_id") or "") or None,
            agent_id=str(context.get("agent_id") or "") or None,
            context=context,
            heuristic_fn=heuristic_fn,
        )
        output = dict(result.output)
        output["policy_task"] = task_name
        output["route"] = result.route
        output["used_tier"] = result.used_tier
        output["context_trimmed"] = result.context_trimmed
        return output

    def plan_move(self, context: dict[str, Any]) -> dict[str, Any]:
        task_name = self._task_for_scope(str(context.get("planning_scope") or "short_action"))
        return self._run_task(task_name=task_name, context=context, heuristic_fn=_heuristic_move)

    def score_poignancy(self, context: dict[str, Any]) -> dict[str, Any]:
        """Score the importance/poignancy of a perceived event (1-10)."""
        return self._run_task(task_name="poignancy", context=context, heuristic_fn=_heuristic_poignancy)

    def generate_conversation(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate a memory-grounded conversation between two agents."""
        return self._run_task(task_name="conversation", context=context, heuristic_fn=_heuristic_conversation)

    def generate_focal_points(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate reflection focal points from recent memory statements."""
        return self._run_task(task_name="focal_points", context=context, heuristic_fn=_heuristic_focal_points)

    def generate_reflection_insights(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate insights with evidence from retrieved memories for a focal point."""
        return self._run_task(
            task_name="reflection_insights", context=context, heuristic_fn=_heuristic_reflection_insights
        )

    def decompose_schedule(self, context: dict[str, Any]) -> dict[str, Any]:
        """Decompose a schedule block into minute-level subtasks."""
        return self._run_task(task_name="schedule_decomp", context=context, heuristic_fn=_heuristic_schedule_decomp)

    def summarize_conversation(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate post-conversation memo and planning thoughts."""
        return self._run_task(
            task_name="conversation_summary", context=context, heuristic_fn=_heuristic_conversation_summary
        )

    def generate_conversation_turn(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate a single conversation turn for one agent."""
        return self._run_task(
            task_name="conversation_turn", context=context, heuristic_fn=_heuristic_conversation_turn
        )

    def generate_daily_schedule(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate a full daily schedule and identity update for an agent."""
        return self._run_task(
            task_name="daily_schedule_gen", context=context, heuristic_fn=_heuristic_daily_schedule_gen
        )
