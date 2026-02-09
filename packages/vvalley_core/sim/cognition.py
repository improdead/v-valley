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


def _heuristic_pronunciatio(context: dict[str, Any]) -> dict[str, Any]:
    """Keyword-based emoji + event triple heuristic."""
    desc = str(context.get("action_description") or "idle").lower()
    agent = str(context.get("agent_name") or "Agent")
    if "sleep" in desc or "bed" in desc:
        return {"emoji": "💤", "event_triple": {"subject": agent, "predicate": "is", "object": "sleeping"}}
    if any(kw in desc for kw in ("eat", "lunch", "dinner", "food", "cook", "breakfast")):
        return {"emoji": "🍽️", "event_triple": {"subject": agent, "predicate": "is", "object": "eating"}}
    if any(kw in desc for kw in ("work", "task", "study", "write", "research")):
        return {"emoji": "💼", "event_triple": {"subject": agent, "predicate": "is", "object": "working"}}
    if any(kw in desc for kw in ("chat", "talk", "social", "convers", "discuss")):
        return {"emoji": "💬", "event_triple": {"subject": agent, "predicate": "is", "object": "chatting"}}
    if any(kw in desc for kw in ("walk", "roam", "explore", "move", "stroll")):
        return {"emoji": "🚶", "event_triple": {"subject": agent, "predicate": "is", "object": "walking"}}
    if any(kw in desc for kw in ("leisure", "relax", "rest", "read", "paint", "play")):
        return {"emoji": "🎨", "event_triple": {"subject": agent, "predicate": "is", "object": "relaxing"}}
    if any(kw in desc for kw in ("morning", "routine", "shower", "brush")):
        return {"emoji": "🌅", "event_triple": {"subject": agent, "predicate": "is", "object": "doing morning routine"}}
    return {"emoji": "💭", "event_triple": {"subject": agent, "predicate": "is", "object": desc[:60]}}


def _heuristic_reaction_policy(context: dict[str, Any]) -> dict[str, Any]:
    """Hash-based reaction policy heuristic (matching runner's existing logic)."""
    a_id = str(context.get("agent_a_id") or "")
    b_id = str(context.get("agent_b_id") or "")
    step = int(context.get("step") or 0)
    relationship = float(context.get("relationship_score") or 0.0)

    talk_threshold = 22 + min(22, int(relationship * 2.5))
    wait_threshold = min(90, talk_threshold + 28)
    gate = _hash_int(f"reaction:{a_id}:{b_id}:{step}") % 100

    if gate < talk_threshold:
        return {"decision": "talk", "reason": "social affinity favors conversation"}
    if gate < wait_threshold:
        return {"decision": "wait", "reason": "deferred social intent", "wait_steps": 2}
    return {"decision": "ignore", "reason": "not the right moment for interaction"}


def _heuristic_wake_up_hour(context: dict[str, Any]) -> dict[str, Any]:
    """Parse wake-up hour from lifestyle string."""
    lifestyle = str(context.get("lifestyle") or "").lower()
    if any(kw in lifestyle for kw in ("early", "5am", "5 am", "6am", "6 am")):
        return {"wake_up_hour": 6}
    if any(kw in lifestyle for kw in ("late", "9am", "9 am", "10am", "10 am")):
        return {"wake_up_hour": 9}
    # Try to parse "wakes at Xam" pattern
    import re
    match = re.search(r'wakes?\s+(?:up\s+)?(?:at|around)\s+(\d{1,2})', lifestyle)
    if match:
        hour = int(match.group(1))
        if 0 <= hour <= 23:
            return {"wake_up_hour": hour}
    return {"wake_up_hour": 7}


def _heuristic_action_address(context: dict[str, Any]) -> dict[str, Any]:
    """Heuristic for 3-step action address resolution (sector→arena→object)."""
    action_desc = str(context.get("action_description") or "idle").lower()
    available_sectors = context.get("available_sectors") or []
    living_area = str(context.get("living_area") or "").lower()

    # Simple keyword matching to pick a sector
    sector = None
    if any(kw in action_desc for kw in ("sleep", "bed", "wake", "morning routine", "shower")):
        # Prefer living area
        for s in available_sectors:
            if living_area and living_area in str(s).lower():
                sector = s
                break
    elif any(kw in action_desc for kw in ("eat", "lunch", "dinner", "breakfast", "coffee", "cafe")):
        for s in available_sectors:
            sl = str(s).lower()
            if any(kw in sl for kw in ("cafe", "pub", "market")):
                sector = s
                break
    elif any(kw in action_desc for kw in ("study", "class", "college", "lecture")):
        for s in available_sectors:
            if "college" in str(s).lower():
                sector = s
                break
    elif any(kw in action_desc for kw in ("work", "supply", "store")):
        for s in available_sectors:
            sl = str(s).lower()
            if any(kw in sl for kw in ("store", "supply", "market")):
                sector = s
                break
    elif any(kw in action_desc for kw in ("park", "walk", "stroll", "jog")):
        for s in available_sectors:
            if "park" in str(s).lower():
                sector = s
                break
    if sector is None and living_area:
        for s in available_sectors:
            if living_area in str(s).lower():
                sector = s
                break
    if sector is None and available_sectors:
        idx = _hash_int(f"addr:{context.get('agent_id')}:{action_desc}") % len(available_sectors)
        sector = available_sectors[idx]

    # Pick arena/object from available lists
    available_arenas = context.get("available_arenas") or []
    arena = available_arenas[0] if available_arenas else None
    available_objects = context.get("available_objects") or []
    game_object = available_objects[0] if available_objects else None

    return {
        "action_sector": sector,
        "action_arena": arena,
        "action_game_object": game_object,
    }


def _heuristic_object_state(context: dict[str, Any]) -> dict[str, Any]:
    """Heuristic for generating object state description when an agent uses an object."""
    agent_name = str(context.get("agent_name") or "Someone")
    action_desc = str(context.get("action_description") or "idle")
    game_object = str(context.get("game_object") or "object")
    return {
        "object_description": f"{game_object} is being used by {agent_name} ({action_desc})",
        "object_event_triple": {"subject": game_object, "predicate": "is used by", "object": agent_name},
    }


def _heuristic_event_reaction(context: dict[str, Any]) -> dict[str, Any]:
    """Heuristic for deciding whether to react to a perceived event (non-chat)."""
    event_desc = str(context.get("event_description") or "").lower()
    agent_activity = str(context.get("agent_activity") or "").lower()
    # Don't react to idle events or if the agent is busy
    if "idle" in event_desc:
        return {"react": False, "reason": "event is idle"}
    if any(kw in agent_activity for kw in ("sleeping", "conversation", "chatting")):
        return {"react": False, "reason": "agent is busy"}
    # Hash-based gate: ~15% chance to react
    gate = _hash_int(f"react:{context.get('agent_id')}:{event_desc}") % 100
    if gate < 15:
        return {"react": True, "reason": f"curious about: {event_desc[:60]}",
                "new_action": f"investigating {event_desc[:40]}"}
    return {"react": False, "reason": "not interesting enough"}


def _heuristic_relationship_summary(context: dict[str, Any]) -> dict[str, Any]:
    """Heuristic relationship narrative for conversation grounding."""
    agent_name = str(context.get("agent_name") or "Agent")
    partner_name = str(context.get("partner_name") or "someone")
    relationship_score = float(context.get("relationship_score") or 0.0)
    past_interactions = context.get("past_interactions") or []
    if relationship_score >= 4.0:
        tone = "close friends who talk regularly"
    elif relationship_score >= 2.0:
        tone = "acquaintances who have chatted before"
    elif past_interactions:
        tone = "people who have met briefly"
    else:
        tone = "meeting for the first time"
    return {
        "summary": f"{agent_name} and {partner_name} are {tone}.",
        "conversation_tone": "warm" if relationship_score >= 3.0 else "polite",
    }


def _heuristic_first_daily_plan(context: dict[str, Any]) -> dict[str, Any]:
    """Heuristic for first-day plan (more exploratory than subsequent days)."""
    agent_name = str(context.get("agent_name") or "Agent")
    innate = str(context.get("innate") or "")
    learned = str(context.get("learned") or "")
    # First day emphasizes exploration and meeting neighbors
    return {
        "schedule": [
            {"description": "morning routine and getting settled", "duration_mins": 60},
            {"description": "explore the neighborhood", "duration_mins": 120},
            {"description": "visit nearby shops and cafes", "duration_mins": 90},
            {"description": "lunch", "duration_mins": 60},
            {"description": "meet neighbors and introduce yourself", "duration_mins": 120},
            {"description": "settle into your home and unpack", "duration_mins": 90},
            {"description": "dinner", "duration_mins": 60},
            {"description": "evening walk around town", "duration_mins": 60},
            {"description": "wind down and prepare for bed", "duration_mins": 60},
        ],
        "currently": f"{agent_name} is exploring the town and getting settled in.",
    }


def _heuristic_schedule_recomposition(context: dict[str, Any]) -> dict[str, Any]:
    """Heuristic for recomposing the daily schedule after a reaction/chat interruption."""
    inserted_act = str(context.get("inserted_activity") or "responding to event")
    inserted_dur = max(5, int(context.get("inserted_duration_mins") or 30))
    current_schedule = context.get("current_schedule") or []
    current_minute = int(context.get("current_minute_of_day") or 0)

    # Build new schedule: keep everything before current_minute, insert the new activity,
    # then compress remaining activities to fit in the rest of the day.
    new_schedule: list[dict[str, Any]] = []
    elapsed = 0
    inserted = False
    for item in current_schedule:
        desc = str(item.get("description") or "idle")
        dur = max(1, int(item.get("duration_mins") or 60))
        if elapsed + dur <= current_minute:
            # Already completed — keep as-is
            new_schedule.append({"description": desc, "duration_mins": dur})
            elapsed += dur
        elif not inserted:
            # Current activity — truncate at interruption point and insert new activity
            passed = max(0, current_minute - elapsed)
            if passed > 0:
                new_schedule.append({"description": desc, "duration_mins": passed})
            new_schedule.append({"description": inserted_act, "duration_mins": inserted_dur})
            remaining_of_block = dur - passed - inserted_dur
            if remaining_of_block > 5:
                new_schedule.append({"description": desc, "duration_mins": remaining_of_block})
            inserted = True
            elapsed += dur
        else:
            # Future activities — keep
            new_schedule.append({"description": desc, "duration_mins": dur})
            elapsed += dur

    if not inserted:
        new_schedule.append({"description": inserted_act, "duration_mins": inserted_dur})

    # Ensure total ~ 1440 mins
    total = sum(item["duration_mins"] for item in new_schedule)
    if total < 1440:
        new_schedule.append({"description": "rest", "duration_mins": 1440 - total})
    elif total > 1440 and len(new_schedule) > 1:
        # Trim last item
        excess = total - 1440
        last_dur = new_schedule[-1]["duration_mins"]
        if last_dur > excess:
            new_schedule[-1]["duration_mins"] = last_dur - excess
        else:
            new_schedule.pop()

    return {"schedule": new_schedule, "inserted_activity": inserted_act, "inserted_duration_mins": inserted_dur}


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

    def generate_pronunciatio(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate an emoji pronunciatio and event triple for the current action."""
        return self._run_task(
            task_name="pronunciatio", context=context, heuristic_fn=_heuristic_pronunciatio
        )

    def decide_to_talk(self, context: dict[str, Any]) -> dict[str, Any]:
        """LLM-powered reaction policy: should two agents talk, wait, or ignore?"""
        return self._run_task(
            task_name="reaction_policy", context=context, heuristic_fn=_heuristic_reaction_policy
        )

    def generate_wake_up_hour(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate wake-up hour for a new day based on lifestyle."""
        return self._run_task(
            task_name="wake_up_hour", context=context, heuristic_fn=_heuristic_wake_up_hour
        )

    def resolve_action_address(self, context: dict[str, Any]) -> dict[str, Any]:
        """3-step action address resolution: sector→arena→game_object."""
        return self._run_task(
            task_name="action_address", context=context, heuristic_fn=_heuristic_action_address
        )

    def generate_object_state(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate object state description when agent interacts with a game object."""
        return self._run_task(
            task_name="object_state", context=context, heuristic_fn=_heuristic_object_state
        )

    def decide_to_react(self, context: dict[str, Any]) -> dict[str, Any]:
        """Decide whether to react to a perceived non-chat event (change plan)."""
        return self._run_task(
            task_name="event_reaction", context=context, heuristic_fn=_heuristic_event_reaction
        )

    def generate_relationship_summary(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate narrative relationship summary for conversation grounding."""
        return self._run_task(
            task_name="relationship_summary", context=context, heuristic_fn=_heuristic_relationship_summary
        )

    def generate_first_daily_plan(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate first-day plan (more exploratory than subsequent days)."""
        return self._run_task(
            task_name="first_daily_plan", context=context, heuristic_fn=_heuristic_first_daily_plan
        )

    def recompose_schedule(self, context: dict[str, Any]) -> dict[str, Any]:
        """Recompose the daily schedule after a reaction/chat interruption."""
        return self._run_task(
            task_name="schedule_recomposition", context=context, heuristic_fn=_heuristic_schedule_recomposition
        )
