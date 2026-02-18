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

    # Pick arena/object from available lists with spatial awareness
    available_arenas = context.get("available_arenas") or []
    available_objects = context.get("available_objects") or []
    agent_x = int(context.get("agent_x") or 0)
    agent_y = int(context.get("agent_y") or 0)
    arena_positions = context.get("arena_positions") or {}
    object_positions = context.get("object_positions") or {}
    arena = _pick_nearest(available_arenas, arena_positions, agent_x, agent_y, action_desc)
    game_object = _pick_nearest(available_objects, object_positions, agent_x, agent_y, action_desc)

    return {
        "action_sector": sector,
        "action_arena": arena,
        "action_game_object": game_object,
    }


def _pick_nearest(
    names: list[str],
    positions: dict[str, tuple[int, int]],
    ax: int,
    ay: int,
    action_desc: str,
) -> str | None:
    """Pick nearest name by keyword match first, then manhattan distance."""
    if not names:
        return None
    # Keyword match first
    words = [w for w in action_desc.split() if len(w) > 3]
    for n in names:
        nl = n.lower()
        if any(w in nl for w in words):
            return n
    # Then closest by manhattan distance
    if positions:
        def dist(n: str) -> int:
            pos = positions.get(n)
            if pos is None:
                return 9999
            return abs(ax - pos[0]) + abs(ay - pos[1])
        return min(names, key=dist)
    return names[0]


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


def _heuristic_contextual_prioritize(context: dict[str, Any]) -> dict[str, Any]:
    scores = context.get("branch_scores") or {}
    if not isinstance(scores, dict) or not scores:
        return {"branch": "routine", "reason": "default routine priority"}
    best_branch = "routine"
    best_score = -99999.0
    for name, score in scores.items():
        try:
            val = float(score)
        except Exception:
            continue
        if val > best_score:
            best_score = val
            best_branch = str(name)
    return {
        "branch": best_branch,
        "reason": f"{best_branch} scored highest in current context",
    }


def _heuristic_identity_evolution(context: dict[str, Any]) -> dict[str, Any]:
    traits = [str(item) for item in (context.get("current_traits") or []) if str(item).strip()]
    habits = [str(item) for item in (context.get("current_habits") or []) if str(item).strip()]
    values = [str(item) for item in (context.get("current_values") or []) if str(item).strip()]
    attitudes = context.get("social_attitudes") or {}
    if not isinstance(attitudes, dict):
        attitudes = {}

    conversation_count = len(context.get("recent_conversations") or [])
    if conversation_count >= 5 and "socially engaged" not in traits:
        traits.append("socially engaged")
    if conversation_count >= 8 and "checks in with neighbors" not in habits:
        habits.append("checks in with neighbors")
    positive = sum(1 for value in attitudes.values() if "trust" in str(value).lower() or "friendly" in str(value).lower())
    if positive >= 2 and "values community" not in values:
        values.append("values community")
    if not traits:
        traits = ["steady"]

    return {
        "updated_traits": traits[:8],
        "updated_habits": habits[:8],
        "updated_values": values[:8],
        "updated_attitudes": attitudes,
        "evolution_narrative": "Recent social experiences have slightly reshaped this agent's identity.",
    }


def _heuristic_social_attitude(context: dict[str, Any]) -> dict[str, Any]:
    summary = str(context.get("conversation_summary") or "").lower()
    if any(token in summary for token in ("argue", "conflict", "lie", "deceive", "distrust")):
        return {"attitude": "wary", "relationship_delta": -1.2}
    if any(token in summary for token in ("warm", "friendly", "helpful", "supportive", "laugh")):
        return {"attitude": "trusting", "relationship_delta": 1.8}
    return {"attitude": "neutral", "relationship_delta": 0.6}


def _pick_hash_candidate(seed: str, candidates: list[str]) -> str | None:
    clean = [str(value).strip() for value in candidates if str(value).strip()]
    if not clean:
        return None
    idx = _hash_int(seed) % len(clean)
    return clean[idx]


def _heuristic_werewolf_target(context: dict[str, Any]) -> dict[str, Any]:
    candidates = [str(value) for value in (context.get("candidate_agent_ids") or []) if str(value)]
    suspicion = context.get("suspicion") or {}
    if not isinstance(suspicion, dict):
        suspicion = {}
    if candidates:
        ranked = sorted(
            candidates,
            key=lambda aid: (-int(suspicion.get(aid, 0)), str(aid)),
        )
        return {"target_agent_id": ranked[0]}
    return {"target_agent_id": None}


def _heuristic_werewolf_protect(context: dict[str, Any]) -> dict[str, Any]:
    candidates = [str(value) for value in (context.get("candidate_agent_ids") or []) if str(value)]
    suspicion = context.get("suspicion") or {}
    if not isinstance(suspicion, dict):
        suspicion = {}
    if not candidates:
        return {"target_agent_id": None}
    ranked = sorted(candidates, key=lambda aid: (-int(suspicion.get(aid, 0)), str(aid)))
    return {"target_agent_id": ranked[0]}


def _heuristic_werewolf_inspect(context: dict[str, Any]) -> dict[str, Any]:
    candidates = [str(value) for value in (context.get("candidate_agent_ids") or []) if str(value)]
    if not candidates:
        return {"target_agent_id": None}
    seed = f"inspect:{context.get('agent_id')}:{context.get('round_number')}:{context.get('step')}"
    return {"target_agent_id": _pick_hash_candidate(seed, candidates)}


def _heuristic_werewolf_speech(context: dict[str, Any]) -> dict[str, Any]:
    role = str(context.get("role") or "").strip().lower()
    suggested = str(context.get("suggested_target") or "").strip()
    alive = [str(value) for value in (context.get("alive_agent_ids") or []) if str(value)]
    target = suggested or _pick_hash_candidate(
        f"speech:{context.get('agent_id')}:{context.get('step')}",
        [aid for aid in alive if aid != str(context.get("agent_id") or "")],
    )
    if role in {"werewolf", "alpha_werewolf"} and target:
        return {"target_agent_id": target, "line": f"{target} feels too quiet. Something is off."}
    if target:
        return {"target_agent_id": target, "line": f"I suspect {target} based on yesterday's vote."}
    return {"target_agent_id": None, "line": "I need more discussion before voting."}


def _heuristic_werewolf_vote(context: dict[str, Any]) -> dict[str, Any]:
    candidates = [str(value) for value in (context.get("alive_agent_ids") or []) if str(value)]
    suspicion = context.get("suspicion") or {}
    if not isinstance(suspicion, dict):
        suspicion = {}
    if not candidates:
        return {"target_agent_id": None}
    ranked = sorted(candidates, key=lambda aid: (-int(suspicion.get(aid, 0)), str(aid)))
    return {"target_agent_id": ranked[0]}


def _card_rank_value(card: str) -> int:
    face = str(card).upper()[:1]
    mapping = {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "T": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
        "A": 14,
    }
    return mapping.get(face, 0)


def _heuristic_anaconda_pass_cards(context: dict[str, Any]) -> dict[str, Any]:
    hand = [str(card).upper() for card in (context.get("hand") or []) if str(card)]
    try:
        count = max(1, int(context.get("count") or 1))
    except Exception:
        count = 1
    if len(hand) <= count:
        return {"cards": hand[:count]}
    ranked = sorted(hand, key=lambda card: (_card_rank_value(card), card))
    return {"cards": ranked[:count]}


def _heuristic_anaconda_bet(context: dict[str, Any]) -> dict[str, Any]:
    stack = max(0, int(context.get("stack") or 0))
    visible = max(0, int(context.get("visible_cards") or 0))
    pot = max(0, int(context.get("pot") or 0))
    if stack <= 0:
        return {"bet": 0, "fold": False, "all_in": True}

    base = 2 + (visible * 2)
    pressure = min(6, pot // 30)
    bet = min(stack, max(0, base + pressure))
    if stack <= max(3, bet // 2) and visible >= 3:
        return {"bet": 0, "fold": True, "all_in": False}
    if stack <= bet:
        return {"bet": stack, "fold": False, "all_in": True}
    return {"bet": bet, "fold": False, "all_in": False}


def _heuristic_blackjack_bet(context: dict[str, Any]) -> dict[str, Any]:
    stack = max(0, int(context.get("stack") or 0))
    min_bet = max(1, int(context.get("min_bet") or 1))
    max_bet = max(min_bet, int(context.get("max_bet") or min_bet))
    hand_index = max(1, int(context.get("hand_index") or 1))
    max_hands = max(hand_index, int(context.get("max_hands") or hand_index))
    if stack <= 0:
        return {"bet": 0}
    if stack < min_bet:
        return {"bet": stack}
    progress = hand_index / max_hands
    anchor = min_bet + int((max_bet - min_bet) * min(1.0, 0.2 + progress * 0.45))
    conservative = max(min_bet, stack // 6)
    proposed = max(min_bet, min(max_bet, max(anchor, conservative)))
    return {"bet": min(stack, proposed)}


def _heuristic_blackjack_action(context: dict[str, Any]) -> dict[str, Any]:
    total = max(0, int(context.get("total") or 0))
    dealer_upcard = str(context.get("dealer_upcard") or "").strip().upper()
    can_double = bool(context.get("can_double"))
    stack = max(0, int(context.get("stack") or 0))
    bet = max(0, int(context.get("bet") or 0))
    dealer_rank = dealer_upcard[0] if dealer_upcard else ""
    dealer_strong = dealer_rank in {"7", "8", "9", "T", "J", "Q", "K", "A"}

    if can_double and total in {10, 11} and stack >= bet:
        return {"action": "double"}
    if total <= 11:
        return {"action": "hit"}
    if total >= 17:
        return {"action": "stand"}
    if total >= 13 and not dealer_strong:
        return {"action": "stand"}
    return {"action": "hit"}


def _heuristic_holdem_action(context: dict[str, Any]) -> dict[str, Any]:
    to_call = max(0, int(context.get("to_call") or 0))
    stack = max(0, int(context.get("stack") or 0))
    can_raise = "raise" in [str(v).strip().lower() for v in (context.get("allowed_actions") or [])]
    if to_call <= 0:
        if can_raise and stack > int(context.get("bet_unit") or 10) * 2:
            return {"action": "raise"}
        return {"action": "check"}
    if to_call > max(1, stack // 2):
        return {"action": "fold"}
    if can_raise and stack > to_call + int(context.get("bet_unit") or 10):
        return {"action": "raise"}
    return {"action": "call"}


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

    def contextual_prioritize(self, context: dict[str, Any]) -> dict[str, Any]:
        """Select which planning branch should dominate this tick."""
        return self._run_task(
            task_name="contextual_prioritize",
            context=context,
            heuristic_fn=_heuristic_contextual_prioritize,
        )

    def evolve_identity(self, context: dict[str, Any]) -> dict[str, Any]:
        """Consolidate repeated social/reflective patterns into evolving identity."""
        return self._run_task(
            task_name="identity_evolution",
            context=context,
            heuristic_fn=_heuristic_identity_evolution,
        )

    def assess_social_attitude(self, context: dict[str, Any]) -> dict[str, Any]:
        """Assess post-conversation attitude and relationship delta."""
        return self._run_task(
            task_name="social_attitude",
            context=context,
            heuristic_fn=_heuristic_social_attitude,
        )

    def werewolf_choose_target(self, context: dict[str, Any]) -> dict[str, Any]:
        """Night kill target selection for werewolf roles."""
        return self._run_task(
            task_name="werewolf_night_kill",
            context=context,
            heuristic_fn=_heuristic_werewolf_target,
        )

    def werewolf_choose_protect(self, context: dict[str, Any]) -> dict[str, Any]:
        """Night protection target selection for doctor role."""
        return self._run_task(
            task_name="werewolf_doctor_protect",
            context=context,
            heuristic_fn=_heuristic_werewolf_protect,
        )

    def werewolf_choose_inspect(self, context: dict[str, Any]) -> dict[str, Any]:
        """Night inspection target selection for seer role."""
        return self._run_task(
            task_name="werewolf_seer_inspect",
            context=context,
            heuristic_fn=_heuristic_werewolf_inspect,
        )

    def werewolf_generate_speech(self, context: dict[str, Any]) -> dict[str, Any]:
        """Day discussion line generation for werewolf scenario."""
        return self._run_task(
            task_name="werewolf_discussion",
            context=context,
            heuristic_fn=_heuristic_werewolf_speech,
        )

    def werewolf_choose_vote(self, context: dict[str, Any]) -> dict[str, Any]:
        """Day vote target selection for werewolf scenario."""
        return self._run_task(
            task_name="werewolf_vote",
            context=context,
            heuristic_fn=_heuristic_werewolf_vote,
        )

    def anaconda_choose_bet(self, context: dict[str, Any]) -> dict[str, Any]:
        """Bet sizing/fold/all-in decision for Anaconda rounds."""
        return self._run_task(
            task_name="anaconda_bet",
            context=context,
            heuristic_fn=_heuristic_anaconda_bet,
        )

    def anaconda_choose_pass_cards(self, context: dict[str, Any]) -> dict[str, Any]:
        """Card passing choice for Anaconda pass rounds."""
        return self._run_task(
            task_name="anaconda_pass",
            context=context,
            heuristic_fn=_heuristic_anaconda_pass_cards,
        )

    def blackjack_choose_bet(self, context: dict[str, Any]) -> dict[str, Any]:
        """Bet sizing choice for tournament blackjack hand start."""
        return self._run_task(
            task_name="blackjack_bet",
            context=context,
            heuristic_fn=_heuristic_blackjack_bet,
        )

    def blackjack_choose_action(self, context: dict[str, Any]) -> dict[str, Any]:
        """Hit/stand/double decision for blackjack player turns."""
        return self._run_task(
            task_name="blackjack_action",
            context=context,
            heuristic_fn=_heuristic_blackjack_action,
        )

    def holdem_choose_action(self, context: dict[str, Any]) -> dict[str, Any]:
        """Fixed-limit action decision for hold'em betting rounds."""
        return self._run_task(
            task_name="holdem_action",
            context=context,
            heuristic_fn=_heuristic_holdem_action,
        )
