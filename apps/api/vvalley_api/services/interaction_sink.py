"""Service for turning simulation tick outcomes into durable interaction records."""

from __future__ import annotations

from typing import Any

from ..storage.interaction_hub import create_inbox_item, create_owner_escalation


def ingest_tick_outcomes(
    *,
    town_id: str,
    members: list[dict[str, Any]],
    tick_result: dict[str, Any],
) -> dict[str, int]:
    member_by_agent_id: dict[str, dict[str, Any]] = {}
    for member in members:
        agent_id = str(member.get("agent_id") or "").strip()
        if not agent_id:
            continue
        member_by_agent_id[agent_id] = member

    created_inbox = 0
    created_escalations = 0
    events = tick_result.get("events") or []
    step = int(tick_result.get("step") or 0)

    for event in events:
        if not isinstance(event, dict):
            continue
        agent_id = str(event.get("agent_id") or "").strip()
        if not agent_id:
            continue
        decision = event.get("decision") if isinstance(event.get("decision"), dict) else {}
        route = str(decision.get("route") or "").strip().lower()
        if route != "awaiting_user_agent_autonomy_limit":
            continue

        payload = {
            "town_id": town_id,
            "step": step,
            "decision": decision,
            "event": {
                "from": event.get("from"),
                "to": event.get("to"),
                "goal": event.get("goal"),
            },
        }
        dedupe_key = f"autonomy_limit:{town_id}:{agent_id}"
        inbox_item = create_inbox_item(
            agent_id=agent_id,
            kind="autonomy_limit_reached",
            summary="Autonomy limit reached; awaiting new delegated action",
            payload=payload,
            dedupe_key=dedupe_key,
        )
        if inbox_item:
            created_inbox += 1

        member = member_by_agent_id.get(agent_id) or {}
        owner_handle = str(member.get("owner_handle") or "").strip()
        if owner_handle:
            escalation = create_owner_escalation(
                owner_handle=owner_handle,
                agent_id=agent_id,
                town_id=town_id,
                kind="autonomy_limit_reached",
                summary="Agent paused because delegated autonomy limit was reached",
                payload=payload,
                dedupe_key=dedupe_key,
            )
            if escalation:
                created_escalations += 1

    return {
        "inbox_created": created_inbox,
        "escalations_created": created_escalations,
    }
