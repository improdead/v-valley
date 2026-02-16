"""Scenario matchmaking and spectator endpoints."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Query

from ..services.scenario_matchmaker import (
    advance_matches,
    cancel_match,
    forfeit_match,
    form_matches,
    get_agent_active_match,
    get_agent_queue_status,
    get_match_payload,
    get_spectator_payload,
    get_town_scenario_manager,
    join_queue,
    leave_queue,
)
from ..storage import scenarios as scenario_store
from ..storage.agents import get_agent_by_api_key, get_agent_town
from ..storage.runtime_control import get_latest_completed_step


logger = logging.getLogger("vvalley_api.scenarios")

router = APIRouter(prefix="/api/v1/scenarios", tags=["scenarios"])
_SCENARIO_MANAGER = get_town_scenario_manager()


class QueueJoinResponse(dict):
    pass


def _require_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    prefix = "bearer "
    if not authorization.lower().startswith(prefix):
        raise HTTPException(status_code=401, detail="Authorization must be Bearer token")
    token = authorization[len(prefix) :].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return token


def _require_agent(authorization: Optional[str]) -> dict[str, Any]:
    api_key = _require_bearer_token(authorization)
    agent = get_agent_by_api_key(api_key)
    if not agent:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return agent


def _scenario_or_404(scenario_key: str) -> dict[str, Any]:
    scenario = scenario_store.get_scenario(scenario_key=scenario_key)
    if not scenario or not bool(scenario.get("enabled", False)):
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_key}")
    return scenario


def _current_step_for_town(town_id: str) -> int:
    tid = str(town_id or "").strip()
    if not tid:
        return 0
    try:
        return max(0, int(get_latest_completed_step(town_id=tid)))
    except Exception:
        return 0


def _current_step_for_match(match_id: str) -> int:
    match = scenario_store.get_match(match_id=str(match_id))
    if not match:
        return 0
    step = _current_step_for_town(str(match.get("town_id") or ""))
    for key in ("end_step", "start_step", "created_step"):
        try:
            candidate = int(match.get(key) or 0)
        except Exception:
            candidate = 0
        if candidate > step:
            step = candidate
    return max(0, int(step))


@router.get("")
@router.get("/")
def list_scenarios() -> dict[str, Any]:
    items = scenario_store.list_scenarios(enabled_only=True)
    return {"count": len(items), "scenarios": items}


@router.get("/me/queue")
def me_queue(authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
    agent = _require_agent(authorization)
    queue = get_agent_queue_status(agent_id=str(agent["id"]))
    return {"ok": True, "agent_id": str(agent["id"]), "count": len(queue), "queue": queue}


@router.get("/me/match")
def me_match(authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
    agent = _require_agent(authorization)
    match = get_agent_active_match(agent_id=str(agent["id"]))
    return {
        "ok": True,
        "agent_id": str(agent["id"]),
        "active": bool(match),
        "match": match,
    }


@router.get("/me/ratings")
def me_ratings(authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
    agent = _require_agent(authorization)
    ratings = scenario_store.list_agent_ratings(agent_id=str(agent["id"]))
    return {
        "ok": True,
        "agent_id": str(agent["id"]),
        "count": len(ratings),
        "ratings": ratings,
    }


@router.get("/me/wallet")
def me_wallet(authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
    agent = _require_agent(authorization)
    wallet = scenario_store.get_wallet(agent_id=str(agent["id"]), create=True)
    return {"ok": True, "agent_id": str(agent["id"]), "wallet": wallet}


@router.get("/matches")
def list_matches(
    town_id: Optional[str] = Query(default=None),
    scenario_key: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    matches = scenario_store.list_matches(town_id=town_id, scenario_key=scenario_key, limit=limit)
    return {"ok": True, "count": len(matches), "matches": matches}


@router.get("/matches/{match_id}")
def match_detail(match_id: str) -> dict[str, Any]:
    payload = get_match_payload(match_id=match_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    return {"ok": True, **payload}


@router.get("/matches/{match_id}/events")
def match_events(match_id: str, limit: int = Query(default=200, ge=1, le=1000)) -> dict[str, Any]:
    match = scenario_store.get_match(match_id=match_id)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    events = scenario_store.list_match_events(match_id=match_id, limit=limit)
    return {"ok": True, "match_id": match_id, "count": len(events), "events": list(reversed(events))}


@router.get("/matches/{match_id}/state")
def match_state(match_id: str) -> dict[str, Any]:
    payload = get_spectator_payload(match_id=match_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    return {"ok": True, **payload}


@router.get("/matches/{match_id}/spectate")
def match_spectate(match_id: str) -> dict[str, Any]:
    payload = get_spectator_payload(match_id=match_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    return {"ok": True, **payload}


@router.post("/matches/{match_id}/forfeit")
def match_forfeit(
    match_id: str,
    authorization: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    agent = _require_agent(authorization)
    current_step = _current_step_for_match(match_id)
    try:
        match = forfeit_match(
            match_id=match_id,
            agent_id=str(agent["id"]),
            current_step=current_step,
            reason="Player forfeited",
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "ok": True,
        "match_id": str(match_id),
        "agent_id": str(agent["id"]),
        "match": match,
    }


@router.post("/matches/{match_id}/cancel")
def match_cancel(
    match_id: str,
    authorization: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    agent = _require_agent(authorization)
    current_step = _current_step_for_match(match_id)
    try:
        match = cancel_match(
            match_id=match_id,
            requested_by_agent_id=str(agent["id"]),
            current_step=current_step,
            reason="Cancelled by participant",
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "ok": True,
        "match_id": str(match_id),
        "agent_id": str(agent["id"]),
        "match": match,
    }


@router.get("/towns/{town_id}/active")
def town_active_matches(town_id: str) -> dict[str, Any]:
    matches = _SCENARIO_MANAGER.active_for_town(town_id=town_id)
    return {"ok": True, "town_id": town_id, "count": len(matches), "matches": matches}


@router.get("/servers")
def list_servers(town_id: Optional[str] = Query(default=None)) -> dict[str, Any]:
    if town_id:
        matches = _SCENARIO_MANAGER.active_for_town(town_id=town_id)
        return {"ok": True, "count": len(matches), "servers": matches}

    raw = scenario_store.list_active_matches(town_id=None)
    servers: list[dict[str, Any]] = []
    for match in raw:
        participants = scenario_store.list_match_participants(match_id=str(match.get("match_id") or ""))
        scenario = scenario_store.get_scenario(scenario_key=str(match.get("scenario_key") or ""))
        state = match.get("state_json") or {}
        servers.append(
            {
                "match_id": str(match.get("match_id") or ""),
                "scenario_key": str(match.get("scenario_key") or ""),
                "status": str(match.get("status") or ""),
                "phase": str(match.get("phase") or ""),
                "round_number": int(match.get("round_number") or 0),
                "participant_count": len(participants),
                "town_id": str(match.get("town_id") or ""),
                "arena_id": str(match.get("arena_id") or ""),
                "buy_in": int((scenario or {}).get("buy_in") or 0),
                "pot": int(state.get("pot") or 0),
            }
        )
    return {"ok": True, "count": len(servers), "servers": servers}


@router.get("/ratings/{scenario_key}")
def leaderboard(scenario_key: str, limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
    scenario = _scenario_or_404(scenario_key)
    items = scenario_store.list_leaderboard(scenario_key=scenario_key, limit=limit)
    return {
        "ok": True,
        "scenario_key": scenario_key,
        "scenario_name": scenario.get("name"),
        "count": len(items),
        "leaderboard": items,
    }


@router.get("/{scenario_key}")
def scenario_detail(scenario_key: str) -> dict[str, Any]:
    scenario = _scenario_or_404(scenario_key)
    return {"scenario": scenario}


@router.get("/{scenario_key}/queue/status")
def queue_status(
    scenario_key: str,
    town_id: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    scenario = _scenario_or_404(scenario_key)
    queue = scenario_store.list_queue(scenario_key=scenario_key, town_id=town_id, status="queued")
    min_players = max(2, int(scenario.get("min_players") or 2))
    queued_count = len(queue)
    missing = max(0, min_players - queued_count)
    estimated_wait_minutes = missing * 2
    return {
        "ok": True,
        "scenario_key": scenario_key,
        "town_id": town_id,
        "queued": queued_count,
        "min_players": min_players,
        "max_players": int(scenario.get("max_players") or min_players),
        "estimated_wait_minutes": estimated_wait_minutes,
    }


@router.post("/{scenario_key}/queue/join")
def queue_join(
    scenario_key: str,
    authorization: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    _scenario_or_404(scenario_key)
    agent = _require_agent(authorization)
    membership = get_agent_town(agent_id=str(agent["id"]))
    if not membership:
        raise HTTPException(status_code=403, detail="Agent must join a town before queueing")
    town_id = str(membership.get("town_id") or "").strip()
    if not town_id:
        raise HTTPException(status_code=403, detail="Agent must join a valid town before queueing")
    current_step = _current_step_for_town(town_id)

    try:
        result = join_queue(
            scenario_key=scenario_key,
            town_id=town_id,
            agent_id=str(agent["id"]),
            current_step=current_step,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    # Opportunistic formation pass.
    formed = form_matches(town_id=town_id, scenario_key=scenario_key, current_step=current_step)
    if formed:
        result["formed_matches"] = formed

    return {
        "ok": True,
        "scenario_key": scenario_key,
        "town_id": town_id,
        "agent_id": str(agent["id"]),
        "result": result,
    }


@router.post("/{scenario_key}/queue/leave")
def queue_leave(
    scenario_key: str,
    authorization: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    _scenario_or_404(scenario_key)
    agent = _require_agent(authorization)
    result = leave_queue(scenario_key=scenario_key, agent_id=str(agent["id"]))
    return {
        "ok": True,
        "scenario_key": scenario_key,
        "agent_id": str(agent["id"]),
        "result": result,
    }


@router.post("/towns/{town_id}/advance")
def advance_town_matches(town_id: str, steps: int = Query(default=1, ge=1, le=60)) -> dict[str, Any]:
    # Admin/ops helper for manual advancement when no town ticks are running.
    matches = []
    for i in range(max(1, int(steps))):
        current = advance_matches(town_id=town_id, current_step=i + 1)
        matches = current
    return {
        "ok": True,
        "town_id": town_id,
        "matches": matches,
    }
