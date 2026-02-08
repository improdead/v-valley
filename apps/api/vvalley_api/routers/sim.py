"""Simulation runtime endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import logging
import uuid

from fastapi import APIRouter, HTTPException, Header, Query
from pydantic import BaseModel, Field

from packages.vvalley_core.maps.map_utils import load_map
from packages.vvalley_core.sim.cognition import CognitionPlanner
from packages.vvalley_core.sim.runner import (
    get_agent_context_snapshot,
    get_agent_memory_snapshot,
    get_simulation_state,
    submit_agent_action,
    tick_simulation,
)

from ..storage.agents import (
    get_agent_by_api_key,
    get_agent_town,
    list_active_agents_in_town,
)
from ..storage.llm_control import get_policy as get_llm_policy
from ..storage.llm_control import insert_call_log
from ..storage.map_versions import get_active_version
from ..storage.runtime_control import (
    complete_tick_batch,
    get_agent_autonomy,
    list_dead_letters,
    get_tick_metrics,
    get_town_runtime,
    list_agent_autonomy,
    record_dead_letter,
    reserve_tick_batch,
    upsert_town_runtime,
)
from ..services.runtime_scheduler import (
    start_town_runtime_scheduler,
    town_runtime_scheduler_status,
)
from ..services.interaction_sink import ingest_tick_outcomes


logger = logging.getLogger("vvalley_api.sim")
router = APIRouter(prefix="/api/v1/sim", tags=["sim"])
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
_COGNITION_PLANNER = CognitionPlanner(policy_lookup=get_llm_policy, log_sink=insert_call_log)
PLANNING_SCOPE_PATTERN = "^(short_action|short|daily_plan|daily|long_term_plan|long_term|long)$"
CONTROL_MODE_PATTERN = "^(external|autopilot|hybrid)$"
NODE_KIND_PATTERN = "^(event|thought|chat|reflection)$"


class TickTownRequest(BaseModel):
    steps: int = Field(default=1, ge=1, le=240)
    planning_scope: str = Field(default="short_action", pattern=PLANNING_SCOPE_PATTERN)
    control_mode: str = Field(default="external", pattern=CONTROL_MODE_PATTERN)
    idempotency_key: Optional[str] = Field(default=None, min_length=1, max_length=180)


class RuntimeStartRequest(BaseModel):
    tick_interval_seconds: int = Field(default=60, ge=1, le=3600)
    steps_per_tick: int = Field(default=1, ge=1, le=240)
    planning_scope: str = Field(default="short_action", pattern=PLANNING_SCOPE_PATTERN)
    control_mode: str = Field(default="hybrid", pattern=CONTROL_MODE_PATTERN)


class AgentActionSocialRequest(BaseModel):
    target_agent_id: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=400)
    transcript: list[list[str]] = Field(default_factory=list)


class AgentActionMemoryNodeRequest(BaseModel):
    kind: str = Field(pattern=NODE_KIND_PATTERN)
    description: str = Field(min_length=1, max_length=600)
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    poignancy: int = Field(default=3, ge=1, le=10)
    evidence_ids: list[str] = Field(default_factory=list)


class SubmitAgentActionRequest(BaseModel):
    planning_scope: str = Field(default="short_action", pattern=PLANNING_SCOPE_PATTERN)
    dx: Optional[int] = Field(default=None, ge=-1, le=1)
    dy: Optional[int] = Field(default=None, ge=-1, le=1)
    target_x: Optional[int] = None
    target_y: Optional[int] = None
    goal_reason: Optional[str] = None
    affordance: Optional[str] = None
    action_description: Optional[str] = None
    action_ttl_steps: Optional[int] = Field(default=None, ge=1, le=240)
    interrupt_on_social: Optional[bool] = None
    social_interrupt_cooldown_steps: Optional[int] = Field(default=None, ge=1, le=240)
    socials: list[AgentActionSocialRequest] = Field(default_factory=list)
    memory_nodes: list[AgentActionMemoryNodeRequest] = Field(default_factory=list)


def _resolve_workspace_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = (WORKSPACE_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if candidate != WORKSPACE_ROOT and WORKSPACE_ROOT not in candidate.parents:
        raise HTTPException(status_code=400, detail="Path must be within workspace")
    return candidate


def _load_runtime_inputs(town_id: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    active = get_active_version(town_id)
    if not active:
        raise HTTPException(
            status_code=404,
            detail=f"No active map version for town: {town_id}",
        )

    map_path = _resolve_workspace_path(str(active["map_json_path"]))
    if not map_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Active map file not found for town '{town_id}': {map_path}",
        )

    map_data = load_map(map_path)
    members = list_active_agents_in_town(town_id)
    return active, map_data, members


def _autonomy_map_for_members(members: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    agent_ids: list[str] = []
    for member in members:
        agent_id = str(member.get("agent_id") or "").strip()
        if not agent_id:
            continue
        agent_ids.append(agent_id)
    return list_agent_autonomy(agent_ids=agent_ids)


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


def _require_agent_membership(*, town_id: str, authorization: Optional[str]) -> dict[str, Any]:
    api_key = _require_bearer_token(authorization)
    agent = get_agent_by_api_key(api_key)
    if not agent:
        raise HTTPException(status_code=401, detail="Invalid API key")
    membership = get_agent_town(agent_id=agent["id"])
    if not membership or membership.get("town_id") != town_id:
        raise HTTPException(status_code=403, detail=f"Agent is not a member of town: {town_id}")
    return agent


@router.get("/towns/{town_id}/state")
def town_state(town_id: str) -> dict[str, Any]:
    logger.info("[SIM] State request: town_id='%s'", town_id)
    active, map_data, members = _load_runtime_inputs(town_id)
    state = get_simulation_state(
        town_id=town_id,
        active_version=active,
        map_data=map_data,
        members=members,
    )
    return {
        "ok": True,
        "town_id": town_id,
        "active_map": {
            "id": active["id"],
            "version": int(active["version"]),
            "map_name": active.get("map_name"),
        },
        "state": state,
    }


@router.get("/towns/{town_id}/agents/{agent_id}/memory")
def town_agent_memory(
    town_id: str,
    agent_id: str,
    limit: int = Query(default=40, ge=1, le=200),
) -> dict[str, Any]:
    logger.info("[SIM] Agent memory request: town_id='%s', agent_id='%s', limit=%d", town_id, agent_id, limit)
    active, map_data, members = _load_runtime_inputs(town_id)
    memory_snapshot = get_agent_memory_snapshot(
        town_id=town_id,
        active_version=active,
        map_data=map_data,
        members=members,
        agent_id=agent_id,
        limit=limit,
    )
    if memory_snapshot is None:
        raise HTTPException(status_code=404, detail=f"Agent not found in town runtime: {agent_id}")
    return {
        "ok": True,
        "town_id": town_id,
        "active_map": {
            "id": active["id"],
            "version": int(active["version"]),
            "map_name": active.get("map_name"),
        },
        "snapshot": memory_snapshot,
    }


@router.get("/towns/{town_id}/agents/me/context")
def town_agent_context(
    town_id: str,
    planning_scope: str = Query(default="short_action", pattern=PLANNING_SCOPE_PATTERN),
    authorization: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    agent = _require_agent_membership(town_id=town_id, authorization=authorization)
    logger.info(
        "[SIM] Agent context request: town_id='%s', agent_id='%s', scope='%s'",
        town_id,
        agent["id"],
        planning_scope,
    )
    active, map_data, members = _load_runtime_inputs(town_id)
    context = get_agent_context_snapshot(
        town_id=town_id,
        active_version=active,
        map_data=map_data,
        members=members,
        agent_id=str(agent["id"]),
        planning_scope=planning_scope,
    )
    if context is None:
        raise HTTPException(status_code=404, detail=f"Agent not found in town runtime: {agent['id']}")
    context["autonomy_contract"] = get_agent_autonomy(agent_id=str(agent["id"]))
    return {
        "ok": True,
        "town_id": town_id,
        "active_map": {
            "id": active["id"],
            "version": int(active["version"]),
            "map_name": active.get("map_name"),
        },
        "context": context,
    }


@router.post("/towns/{town_id}/agents/me/action")
def town_agent_action(
    town_id: str,
    req: SubmitAgentActionRequest,
    authorization: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    agent = _require_agent_membership(town_id=town_id, authorization=authorization)
    logger.info(
        "[SIM] Agent action submit: town_id='%s', agent_id='%s', scope='%s'",
        town_id,
        agent["id"],
        req.planning_scope,
    )
    active, map_data, members = _load_runtime_inputs(town_id)
    action_payload = (
        req.model_dump(exclude_none=True)
        if hasattr(req, "model_dump")
        else req.dict(exclude_none=True)
    )
    accepted = submit_agent_action(
        town_id=town_id,
        active_version=active,
        map_data=map_data,
        members=members,
        agent_id=str(agent["id"]),
        action=action_payload,
    )
    if accepted is None:
        raise HTTPException(status_code=404, detail=f"Agent not found in town runtime: {agent['id']}")
    return {
        "ok": True,
        "town_id": town_id,
        "active_map": {
            "id": active["id"],
            "version": int(active["version"]),
            "map_name": active.get("map_name"),
        },
        "accepted": accepted,
    }


@router.post("/towns/{town_id}/tick")
def town_tick(town_id: str, req: TickTownRequest) -> dict[str, Any]:
    logger.info(
        "[SIM] Tick request: town_id='%s', steps=%d, control_mode='%s'",
        town_id,
        req.steps,
        req.control_mode,
    )
    active, map_data, members = _load_runtime_inputs(town_id)
    idempotency_key = str(req.idempotency_key or "").strip()
    idempotent_request = bool(idempotency_key)
    batch_row_key = (
        f"manual:{idempotency_key}"
        if idempotent_request
        else f"manual:auto:{uuid.uuid4().hex}"
    )

    reserve = reserve_tick_batch(
        town_id=town_id,
        batch_key=batch_row_key,
        step_before=None,
    )
    if idempotent_request and reserve.get("state") == "existing":
        status = str(reserve.get("status") or "")
        if status == "completed" and isinstance(reserve.get("result"), dict):
            return {
                "ok": True,
                "town_id": town_id,
                "active_map": {
                    "id": active["id"],
                    "version": int(active["version"]),
                    "map_name": active.get("map_name"),
                },
                "tick": reserve["result"],
                "interaction_ingest": {"inbox_created": 0, "escalations_created": 0},
                "idempotent_replay": True,
                "tick_metrics_24h": get_tick_metrics(town_id=town_id, window_hours=24),
            }
        if status == "pending":
            raise HTTPException(status_code=409, detail="Tick batch with this idempotency_key is still pending")
    elif not idempotent_request and reserve.get("state") == "existing":
        # Extremely unlikely UUID collision fallback.
        batch_row_key = f"manual:auto:{uuid.uuid4().hex}"
        reserve_tick_batch(
            town_id=town_id,
            batch_key=batch_row_key,
            step_before=None,
        )

    autonomy_map = _autonomy_map_for_members(members)
    try:
        ticked = tick_simulation(
            town_id=town_id,
            active_version=active,
            map_data=map_data,
            members=members,
            steps=req.steps,
            planning_scope=req.planning_scope,
            control_mode=req.control_mode,
            planner=_COGNITION_PLANNER.plan_move,
            agent_autonomy=autonomy_map,
        )
        try:
            interaction_ingest = ingest_tick_outcomes(
                town_id=town_id,
                members=members,
                tick_result=ticked,
            )
        except Exception as sink_exc:
            record_dead_letter(
                town_id=town_id,
                stage="interaction_sink_manual_tick",
                payload={"batch_key": batch_row_key},
                error=f"{sink_exc.__class__.__name__}: {sink_exc}",
            )
            interaction_ingest = {"inbox_created": 0, "escalations_created": 0}
        complete_tick_batch(
            town_id=town_id,
            batch_key=batch_row_key,
            success=True,
            step_after=int(ticked.get("step") or 0),
            result=ticked,
            error=None,
        )
    except Exception as exc:
        complete_tick_batch(
            town_id=town_id,
            batch_key=batch_row_key,
            success=False,
            step_after=None,
            result=None,
            error=f"{exc.__class__.__name__}: {exc}",
        )
        raise
    return {
        "ok": True,
        "town_id": town_id,
        "active_map": {
            "id": active["id"],
            "version": int(active["version"]),
            "map_name": active.get("map_name"),
        },
        "tick": ticked,
        "interaction_ingest": interaction_ingest,
        "idempotent_replay": False,
        "tick_metrics_24h": get_tick_metrics(town_id=town_id, window_hours=24),
    }


@router.post("/towns/{town_id}/runtime/start")
def start_town_runtime(town_id: str, req: RuntimeStartRequest) -> dict[str, Any]:
    logger.info(
        "[SIM] Runtime start request: town_id='%s', interval=%ds, steps=%d, scope='%s', mode='%s'",
        town_id,
        req.tick_interval_seconds,
        req.steps_per_tick,
        req.planning_scope,
        req.control_mode,
    )
    # Validate town can run by resolving runtime inputs once.
    _load_runtime_inputs(town_id)
    control = upsert_town_runtime(
        town_id=town_id,
        enabled=True,
        tick_interval_seconds=req.tick_interval_seconds,
        steps_per_tick=req.steps_per_tick,
        planning_scope=req.planning_scope,
        control_mode=req.control_mode,
    )
    start_town_runtime_scheduler()
    return {
        "ok": True,
        "town_id": town_id,
        "runtime": control,
        "scheduler": town_runtime_scheduler_status(),
    }


@router.post("/towns/{town_id}/runtime/stop")
def stop_town_runtime(town_id: str) -> dict[str, Any]:
    logger.info("[SIM] Runtime stop request: town_id='%s'", town_id)
    _load_runtime_inputs(town_id)
    control = upsert_town_runtime(town_id=town_id, enabled=False)
    return {
        "ok": True,
        "town_id": town_id,
        "runtime": control,
        "scheduler": town_runtime_scheduler_status(),
    }


@router.get("/towns/{town_id}/runtime/status")
def town_runtime_status(town_id: str) -> dict[str, Any]:
    logger.info("[SIM] Runtime status request: town_id='%s'", town_id)
    _load_runtime_inputs(town_id)
    return {
        "ok": True,
        "town_id": town_id,
        "runtime": get_town_runtime(town_id=town_id),
        "tick_metrics_24h": get_tick_metrics(town_id=town_id, window_hours=24),
        "scheduler": town_runtime_scheduler_status(),
    }


@router.get("/towns/{town_id}/dead-letters")
def town_dead_letters(
    town_id: str,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    logger.info("[SIM] Dead letter request: town_id='%s', limit=%d", town_id, limit)
    _load_runtime_inputs(town_id)
    items = list_dead_letters(town_id=town_id, limit=limit)
    return {
        "ok": True,
        "town_id": town_id,
        "count": len(items),
        "items": items,
    }
