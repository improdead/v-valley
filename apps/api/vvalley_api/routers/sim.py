"""Simulation runtime endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from packages.vvalley_core.maps.map_utils import load_map
from packages.vvalley_core.sim.cognition import CognitionPlanner
from packages.vvalley_core.sim.runner import (
    get_agent_memory_snapshot,
    get_simulation_state,
    tick_simulation,
)

from ..storage.agents import list_active_agents_in_town
from ..storage.llm_control import get_policy as get_llm_policy
from ..storage.llm_control import insert_call_log
from ..storage.map_versions import get_active_version


logger = logging.getLogger("vvalley_api.sim")
router = APIRouter(prefix="/api/v1/sim", tags=["sim"])
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
_COGNITION_PLANNER = CognitionPlanner(policy_lookup=get_llm_policy, log_sink=insert_call_log)


class TickTownRequest(BaseModel):
    steps: int = Field(default=1, ge=1, le=240)
    planning_scope: str = Field(default="short_action", pattern="^(short_action|short|daily_plan|daily|long_term_plan|long_term|long)$")


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


@router.post("/towns/{town_id}/tick")
def town_tick(town_id: str, req: TickTownRequest) -> dict[str, Any]:
    logger.info("[SIM] Tick request: town_id='%s', steps=%d", town_id, req.steps)
    active, map_data, members = _load_runtime_inputs(town_id)
    ticked = tick_simulation(
        town_id=town_id,
        active_version=active,
        map_data=map_data,
        members=members,
        steps=req.steps,
        planning_scope=req.planning_scope,
        planner=_COGNITION_PLANNER.plan_move,
    )
    return {
        "ok": True,
        "town_id": town_id,
        "active_map": {
            "id": active["id"],
            "version": int(active["version"]),
            "map_name": active.get("map_name"),
        },
        "tick": ticked,
    }
