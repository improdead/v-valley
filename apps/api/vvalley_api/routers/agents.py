"""Agent onboarding and ownership endpoints."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from ..storage.agents import (
    claim_agent,
    count_active_agents_in_town,
    get_agent_town,
    get_agent_by_api_key,
    join_agent_town,
    leave_agent_town,
    register_agent,
    rotate_api_key,
)
from ..storage.map_versions import list_versions, list_town_ids
from ..storage.runtime_control import (
    get_agent_autonomy,
    upsert_agent_autonomy,
)

logger = logging.getLogger("vvalley_api.agents")


router = APIRouter(prefix="/api/v1/agents", tags=["agents"])
MAX_AGENTS_PER_TOWN = 25


class RegisterAgentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: Optional[str] = None
    personality: Optional[dict[str, Any]] = None
    owner_handle: Optional[str] = None
    auto_claim: bool = False


class ClaimAgentRequest(BaseModel):
    claim_token: str
    verification_code: str
    owner_handle: Optional[str] = None


class JoinTownRequest(BaseModel):
    town_id: str = Field(min_length=1, max_length=80)


class UpdateAutonomyRequest(BaseModel):
    mode: str = Field(default="manual", pattern="^(manual|delegated|autonomous)$")
    allowed_scopes: list[str] = Field(default_factory=lambda: ["short_action", "daily_plan", "long_term_plan"])
    max_autonomous_ticks: int = Field(default=0, ge=0, le=100000)
    escalation_policy: Optional[dict[str, Any]] = None


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


def _setup_info_payload() -> dict[str, Any]:
    return {
        "autonomous_agents": True,
        "requires_user_vvalley_api_key": False,
        "supports_auto_claim": True,
        "recommended_register_payload": {
            "name": "MyAgent",
            "owner_handle": "your_handle",
            "auto_claim": True,
        },
        "notes": [
            "V-Valley API keys are issued by /api/v1/agents/register.",
            "Users should not manually create a V-Valley API key.",
            "Agents should store their issued API key and use it autonomously.",
            "No X/Twitter integration is required for core onboarding in this repo.",
            "Claimed agents can join a town via /api/v1/agents/me/join-town.",
        ],
    }


@router.get("/setup-info")
def agent_setup_info() -> dict[str, Any]:
    logger.info("[AGENTS] Setup info requested")
    return _setup_info_payload()


@router.post("/register")
def register_agent_endpoint(req: RegisterAgentRequest, request: Request) -> dict[str, Any]:
    logger.info("[AGENTS] Registering new agent: name='%s', owner_handle='%s', auto_claim=%s", 
                req.name, req.owner_handle, req.auto_claim)
    
    created = register_agent(
        name=req.name.strip(),
        description=req.description,
        personality=req.personality,
    )
    logger.info("[AGENTS] Agent registered: id=%s, name='%s'", created["id"], created["name"])

    claimed = None
    if req.auto_claim:
        logger.info("[AGENTS] Auto-claiming agent: id=%s, owner_handle='%s'", created["id"], req.owner_handle)
        try:
            claimed = claim_agent(
                claim_token=created["claim_token"],
                verification_code=created["verification_code"],
                owner_handle=req.owner_handle,
            )
            logger.info("[AGENTS] Agent auto-claimed successfully: id=%s", created["id"])
        except ValueError as e:
            logger.warning("[AGENTS] Auto-claim failed for agent id=%s: %s", created["id"], e)
            claimed = None
        if claimed:
            created["claim_status"] = claimed["claim_status"]
            created["claimed"] = bool(claimed.get("claimed", False))
            created["owner_handle"] = claimed.get("owner_handle")
            created["claimed_at"] = claimed.get("claimed_at")

    claim_url = f"{str(request.base_url).rstrip('/')}/claim/{created['claim_token']}"
    logger.info("[AGENTS] Agent registration complete: id=%s, claimed=%s", created["id"], bool(created.get("claimed", False)))
    
    return {
        "agent": {
            "id": created["id"],
            "name": created["name"],
            "description": created.get("description"),
            "status": created["claim_status"],
            "claimed": bool(created.get("claimed", False)),
            "api_key": created["api_key"],
            "claim_token": created["claim_token"],
            "verification_code": created["verification_code"],
            "claim_url": claim_url,
            "owner_handle": created.get("owner_handle"),
            "claimed_at": created.get("claimed_at"),
        },
        "setup": _setup_info_payload(),
    }


@router.post("/claim")
def claim_agent_endpoint(req: ClaimAgentRequest) -> dict[str, Any]:
    logger.info("[AGENTS] Claim attempt: claim_token='%s...', owner_handle='%s'", 
                req.claim_token[:20] if req.claim_token else "", req.owner_handle)
    
    try:
        claimed = claim_agent(
            claim_token=req.claim_token,
            verification_code=req.verification_code,
            owner_handle=req.owner_handle,
        )
    except ValueError as e:
        logger.warning("[AGENTS] Claim failed - invalid verification code: %s", e)
        raise HTTPException(status_code=400, detail="Invalid verification code")

    if not claimed:
        logger.warning("[AGENTS] Claim failed - token not found: claim_token='%s...'", 
                      req.claim_token[:20] if req.claim_token else "")
        raise HTTPException(status_code=404, detail="Claim token not found")

    logger.info("[AGENTS] Agent claimed successfully: id=%s, name='%s', owner_handle='%s'", 
                claimed["id"], claimed["name"], claimed.get("owner_handle"))
    
    return {
        "ok": True,
        "agent": {
            "id": claimed["id"],
            "name": claimed["name"],
            "status": claimed["claim_status"],
            "claimed": bool(claimed.get("claimed", False)),
            "owner_handle": claimed.get("owner_handle"),
            "claimed_at": claimed.get("claimed_at"),
        },
    }


@router.get("/me")
def agent_me(authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
    api_key = _require_bearer_token(authorization)
    logger.debug("[AGENTS] Get agent info request: api_key_prefix='%s...'", api_key[:16] if api_key else "")
    
    agent = get_agent_by_api_key(api_key)
    if not agent:
        logger.warning("[AGENTS] Get agent info failed - invalid API key: prefix='%s...'", api_key[:16] if api_key else "")
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    current_town = get_agent_town(agent_id=agent["id"])
    logger.info("[AGENTS] Agent info retrieved: id=%s, name='%s', current_town='%s'", 
                agent["id"], agent["name"], current_town.get("town_id") if current_town else None)
    
    return {
        "agent": {
            "id": agent["id"],
            "name": agent["name"],
            "description": agent.get("description"),
            "status": agent["claim_status"],
            "claimed": bool(agent.get("claimed", False)),
            "owner_handle": agent.get("owner_handle"),
            "created_at": agent.get("created_at"),
            "claimed_at": agent.get("claimed_at"),
            "personality": agent.get("personality", {}),
            "current_town": current_town,
        }
    }


@router.get("/me/autonomy")
def get_me_autonomy(authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
    api_key = _require_bearer_token(authorization)
    agent = get_agent_by_api_key(api_key)
    if not agent:
        raise HTTPException(status_code=401, detail="Invalid API key")
    contract = get_agent_autonomy(agent_id=str(agent["id"]))
    return {
        "ok": True,
        "agent_id": str(agent["id"]),
        "autonomy": contract,
    }


@router.put("/me/autonomy")
def put_me_autonomy(
    req: UpdateAutonomyRequest,
    authorization: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    api_key = _require_bearer_token(authorization)
    agent = get_agent_by_api_key(api_key)
    if not agent:
        raise HTTPException(status_code=401, detail="Invalid API key")
    saved = upsert_agent_autonomy(
        agent_id=str(agent["id"]),
        mode=req.mode,
        allowed_scopes=list(req.allowed_scopes),
        max_autonomous_ticks=int(req.max_autonomous_ticks),
        escalation_policy=req.escalation_policy,
    )
    if saved is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {
        "ok": True,
        "agent_id": str(agent["id"]),
        "autonomy": saved,
    }


@router.post("/me/rotate-key")
def rotate_agent_key(authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
    api_key = _require_bearer_token(authorization)
    logger.info("[AGENTS] API key rotation requested: api_key_prefix='%s...'", api_key[:16] if api_key else "")
    
    rotated = rotate_api_key(api_key)
    if not rotated:
        logger.warning("[AGENTS] Key rotation failed - invalid API key: prefix='%s...'", api_key[:16] if api_key else "")
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    logger.info("[AGENTS] API key rotated successfully: agent_id=%s", rotated["id"])
    return {
        "ok": True,
        "agent_id": rotated["id"],
        "api_key": rotated["api_key"],
        "api_key_prefix": rotated["api_key"][:16],
    }


@router.post("/me/join-town")
def join_town(req: JoinTownRequest, authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
    api_key = _require_bearer_token(authorization)
    logger.info("[AGENTS] Join town request: town_id='%s'", req.town_id)

    agent = get_agent_by_api_key(api_key)
    if not agent:
        logger.warning("[AGENTS] Join town failed - invalid API key")
        raise HTTPException(status_code=401, detail="Invalid API key")

    logger.debug("[AGENTS] Agent joining town: agent_id=%s, name='%s'", agent["id"], agent["name"])

    if not bool(agent.get("claimed", False)):
        logger.warning("[AGENTS] Join town failed - agent not claimed: agent_id=%s", agent["id"])
        raise HTTPException(status_code=403, detail="Agent must be claimed before joining a town")

    town_id = req.town_id.strip()
    if not town_id:
        logger.warning("[AGENTS] Join town failed - empty town_id")
        raise HTTPException(status_code=400, detail="town_id is required")

    if not list_versions(town_id):
        logger.warning("[AGENTS] Join town failed - town not found: town_id='%s'", town_id)
        raise HTTPException(status_code=404, detail=f"Town not found: {town_id}")

    current_town = get_agent_town(agent_id=agent["id"])
    if not current_town or current_town.get("town_id") != town_id:
        current_count = count_active_agents_in_town(town_id)
        if current_count >= MAX_AGENTS_PER_TOWN:
            logger.warning("[AGENTS] Join town failed - town full: town_id='%s', current=%d, max=%d",
                          town_id, current_count, MAX_AGENTS_PER_TOWN)
            raise HTTPException(
                status_code=409,
                detail=f"Town is full (max {MAX_AGENTS_PER_TOWN} agents): {town_id}",
            )

    membership = join_agent_town(agent_id=agent["id"], town_id=town_id)
    if not membership:
        logger.error("[AGENTS] Join town failed - membership creation failed: agent_id=%s, town_id='%s'",
                    agent["id"], town_id)
        raise HTTPException(status_code=404, detail="Agent not found")

    logger.info("[AGENTS] Agent joined town successfully: agent_id=%s, agent_name='%s', town_id='%s'",
                agent["id"], agent["name"], town_id)

    return {
        "ok": True,
        "membership": membership,
        "agent": {
            "id": agent["id"],
            "name": agent["name"],
            "status": agent["claim_status"],
            "claimed": bool(agent.get("claimed", False)),
            "current_town": membership,
        },
    }


@router.post("/me/auto-join")
def auto_join_town(authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
    """Join the best available town automatically (least populated with capacity)."""
    api_key = _require_bearer_token(authorization)
    logger.info("[AGENTS] Auto-join request")

    agent = get_agent_by_api_key(api_key)
    if not agent:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not bool(agent.get("claimed", False)):
        raise HTTPException(status_code=403, detail="Agent must be claimed before joining a town")

    current = get_agent_town(agent_id=agent["id"])
    if current:
        return {
            "ok": True,
            "already_joined": True,
            "membership": current,
            "agent": {
                "id": agent["id"],
                "name": agent["name"],
                "status": agent["claim_status"],
                "claimed": True,
                "current_town": current,
            },
        }

    town_ids = list_town_ids()
    if not town_ids:
        raise HTTPException(status_code=404, detail="No towns available. An operator must create a town first.")

    best_town = None
    best_pop = MAX_AGENTS_PER_TOWN + 1
    for tid in town_ids:
        pop = count_active_agents_in_town(tid)
        if pop < MAX_AGENTS_PER_TOWN and pop < best_pop:
            best_town = tid
            best_pop = pop

    if not best_town:
        raise HTTPException(status_code=409, detail="All towns are full")

    membership = join_agent_town(agent_id=agent["id"], town_id=best_town)
    if not membership:
        raise HTTPException(status_code=500, detail="Failed to join town")

    logger.info("[AGENTS] Auto-joined: agent_id=%s, town_id='%s' (pop=%d)", agent["id"], best_town, best_pop)
    return {
        "ok": True,
        "membership": membership,
        "agent": {
            "id": agent["id"],
            "name": agent["name"],
            "status": agent["claim_status"],
            "claimed": True,
            "current_town": membership,
        },
    }


@router.post("/me/leave-town")
def leave_town(authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
    api_key = _require_bearer_token(authorization)
    logger.info("[AGENTS] Leave town request received")
    
    agent = get_agent_by_api_key(api_key)
    if not agent:
        logger.warning("[AGENTS] Leave town failed - invalid API key")
        raise HTTPException(status_code=401, detail="Invalid API key")

    logger.debug("[AGENTS] Agent leaving town: agent_id=%s, name='%s'", agent["id"], agent["name"])
    
    left = leave_agent_town(agent_id=agent["id"])
    if left:
        logger.info("[AGENTS] Agent left town successfully: agent_id=%s, name='%s'", agent["id"], agent["name"])
    else:
        logger.info("[AGENTS] Agent leave town - no active membership: agent_id=%s", agent["id"])
    
    return {
        "ok": True,
        "left": bool(left),
        "agent": {
            "id": agent["id"],
            "name": agent["name"],
            "status": agent["claim_status"],
            "claimed": bool(agent.get("claimed", False)),
            "current_town": None,
        },
    }
