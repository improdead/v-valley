"""Owner escalation queue endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from ..storage.agents import get_agent_by_api_key
from ..storage.interaction_hub import list_owner_escalations, resolve_owner_escalation


router = APIRouter(prefix="/api/v1/owners", tags=["owners"])


class ResolveEscalationRequest(BaseModel):
    resolution_note: Optional[str] = Field(default=None, max_length=500)


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


def _require_owner_handle(authorization: Optional[str]) -> str:
    api_key = _require_bearer_token(authorization)
    agent = get_agent_by_api_key(api_key)
    if not agent:
        raise HTTPException(status_code=401, detail="Invalid API key")
    owner_handle = str(agent.get("owner_handle") or "").strip()
    if not owner_handle:
        raise HTTPException(status_code=403, detail="Owner handle is not set for this agent")
    return owner_handle


@router.get("/me/escalations")
def get_owner_escalations(
    status: list[str] = Query(default_factory=list),
    limit: int = Query(default=50, ge=1, le=200),
    authorization: Optional[str] = Header(default=None),
) -> dict[str, object]:
    owner_handle = _require_owner_handle(authorization)
    items = list_owner_escalations(owner_handle=owner_handle, statuses=status, limit=limit)
    return {
        "ok": True,
        "owner_handle": owner_handle,
        "count": len(items),
        "items": items,
    }


@router.post("/me/escalations/{escalation_id}/resolve")
def resolve_escalation(
    escalation_id: str,
    req: ResolveEscalationRequest,
    authorization: Optional[str] = Header(default=None),
) -> dict[str, object]:
    owner_handle = _require_owner_handle(authorization)
    updated = resolve_owner_escalation(
        owner_handle=owner_handle,
        escalation_id=str(escalation_id),
        resolution_note=req.resolution_note,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Escalation not found")
    return {
        "ok": True,
        "item": updated,
    }
