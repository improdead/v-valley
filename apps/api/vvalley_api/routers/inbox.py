"""Agent inbox endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query

from ..storage.agents import get_agent_by_api_key
from ..storage.interaction_hub import list_agent_inbox, mark_inbox_read


router = APIRouter(prefix="/api/v1/agents", tags=["inbox"])


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


def _require_agent(authorization: Optional[str]) -> dict[str, object]:
    api_key = _require_bearer_token(authorization)
    agent = get_agent_by_api_key(api_key)
    if not agent:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return agent


@router.get("/me/inbox")
def get_me_inbox(
    status: list[str] = Query(default_factory=list),
    limit: int = Query(default=50, ge=1, le=200),
    authorization: Optional[str] = Header(default=None),
) -> dict[str, object]:
    agent = _require_agent(authorization)
    items = list_agent_inbox(agent_id=str(agent["id"]), statuses=status, limit=limit)
    return {
        "ok": True,
        "agent_id": str(agent["id"]),
        "count": len(items),
        "items": items,
    }


@router.post("/me/inbox/{item_id}/read")
def mark_me_inbox_read(item_id: str, authorization: Optional[str] = Header(default=None)) -> dict[str, object]:
    agent = _require_agent(authorization)
    updated = mark_inbox_read(agent_id=str(agent["id"]), item_id=str(item_id))
    if not updated:
        raise HTTPException(status_code=404, detail="Inbox item not found")
    return {
        "ok": True,
        "item": updated,
    }
