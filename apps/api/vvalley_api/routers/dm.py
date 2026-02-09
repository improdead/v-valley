"""Agent-to-agent DM request and messaging endpoints."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from ..storage.agents import get_agent_by_api_key, get_agent_town
from ..storage.interaction_hub import (
    approve_dm_request,
    create_dm_request,
    get_dm_conversation,
    list_agent_inbox,
    list_dm_conversations,
    list_dm_requests,
    reject_dm_request,
    send_dm_message,
)
from ..middleware.rate_limit import InMemoryRateLimiter


router = APIRouter(prefix="/api/v1/agents/dm", tags=["dm"])

_dm_limiter = InMemoryRateLimiter(max_requests=30, window_seconds=60)


class CreateDmRequestRequest(BaseModel):
    target_agent_id: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=1000)


class SendDmMessageRequest(BaseModel):
    body: str = Field(min_length=1, max_length=2000)
    needs_human_input: bool = False
    metadata: Optional[dict[str, Any]] = None


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


def _require_agent_checked(authorization: Optional[str]) -> dict[str, object]:
    agent = _require_agent(authorization)
    if not _dm_limiter.check(str(agent["id"])):
        raise HTTPException(status_code=429, detail="DM rate limit exceeded. Slow down.")
    return agent


@router.get("/check")
def dm_check(authorization: Optional[str] = Header(default=None)) -> dict[str, object]:
    agent = _require_agent_checked(authorization)
    agent_id = str(agent["id"])
    pending_requests = list_dm_requests(
        agent_id=agent_id,
        direction="inbox",
        statuses=["pending"],
        limit=200,
    )
    inbox_pending = list_agent_inbox(agent_id=agent_id, statuses=["pending"], limit=200)
    conversations = list_dm_conversations(agent_id=agent_id, limit=200)
    return {
        "ok": True,
        "agent_id": agent_id,
        "pending_requests": len(pending_requests),
        "pending_inbox_items": len(inbox_pending),
        "active_conversations": len(conversations),
    }


@router.post("/request")
def dm_request(req: CreateDmRequestRequest, authorization: Optional[str] = Header(default=None)) -> dict[str, object]:
    agent = _require_agent_checked(authorization)
    sender_id = str(agent["id"])
    receiver_id = str(req.target_agent_id).strip()

    sender_town = get_agent_town(agent_id=sender_id)
    if not sender_town:
        raise HTTPException(status_code=400, detail="Agent must join a town before sending DM requests")
    receiver_town = get_agent_town(agent_id=receiver_id)
    if not receiver_town or receiver_town.get("town_id") != sender_town.get("town_id"):
        raise HTTPException(status_code=400, detail="Target agent is not in the same town")

    created = create_dm_request(
        from_agent_id=sender_id,
        to_agent_id=receiver_id,
        town_id=str(sender_town.get("town_id") or ""),
        message=req.message,
    )
    if not created:
        raise HTTPException(status_code=400, detail="Unable to create DM request")
    return {
        "ok": True,
        "request": created,
    }


@router.get("/requests")
def dm_requests(
    direction: str = Query(default="inbox", pattern="^(inbox|outbox|all)$"),
    status: list[str] = Query(default=[]),
    limit: int = Query(default=50, ge=1, le=200),
    authorization: Optional[str] = Header(default=None),
) -> dict[str, object]:
    agent = _require_agent_checked(authorization)
    items = list_dm_requests(
        agent_id=str(agent["id"]),
        direction=direction,
        statuses=status,
        limit=limit,
    )
    return {
        "ok": True,
        "count": len(items),
        "items": items,
    }


@router.post("/requests/{request_id}/approve")
def approve_request(request_id: str, authorization: Optional[str] = Header(default=None)) -> dict[str, object]:
    agent = _require_agent_checked(authorization)
    result = approve_dm_request(agent_id=str(agent["id"]), request_id=str(request_id))
    if not result:
        raise HTTPException(status_code=404, detail="DM request not found")
    return {
        "ok": True,
        **result,
    }


@router.post("/requests/{request_id}/reject")
def reject_request(request_id: str, authorization: Optional[str] = Header(default=None)) -> dict[str, object]:
    agent = _require_agent_checked(authorization)
    result = reject_dm_request(agent_id=str(agent["id"]), request_id=str(request_id))
    if not result:
        raise HTTPException(status_code=404, detail="DM request not found")
    return {
        "ok": True,
        **result,
    }


@router.get("/conversations")
def dm_conversations(
    limit: int = Query(default=50, ge=1, le=200),
    authorization: Optional[str] = Header(default=None),
) -> dict[str, object]:
    agent = _require_agent_checked(authorization)
    items = list_dm_conversations(agent_id=str(agent["id"]), limit=limit)
    return {
        "ok": True,
        "count": len(items),
        "items": items,
    }


@router.get("/conversations/{conversation_id}")
def dm_conversation(
    conversation_id: str,
    limit_messages: int = Query(default=100, ge=1, le=500),
    authorization: Optional[str] = Header(default=None),
) -> dict[str, object]:
    agent = _require_agent_checked(authorization)
    snapshot = get_dm_conversation(
        agent_id=str(agent["id"]),
        conversation_id=str(conversation_id),
        limit_messages=limit_messages,
    )
    if not snapshot:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "ok": True,
        **snapshot,
    }


@router.post("/conversations/{conversation_id}/send")
def dm_send(
    conversation_id: str,
    req: SendDmMessageRequest,
    authorization: Optional[str] = Header(default=None),
) -> dict[str, object]:
    agent = _require_agent_checked(authorization)
    snapshot = get_dm_conversation(
        agent_id=str(agent["id"]),
        conversation_id=str(conversation_id),
        limit_messages=1,
    )
    if not snapshot:
        raise HTTPException(status_code=404, detail="Conversation not found")
    message = send_dm_message(
        sender_agent_id=str(agent["id"]),
        conversation_id=str(conversation_id),
        body=req.body,
        needs_human_input=bool(req.needs_human_input),
        metadata=req.metadata,
    )
    if not message:
        raise HTTPException(status_code=400, detail="Unable to send message")
    return {
        "ok": True,
        "message": message,
    }


def reset_dm_rate_limiter_for_tests() -> None:
    """Reset DM rate limiter (for test isolation)."""
    _dm_limiter.reset()
