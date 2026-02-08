"""LLM control-plane endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from packages.vvalley_core.llm.policy import (
    fun_low_cost_preset_policies,
    normalize_policy_row,
    situational_default_preset_policies,
)

from ..storage.llm_control import get_policy, list_call_logs, list_policies, upsert_policy


router = APIRouter(prefix="/api/v1/llm", tags=["llm"])


class UpsertPolicyRequest(BaseModel):
    model_tier: str = Field(pattern="^(strong|fast|cheap|heuristic)$")
    max_input_tokens: int = Field(ge=1, le=200000)
    max_output_tokens: int = Field(ge=1, le=64000)
    temperature: float = Field(ge=0.0, le=2.0)
    timeout_ms: int = Field(ge=100, le=120000)
    retry_limit: int = Field(ge=0, le=10)
    enable_prompt_cache: bool = True


@router.get("/policies")
def get_policies() -> dict:
    policies = [p.as_dict() for p in list_policies()]
    return {"count": len(policies), "policies": policies}


@router.get("/policies/{task_name}")
def get_policy_by_task(task_name: str) -> dict:
    policy = get_policy(task_name)
    return {"task_name": task_name, "policy": policy.as_dict()}


@router.put("/policies/{task_name}")
def put_policy(task_name: str, req: UpsertPolicyRequest) -> dict:
    normalized = normalize_policy_row(
        task_name,
        {
            "model_tier": req.model_tier,
            "max_input_tokens": req.max_input_tokens,
            "max_output_tokens": req.max_output_tokens,
            "temperature": req.temperature,
            "timeout_ms": req.timeout_ms,
            "retry_limit": req.retry_limit,
            "enable_prompt_cache": req.enable_prompt_cache,
        },
    )
    saved = upsert_policy(normalized)
    return {"ok": True, "policy": saved.as_dict()}


@router.get("/logs")
def get_logs(
    limit: int = Query(default=50, ge=1, le=500),
    task_name: Optional[str] = Query(default=None),
) -> dict:
    rows = list_call_logs(limit=limit, task_name=task_name)
    return {"count": len(rows), "logs": rows}


@router.post("/presets/fun-low-cost")
def apply_fun_low_cost_preset() -> dict:
    saved = []
    for policy in fun_low_cost_preset_policies():
        saved_policy = upsert_policy(policy)
        saved.append(saved_policy.as_dict())
    return {
        "ok": True,
        "preset": "fun-low-cost",
        "count": len(saved),
        "policies": saved,
    }


@router.post("/presets/situational-default")
def apply_situational_default_preset() -> dict:
    saved = []
    for policy in situational_default_preset_policies():
        saved_policy = upsert_policy(policy)
        saved.append(saved_policy.as_dict())
    return {
        "ok": True,
        "preset": "situational-default",
        "count": len(saved),
        "policies": saved,
    }
