"""Policy-aware cognition task runner with fallback and logging hooks."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable
import json
import uuid

from .policy import (
    TaskPolicy,
    default_policy_for_task,
    resolve_fallback_chain,
    trim_context_to_budget,
)


PolicyLookup = Callable[[str], TaskPolicy]
LogSink = Callable[[dict[str, Any]], None]
HeuristicFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class TaskExecutionResult:
    task_name: str
    route: str
    used_tier: str
    output: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    context_trimmed: bool
    policy: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "route": self.route,
            "used_tier": self.used_tier,
            "output": self.output,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "context_trimmed": self.context_trimmed,
            "policy": self.policy,
        }


class PolicyTaskRunner:
    """Runs cognition tasks under explicit policy constraints."""

    def __init__(self, *, policy_lookup: PolicyLookup | None = None, log_sink: LogSink | None = None) -> None:
        self._policy_lookup = policy_lookup or default_policy_for_task
        self._log_sink = log_sink

    def run(
        self,
        *,
        task_name: str,
        town_id: str | None,
        agent_id: str | None,
        context: dict[str, Any],
        heuristic_fn: HeuristicFn,
    ) -> TaskExecutionResult:
        policy = self._policy_lookup(task_name)
        if not isinstance(policy, TaskPolicy):
            policy = default_policy_for_task(task_name)

        context_text = json.dumps(context, sort_keys=True, separators=(",", ":"))
        bounded_context, trimmed, prompt_tokens = trim_context_to_budget(
            context_text,
            policy.max_input_tokens,
        )
        _ = bounded_context  # placeholder for future provider execution.

        fallback_chain = resolve_fallback_chain(policy.model_tier)
        selected_tier = fallback_chain[-1]  # currently always final fallback: heuristic
        start = perf_counter()
        output = heuristic_fn(context)
        latency_ms = int((perf_counter() - start) * 1000)
        completion_tokens = max(1, len(json.dumps(output, separators=(",", ":"))) // 4)

        result = TaskExecutionResult(
            task_name=task_name,
            route="heuristic",
            used_tier=selected_tier,
            output=output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            context_trimmed=trimmed,
            policy=policy.as_dict(),
        )
        self._emit_log(
            town_id=town_id,
            agent_id=agent_id,
            task_name=task_name,
            model_name=f"{selected_tier}:heuristic",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            success=True,
            error_code=None,
        )
        return result

    def _emit_log(
        self,
        *,
        town_id: str | None,
        agent_id: str | None,
        task_name: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        success: bool,
        error_code: str | None,
    ) -> None:
        if not self._log_sink:
            return
        self._log_sink(
            {
                "id": str(uuid.uuid4()),
                "town_id": town_id,
                "agent_id": agent_id,
                "task_name": task_name,
                "model_name": model_name,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "cached_tokens": 0,
                "latency_ms": int(latency_ms),
                "success": bool(success),
                "error_code": error_code,
            }
        )
