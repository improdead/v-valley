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
from .providers import (
    ProviderExecutionError,
    ProviderExecutionResult,
    ProviderUnavailableError,
    execute_tier_model,
)


PolicyLookup = Callable[[str], TaskPolicy]
LogSink = Callable[[dict[str, Any]], None]
HeuristicFn = Callable[[dict[str, Any]], dict[str, Any]]
ProviderInvoker = Callable[..., ProviderExecutionResult]


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

    def __init__(
        self,
        *,
        policy_lookup: PolicyLookup | None = None,
        log_sink: LogSink | None = None,
        provider_invoker: ProviderInvoker | None = None,
    ) -> None:
        self._policy_lookup = policy_lookup or default_policy_for_task
        self._log_sink = log_sink
        self._provider_invoker = provider_invoker or execute_tier_model

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
        fallback_chain = resolve_fallback_chain(policy.model_tier)
        for tier in fallback_chain:
            if tier == "heuristic":
                break
            for _ in range(max(1, policy.retry_limit + 1)):
                start = perf_counter()
                try:
                    provider_result = self._provider_invoker(
                        tier=tier,
                        task_name=task_name,
                        bounded_context_text=bounded_context,
                        temperature=policy.temperature,
                        max_output_tokens=policy.max_output_tokens,
                        timeout_ms=policy.timeout_ms,
                    )
                    latency_ms = int((perf_counter() - start) * 1000)
                    completion_tokens = int(
                        provider_result.completion_tokens
                        if provider_result.completion_tokens is not None
                        else max(1, len(json.dumps(provider_result.output, separators=(",", ":"))) // 4)
                    )
                    result = TaskExecutionResult(
                        task_name=task_name,
                        route="provider",
                        used_tier=tier,
                        output=dict(provider_result.output),
                        prompt_tokens=int(provider_result.prompt_tokens or prompt_tokens),
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                        context_trimmed=trimmed,
                        policy=policy.as_dict(),
                    )
                    self._emit_log(
                        town_id=town_id,
                        agent_id=agent_id,
                        task_name=task_name,
                        model_name=provider_result.model_name,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        latency_ms=latency_ms,
                        success=True,
                        error_code=None,
                    )
                    return result
                except ProviderUnavailableError as exc:
                    latency_ms = int((perf_counter() - start) * 1000)
                    self._emit_log(
                        town_id=town_id,
                        agent_id=agent_id,
                        task_name=task_name,
                        model_name=exc.model_name or f"{tier}:provider",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        latency_ms=latency_ms,
                        success=False,
                        error_code=exc.error_code,
                    )
                    break
                except ProviderExecutionError as exc:
                    latency_ms = int((perf_counter() - start) * 1000)
                    self._emit_log(
                        town_id=town_id,
                        agent_id=agent_id,
                        task_name=task_name,
                        model_name=exc.model_name or f"{tier}:provider",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        latency_ms=latency_ms,
                        success=False,
                        error_code=exc.error_code,
                    )
                    continue
                except Exception as exc:
                    latency_ms = int((perf_counter() - start) * 1000)
                    self._emit_log(
                        town_id=town_id,
                        agent_id=agent_id,
                        task_name=task_name,
                        model_name=f"{tier}:provider",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        latency_ms=latency_ms,
                        success=False,
                        error_code=f"provider_exception:{exc.__class__.__name__}",
                    )
                    continue

        start = perf_counter()
        output = heuristic_fn(context)
        latency_ms = int((perf_counter() - start) * 1000)
        completion_tokens = max(1, len(json.dumps(output, separators=(",", ":"))) // 4)
        result = TaskExecutionResult(
            task_name=task_name,
            route="heuristic",
            used_tier="heuristic",
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
            model_name="heuristic:local",
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
