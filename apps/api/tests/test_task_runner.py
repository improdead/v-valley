#!/usr/bin/env python3

from __future__ import annotations

import unittest

from packages.vvalley_core.llm.policy import TaskPolicy
from packages.vvalley_core.llm.providers import (
    ProviderExecutionError,
    ProviderExecutionResult,
    ProviderUnavailableError,
)
from packages.vvalley_core.llm.task_runner import PolicyTaskRunner


def _heuristic_move(_: dict) -> dict:
    return {"dx": 0, "dy": 1, "reason": "heuristic"}


class PolicyTaskRunnerTests(unittest.TestCase):
    @staticmethod
    def _policy(task_name: str, model_tier: str = "strong", retry_limit: int = 1) -> TaskPolicy:
        return TaskPolicy(
            task_name=task_name,
            model_tier=model_tier,
            max_input_tokens=640,
            max_output_tokens=80,
            temperature=0.2,
            timeout_ms=1200,
            retry_limit=retry_limit,
        )

    def test_runner_uses_provider_before_heuristic(self) -> None:
        calls: list[str] = []
        logs: list[dict] = []

        def fake_provider(**kwargs):
            calls.append(kwargs["tier"])
            return ProviderExecutionResult(
                output={"dx": 1, "dy": 0, "reason": "provider_move"},
                model_name="openai_compatible:test-model",
                prompt_tokens=12,
                completion_tokens=8,
            )

        runner = PolicyTaskRunner(
            policy_lookup=lambda _: self._policy("short_action", model_tier="strong", retry_limit=1),
            log_sink=logs.append,
            provider_invoker=fake_provider,
        )
        result = runner.run(
            task_name="short_action",
            town_id="town-a",
            agent_id="agent-a",
            context={"agent_id": "agent-a", "step": 1},
            heuristic_fn=_heuristic_move,
        )

        self.assertEqual(calls, ["strong"])
        self.assertEqual(result.route, "provider")
        self.assertEqual(result.used_tier, "strong")
        self.assertEqual(result.output["reason"], "provider_move")
        self.assertTrue(logs[-1]["success"])
        self.assertEqual(logs[-1]["model_name"], "openai_compatible:test-model")

    def test_runner_falls_back_to_heuristic_when_provider_unavailable(self) -> None:
        calls: list[str] = []
        logs: list[dict] = []

        def fake_provider(**kwargs):
            calls.append(kwargs["tier"])
            raise ProviderUnavailableError(
                "missing configuration",
                error_code="missing_model",
            )

        runner = PolicyTaskRunner(
            policy_lookup=lambda _: self._policy("short_action", model_tier="strong", retry_limit=1),
            log_sink=logs.append,
            provider_invoker=fake_provider,
        )
        result = runner.run(
            task_name="short_action",
            town_id="town-b",
            agent_id="agent-b",
            context={"agent_id": "agent-b", "step": 2},
            heuristic_fn=_heuristic_move,
        )

        self.assertEqual(calls, ["strong", "fast", "cheap"])
        self.assertEqual(result.route, "heuristic")
        self.assertEqual(result.used_tier, "heuristic")
        self.assertEqual(result.output["reason"], "heuristic")
        self.assertFalse(logs[0]["success"])
        self.assertEqual(logs[0]["error_code"], "missing_model")
        self.assertTrue(logs[-1]["success"])
        self.assertEqual(logs[-1]["model_name"], "heuristic:local")

    def test_runner_retries_provider_then_uses_heuristic(self) -> None:
        call_count = 0

        def fake_provider(**kwargs):
            nonlocal call_count
            call_count += 1
            raise ProviderExecutionError(
                "network timeout",
                error_code="network_error",
                model_name="openai_compatible:test-timeout-model",
            )

        runner = PolicyTaskRunner(
            policy_lookup=lambda _: self._policy("short_action", model_tier="cheap", retry_limit=1),
            provider_invoker=fake_provider,
        )
        result = runner.run(
            task_name="short_action",
            town_id="town-c",
            agent_id="agent-c",
            context={"agent_id": "agent-c", "step": 3},
            heuristic_fn=_heuristic_move,
        )

        self.assertEqual(call_count, 2)
        self.assertEqual(result.route, "heuristic")
        self.assertEqual(result.used_tier, "heuristic")


if __name__ == "__main__":
    unittest.main()
