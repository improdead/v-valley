#!/usr/bin/env python3

from __future__ import annotations

import os
import unittest

from packages.vvalley_core.llm.providers import (
    DEFAULT_MODEL_BY_TIER,
    ProviderExecutionError,
    ProviderUnavailableError,
    _tier_provider_config,
    _validate_output_schema,
)


class ProviderConfigTests(unittest.TestCase):
    _env_keys = (
        "VVALLEY_LLM_MODEL",
        "VVALLEY_LLM_STRONG_MODEL",
        "VVALLEY_LLM_FAST_MODEL",
        "VVALLEY_LLM_CHEAP_MODEL",
        "VVALLEY_LLM_PROVIDER",
        "VVALLEY_LLM_STRONG_PROVIDER",
        "VVALLEY_LLM_FAST_PROVIDER",
        "VVALLEY_LLM_CHEAP_PROVIDER",
        "VVALLEY_LLM_BASE_URL",
        "VVALLEY_LLM_STRONG_BASE_URL",
        "VVALLEY_LLM_FAST_BASE_URL",
        "VVALLEY_LLM_CHEAP_BASE_URL",
        "VVALLEY_LLM_API_KEY",
        "VVALLEY_LLM_STRONG_API_KEY",
        "VVALLEY_LLM_FAST_API_KEY",
        "VVALLEY_LLM_CHEAP_API_KEY",
        "VVALLEY_LLM_ALLOW_EMPTY_API_KEY",
        "OPENAI_API_KEY",
    )

    def setUp(self) -> None:
        self._env_backup = {k: os.environ.get(k) for k in self._env_keys}
        for key in self._env_keys:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_default_models_follow_latest_tier_mapping(self) -> None:
        os.environ["VVALLEY_LLM_ALLOW_EMPTY_API_KEY"] = "1"

        strong = _tier_provider_config("strong")
        fast = _tier_provider_config("fast")
        cheap = _tier_provider_config("cheap")

        self.assertEqual(strong.model, DEFAULT_MODEL_BY_TIER["strong"])
        self.assertEqual(fast.model, DEFAULT_MODEL_BY_TIER["fast"])
        self.assertEqual(cheap.model, DEFAULT_MODEL_BY_TIER["cheap"])

    def test_tier_specific_model_override_precedence(self) -> None:
        os.environ["VVALLEY_LLM_ALLOW_EMPTY_API_KEY"] = "1"
        os.environ["VVALLEY_LLM_MODEL"] = "global-model"
        os.environ["VVALLEY_LLM_FAST_MODEL"] = "fast-tier-model"

        fast = _tier_provider_config("fast")
        cheap = _tier_provider_config("cheap")

        self.assertEqual(fast.model, "fast-tier-model")
        self.assertEqual(cheap.model, "global-model")

    def test_missing_api_key_is_rejected_by_default(self) -> None:
        with self.assertRaises(ProviderUnavailableError) as raised:
            _tier_provider_config("strong")
        self.assertEqual(raised.exception.error_code, "missing_api_key")

    def test_schema_validation_accepts_valid_reaction_policy(self) -> None:
        _validate_output_schema(
            "reaction_policy",
            {
                "decision": "talk",
                "reason": "nearby ally available",
                "wait_steps": 1,
            },
        )

    def test_schema_validation_rejects_invalid_reaction_policy(self) -> None:
        with self.assertRaises(ProviderExecutionError) as raised:
            _validate_output_schema(
                "reaction_policy",
                {
                    "decision": "dance",
                    "reason": "invalid enum value",
                },
            )
        self.assertEqual(raised.exception.error_code, "invalid_schema_output")

    def test_schema_validation_rejects_boolean_for_integer_fields(self) -> None:
        with self.assertRaises(ProviderExecutionError) as raised:
            _validate_output_schema(
                "short_action",
                {
                    "dx": True,
                    "dy": 0,
                    "reason": "bool should not pass integer validation",
                },
            )
        self.assertEqual(raised.exception.error_code, "invalid_schema_output")


if __name__ == "__main__":
    unittest.main()
