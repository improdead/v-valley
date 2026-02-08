#!/usr/bin/env python3

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
TEST_DB_DIR = tempfile.TemporaryDirectory(dir=ROOT)
TEST_DB_PATH = Path(TEST_DB_DIR.name) / "test_llm_vvalley.db"
os.environ["VVALLEY_DB_PATH"] = str(TEST_DB_PATH)

from apps.api.vvalley_api.main import app
from apps.api.vvalley_api.storage.agents import reset_backend_cache_for_tests as reset_agents_backend
from apps.api.vvalley_api.storage.llm_control import reset_backend_cache_for_tests as reset_llm_backend
from apps.api.vvalley_api.storage.map_versions import reset_backend_cache_for_tests as reset_maps_backend
from packages.vvalley_core.sim.runner import reset_simulation_states_for_tests


class LlmApiTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["VVALLEY_DB_PATH"] = str(TEST_DB_PATH)
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        reset_agents_backend()
        reset_maps_backend()
        reset_llm_backend()
        reset_simulation_states_for_tests()
        self.client = TestClient(app)

    def test_default_policies_are_exposed(self) -> None:
        resp = self.client.get("/api/v1/llm/policies")
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertGreaterEqual(payload["count"], 4)
        plan_policy = next(
            p for p in payload["policies"] if p["task_name"] == "plan_move"
        )
        self.assertEqual(plan_policy["model_tier"], "cheap")
        self.assertIn("heuristic", plan_policy["fallback_chain"])

    def test_upsert_policy_then_get(self) -> None:
        put = self.client.put(
            "/api/v1/llm/policies/plan_move",
            json={
                "model_tier": "cheap",
                "max_input_tokens": 640,
                "max_output_tokens": 64,
                "temperature": 0.1,
                "timeout_ms": 1800,
                "retry_limit": 1,
                "enable_prompt_cache": True,
            },
        )
        self.assertEqual(put.status_code, 200)
        policy = put.json()["policy"]
        self.assertEqual(policy["task_name"], "plan_move")
        self.assertEqual(policy["model_tier"], "cheap")

        fetched = self.client.get("/api/v1/llm/policies/plan_move")
        self.assertEqual(fetched.status_code, 200)
        fetched_policy = fetched.json()["policy"]
        self.assertEqual(fetched_policy["model_tier"], "cheap")
        self.assertEqual(int(fetched_policy["max_input_tokens"]), 640)

    def test_tick_generates_llm_task_logs(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        published = self.client.post(
            "/api/v1/maps/publish-version",
            json={
                "town_id": town_id,
                "map_path": "assets/templates/starter_town/map.json",
                "map_name": "starter",
            },
        )
        self.assertEqual(published.status_code, 200)

        agent = self.client.post(
            "/api/v1/agents/register",
            json={"name": "PlannerOne", "owner_handle": "owner-a", "auto_claim": True},
        ).json()["agent"]
        joined = self.client.post(
            "/api/v1/agents/me/join-town",
            json={"town_id": town_id},
            headers={"Authorization": f"Bearer {agent['api_key']}"},
        )
        self.assertEqual(joined.status_code, 200)

        ticked = self.client.post(f"/api/v1/sim/towns/{town_id}/tick", json={"steps": 2})
        self.assertEqual(ticked.status_code, 200)
        events = ticked.json()["tick"]["events"]
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[0]["decision"]["route"], "heuristic")

        logs = self.client.get("/api/v1/llm/logs", params={"task_name": "short_action", "limit": 20})
        self.assertEqual(logs.status_code, 200)
        payload = logs.json()
        self.assertGreaterEqual(payload["count"], 1)
        first = payload["logs"][0]
        self.assertEqual(first["task_name"], "short_action")
        self.assertTrue(first["model_name"].endswith(":heuristic"))
        self.assertTrue(bool(first["success"]))

    def test_tick_scope_routes_to_daily_or_long_term_policy(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        published = self.client.post(
            "/api/v1/maps/publish-version",
            json={
                "town_id": town_id,
                "map_path": "assets/templates/starter_town/map.json",
                "map_name": "starter",
            },
        )
        self.assertEqual(published.status_code, 200)

        agent = self.client.post(
            "/api/v1/agents/register",
            json={"name": "PlannerScope", "owner_handle": "owner-s", "auto_claim": True},
        ).json()["agent"]
        joined = self.client.post(
            "/api/v1/agents/me/join-town",
            json={"town_id": town_id},
            headers={"Authorization": f"Bearer {agent['api_key']}"},
        )
        self.assertEqual(joined.status_code, 200)

        daily_tick = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 1, "planning_scope": "daily_plan"},
        )
        self.assertEqual(daily_tick.status_code, 200)
        self.assertEqual(daily_tick.json()["tick"]["events"][0]["decision"]["policy_task"], "daily_plan")

        long_tick = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 1, "planning_scope": "long_term_plan"},
        )
        self.assertEqual(long_tick.status_code, 200)
        self.assertEqual(long_tick.json()["tick"]["events"][0]["decision"]["policy_task"], "long_term_plan")

    def test_apply_fun_low_cost_preset_endpoint(self) -> None:
        resp = self.client.post("/api/v1/llm/presets/fun-low-cost")
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["preset"], "fun-low-cost")
        self.assertGreaterEqual(payload["count"], 4)

    def test_apply_situational_default_preset_endpoint(self) -> None:
        resp = self.client.post("/api/v1/llm/presets/situational-default")
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["preset"], "situational-default")
        self.assertGreaterEqual(payload["count"], 6)


if __name__ == "__main__":
    unittest.main()
