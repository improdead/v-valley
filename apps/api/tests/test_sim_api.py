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
TEST_DB_PATH = Path(TEST_DB_DIR.name) / "test_sim_vvalley.db"
os.environ["VVALLEY_DB_PATH"] = str(TEST_DB_PATH)

from apps.api.vvalley_api.main import app
from apps.api.vvalley_api.storage.agents import reset_backend_cache_for_tests as reset_agents_backend
from apps.api.vvalley_api.storage.llm_control import reset_backend_cache_for_tests as reset_llm_backend
from apps.api.vvalley_api.storage.map_versions import reset_backend_cache_for_tests as reset_maps_backend
from packages.vvalley_core.sim.runner import reset_simulation_states_for_tests


class SimApiTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["VVALLEY_DB_PATH"] = str(TEST_DB_PATH)
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        reset_agents_backend()
        reset_maps_backend()
        reset_llm_backend()
        reset_simulation_states_for_tests()
        self.client = TestClient(app)

    def _publish_town(self, town_id: str, *, map_name: str = "starter") -> None:
        resp = self.client.post(
            "/api/v1/maps/publish-version",
            json={
                "town_id": town_id,
                "map_path": "assets/templates/starter_town/map.json",
                "map_name": map_name,
            },
        )
        self.assertEqual(resp.status_code, 200)

    def _register_and_join(self, town_id: str, name: str, owner: str) -> dict[str, str]:
        registered = self.client.post(
            "/api/v1/agents/register",
            json={"name": name, "owner_handle": owner, "auto_claim": True},
        )
        self.assertEqual(registered.status_code, 200)
        payload = registered.json()["agent"]
        joined = self.client.post(
            "/api/v1/agents/me/join-town",
            json={"town_id": town_id},
            headers={"Authorization": f"Bearer {payload['api_key']}"},
        )
        self.assertEqual(joined.status_code, 200)
        return payload

    def test_sim_state_requires_active_map(self) -> None:
        missing = self.client.get("/api/v1/sim/towns/does-not-exist/state")
        self.assertEqual(missing.status_code, 404)

    def test_state_and_tick_flow(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)
        self._register_and_join(town_id, "Alice", "owner-a")
        self._register_and_join(town_id, "Bob", "owner-b")

        state_resp = self.client.get(f"/api/v1/sim/towns/{town_id}/state")
        self.assertEqual(state_resp.status_code, 200)
        state_payload = state_resp.json()["state"]
        self.assertEqual(state_payload["npc_count"], 2)
        self.assertEqual(state_payload["step"], 0)
        self.assertEqual(state_payload["max_agents"], 25)
        self.assertEqual(state_payload["map_version"], 1)

        tick_resp = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 3},
        )
        self.assertEqual(tick_resp.status_code, 200)
        tick = tick_resp.json()["tick"]
        self.assertEqual(tick["steps_run"], 3)
        self.assertEqual(tick["step"], 3)
        self.assertEqual(tick["event_count"], 6)
        self.assertIn("social_event_count", tick)
        self.assertIn("social_events", tick)
        self.assertIn("reflection_event_count", tick)
        self.assertIn("reflection_events", tick)
        self.assertGreaterEqual(len(tick["events"]), 1)

        state_after = self.client.get(f"/api/v1/sim/towns/{town_id}/state")
        self.assertEqual(state_after.status_code, 200)
        after_payload = state_after.json()["state"]
        self.assertEqual(after_payload["step"], 3)
        self.assertEqual(after_payload["npc_count"], 2)
        for npc in after_payload["npcs"]:
            self.assertGreaterEqual(int(npc["x"]), 0)
            self.assertGreaterEqual(int(npc["y"]), 0)
            self.assertLess(int(npc["x"]), int(after_payload["map_width"]))
            self.assertLess(int(npc["y"]), int(after_payload["map_height"]))
            self.assertIn("memory_summary", npc)
            self.assertGreaterEqual(int(npc["memory_summary"]["total_nodes"]), 1)

    def test_state_resets_when_active_map_changes(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id, map_name="v1")
        self._register_and_join(town_id, "Runner", "owner-r")

        first_tick = self.client.post(f"/api/v1/sim/towns/{town_id}/tick", json={"steps": 2})
        self.assertEqual(first_tick.status_code, 200)
        self.assertEqual(first_tick.json()["tick"]["step"], 2)

        self._publish_town(town_id, map_name="v2")
        activate = self.client.post(f"/api/v1/maps/versions/{town_id}/2/activate")
        self.assertEqual(activate.status_code, 200)

        state = self.client.get(f"/api/v1/sim/towns/{town_id}/state")
        self.assertEqual(state.status_code, 200)
        payload = state.json()["state"]
        self.assertEqual(payload["map_version"], 2)
        self.assertEqual(payload["step"], 0)

    def test_agent_memory_snapshot_endpoint(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)
        agent = self._register_and_join(town_id, "MemoryTester", "owner-m")

        tick_resp = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 2, "planning_scope": "daily_plan"},
        )
        self.assertEqual(tick_resp.status_code, 200)

        memory_resp = self.client.get(
            f"/api/v1/sim/towns/{town_id}/agents/{agent['id']}/memory",
            params={"limit": 25},
        )
        self.assertEqual(memory_resp.status_code, 200)
        snapshot = memory_resp.json()["snapshot"]
        self.assertEqual(snapshot["agent"]["agent_id"], agent["id"])
        self.assertIn("memory", snapshot["agent"])
        self.assertGreaterEqual(snapshot["agent"]["memory"]["summary"]["total_nodes"], 1)
        self.assertGreaterEqual(len(snapshot["agent"]["memory"]["nodes"]), 1)


if __name__ == "__main__":
    unittest.main()
