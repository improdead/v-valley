#!/usr/bin/env python3

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
TEST_DB_DIR = tempfile.TemporaryDirectory(dir=ROOT)
TEST_DB_PATH = Path(TEST_DB_DIR.name) / "test_legacy_vvalley.db"
os.environ["VVALLEY_DB_PATH"] = str(TEST_DB_PATH)

from apps.api.vvalley_api.main import app
from apps.api.vvalley_api.routers.agents import reset_rate_limiter_for_tests as reset_rate_limiter
from apps.api.vvalley_api.storage.agents import reset_backend_cache_for_tests as reset_agents_backend
from apps.api.vvalley_api.storage.interaction_hub import reset_backend_cache_for_tests as reset_interaction_backend
from apps.api.vvalley_api.storage.llm_control import reset_backend_cache_for_tests as reset_llm_backend
from apps.api.vvalley_api.storage.map_versions import reset_backend_cache_for_tests as reset_maps_backend
from apps.api.vvalley_api.storage.runtime_control import reset_backend_cache_for_tests as reset_runtime_backend
from apps.api.vvalley_api.storage.scenarios import reset_backend_cache_for_tests as reset_scenarios_backend


class LegacyApiTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["VVALLEY_DB_PATH"] = str(TEST_DB_PATH)
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        reset_agents_backend()
        reset_maps_backend()
        reset_llm_backend()
        reset_runtime_backend()
        reset_interaction_backend()
        reset_scenarios_backend()
        reset_rate_limiter()
        self.client = TestClient(app)

    def test_list_legacy_simulations_and_replay_events(self) -> None:
        sims_resp = self.client.get("/api/v1/legacy/simulations")
        self.assertEqual(sims_resp.status_code, 200)
        payload = sims_resp.json()
        self.assertGreaterEqual(payload["count"], 1)

        sim_id = payload["simulations"][0]["simulation_id"]
        replay_resp = self.client.post(
            f"/api/v1/legacy/simulations/{sim_id}/events",
            json={"start_step": 0, "max_steps": 2},
        )
        self.assertEqual(replay_resp.status_code, 200)
        replay = replay_resp.json()
        self.assertEqual(replay["simulation_id"], sim_id)
        self.assertGreaterEqual(replay["event_count"], 1)
        self.assertGreaterEqual(len(replay["persona_names"]), 1)

    def test_import_the_ville_and_publish(self) -> None:
        town_id = "legacy_the_ville_test"
        resp = self.client.post(
            "/api/v1/legacy/import-the-ville",
            json={
                "town_id": town_id,
                "publish_version_record": True,
            },
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["published"])
        self.assertEqual(payload["town_id"], town_id)
        self.assertEqual(payload["version"]["version"], 1)

        detail = self.client.get(f"/api/v1/towns/{town_id}")
        self.assertEqual(detail.status_code, 200)
        detail_payload = detail.json()
        self.assertEqual(detail_payload["town"]["town_id"], town_id)
        self.assertEqual(detail_payload["town"]["active_map_version"]["version"], 1)


if __name__ == "__main__":
    unittest.main()
