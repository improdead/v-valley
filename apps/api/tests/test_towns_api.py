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
TEST_DB_PATH = Path(TEST_DB_DIR.name) / "test_towns_vvalley.db"
os.environ["VVALLEY_DB_PATH"] = str(TEST_DB_PATH)

from apps.api.vvalley_api.main import app
from apps.api.vvalley_api.routers.agents import reset_rate_limiter_for_tests as reset_rate_limiter
from apps.api.vvalley_api.storage.agents import reset_backend_cache_for_tests as reset_agents_backend
from apps.api.vvalley_api.storage.interaction_hub import reset_backend_cache_for_tests as reset_interaction_backend
from apps.api.vvalley_api.storage.llm_control import reset_backend_cache_for_tests as reset_llm_backend
from apps.api.vvalley_api.storage.map_versions import reset_backend_cache_for_tests as reset_maps_backend
from apps.api.vvalley_api.storage.runtime_control import reset_backend_cache_for_tests as reset_runtime_backend


class TownsApiTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["VVALLEY_DB_PATH"] = str(TEST_DB_PATH)
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        reset_agents_backend()
        reset_maps_backend()
        reset_llm_backend()
        reset_runtime_backend()
        reset_interaction_backend()
        reset_rate_limiter()
        self.client = TestClient(app)

    def test_town_directory_empty_then_populated(self) -> None:
        empty = self.client.get("/api/v1/towns")
        self.assertEqual(empty.status_code, 200)
        self.assertEqual(empty.json()["count"], 0)

        town_id = f"town-{uuid.uuid4().hex[:8]}"
        publish = self.client.post(
            "/api/v1/maps/publish-version",
            json={
                "town_id": town_id,
                "map_path": "assets/templates/starter_town/map.json",
                "map_name": "starter",
            },
        )
        self.assertEqual(publish.status_code, 200)
        self.assertTrue(publish.json()["ok"])

        directory = self.client.get("/api/v1/towns")
        self.assertEqual(directory.status_code, 200)
        payload = directory.json()
        self.assertEqual(payload["count"], 1)
        town = payload["towns"][0]
        self.assertEqual(town["town_id"], town_id)
        self.assertTrue(town["observer_mode"])
        self.assertEqual(town["max_agents"], 25)
        self.assertEqual(town["population"], 0)
        self.assertEqual(town["active_map_version"]["version"], 1)

    def test_town_detail_includes_versions(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self.client.post(
            "/api/v1/maps/publish-version",
            json={
                "town_id": town_id,
                "map_path": "assets/templates/starter_town/map.json",
                "map_name": "starter",
            },
        )
        self.client.post(
            "/api/v1/maps/publish-version",
            json={
                "town_id": town_id,
                "map_path": "assets/templates/starter_town/map.json",
                "map_name": "starter-v2",
            },
        )

        detail = self.client.get(f"/api/v1/towns/{town_id}")
        self.assertEqual(detail.status_code, 200)
        payload = detail.json()
        self.assertEqual(payload["town"]["town_id"], town_id)
        self.assertEqual(len(payload["map_versions"]), 2)
        self.assertEqual(payload["map_versions"][0]["version"], 2)

        missing = self.client.get("/api/v1/towns/does-not-exist")
        self.assertEqual(missing.status_code, 404)

    def test_town_population_reflects_joined_agents(self) -> None:
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

        registered = self.client.post(
            "/api/v1/agents/register",
            json={"name": "ResidentAgent", "owner_handle": "dekai", "auto_claim": True},
        ).json()["agent"]

        joined = self.client.post(
            "/api/v1/agents/me/join-town",
            json={"town_id": town_id},
            headers={"Authorization": f"Bearer {registered['api_key']}"},
        )
        self.assertEqual(joined.status_code, 200)

        directory = self.client.get("/api/v1/towns").json()
        self.assertEqual(directory["count"], 1)
        self.assertEqual(directory["towns"][0]["population"], 1)

        detail = self.client.get(f"/api/v1/towns/{town_id}").json()
        self.assertEqual(detail["town"]["population"], 1)


if __name__ == "__main__":
    unittest.main()
