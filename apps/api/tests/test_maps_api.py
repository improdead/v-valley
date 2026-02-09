#!/usr/bin/env python3

from __future__ import annotations

import os
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
TEST_DB_DIR = tempfile.TemporaryDirectory(dir=ROOT)
os.environ["VVALLEY_DB_PATH"] = str(Path(TEST_DB_DIR.name) / "test_vvalley.db")

from apps.api.vvalley_api.main import app
from apps.api.vvalley_api.routers.agents import reset_rate_limiter_for_tests as reset_rate_limiter
from apps.api.vvalley_api.storage.agents import reset_backend_cache_for_tests as reset_agents_backend
from apps.api.vvalley_api.storage.interaction_hub import reset_backend_cache_for_tests as reset_interaction_backend
from apps.api.vvalley_api.storage.llm_control import reset_backend_cache_for_tests as reset_llm_backend
from apps.api.vvalley_api.storage.map_versions import reset_backend_cache_for_tests as reset_maps_backend
from apps.api.vvalley_api.storage.runtime_control import reset_backend_cache_for_tests as reset_runtime_backend

STARTER_MAP = ROOT / "assets/templates/starter_town/map.json"


class MapsApiTests(unittest.TestCase):
    def setUp(self) -> None:
        db_path = Path(os.environ["VVALLEY_DB_PATH"])
        if db_path.exists():
            db_path.unlink()
        reset_agents_backend()
        reset_maps_backend()
        reset_llm_backend()
        reset_runtime_backend()
        reset_interaction_backend()
        reset_rate_limiter()
        self.client = TestClient(app)

    def test_healthz(self) -> None:
        resp = self.client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("status"), "ok")

    def test_validate_file_ok(self) -> None:
        resp = self.client.post(
            "/api/v1/maps/validate-file",
            json={"map_path": "assets/templates/starter_town/map.json"},
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["errors"], [])

    def test_validate_file_missing(self) -> None:
        resp = self.client.post(
            "/api/v1/maps/validate-file",
            json={"map_path": "assets/templates/does_not_exist/map.json"},
        )
        self.assertEqual(resp.status_code, 404)

    def test_bake_file_writes_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            out = Path(td) / "starter.nav.json"
            resp = self.client.post(
                "/api/v1/maps/bake-file",
                json={
                    "map_path": "assets/templates/starter_town/map.json",
                    "out_path": str(out.relative_to(ROOT)),
                    "allow_invalid": False,
                },
            )
            self.assertEqual(resp.status_code, 200)
            payload = resp.json()
            self.assertTrue(payload["ok"])
            self.assertTrue(out.exists())

    def test_customize_file_writes_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            out = Path(td) / "starter.custom.json"
            resp = self.client.post(
                "/api/v1/maps/customize-file",
                json={
                    "map_path": "assets/templates/starter_town/map.json",
                    "config_path": "assets/templates/starter_town/customization.example.json",
                    "out_path": str(out.relative_to(ROOT)),
                    "validate_after": True,
                },
            )
            self.assertEqual(resp.status_code, 200)
            payload = resp.json()
            self.assertTrue(payload["ok"])
            self.assertTrue(out.exists())

    def test_publish_list_activate_versions(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"

        publish_1 = self.client.post(
            "/api/v1/maps/publish-version",
            json={
                "town_id": town_id,
                "map_path": "assets/templates/starter_town/map.json",
                "map_name": "starter",
            },
        )
        self.assertEqual(publish_1.status_code, 200)
        p1 = publish_1.json()
        self.assertTrue(p1["ok"])
        self.assertEqual(p1["version"]["version"], 1)
        self.assertTrue(p1["version"]["is_active"])

        publish_2 = self.client.post(
            "/api/v1/maps/publish-version",
            json={
                "town_id": town_id,
                "map_path": "assets/templates/starter_town/map.json",
                "map_name": "starter-again",
            },
        )
        self.assertEqual(publish_2.status_code, 200)
        p2 = publish_2.json()
        self.assertTrue(p2["ok"])
        self.assertEqual(p2["version"]["version"], 2)
        self.assertFalse(p2["version"]["is_active"])

        listed = self.client.get(f"/api/v1/maps/versions/{town_id}")
        self.assertEqual(listed.status_code, 200)
        versions = listed.json()["versions"]
        self.assertEqual(len(versions), 2)
        self.assertEqual(versions[0]["version"], 2)
        self.assertEqual(versions[1]["version"], 1)

        active_before = self.client.get(f"/api/v1/maps/versions/{town_id}/active")
        self.assertEqual(active_before.status_code, 200)
        self.assertEqual(active_before.json()["active"]["version"], 1)

        activate = self.client.post(f"/api/v1/maps/versions/{town_id}/2/activate")
        self.assertEqual(activate.status_code, 200)
        self.assertTrue(activate.json()["ok"])
        self.assertEqual(activate.json()["active"]["version"], 2)

        active_after = self.client.get(f"/api/v1/maps/versions/{town_id}/active")
        self.assertEqual(active_after.status_code, 200)
        self.assertEqual(active_after.json()["active"]["version"], 2)

    def test_list_templates(self) -> None:
        resp = self.client.get("/api/v1/maps/templates")
        self.assertEqual(resp.status_code, 200)

        payload = resp.json()
        self.assertGreaterEqual(payload["count"], 1)
        template_ids = [item["template_id"] for item in payload["templates"]]
        self.assertIn("starter_town", template_ids)

    def test_scaffold_template_with_inline_customization(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            out_map = Path(td) / "oakville.draft.map.json"
            out_nav = Path(td) / "oakville.draft.nav.json"

            resp = self.client.post(
                "/api/v1/maps/templates/scaffold",
                json={
                    "template_id": "starter_town",
                    "town_id": f"oakville-{uuid.uuid4().hex[:6]}",
                    "out_map_path": str(out_map.relative_to(ROOT)),
                    "out_nav_path": str(out_nav.relative_to(ROOT)),
                    "customization": {
                        "map_properties": {"town_theme": "forest"},
                        "locations": [
                            {
                                "name": "Johnson Park",
                                "affordances": ["leisure", "social"],
                                "hours": "06:00-22:00",
                            }
                        ],
                    },
                },
            )
            self.assertEqual(resp.status_code, 200)
            payload = resp.json()
            self.assertTrue(payload["ok"])
            self.assertTrue(out_map.exists())
            self.assertTrue(out_nav.exists())

            map_data = json.loads(out_map.read_text(encoding="utf-8"))
            props = {p["name"]: p["value"] for p in map_data.get("properties", [])}
            self.assertEqual(props.get("town_theme"), "forest")

    def test_first_map_end_to_end_pipeline(self) -> None:
        town_id = f"oakville-{uuid.uuid4().hex[:8]}"

        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            out_map = Path(td) / "oakville.draft.map.json"
            out_nav = Path(td) / "oakville.draft.nav.json"
            rebaked_nav = Path(td) / "oakville.draft.rebaked.nav.json"

            scaffold = self.client.post(
                "/api/v1/maps/templates/scaffold",
                json={
                    "template_id": "starter_town",
                    "town_id": town_id,
                    "out_map_path": str(out_map.relative_to(ROOT)),
                    "out_nav_path": str(out_nav.relative_to(ROOT)),
                    "config_path": "assets/templates/starter_town/customization.example.json",
                    "validate_after": True,
                },
            )
            self.assertEqual(scaffold.status_code, 200)
            s_payload = scaffold.json()
            self.assertTrue(s_payload["ok"])
            self.assertTrue(out_map.exists())
            self.assertTrue(out_nav.exists())

            validated = self.client.post(
                "/api/v1/maps/validate-file",
                json={"map_path": str(out_map.relative_to(ROOT))},
            )
            self.assertEqual(validated.status_code, 200)
            self.assertTrue(validated.json()["ok"])

            baked = self.client.post(
                "/api/v1/maps/bake-file",
                json={
                    "map_path": str(out_map.relative_to(ROOT)),
                    "out_path": str(rebaked_nav.relative_to(ROOT)),
                    "allow_invalid": False,
                },
            )
            self.assertEqual(baked.status_code, 200)
            self.assertTrue(baked.json()["ok"])
            self.assertTrue(rebaked_nav.exists())

            published = self.client.post(
                "/api/v1/maps/publish-version",
                json={
                    "town_id": town_id,
                    "map_path": str(out_map.relative_to(ROOT)),
                    "map_name": "starter-e2e",
                },
            )
            self.assertEqual(published.status_code, 200)
            p_payload = published.json()
            self.assertTrue(p_payload["ok"])
            self.assertEqual(p_payload["version"]["version"], 1)

            active = self.client.get(f"/api/v1/maps/versions/{town_id}/active")
            self.assertEqual(active.status_code, 200)
            self.assertEqual(active.json()["active"]["version"], 1)


if __name__ == "__main__":
    unittest.main()
