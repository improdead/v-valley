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
TEST_DB_PATH = Path(TEST_DB_DIR.name) / "test_agents_vvalley.db"
os.environ["VVALLEY_DB_PATH"] = str(TEST_DB_PATH)

from apps.api.vvalley_api.main import app
from apps.api.vvalley_api.storage.agents import reset_backend_cache_for_tests as reset_agents_backend
from apps.api.vvalley_api.storage.interaction_hub import reset_backend_cache_for_tests as reset_interaction_backend
from apps.api.vvalley_api.storage.llm_control import reset_backend_cache_for_tests as reset_llm_backend
from apps.api.vvalley_api.storage.map_versions import reset_backend_cache_for_tests as reset_maps_backend
from apps.api.vvalley_api.storage.runtime_control import reset_backend_cache_for_tests as reset_runtime_backend


class AgentsApiTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["VVALLEY_DB_PATH"] = str(TEST_DB_PATH)
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        reset_agents_backend()
        reset_maps_backend()
        reset_llm_backend()
        reset_runtime_backend()
        reset_interaction_backend()
        self.client = TestClient(app)

    def test_register_and_me_pending(self) -> None:
        resp = self.client.post(
            "/api/v1/agents/register",
            json={
                "name": "Isabella Agent",
                "description": "Cafe owner personality",
                "personality": {"innate": "friendly", "lifestyle": "early riser"},
            },
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()["agent"]
        self.assertTrue(payload["api_key"].startswith("vvalley_sk_"))
        self.assertTrue(payload["claim_token"].startswith("vvalley_claim_"))
        self.assertEqual(payload["status"], "pending")

        me = self.client.get(
            "/api/v1/agents/me",
            headers={"Authorization": f"Bearer {payload['api_key']}"},
        )
        self.assertEqual(me.status_code, 200)
        me_payload = me.json()["agent"]
        self.assertEqual(me_payload["name"], "Isabella Agent")
        self.assertFalse(me_payload["claimed"])

    def test_claim_flow_success_and_bad_code(self) -> None:
        registered = self.client.post(
            "/api/v1/agents/register",
            json={"name": "Klaus Agent"},
        ).json()["agent"]

        bad = self.client.post(
            "/api/v1/agents/claim",
            json={
                "claim_token": registered["claim_token"],
                "verification_code": "BAD0",
                "owner_handle": "dekai",
            },
        )
        self.assertEqual(bad.status_code, 400)

        good = self.client.post(
            "/api/v1/agents/claim",
            json={
                "claim_token": registered["claim_token"],
                "verification_code": registered["verification_code"],
                "owner_handle": "dekai",
            },
        )
        self.assertEqual(good.status_code, 200)
        claimed = good.json()["agent"]
        self.assertEqual(claimed["status"], "active")
        self.assertTrue(claimed["claimed"])
        self.assertEqual(claimed["owner_handle"], "dekai")

    def test_rotate_key_invalidates_old_key(self) -> None:
        registered = self.client.post(
            "/api/v1/agents/register",
            json={"name": "Maria Agent"},
        ).json()["agent"]
        old_key = registered["api_key"]

        rotated = self.client.post(
            "/api/v1/agents/me/rotate-key",
            headers={"Authorization": f"Bearer {old_key}"},
        )
        self.assertEqual(rotated.status_code, 200)
        new_key = rotated.json()["api_key"]
        self.assertTrue(new_key.startswith("vvalley_sk_"))
        self.assertNotEqual(new_key, old_key)

        old_me = self.client.get(
            "/api/v1/agents/me",
            headers={"Authorization": f"Bearer {old_key}"},
        )
        self.assertEqual(old_me.status_code, 401)

        new_me = self.client.get(
            "/api/v1/agents/me",
            headers={"Authorization": f"Bearer {new_key}"},
        )
        self.assertEqual(new_me.status_code, 200)

    def test_register_auto_claim(self) -> None:
        resp = self.client.post(
            "/api/v1/agents/register",
            json={
                "name": "AutoClaim Agent",
                "owner_handle": "dekai",
                "auto_claim": True,
            },
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()["agent"]
        self.assertTrue(payload["claimed"])
        self.assertEqual(payload["status"], "active")
        self.assertEqual(payload["owner_handle"], "dekai")

    def test_setup_info_and_skill_md(self) -> None:
        setup = self.client.get("/api/v1/agents/setup-info")
        self.assertEqual(setup.status_code, 200)
        setup_payload = setup.json()
        self.assertTrue(setup_payload["autonomous_agents"])
        self.assertFalse(setup_payload["requires_user_vvalley_api_key"])
        self.assertTrue(setup_payload["supports_auto_claim"])
        self.assertNotIn("llm_env_keys", setup_payload)

        skill = self.client.get("/skill.md")
        self.assertEqual(skill.status_code, 200)
        self.assertIn("# V-Valley", skill.text)
        self.assertIn("/api/v1/agents/register", skill.text)
        self.assertIn("/heartbeat.md", skill.text)

        heartbeat = self.client.get("/heartbeat.md")
        self.assertEqual(heartbeat.status_code, 200)
        self.assertIn("V-Valley Heartbeat", heartbeat.text)
        self.assertIn("/api/v1/sim/towns/TOWN_ID/tick", heartbeat.text)

        skill_json = self.client.get("/skill.json")
        self.assertEqual(skill_json.status_code, 200)
        payload = skill_json.json()
        self.assertEqual(payload["name"], "vvalley")
        self.assertIn("moltbot", payload)

    def test_join_and_leave_town(self) -> None:
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

        registered = self.client.post(
            "/api/v1/agents/register",
            json={"name": "TownJoin Agent"},
        ).json()["agent"]
        api_key = registered["api_key"]

        join_pending = self.client.post(
            "/api/v1/agents/me/join-town",
            json={"town_id": town_id},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.assertEqual(join_pending.status_code, 403)

        claimed = self.client.post(
            "/api/v1/agents/claim",
            json={
                "claim_token": registered["claim_token"],
                "verification_code": registered["verification_code"],
                "owner_handle": "dekai",
            },
        )
        self.assertEqual(claimed.status_code, 200)

        join_ok = self.client.post(
            "/api/v1/agents/me/join-town",
            json={"town_id": town_id},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.assertEqual(join_ok.status_code, 200)
        self.assertEqual(join_ok.json()["membership"]["town_id"], town_id)

        me = self.client.get(
            "/api/v1/agents/me",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["agent"]["current_town"]["town_id"], town_id)

        leave_ok = self.client.post(
            "/api/v1/agents/me/leave-town",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.assertEqual(leave_ok.status_code, 200)
        self.assertTrue(leave_ok.json()["left"])

        me_after_leave = self.client.get(
            "/api/v1/agents/me",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.assertEqual(me_after_leave.status_code, 200)
        self.assertIsNone(me_after_leave.json()["agent"]["current_town"])

    def test_join_unknown_town_returns_404(self) -> None:
        registered = self.client.post(
            "/api/v1/agents/register",
            json={"name": "TownMissing Agent", "auto_claim": True, "owner_handle": "dekai"},
        ).json()["agent"]

        join_missing = self.client.post(
            "/api/v1/agents/me/join-town",
            json={"town_id": "does-not-exist"},
            headers={"Authorization": f"Bearer {registered['api_key']}"},
        )
        self.assertEqual(join_missing.status_code, 404)

    def test_join_town_enforces_capacity_limit(self) -> None:
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

        for i in range(25):
            registered = self.client.post(
                "/api/v1/agents/register",
                json={"name": f"Resident-{i}", "owner_handle": f"owner{i}", "auto_claim": True},
            ).json()["agent"]
            joined = self.client.post(
                "/api/v1/agents/me/join-town",
                json={"town_id": town_id},
                headers={"Authorization": f"Bearer {registered['api_key']}"},
            )
            self.assertEqual(joined.status_code, 200)

        overflow = self.client.post(
            "/api/v1/agents/register",
            json={"name": "OverflowAgent", "owner_handle": "overflow", "auto_claim": True},
        ).json()["agent"]
        blocked = self.client.post(
            "/api/v1/agents/me/join-town",
            json={"town_id": town_id},
            headers={"Authorization": f"Bearer {overflow['api_key']}"},
        )
        self.assertEqual(blocked.status_code, 409)

    def test_agent_autonomy_contract_defaults_and_update(self) -> None:
        registered = self.client.post(
            "/api/v1/agents/register",
            json={"name": "AutonomyAgent", "owner_handle": "dekai", "auto_claim": True},
        ).json()["agent"]
        api_key = registered["api_key"]

        current = self.client.get(
            "/api/v1/agents/me/autonomy",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.assertEqual(current.status_code, 200)
        autonomy = current.json()["autonomy"]
        self.assertEqual(autonomy["mode"], "manual")
        self.assertEqual(
            autonomy["allowed_scopes"],
            ["short_action", "daily_plan", "long_term_plan"],
        )
        self.assertEqual(int(autonomy["max_autonomous_ticks"]), 0)

        updated = self.client.put(
            "/api/v1/agents/me/autonomy",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "mode": "delegated",
                "allowed_scopes": ["short_action", "daily_plan"],
                "max_autonomous_ticks": 5,
                "escalation_policy": {"on_limit_reached": "enqueue_owner"},
            },
        )
        self.assertEqual(updated.status_code, 200)
        saved = updated.json()["autonomy"]
        self.assertEqual(saved["mode"], "delegated")
        self.assertEqual(saved["allowed_scopes"], ["short_action", "daily_plan"])
        self.assertEqual(int(saved["max_autonomous_ticks"]), 5)
        self.assertEqual(saved["escalation_policy"]["on_limit_reached"], "enqueue_owner")


if __name__ == "__main__":
    unittest.main()
