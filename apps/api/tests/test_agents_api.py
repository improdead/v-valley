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

    def test_list_characters_endpoint(self) -> None:
        resp = self.client.get("/api/v1/agents/characters")
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["count"], 25)
        self.assertEqual(len(payload["characters"]), 25)
        self.assertIn("Isabella_Rodriguez", payload["characters"])
        self.assertIn("Klaus_Mueller", payload["characters"])

    def test_register_with_explicit_sprite(self) -> None:
        resp = self.client.post(
            "/api/v1/agents/register",
            json={
                "name": "SpriteTest Agent",
                "sprite_name": "Isabella_Rodriguez",
            },
        )
        self.assertEqual(resp.status_code, 200)
        agent = resp.json()["agent"]
        self.assertEqual(agent["sprite_name"], "Isabella_Rodriguez")

    def test_register_assigns_random_sprite_when_none(self) -> None:
        resp = self.client.post(
            "/api/v1/agents/register",
            json={"name": "NoSprite Agent"},
        )
        self.assertEqual(resp.status_code, 200)
        agent = resp.json()["agent"]
        self.assertIsNotNone(agent["sprite_name"])
        self.assertIn(agent["sprite_name"], [
            "Abigail_Chen", "Adam_Smith", "Arthur_Burton", "Ayesha_Khan",
            "Carlos_Gomez", "Carmen_Ortiz", "Eddy_Lin", "Francisco_Lopez",
            "Giorgio_Rossi", "Hailey_Johnson", "Isabella_Rodriguez", "Jane_Moreno",
            "Jennifer_Moore", "John_Lin", "Klaus_Mueller", "Latoya_Williams",
            "Maria_Lopez", "Mei_Lin", "Rajiv_Patel", "Ryan_Park",
            "Sam_Moore", "Tamara_Taylor", "Tom_Moreno", "Wolfgang_Schulz",
            "Yuriko_Yamamoto",
        ])

    def test_register_empty_name_gets_default(self) -> None:
        resp = self.client.post(
            "/api/v1/agents/register",
            json={"name": ""},
        )
        self.assertEqual(resp.status_code, 200)
        agent = resp.json()["agent"]
        # Should get fallback name "Agent"
        self.assertEqual(agent["name"], "Agent")

    def test_persona_stored_as_personality(self) -> None:
        """Register with explicit personality, verify it flows to NpcState via sim state."""
        persona = {
            "first_name": "Isabella",
            "last_name": "Rodriguez",
            "age": 34,
            "innate": "friendly, outgoing, hospitable",
            "learned": "Isabella Rodriguez is a cafe owner who loves her community.",
            "lifestyle": "goes to bed around 11pm, wakes at 6am",
            "daily_plan_req": "1. Wake up. 2. Open cafe. 3. Serve customers.",
            "daily_req": ["wake up at 6am", "open cafe at 8am", "serve customers"],
        }
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self.client.post(
            "/api/v1/maps/publish-version",
            json={
                "town_id": town_id,
                "map_path": "assets/templates/starter_town/map.json",
                "map_name": "starter",
            },
        )

        registered = self.client.post(
            "/api/v1/agents/register",
            json={
                "name": "Isabella Rodriguez",
                "personality": persona,
                "sprite_name": "Isabella_Rodriguez",
                "owner_handle": "dekai",
                "auto_claim": True,
            },
        ).json()["agent"]
        api_key = registered["api_key"]

        self.client.post(
            "/api/v1/agents/me/join-town",
            json={"town_id": town_id},
            headers={"Authorization": f"Bearer {api_key}"},
        )

        state = self.client.get(f"/api/v1/sim/towns/{town_id}/state")
        self.assertEqual(state.status_code, 200)
        npcs = state.json()["state"]["npcs"]
        self.assertTrue(len(npcs) >= 1)
        isabella = [n for n in npcs if n["name"] == "Isabella Rodriguez"]
        self.assertEqual(len(isabella), 1)
        npc = isabella[0]
        self.assertEqual(npc["sprite_name"], "Isabella_Rodriguez")
        self.assertIsNotNone(npc["persona"])
        self.assertEqual(npc["persona"]["innate"], "friendly, outgoing, hospitable")
        self.assertEqual(npc["persona"]["age"], 34)


class PersonaBootstrapTests(unittest.TestCase):
    """Unit tests for persona-aware memory bootstrap."""

    def test_bootstrap_without_persona(self) -> None:
        from packages.vvalley_core.sim.memory import AgentMemory

        mem = AgentMemory.bootstrap(agent_id="a1", agent_name="Test Agent", step=0)
        self.assertEqual(mem.agent_name, "Test Agent")
        # Should have generic join thought
        self.assertTrue(len(mem.seq_event) > 0 or len(mem.seq_thought) > 0)

    def test_bootstrap_with_persona(self) -> None:
        from packages.vvalley_core.sim.memory import AgentMemory

        persona = {
            "first_name": "Klaus",
            "last_name": "Mueller",
            "age": 20,
            "innate": "curious, analytical, studious",
            "learned": "Klaus Mueller is a student at Oak Hill College studying chemistry.",
            "lifestyle": "goes to bed around midnight, wakes at 8am",
            "daily_req": ["wake up at 8am", "go to class", "study at library"],
        }
        mem = AgentMemory.bootstrap(
            agent_id="k1", agent_name="Klaus Mueller", step=0, persona=persona
        )

        # Should have personalized daily_req
        self.assertEqual(mem.scratch.daily_req, ["wake up at 8am", "go to class", "study at library"])

        # Should have backstory in long_term_goals
        self.assertTrue(any("chemistry" in g for g in mem.scratch.long_term_goals))

        # Should have at least 2 thoughts (identity + backstory)
        self.assertTrue(len(mem.seq_thought) >= 2)

        # Check identity thought content
        all_descs = [mem.id_to_node[nid].description for nid in mem.seq_thought]
        identity_found = any("curious" in d and "Klaus Mueller" in d for d in all_descs)
        backstory_found = any("chemistry" in d for d in all_descs)
        self.assertTrue(identity_found, f"Identity thought not found in {all_descs}")
        self.assertTrue(backstory_found, f"Backstory thought not found in {all_descs}")


class ConstraintTests(unittest.TestCase):
    """Tests for one-agent-per-owner and sprite uniqueness constraints."""

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

    def test_one_agent_per_owner(self) -> None:
        resp1 = self.client.post(
            "/api/v1/agents/register",
            json={"name": "First Agent", "owner_handle": "alice", "auto_claim": True},
        )
        self.assertEqual(resp1.status_code, 200)

        resp2 = self.client.post(
            "/api/v1/agents/register",
            json={"name": "Second Agent", "owner_handle": "alice", "auto_claim": True},
        )
        self.assertEqual(resp2.status_code, 409)
        self.assertIn("already has a registered agent", resp2.json()["detail"])

    def test_dekai_can_register_multiple_agents(self) -> None:
        resp1 = self.client.post(
            "/api/v1/agents/register",
            json={"name": "Test Agent 1", "owner_handle": "dekai", "auto_claim": True},
        )
        self.assertEqual(resp1.status_code, 200)

        resp2 = self.client.post(
            "/api/v1/agents/register",
            json={"name": "Test Agent 2", "owner_handle": "dekai", "auto_claim": True},
        )
        self.assertEqual(resp2.status_code, 200)

    def test_sprite_uniqueness_in_town(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self.client.post(
            "/api/v1/maps/publish-version",
            json={
                "town_id": town_id,
                "map_path": "assets/templates/starter_town/map.json",
                "map_name": "starter",
            },
        )

        a1 = self.client.post(
            "/api/v1/agents/register",
            json={"name": "Agent A", "sprite_name": "Klaus_Mueller", "owner_handle": "dekai", "auto_claim": True},
        ).json()["agent"]
        a2 = self.client.post(
            "/api/v1/agents/register",
            json={"name": "Agent B", "sprite_name": "Klaus_Mueller", "owner_handle": "dekai", "auto_claim": True},
        ).json()["agent"]

        j1 = self.client.post(
            "/api/v1/agents/me/join-town",
            json={"town_id": town_id},
            headers={"Authorization": f"Bearer {a1['api_key']}"},
        )
        self.assertEqual(j1.status_code, 200)

        j2 = self.client.post(
            "/api/v1/agents/me/join-town",
            json={"town_id": town_id},
            headers={"Authorization": f"Bearer {a2['api_key']}"},
        )
        self.assertEqual(j2.status_code, 200)

        state = self.client.get(f"/api/v1/sim/towns/{town_id}/state")
        npcs = state.json()["state"]["npcs"]
        sprites = [n["sprite_name"] for n in npcs]
        self.assertEqual(len(set(sprites)), 2, f"Expected 2 unique sprites, got {sprites}")


if __name__ == "__main__":
    unittest.main()
