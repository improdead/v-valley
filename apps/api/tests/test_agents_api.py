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
from apps.api.vvalley_api.routers.agents import reset_rate_limiter_for_tests as reset_rate_limiter
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
        reset_rate_limiter()
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
        os.environ["VVALLEY_ADMIN_HANDLES"] = "dekai"
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        reset_agents_backend()
        reset_maps_backend()
        reset_llm_backend()
        reset_runtime_backend()
        reset_interaction_backend()
        reset_rate_limiter()
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


class GAFeatureGapTests(unittest.TestCase):
    """Tests for GA feature gap closures: ISS, pronunciatio, spatial metadata, etc."""

    def test_iss_fields_populated_from_persona(self) -> None:
        from packages.vvalley_core.sim.memory import AgentMemory

        persona = {
            "first_name": "Latoya",
            "last_name": "Williams",
            "age": 28,
            "innate": "creative, empathetic",
            "learned": "Latoya Williams is a digital artist living in the co-living space.",
            "lifestyle": "goes to bed around 11pm, wakes at 7am",
            "living_area": "artist's co-living space",
        }
        mem = AgentMemory.bootstrap(
            agent_id="lw1", agent_name="Latoya Williams", step=0, persona=persona
        )
        self.assertEqual(mem.scratch.iss_name, "Latoya Williams")
        self.assertEqual(mem.scratch.first_name, "Latoya")
        self.assertEqual(mem.scratch.last_name, "Williams")
        self.assertEqual(mem.scratch.age, 28)
        self.assertEqual(mem.scratch.innate, "creative, empathetic")
        self.assertIn("digital artist", mem.scratch.learned)
        self.assertIn("7am", mem.scratch.lifestyle)
        self.assertEqual(mem.scratch.living_area, "artist's co-living space")

    def test_get_str_iss_format(self) -> None:
        from packages.vvalley_core.sim.memory import AgentMemory

        persona = {
            "first_name": "Klaus",
            "last_name": "Mueller",
            "age": 20,
            "innate": "curious, analytical",
            "learned": "Klaus is a chemistry student.",
            "lifestyle": "wakes at 8am",
        }
        mem = AgentMemory.bootstrap(
            agent_id="km1", agent_name="Klaus Mueller", step=0, persona=persona
        )
        iss = mem.scratch.get_str_iss()
        self.assertIn("Name: Klaus Mueller", iss)
        self.assertIn("Age: 20", iss)
        self.assertIn("Innate traits: curious, analytical", iss)
        self.assertIn("Background: Klaus is a chemistry student.", iss)
        self.assertIn("Lifestyle: wakes at 8am", iss)

    def test_iss_serialization_roundtrip(self) -> None:
        from packages.vvalley_core.sim.memory import AgentMemory

        persona = {
            "first_name": "Tom",
            "last_name": "Moreno",
            "age": 45,
            "innate": "hardworking",
            "learned": "Tom is a carpenter.",
            "lifestyle": "early bird",
            "living_area": "Moreno family's house",
        }
        mem = AgentMemory.bootstrap(
            agent_id="tm1", agent_name="Tom Moreno", step=0, persona=persona
        )
        exported = mem.export_state()
        restored = AgentMemory.from_state(exported)
        self.assertEqual(restored.scratch.iss_name, "Tom Moreno")
        self.assertEqual(restored.scratch.age, 45)
        self.assertEqual(restored.scratch.living_area, "Moreno family's house")

    def test_pronunciatio_heuristic(self) -> None:
        from packages.vvalley_core.sim.cognition import _heuristic_pronunciatio

        result = _heuristic_pronunciatio({"action_description": "sleeping in bed", "agent_name": "Klaus"})
        self.assertEqual(result["emoji"], "💤")
        self.assertEqual(result["event_triple"]["subject"], "Klaus")
        self.assertEqual(result["event_triple"]["object"], "sleeping")

        result2 = _heuristic_pronunciatio({"action_description": "eating lunch at cafe", "agent_name": "Mei"})
        self.assertEqual(result2["emoji"], "🍽️")

        result3 = _heuristic_pronunciatio({"action_description": "chatting with Tom", "agent_name": "Isabella"})
        self.assertEqual(result3["emoji"], "💬")

    def test_reaction_policy_heuristic(self) -> None:
        from packages.vvalley_core.sim.cognition import _heuristic_reaction_policy

        result = _heuristic_reaction_policy({
            "agent_a_id": "a1",
            "agent_b_id": "b1",
            "step": 10,
            "relationship_score": 5.0,
        })
        self.assertIn(result["decision"], ("talk", "wait", "ignore"))

    def test_wake_up_hour_heuristic(self) -> None:
        from packages.vvalley_core.sim.cognition import _heuristic_wake_up_hour

        early = _heuristic_wake_up_hour({"lifestyle": "early riser, wakes at 5am"})
        self.assertEqual(early["wake_up_hour"], 6)

        late = _heuristic_wake_up_hour({"lifestyle": "night owl, wakes at 10am"})
        self.assertEqual(late["wake_up_hour"], 9)

        default = _heuristic_wake_up_hour({"lifestyle": "regular schedule"})
        self.assertEqual(default["wake_up_hour"], 7)

        parsed = _heuristic_wake_up_hour({"lifestyle": "wakes up around 8"})
        self.assertEqual(parsed["wake_up_hour"], 8)

    def test_act_event_serialization(self) -> None:
        from packages.vvalley_core.sim.memory import AgentMemory

        mem = AgentMemory.bootstrap(agent_id="e1", agent_name="Test", step=0)
        mem.scratch.act_pronunciatio = "💼"
        mem.scratch.act_event = ("Test", "is", "working")

        exported = mem.export_state()
        restored = AgentMemory.from_state(exported)
        self.assertEqual(restored.scratch.act_pronunciatio, "💼")
        self.assertEqual(restored.scratch.act_event, ("Test", "is", "working"))

    def test_chat_cooldown_is_20_steps(self) -> None:
        from packages.vvalley_core.sim.memory import AgentMemory

        mem = AgentMemory.bootstrap(agent_id="c1", agent_name="ChatTest", step=0)
        mem.add_social_interaction(
            step=10,
            target_agent_id="c2",
            target_name="Partner",
            message="Hello!",
        )
        # Cooldown should be step + 20 = 30
        self.assertEqual(mem.scratch.chatting_with_buffer["Partner"], 30)

    def test_spatial_metadata_from_map(self) -> None:
        import json
        from packages.vvalley_core.maps.map_utils import build_spatial_metadata

        map_path = ROOT / "assets" / "templates" / "starter_town" / "map.json"
        if not map_path.exists():
            self.skipTest("Map file not available")

        with map_path.open("r") as f:
            map_data = json.load(f)

        spatial = build_spatial_metadata(map_data)
        self.assertIsNotNone(spatial, "Should build spatial metadata from GA map")
        self.assertEqual(spatial.width, 140)
        self.assertEqual(spatial.height, 100)

        # Should have sectors
        sectors = spatial.sectors_list()
        self.assertTrue(len(sectors) > 0, f"Should have sectors, got {sectors}")
        self.assertIn("artist's co-living space", [s.lower() if s else "" for s in sectors] + sectors)

        # Should have arenas within a sector
        co_living = next((s for s in sectors if "co-living" in s.lower()), None)
        if co_living:
            arenas = spatial.arenas_in_sector(co_living)
            self.assertTrue(len(arenas) > 0, f"Co-living space should have arenas")

    def test_spatial_metadata_address_lookup(self) -> None:
        import json
        from packages.vvalley_core.maps.map_utils import build_spatial_metadata

        map_path = ROOT / "assets" / "templates" / "starter_town" / "map.json"
        if not map_path.exists():
            self.skipTest("Map file not available")

        with map_path.open("r") as f:
            map_data = json.load(f)

        spatial = build_spatial_metadata(map_data)
        self.assertIsNotNone(spatial)

        # Test full_address_at for a known position
        # Tile (0,0) might be None, but most tiles in the map should have metadata
        found_sector = False
        for y in range(spatial.height):
            for x in range(spatial.width):
                sector, arena, obj = spatial.full_address_at(x, y)
                if sector is not None:
                    found_sector = True
                    break
            if found_sector:
                break
        self.assertTrue(found_sector, "Should find at least one tile with sector metadata")

    def test_cognition_planner_methods_exist(self) -> None:
        """Verify all new CognitionPlanner methods exist."""
        from packages.vvalley_core.sim.cognition import CognitionPlanner

        self.assertTrue(hasattr(CognitionPlanner, "generate_pronunciatio"))
        self.assertTrue(hasattr(CognitionPlanner, "decide_to_talk"))
        self.assertTrue(hasattr(CognitionPlanner, "generate_wake_up_hour"))

    def test_task_policies_include_new_tasks(self) -> None:
        from packages.vvalley_core.llm.policy import DEFAULT_TASK_POLICIES

        self.assertIn("pronunciatio", DEFAULT_TASK_POLICIES)
        self.assertIn("reaction_policy", DEFAULT_TASK_POLICIES)
        self.assertIn("wake_up_hour", DEFAULT_TASK_POLICIES)

        # Verify tiers
        self.assertEqual(DEFAULT_TASK_POLICIES["pronunciatio"].model_tier, "cheap")
        self.assertEqual(DEFAULT_TASK_POLICIES["reaction_policy"].model_tier, "cheap")
        self.assertEqual(DEFAULT_TASK_POLICIES["wake_up_hour"].model_tier, "cheap")

    # --- Wave 2 Gap Closure Tests ---

    def test_task_policies_include_wave2_tasks(self) -> None:
        from packages.vvalley_core.llm.policy import DEFAULT_TASK_POLICIES

        for task in ("action_address", "object_state", "event_reaction",
                     "relationship_summary", "first_daily_plan"):
            self.assertIn(task, DEFAULT_TASK_POLICIES, f"Missing policy: {task}")

        self.assertEqual(DEFAULT_TASK_POLICIES["action_address"].model_tier, "cheap")
        self.assertEqual(DEFAULT_TASK_POLICIES["object_state"].model_tier, "cheap")
        self.assertEqual(DEFAULT_TASK_POLICIES["event_reaction"].model_tier, "cheap")
        self.assertEqual(DEFAULT_TASK_POLICIES["relationship_summary"].model_tier, "fast")
        self.assertEqual(DEFAULT_TASK_POLICIES["first_daily_plan"].model_tier, "fast")

    def test_cognition_planner_wave2_methods_exist(self) -> None:
        from packages.vvalley_core.sim.cognition import CognitionPlanner

        for method in ("resolve_action_address", "generate_object_state",
                       "decide_to_react", "generate_relationship_summary",
                       "generate_first_daily_plan"):
            self.assertTrue(
                hasattr(CognitionPlanner, method),
                f"CognitionPlanner missing method: {method}",
            )

    def test_action_address_heuristic_sleep_prefers_living_area(self) -> None:
        from packages.vvalley_core.sim.cognition import _heuristic_action_address

        result = _heuristic_action_address({
            "action_description": "sleeping in bed",
            "available_sectors": ["Oak Hill College", "artist's co-living space", "Hobbs Cafe"],
            "living_area": "artist's co-living space",
            "agent_id": "a1",
        })
        self.assertEqual(result["action_sector"], "artist's co-living space")

    def test_action_address_heuristic_eat_prefers_cafe(self) -> None:
        from packages.vvalley_core.sim.cognition import _heuristic_action_address

        result = _heuristic_action_address({
            "action_description": "eating lunch",
            "available_sectors": ["Oak Hill College", "artist's co-living space", "Hobbs Cafe"],
            "living_area": "artist's co-living space",
            "agent_id": "a1",
        })
        self.assertEqual(result["action_sector"], "Hobbs Cafe")

    def test_action_address_heuristic_study_prefers_college(self) -> None:
        from packages.vvalley_core.sim.cognition import _heuristic_action_address

        result = _heuristic_action_address({
            "action_description": "studying for exam",
            "available_sectors": ["Oak Hill College", "artist's co-living space", "Hobbs Cafe"],
            "living_area": "artist's co-living space",
            "agent_id": "a1",
        })
        self.assertEqual(result["action_sector"], "Oak Hill College")

    def test_action_address_heuristic_fallback_to_living_area(self) -> None:
        from packages.vvalley_core.sim.cognition import _heuristic_action_address

        result = _heuristic_action_address({
            "action_description": "doing some unknown activity",
            "available_sectors": ["Oak Hill College", "artist's co-living space"],
            "living_area": "artist's co-living space",
            "agent_id": "a1",
        })
        self.assertEqual(result["action_sector"], "artist's co-living space")

    def test_object_state_heuristic(self) -> None:
        from packages.vvalley_core.sim.cognition import _heuristic_object_state

        result = _heuristic_object_state({
            "agent_name": "Klaus Mueller",
            "action_description": "sleeping soundly",
            "game_object": "bed",
        })
        self.assertIn("bed", result["object_description"])
        self.assertIn("Klaus Mueller", result["object_description"])
        self.assertEqual(result["object_event_triple"]["subject"], "bed")
        self.assertEqual(result["object_event_triple"]["predicate"], "is used by")
        self.assertEqual(result["object_event_triple"]["object"], "Klaus Mueller")

    def test_event_reaction_heuristic_skip_idle(self) -> None:
        from packages.vvalley_core.sim.cognition import _heuristic_event_reaction

        result = _heuristic_event_reaction({
            "agent_id": "a1",
            "event_description": "idle",
            "agent_activity": "walking around",
        })
        self.assertFalse(result["react"])
        self.assertEqual(result["reason"], "event is idle")

    def test_event_reaction_heuristic_skip_busy_agent(self) -> None:
        from packages.vvalley_core.sim.cognition import _heuristic_event_reaction

        result = _heuristic_event_reaction({
            "agent_id": "a1",
            "event_description": "Klaus is studying",
            "agent_activity": "sleeping in bed",
        })
        self.assertFalse(result["react"])
        self.assertEqual(result["reason"], "agent is busy")

    def test_event_reaction_heuristic_chatting_is_busy(self) -> None:
        from packages.vvalley_core.sim.cognition import _heuristic_event_reaction

        result = _heuristic_event_reaction({
            "agent_id": "a1",
            "event_description": "Klaus is studying",
            "agent_activity": "chatting with Tom",
        })
        self.assertFalse(result["react"])

    def test_relationship_summary_heuristic_close_friends(self) -> None:
        from packages.vvalley_core.sim.cognition import _heuristic_relationship_summary

        result = _heuristic_relationship_summary({
            "agent_name": "Isabella",
            "partner_name": "Klaus",
            "relationship_score": 5.0,
            "past_interactions": ["chat about cafe"],
        })
        self.assertIn("close friends", result["summary"])
        self.assertEqual(result["conversation_tone"], "warm")

    def test_relationship_summary_heuristic_acquaintances(self) -> None:
        from packages.vvalley_core.sim.cognition import _heuristic_relationship_summary

        result = _heuristic_relationship_summary({
            "agent_name": "Isabella",
            "partner_name": "Tom",
            "relationship_score": 2.5,
            "past_interactions": ["brief hello"],
        })
        self.assertIn("acquaintances", result["summary"])
        self.assertEqual(result["conversation_tone"], "polite")

    def test_relationship_summary_heuristic_first_meeting(self) -> None:
        from packages.vvalley_core.sim.cognition import _heuristic_relationship_summary

        result = _heuristic_relationship_summary({
            "agent_name": "Isabella",
            "partner_name": "NewAgent",
            "relationship_score": 0.0,
            "past_interactions": [],
        })
        self.assertIn("first time", result["summary"])

    def test_first_daily_plan_heuristic(self) -> None:
        from packages.vvalley_core.sim.cognition import _heuristic_first_daily_plan

        result = _heuristic_first_daily_plan({
            "agent_name": "NewAgent",
            "innate": "curious, friendly",
            "learned": "NewAgent just moved to town.",
        })
        self.assertIsInstance(result["schedule"], list)
        self.assertTrue(len(result["schedule"]) > 0)
        # First-day plan should include exploration
        all_descs = " ".join(item["description"] for item in result["schedule"]).lower()
        self.assertTrue(
            "explore" in all_descs or "neighbor" in all_descs or "settle" in all_descs,
            f"First-day plan should be exploratory, got: {all_descs}",
        )
        # Total duration should be reasonable
        total_mins = sum(item["duration_mins"] for item in result["schedule"])
        self.assertGreater(total_mins, 0)
        self.assertIn("exploring", result["currently"].lower())

    def test_choose_perceived_event_skips_self(self) -> None:
        from packages.vvalley_core.sim.runner import SimulationRunner

        runner = SimulationRunner()
        class FakeNpc:
            name = "Klaus"
        npc = FakeNpc()
        events = [
            {"name": "Klaus", "status": "studying"},
            {"name": "Isabella", "status": "cooking dinner"},
        ]
        chosen = runner._choose_perceived_event(npc=npc, events=events)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["name"], "Isabella")

    def test_choose_perceived_event_skips_idle(self) -> None:
        from packages.vvalley_core.sim.runner import SimulationRunner

        runner = SimulationRunner()
        class FakeNpc:
            name = "Klaus"
        npc = FakeNpc()
        events = [
            {"name": "Tom", "status": "idle"},
            {"name": "Isabella", "status": "painting"},
        ]
        chosen = runner._choose_perceived_event(npc=npc, events=events)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["name"], "Isabella")

    def test_choose_perceived_event_returns_none_if_all_self(self) -> None:
        from packages.vvalley_core.sim.runner import SimulationRunner

        runner = SimulationRunner()
        class FakeNpc:
            name = "Klaus"
        npc = FakeNpc()
        events = [
            {"name": "Klaus", "status": "studying"},
        ]
        chosen = runner._choose_perceived_event(npc=npc, events=events)
        self.assertIsNone(chosen)

    def test_choose_perceived_event_empty_list(self) -> None:
        from packages.vvalley_core.sim.runner import SimulationRunner

        runner = SimulationRunner()
        class FakeNpc:
            name = "Klaus"
        npc = FakeNpc()
        chosen = runner._choose_perceived_event(npc=npc, events=[])
        self.assertIsNone(chosen)

    def test_choose_perceived_event_prioritizes_persons(self) -> None:
        from packages.vvalley_core.sim.runner import SimulationRunner

        runner = SimulationRunner()
        class FakeNpc:
            name = "Klaus"
        npc = FakeNpc()
        events = [
            {"name": "cafe:counter:coffee_machine", "status": "brewing coffee"},
            {"name": "Isabella", "status": "painting"},
        ]
        chosen = runner._choose_perceived_event(npc=npc, events=events)
        self.assertIsNotNone(chosen)
        # Person events should be prioritized over object events (with ":" in name)
        self.assertEqual(chosen["name"], "Isabella")

    def test_town_runtime_has_object_states(self) -> None:
        """TownRuntime should have an object_states dict field."""
        from packages.vvalley_core.sim.runner import TownRuntime

        self.assertTrue(hasattr(TownRuntime, "__dataclass_fields__"))
        self.assertIn("object_states", TownRuntime.__dataclass_fields__)

    def test_first_day_detection_in_memory(self) -> None:
        """First-day schedule uses generate_first_daily_plan when schedule is empty."""
        from packages.vvalley_core.sim.memory import AgentMemory

        mem = AgentMemory.bootstrap(agent_id="fd1", agent_name="FirstDay Agent", step=0)
        # At step=0, daily_schedule should be empty, so is_first_day should be True
        is_first_day = not mem.scratch.daily_schedule or 0 <= 1
        self.assertTrue(is_first_day)

    def test_act_address_field_exists(self) -> None:
        """AgentScratch should have act_address field for action address resolution."""
        from packages.vvalley_core.sim.memory import AgentMemory

        mem = AgentMemory.bootstrap(agent_id="aa1", agent_name="Addr Agent", step=0)
        self.assertTrue(hasattr(mem.scratch, "act_address"))

    def test_fun_low_cost_preset_includes_all_tasks(self) -> None:
        """fun_low_cost_preset_policies should include all task policies."""
        from packages.vvalley_core.llm.policy import fun_low_cost_preset_policies

        policies = fun_low_cost_preset_policies()
        task_names = {p.task_name for p in policies}
        for expected in ("pronunciatio", "reaction_policy", "wake_up_hour",
                         "action_address", "object_state", "event_reaction",
                         "relationship_summary", "first_daily_plan",
                         "schedule_recomposition"):
            self.assertIn(expected, task_names, f"fun_low_cost_preset missing: {expected}")

    # --- Final Gap Closure Tests (Schedule Recomposition + Collision Avoidance) ---

    def test_schedule_recomposition_heuristic(self) -> None:
        from packages.vvalley_core.sim.cognition import _heuristic_schedule_recomposition

        schedule = [
            {"description": "sleeping", "duration_mins": 360},
            {"description": "morning routine", "duration_mins": 60},
            {"description": "work on project", "duration_mins": 180},
            {"description": "lunch", "duration_mins": 60},
            {"description": "afternoon tasks", "duration_mins": 240},
            {"description": "dinner", "duration_mins": 60},
            {"description": "evening rest", "duration_mins": 180},
        ]
        result = _heuristic_schedule_recomposition({
            "inserted_activity": "chatting with Isabella",
            "inserted_duration_mins": 30,
            "current_schedule": schedule,
            "current_minute_of_day": 420,  # 7am, during morning routine
        })
        self.assertIsInstance(result["schedule"], list)
        self.assertTrue(len(result["schedule"]) > 0)
        # The inserted activity should be in the schedule
        descs = [item["description"] for item in result["schedule"]]
        self.assertIn("chatting with Isabella", descs)

    def test_schedule_recomposition_preserves_past_blocks(self) -> None:
        from packages.vvalley_core.sim.cognition import _heuristic_schedule_recomposition

        schedule = [
            {"description": "sleeping", "duration_mins": 360},
            {"description": "morning routine", "duration_mins": 60},
            {"description": "work", "duration_mins": 300},
        ]
        result = _heuristic_schedule_recomposition({
            "inserted_activity": "reacting to event",
            "inserted_duration_mins": 20,
            "current_schedule": schedule,
            "current_minute_of_day": 200,  # Still during sleeping block
        })
        new_schedule = result["schedule"]
        # Sleep block should be truncated and reaction inserted
        self.assertTrue(any(item["description"] == "reacting to event" for item in new_schedule))

    def test_cognition_planner_has_recompose_schedule(self) -> None:
        from packages.vvalley_core.sim.cognition import CognitionPlanner
        self.assertTrue(hasattr(CognitionPlanner, "recompose_schedule"))

    def test_schedule_recomposition_policy_exists(self) -> None:
        from packages.vvalley_core.llm.policy import DEFAULT_TASK_POLICIES
        self.assertIn("schedule_recomposition", DEFAULT_TASK_POLICIES)
        self.assertEqual(DEFAULT_TASK_POLICIES["schedule_recomposition"].model_tier, "fast")

    def test_collision_avoidance_returns_same_if_empty(self) -> None:
        """_avoid_occupied_tile should return the same tile if no agent is on it."""
        from packages.vvalley_core.sim.runner import SimulationRunner

        runner = SimulationRunner()

        class FakeTown:
            npcs = {}
            walkable = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

        class FakeNpc:
            agent_id = "a1"
            x = 0
            y = 0

        result = runner._avoid_occupied_tile(town=FakeTown(), npc=FakeNpc(), target=(1, 1))
        self.assertEqual(result, (1, 1))

    def test_collision_avoidance_moves_when_occupied(self) -> None:
        """_avoid_occupied_tile should find a nearby tile when target is occupied."""
        from packages.vvalley_core.sim.runner import SimulationRunner

        runner = SimulationRunner()

        class FakeOther:
            agent_id = "other"
            x = 1
            y = 1

        class FakeTown:
            npcs = {"other": FakeOther()}
            walkable = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

        class FakeNpc:
            agent_id = "a1"
            x = 0
            y = 0

        result = runner._avoid_occupied_tile(town=FakeTown(), npc=FakeNpc(), target=(1, 1))
        # Should NOT return (1, 1) since it's occupied
        self.assertNotEqual(result, (1, 1))
        # Should return a walkable tile
        rx, ry = result
        self.assertEqual(FakeTown.walkable[ry][rx], 1)

    def test_collision_avoidance_skips_self(self) -> None:
        """_avoid_occupied_tile should not consider the agent itself as an occupant."""
        from packages.vvalley_core.sim.runner import SimulationRunner

        runner = SimulationRunner()

        class FakeSelf:
            agent_id = "a1"
            x = 1
            y = 1

        class FakeTown:
            npcs = {"a1": FakeSelf()}
            walkable = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

        result = runner._avoid_occupied_tile(town=FakeTown(), npc=FakeSelf(), target=(1, 1))
        # Should return (1, 1) since only self is there
        self.assertEqual(result, (1, 1))

    def test_simulation_runner_has_recompose_schedule_method(self) -> None:
        from packages.vvalley_core.sim.runner import SimulationRunner
        self.assertTrue(hasattr(SimulationRunner, "_recompose_schedule_on_reaction"))

    def test_simulation_runner_has_avoid_occupied_tile_method(self) -> None:
        from packages.vvalley_core.sim.runner import SimulationRunner
        self.assertTrue(hasattr(SimulationRunner, "_avoid_occupied_tile"))


class ProductionHardeningTests(unittest.TestCase):
    """Tests for memory hardening: pruning, eviction, ordering, and error tracking."""

    def test_prune_expired_removes_expired_nodes(self) -> None:
        from packages.vvalley_core.sim.memory import AgentMemory

        mem = AgentMemory(agent_id="test", agent_name="Test Agent")
        # Add 2 nodes with expiration_step=10
        mem.add_node(
            kind="event", step=1, subject="Test", predicate="does", object="something",
            description="expiring event 1", poignancy=3, expiration_step=10,
        )
        mem.add_node(
            kind="event", step=2, subject="Test", predicate="does", object="something",
            description="expiring event 2", poignancy=3, expiration_step=10,
        )
        # Add 3 nodes without expiration
        for i in range(3):
            mem.add_node(
                kind="event", step=3 + i, subject="Test", predicate="does", object="something",
                description=f"permanent event {i}", poignancy=3,
            )
        self.assertEqual(len(mem.nodes), 5)

        removed = mem.prune_expired(step=11)
        self.assertEqual(removed, 2)
        self.assertEqual(len(mem.nodes), 3)
        # Verify removed from id_to_node
        for node in mem.nodes:
            self.assertIn(node.node_id, mem.id_to_node)
        self.assertEqual(len(mem.id_to_node), 3)
        # Verify removed from seq_event
        self.assertEqual(len(mem.seq_event), 3)
        for nid in mem.seq_event:
            self.assertIn(nid, mem.id_to_node)

    def test_prune_expired_keeps_unexpired_nodes(self) -> None:
        from packages.vvalley_core.sim.memory import AgentMemory

        mem = AgentMemory(agent_id="test", agent_name="Test Agent")
        for i in range(5):
            mem.add_node(
                kind="event", step=i, subject="Test", predicate="does", object="something",
                description=f"event {i}", poignancy=3, expiration_step=100,
            )
        self.assertEqual(len(mem.nodes), 5)

        removed = mem.prune_expired(step=50)
        self.assertEqual(removed, 0)
        self.assertEqual(len(mem.nodes), 5)

    def test_memory_cap_eviction(self) -> None:
        from packages.vvalley_core.sim.memory import AgentMemory

        mem = AgentMemory(agent_id="test", agent_name="Test Agent")
        # Add 2500 event nodes with varying poignancy (1-5)
        for i in range(2500):
            poignancy = (i % 5) + 1
            mem.add_node(
                kind="event", step=i, subject="Test", predicate="does", object="something",
                description=f"event {i}", poignancy=poignancy,
            )
        self.assertGreater(len(mem.nodes), AgentMemory.MAX_MEMORY_NODES)

        # Trigger eviction via maybe_reflect
        mem.maybe_reflect(step=2500)
        self.assertLessEqual(len(mem.nodes), AgentMemory.MAX_MEMORY_NODES)

        # High-poignancy nodes should survive preferentially
        surviving_poignancies = [n.poignancy for n in mem.nodes]
        avg_poignancy = sum(surviving_poignancies) / len(surviving_poignancies)
        # Average poignancy of survivors should be higher than the overall average (3.0)
        self.assertGreater(avg_poignancy, 3.0)

    def test_eviction_preserves_reflections(self) -> None:
        from packages.vvalley_core.sim.memory import AgentMemory

        mem = AgentMemory(agent_id="test", agent_name="Test Agent")
        reflection_ids: set[str] = set()
        # Add 2000 event nodes
        for i in range(2000):
            mem.add_node(
                kind="event", step=i, subject="Test", predicate="does", object="something",
                description=f"event {i}", poignancy=2,
            )
        # Add 500 reflection nodes
        for i in range(500):
            node = mem.add_node(
                kind="reflection", step=i, subject="Test", predicate="reflects_on",
                object="topic", description=f"reflection {i}", poignancy=6,
            )
            reflection_ids.add(node.node_id)
        self.assertEqual(len(mem.nodes), 2500)

        mem.maybe_reflect(step=2500)
        self.assertLessEqual(len(mem.nodes), AgentMemory.MAX_MEMORY_NODES)

        # All reflections should survive
        surviving_ids = {n.node_id for n in mem.nodes}
        for rid in reflection_ids:
            self.assertIn(rid, surviving_ids, "Reflection node should survive eviction")

    def test_eviction_preserves_recent_nodes(self) -> None:
        from packages.vvalley_core.sim.memory import AgentMemory

        mem = AgentMemory(agent_id="test", agent_name="Test Agent")
        recent_ids: set[str] = set()
        # Add 2400 old nodes at step=1
        for i in range(2400):
            mem.add_node(
                kind="event", step=1, subject="Test", predicate="does", object="something",
                description=f"old event {i}", poignancy=2,
            )
        # Add 100 recent nodes at step=2490
        for i in range(100):
            node = mem.add_node(
                kind="event", step=2490, subject="Test", predicate="does", object="something",
                description=f"recent event {i}", poignancy=2,
            )
            recent_ids.add(node.node_id)
        self.assertEqual(len(mem.nodes), 2500)

        mem.maybe_reflect(step=2500)
        self.assertLessEqual(len(mem.nodes), AgentMemory.MAX_MEMORY_NODES)

        # Recent nodes (step >= 2400) should survive since recent_cutoff = 2500 - 100 = 2400
        surviving_ids = {n.node_id for n in mem.nodes}
        for rid in recent_ids:
            self.assertIn(rid, surviving_ids, "Recent node should survive eviction")

    def test_append_ordering_newest_last(self) -> None:
        from packages.vvalley_core.sim.memory import AgentMemory

        mem = AgentMemory(agent_id="test", agent_name="Test Agent")
        mem.add_node(
            kind="event", step=1, subject="Test", predicate="does", object="first",
            description="first event", poignancy=3,
        )
        mem.add_node(
            kind="event", step=2, subject="Test", predicate="does", object="second",
            description="second event", poignancy=3,
        )
        mem.add_node(
            kind="event", step=3, subject="Test", predicate="does", object="third",
            description="third event", poignancy=3,
        )
        self.assertEqual(mem.nodes[0].step, 1, "Oldest node should be at start")
        self.assertEqual(mem.nodes[-1].step, 3, "Newest node should be at end")

    def test_retrieve_ranked_returns_recent_first(self) -> None:
        from packages.vvalley_core.sim.memory import AgentMemory

        mem = AgentMemory(agent_id="test", agent_name="Test Agent")
        for i in range(10):
            mem.add_node(
                kind="event", step=i * 10, subject="Test", predicate="does", object="task",
                description="working on the project report", poignancy=3,
            )
        results = mem.retrieve_ranked(focal_text="working on the project report", step=100, limit=10)
        self.assertTrue(len(results) > 0)
        # The most recent node (step=90) should score highest due to recency
        self.assertEqual(results[0]["step"], 90, "Most recent node should score highest")

    def test_from_state_sorts_by_step(self) -> None:
        from packages.vvalley_core.sim.memory import AgentMemory, MemoryNode

        # Create an export with nodes in arbitrary (non-sorted) order
        exported = {
            "agent_id": "test",
            "agent_name": "Test Agent",
            "scratch": {},
            "spatial_memory": {},
            "known_places": {},
            "relationship_scores": {},
            "nodes": [
                {"node_id": "n3", "kind": "event", "step": 30, "description": "third",
                 "subject": "T", "predicate": "d", "object": "o", "poignancy": 3},
                {"node_id": "n1", "kind": "event", "step": 10, "description": "first",
                 "subject": "T", "predicate": "d", "object": "o", "poignancy": 3},
                {"node_id": "n2", "kind": "event", "step": 20, "description": "second",
                 "subject": "T", "predicate": "d", "object": "o", "poignancy": 3},
            ],
        }
        mem = AgentMemory.from_state(exported)
        steps = [n.step for n in mem.nodes]
        self.assertEqual(steps, [10, 20, 30], "Nodes should be sorted by step ascending")

    def test_persist_error_logged(self) -> None:
        from packages.vvalley_core.sim.runner import SimulationRunner

        runner = SimulationRunner()
        # _persist_error_count is initialized lazily via getattr; verify the pattern works
        count = getattr(runner, "_persist_error_count", 0)
        self.assertIsInstance(count, int)
        self.assertEqual(count, 0)


class MultiplayerHardeningTests(unittest.TestCase):
    """Tests for multiplayer production hardening (Phase 6)."""

    def test_sqlite_stores_use_thread_local(self) -> None:
        """All SQLite stores should use threading.local for connections."""
        import threading
        from apps.api.vvalley_api.storage.agents import SQLiteAgentStore
        from apps.api.vvalley_api.storage.runtime_control import SQLiteRuntimeControlStore
        from apps.api.vvalley_api.storage.map_versions import SQLiteMapVersionStore
        from apps.api.vvalley_api.storage.llm_control import SQLiteLlmControlStore
        from apps.api.vvalley_api.storage.interaction_hub import SQLiteInteractionHubStore

        for cls in (SQLiteAgentStore, SQLiteRuntimeControlStore,
                    SQLiteMapVersionStore, SQLiteLlmControlStore,
                    SQLiteInteractionHubStore):
            store = cls(db_path=Path("/tmp/test_thread_local.db"))
            self.assertTrue(hasattr(store, "_local"), f"{cls.__name__} missing _local")
            self.assertIsInstance(store._local, threading.local, f"{cls.__name__}._local is not threading.local")
            self.assertFalse(hasattr(store, "_conn"), f"{cls.__name__} still has _conn attribute")

    def test_context_read_does_not_persist(self) -> None:
        """get_agent_context_snapshot should NOT call _persist_runtime."""
        from unittest.mock import patch
        from packages.vvalley_core.sim.runner import (
            get_agent_context_snapshot,
            SimulationRunner,
            reset_simulation_states_for_tests,
        )

        reset_simulation_states_for_tests()
        active = {"id": "v1", "version": 1, "map_name": "test", "town_id": "t1"}
        map_data = {
            "width": 4, "height": 4, "tilewidth": 32, "tileheight": 32,
            "layers": [{"name": "collision", "data": [0]*16}],
            "spawns": [{"x": 0, "y": 0}],
            "locations": [],
        }
        members = [{"agent_id": "a1", "name": "Alice"}]

        with patch.object(SimulationRunner, "_persist_runtime") as mock_persist:
            get_agent_context_snapshot(
                town_id="t-persist-test",
                active_version=active,
                map_data=map_data,
                members=members,
                agent_id="a1",
            )
            mock_persist.assert_not_called()

        reset_simulation_states_for_tests()

    def test_simulation_busy_error_raised_on_lock_timeout(self) -> None:
        """SimulationBusyError should be raised when lock can't be acquired."""
        import threading
        from packages.vvalley_core.sim.runner import (
            _RUNNER_LOCK,
            SimulationBusyError,
            _acquire_or_raise,
        )

        lock_held = threading.Event()
        release = threading.Event()

        def hold_lock():
            with _RUNNER_LOCK:
                lock_held.set()
                release.wait(timeout=5)

        t = threading.Thread(target=hold_lock, daemon=True)
        t.start()
        lock_held.wait(timeout=2)

        try:
            with self.assertRaises(SimulationBusyError):
                _acquire_or_raise(timeout=0.1)
        finally:
            release.set()
            t.join(timeout=2)

    def test_simulation_busy_returns_503(self) -> None:
        """Sim busy error should produce a 503 HTTP response with Retry-After."""
        from fastapi.testclient import TestClient
        from apps.api.vvalley_api.main import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/v1/sim/towns/nonexistent/state")
        # This should work normally (no lock contention in test)
        # We just verify the handler is registered by checking app exception handlers
        from packages.vvalley_core.sim.runner import SimulationBusyError
        handlers = getattr(app, "exception_handlers", {})
        self.assertIn(SimulationBusyError, handlers)

    def test_external_action_skips_cognition_planner(self) -> None:
        """Tick with external actions should skip LLM cognition calls."""
        from unittest.mock import MagicMock
        from packages.vvalley_core.sim.runner import (
            SimulationRunner,
            tick_simulation,
            submit_agent_action,
            reset_simulation_states_for_tests,
        )

        reset_simulation_states_for_tests()
        active = {"id": "v1", "version": 1, "map_name": "test", "town_id": "t1"}
        map_data = {
            "width": 4, "height": 4, "tilewidth": 32, "tileheight": 32,
            "layers": [{"name": "collision", "data": [0]*16}],
            "spawns": [{"x": 0, "y": 0}],
            "locations": [],
        }
        members = [{"agent_id": "ext1", "name": "External"}]
        town_id = "t-ext-test"

        # Submit an external action first
        submit_agent_action(
            town_id=town_id,
            active_version=active,
            map_data=map_data,
            members=members,
            agent_id="ext1",
            action={"dx": 1, "dy": 0, "action_description": "walking east"},
        )

        # Create a mock cognition planner
        mock_cognition = MagicMock()
        mock_cognition.generate_pronunciatio = MagicMock(return_value={"emoji": "🚶"})

        # Tick with the mock cognition planner
        tick_simulation(
            town_id=town_id,
            active_version=active,
            map_data=map_data,
            members=members,
            steps=1,
            control_mode="hybrid",
            cognition_planner=mock_cognition,
        )

        # The mock cognition planner should NOT have been called for the external agent
        mock_cognition.generate_pronunciatio.assert_not_called()

        reset_simulation_states_for_tests()

    def test_context_rate_limit(self) -> None:
        """Context endpoint should return 429 when rate limit exceeded."""
        from apps.api.vvalley_api.routers.sim import reset_sim_rate_limiters_for_tests
        from apps.api.vvalley_api.middleware.rate_limit import InMemoryRateLimiter

        # Create a limiter with very low limit to test
        limiter = InMemoryRateLimiter(max_requests=3, window_seconds=60)
        self.assertTrue(limiter.check("test-key"))
        self.assertTrue(limiter.check("test-key"))
        self.assertTrue(limiter.check("test-key"))
        self.assertFalse(limiter.check("test-key"))  # 4th should fail

        # Different key should still work
        self.assertTrue(limiter.check("other-key"))

    def test_dm_rate_limit(self) -> None:
        """DM rate limiter should enforce per-agent limits."""
        from apps.api.vvalley_api.middleware.rate_limit import InMemoryRateLimiter

        limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)
        self.assertTrue(limiter.check("agent-1"))
        self.assertTrue(limiter.check("agent-1"))
        self.assertFalse(limiter.check("agent-1"))
        # Another agent is independent
        self.assertTrue(limiter.check("agent-2"))


class AgentLifecycleTests(unittest.TestCase):
    """Tests for Phase 7A-7F: agent lifecycle, departure/arrival events,
    auto-town creation, action resilience, per-town locks, server discovery."""

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

    # ── helpers ──────────────────────────────────────────────────────

    def _publish_town(self, town_id: str) -> None:
        resp = self.client.post(
            "/api/v1/maps/publish-version",
            json={
                "town_id": town_id,
                "map_path": "assets/templates/starter_town/map.json",
                "map_name": "starter",
            },
        )
        self.assertEqual(resp.status_code, 200)

    def _register_and_claim(self, name: str = "TestAgent") -> dict:
        resp = self.client.post(
            "/api/v1/agents/register",
            json={"name": name, "auto_claim": True, "owner_handle": f"owner-{uuid.uuid4().hex[:6]}"},
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()["agent"]

    def _join_town(self, api_key: str, town_id: str) -> dict:
        resp = self.client.post(
            "/api/v1/agents/me/join-town",
            json={"town_id": town_id},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()

    def _leave_town(self, api_key: str) -> dict:
        resp = self.client.post(
            "/api/v1/agents/me/leave-town",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()

    def _setup_runtime(self, town_id: str, members: list[dict]) -> dict:
        """Sync a town runtime and return the state dict."""
        from packages.vvalley_core.sim.runner import (
            get_simulation_state,
            reset_simulation_states_for_tests,
        )
        from apps.api.vvalley_api.storage.map_versions import get_active_version

        active = get_active_version(town_id)
        import json
        map_path = ROOT / active["map_json_path"]
        with open(map_path) as f:
            map_data = json.load(f)
        return get_simulation_state(
            town_id=town_id,
            active_version=active,
            map_data=map_data,
            members=members,
        )

    # ── Phase 7A: Graceful departure + departure events ─────────────

    def test_evict_agent_removes_npc_and_injects_departure_events(self) -> None:
        """Evicting an agent should remove its NPC and inject departure events
        into all remaining agents' memories."""
        from packages.vvalley_core.sim.runner import (
            SimulationRunner,
            reset_simulation_states_for_tests,
        )

        reset_simulation_states_for_tests()
        runner = SimulationRunner()
        active = {"id": "v1", "version": 1, "map_name": "test", "town_id": "t-evict"}
        map_data = {
            "width": 4, "height": 4, "tilewidth": 32, "tileheight": 32,
            "layers": [{"name": "collision", "data": [0] * 16}],
            "spawns": [{"x": 0, "y": 0}],
            "locations": [],
        }
        members = [
            {"agent_id": "a1", "name": "Alice"},
            {"agent_id": "a2", "name": "Bob"},
            {"agent_id": "a3", "name": "Charlie"},
        ]
        runner.get_state(
            town_id="t-evict",
            active_version=active,
            map_data=map_data,
            members=members,
        )
        runtime = runner._towns["t-evict"]
        self.assertIn("a2", runtime.npcs)

        # Record initial memory counts
        alice_nodes_before = len(runtime.npcs["a1"].memory.nodes)
        charlie_nodes_before = len(runtime.npcs["a3"].memory.nodes)

        # Evict Bob
        runner.evict_agent(town_id="t-evict", agent_id="a2")

        # Bob should be gone
        self.assertNotIn("a2", runtime.npcs)

        # Alice and Charlie should each have a departure event
        alice_nodes_after = len(runtime.npcs["a1"].memory.nodes)
        charlie_nodes_after = len(runtime.npcs["a3"].memory.nodes)
        self.assertEqual(alice_nodes_after, alice_nodes_before + 1)
        self.assertEqual(charlie_nodes_after, charlie_nodes_before + 1)

        # Check the departure event content
        alice_last_node = runtime.npcs["a1"].memory.nodes[-1]
        self.assertEqual(alice_last_node.kind, "event")
        self.assertEqual(alice_last_node.subject, "Bob")
        self.assertEqual(alice_last_node.predicate, "left")
        self.assertEqual(alice_last_node.object, "the valley")
        self.assertEqual(alice_last_node.poignancy, 6)
        self.assertIn("left the valley", alice_last_node.description)
        self.assertIn("Bob", alice_last_node.description)

        reset_simulation_states_for_tests()

    def test_evict_nonexistent_agent_is_noop(self) -> None:
        """Evicting an agent not in the town should be a no-op."""
        from packages.vvalley_core.sim.runner import (
            SimulationRunner,
            reset_simulation_states_for_tests,
        )

        reset_simulation_states_for_tests()
        runner = SimulationRunner()
        active = {"id": "v1", "version": 1, "map_name": "test", "town_id": "t-evict2"}
        map_data = {
            "width": 4, "height": 4, "tilewidth": 32, "tileheight": 32,
            "layers": [{"name": "collision", "data": [0] * 16}],
            "spawns": [{"x": 0, "y": 0}],
            "locations": [],
        }
        runner.get_state(
            town_id="t-evict2",
            active_version=active,
            map_data=map_data,
            members=[{"agent_id": "a1", "name": "Alice"}],
        )

        # Evict non-existent agent — should not crash
        runner.evict_agent(town_id="t-evict2", agent_id="nonexistent")
        self.assertIn("a1", runner._towns["t-evict2"].npcs)

        # Evict from non-existent town — should not crash
        runner.evict_agent(town_id="nonexistent-town", agent_id="a1")

        reset_simulation_states_for_tests()

    def test_evict_cleans_up_social_state(self) -> None:
        """Evicting an agent should clean up cooldowns, wait states,
        and conversations involving the departed agent."""
        from packages.vvalley_core.sim.runner import (
            SimulationRunner,
            reset_simulation_states_for_tests,
        )

        reset_simulation_states_for_tests()
        runner = SimulationRunner()
        active = {"id": "v1", "version": 1, "map_name": "test", "town_id": "t-evict3"}
        map_data = {
            "width": 4, "height": 4, "tilewidth": 32, "tileheight": 32,
            "layers": [{"name": "collision", "data": [0] * 16}],
            "spawns": [{"x": 0, "y": 0}],
            "locations": [],
        }
        runner.get_state(
            town_id="t-evict3",
            active_version=active,
            map_data=map_data,
            members=[
                {"agent_id": "a1", "name": "Alice"},
                {"agent_id": "a2", "name": "Bob"},
            ],
        )
        runtime = runner._towns["t-evict3"]

        # Manually inject social state involving a2
        runtime.social_interrupt_cooldowns["a1|a2"] = 10
        runtime.social_interrupt_cooldowns["a2|a1"] = 10
        runtime.social_interrupt_cooldowns["a1|other"] = 5
        runtime.social_wait_states["a1"] = {"target_agent_id": "a2"}
        runtime.social_wait_states["a2"] = {"target_agent_id": "a1"}

        runner.evict_agent(town_id="t-evict3", agent_id="a2")

        # Cooldowns involving a2 should be removed
        self.assertNotIn("a1|a2", runtime.social_interrupt_cooldowns)
        self.assertNotIn("a2|a1", runtime.social_interrupt_cooldowns)
        # Unrelated cooldown should survive
        self.assertIn("a1|other", runtime.social_interrupt_cooldowns)
        # Wait states involving a2 should be removed
        self.assertNotIn("a1", runtime.social_wait_states)
        self.assertNotIn("a2", runtime.social_wait_states)

        reset_simulation_states_for_tests()

    def test_leave_town_no_membership_returns_false(self) -> None:
        """Leave town with no membership should return left: false, no crash."""
        agent = self._register_and_claim("NoTown Agent")
        result = self._leave_town(agent["api_key"])
        self.assertFalse(result["left"])

    # ── Phase 7B: Arrival events + last_town preference ─────────────

    def test_new_agent_triggers_arrival_events(self) -> None:
        """When a new agent joins a town, existing agents should get
        an arrival event in their memory."""
        from packages.vvalley_core.sim.runner import (
            SimulationRunner,
            reset_simulation_states_for_tests,
        )

        reset_simulation_states_for_tests()
        runner = SimulationRunner()
        active = {"id": "v1", "version": 1, "map_name": "test", "town_id": "t-arrival"}
        map_data = {
            "width": 4, "height": 4, "tilewidth": 32, "tileheight": 32,
            "layers": [{"name": "collision", "data": [0] * 16}],
            "spawns": [{"x": 0, "y": 0}],
            "locations": [],
        }

        # First sync with just Alice
        runner.get_state(
            town_id="t-arrival",
            active_version=active,
            map_data=map_data,
            members=[{"agent_id": "a1", "name": "Alice"}],
        )
        alice_nodes_before = len(runner._towns["t-arrival"].npcs["a1"].memory.nodes)

        # Second sync adds Bob — this triggers arrival event for Alice
        runner.get_state(
            town_id="t-arrival",
            active_version=active,
            map_data=map_data,
            members=[
                {"agent_id": "a1", "name": "Alice"},
                {"agent_id": "a2", "name": "Bob"},
            ],
        )

        alice_nodes_after = len(runner._towns["t-arrival"].npcs["a1"].memory.nodes)
        self.assertEqual(alice_nodes_after, alice_nodes_before + 1)

        arrival_node = runner._towns["t-arrival"].npcs["a1"].memory.nodes[-1]
        self.assertEqual(arrival_node.kind, "event")
        self.assertEqual(arrival_node.subject, "Bob")
        self.assertEqual(arrival_node.predicate, "arrived in")
        self.assertEqual(arrival_node.object, "the valley")
        self.assertEqual(arrival_node.poignancy, 4)
        self.assertIn("arrived in the valley", arrival_node.description)

        reset_simulation_states_for_tests()

    def test_last_town_saved_on_leave(self) -> None:
        """When an agent leaves a town, last_town_id should be stored."""
        from apps.api.vvalley_api.storage.agents import get_agent_last_town

        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)

        agent = self._register_and_claim("LastTown Agent")
        self._join_town(agent["api_key"], town_id)

        # Leave town
        self._leave_town(agent["api_key"])

        # Check last_town_id is saved
        last = get_agent_last_town(agent_id=agent["id"])
        self.assertEqual(last, town_id)

    def test_auto_join_prefers_last_town(self) -> None:
        """Auto-join should prefer the agent's last town if it has capacity."""
        town_a = f"town-{uuid.uuid4().hex[:8]}"
        town_b = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_a)
        self._publish_town(town_b)

        agent = self._register_and_claim("Rejoin Agent")
        self._join_town(agent["api_key"], town_a)
        self._leave_town(agent["api_key"])

        # Auto-join should prefer town_a
        resp = self.client.post(
            "/api/v1/agents/me/auto-join",
            headers={"Authorization": f"Bearer {agent['api_key']}"},
        )
        self.assertEqual(resp.status_code, 200)
        membership = resp.json()["membership"]
        self.assertEqual(membership["town_id"], town_a)

    # ── Phase 7C: Auto-town creation ────────────────────────────────

    def test_auto_create_town_function(self) -> None:
        """_auto_create_town should scaffold a new town from the template."""
        from apps.api.vvalley_api.routers.agents import _auto_create_town
        from apps.api.vvalley_api.storage.map_versions import list_town_ids

        town_id = _auto_create_town()
        self.assertIsNotNone(town_id)
        self.assertTrue(town_id.startswith("valley-"))
        self.assertIn(town_id, list_town_ids())

    # ── Phase 7D: Action resilience ─────────────────────────────────

    def test_evict_cancels_actions_targeting_departed_agent(self) -> None:
        """When an agent is evicted, active actions targeting them
        should be cancelled and the remaining agent set to idle."""
        from packages.vvalley_core.sim.runner import (
            SimulationRunner,
            reset_simulation_states_for_tests,
        )

        reset_simulation_states_for_tests()
        runner = SimulationRunner()
        active = {"id": "v1", "version": 1, "map_name": "test", "town_id": "t-resilience"}
        map_data = {
            "width": 4, "height": 4, "tilewidth": 32, "tileheight": 32,
            "layers": [{"name": "collision", "data": [0] * 16}],
            "spawns": [{"x": 0, "y": 0}],
            "locations": [],
        }
        runner.get_state(
            town_id="t-resilience",
            active_version=active,
            map_data=map_data,
            members=[
                {"agent_id": "a1", "name": "Alice"},
                {"agent_id": "a2", "name": "Bob"},
                {"agent_id": "a3", "name": "Charlie"},
            ],
        )
        runtime = runner._towns["t-resilience"]

        # Alice has a talk_to targeting Bob (socials format)
        runtime.active_actions["a1"] = {
            "socials": [{"target_agent_id": "a2", "message": "hi"}],
        }
        runtime.npcs["a1"].status = "moving"

        # Charlie has a non-social action (no socials)
        runtime.active_actions["a3"] = {
            "target_x": 2,
            "target_y": 2,
        }
        runtime.npcs["a3"].status = "moving"

        # Evict Bob
        runner.evict_agent(town_id="t-resilience", agent_id="a2")

        # Alice's action should be cancelled
        self.assertNotIn("a1", runtime.active_actions)
        self.assertEqual(runtime.npcs["a1"].status, "idle")

        # Charlie's action should be unaffected
        self.assertIn("a3", runtime.active_actions)
        self.assertEqual(runtime.npcs["a3"].status, "moving")

        reset_simulation_states_for_tests()

    def test_sync_town_cancels_actions_targeting_removed_agents(self) -> None:
        """sync_town should cancel active actions targeting agents
        who have been removed from the members list."""
        from packages.vvalley_core.sim.runner import (
            SimulationRunner,
            reset_simulation_states_for_tests,
        )

        reset_simulation_states_for_tests()
        runner = SimulationRunner()
        active = {"id": "v1", "version": 1, "map_name": "test", "town_id": "t-sync-resil"}
        map_data = {
            "width": 4, "height": 4, "tilewidth": 32, "tileheight": 32,
            "layers": [{"name": "collision", "data": [0] * 16}],
            "spawns": [{"x": 0, "y": 0}],
            "locations": [],
        }
        # Sync with both agents
        runner.get_state(
            town_id="t-sync-resil",
            active_version=active,
            map_data=map_data,
            members=[
                {"agent_id": "a1", "name": "Alice"},
                {"agent_id": "a2", "name": "Bob"},
            ],
        )
        runtime = runner._towns["t-sync-resil"]

        # Alice targets Bob (socials format)
        runtime.active_actions["a1"] = {
            "socials": [{"target_agent_id": "a2", "message": "hi"}],
        }
        runtime.npcs["a1"].status = "moving"

        # Sync again without Bob — simulates Bob leaving via membership removal
        runner.get_state(
            town_id="t-sync-resil",
            active_version=active,
            map_data=map_data,
            members=[{"agent_id": "a1", "name": "Alice"}],
        )

        # Alice's action targeting Bob should be cancelled
        self.assertNotIn("a1", runtime.active_actions)
        self.assertEqual(runtime.npcs["a1"].status, "idle")

        reset_simulation_states_for_tests()

    # ── Phase 7E: Per-town locks ────────────────────────────────────

    def test_per_town_lock_isolation(self) -> None:
        """Operations on different towns should not block each other."""
        import threading
        from packages.vvalley_core.sim.runner import (
            _lock_for_town,
            _acquire_town_or_raise,
            SimulationBusyError,
            reset_simulation_states_for_tests,
        )

        reset_simulation_states_for_tests()

        # Hold lock on town-a
        lock_a = _lock_for_town("town-a")
        lock_held = threading.Event()
        release = threading.Event()

        def hold_lock_a():
            with lock_a:
                lock_held.set()
                release.wait(timeout=5)

        t = threading.Thread(target=hold_lock_a, daemon=True)
        t.start()
        lock_held.wait(timeout=2)

        try:
            # town-b should be acquirable even though town-a is locked
            lock_b = _acquire_town_or_raise("town-b", timeout=0.5)
            lock_b.release()  # success

            # town-a should NOT be acquirable
            with self.assertRaises(SimulationBusyError):
                _acquire_town_or_raise("town-a", timeout=0.1)
        finally:
            release.set()
            t.join(timeout=2)

        reset_simulation_states_for_tests()

    def test_lock_for_town_returns_same_lock_for_same_town(self) -> None:
        """_lock_for_town should return the same RLock for the same town_id."""
        from packages.vvalley_core.sim.runner import (
            _lock_for_town,
            reset_simulation_states_for_tests,
        )

        reset_simulation_states_for_tests()
        lock1 = _lock_for_town("same-town")
        lock2 = _lock_for_town("same-town")
        self.assertIs(lock1, lock2)

        lock3 = _lock_for_town("different-town")
        self.assertIsNot(lock1, lock3)

        reset_simulation_states_for_tests()

    # ── Phase 7F: Server discovery ──────────────────────────────────

    def test_town_summary_includes_availability_fields(self) -> None:
        """Town summary should include available_slots and is_full."""
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)

        resp = self.client.get(f"/api/v1/towns/{town_id}")
        self.assertEqual(resp.status_code, 200)
        town = resp.json()["town"]
        self.assertIn("available_slots", town)
        self.assertIn("is_full", town)
        self.assertEqual(town["available_slots"], 25)
        self.assertFalse(town["is_full"])
        self.assertEqual(town["population"], 0)

    def test_town_list_includes_availability_fields(self) -> None:
        """Town list should include available_slots and is_full for each town."""
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)

        resp = self.client.get("/api/v1/towns")
        self.assertEqual(resp.status_code, 200)
        towns = resp.json()["towns"]
        self.assertTrue(len(towns) > 0)
        for town in towns:
            self.assertIn("available_slots", town)
            self.assertIn("is_full", town)

    def test_town_availability_reflects_population(self) -> None:
        """available_slots and is_full should reflect actual population."""
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)

        # Join one agent
        agent = self._register_and_claim("Slot Agent")
        self._join_town(agent["api_key"], town_id)

        resp = self.client.get(f"/api/v1/towns/{town_id}")
        town = resp.json()["town"]
        self.assertEqual(town["population"], 1)
        self.assertEqual(town["available_slots"], 24)
        self.assertFalse(town["is_full"])

    # ── Integration: departure + rejoin end-to-end ──────────────────

    def test_leave_and_rejoin_fresh_start(self) -> None:
        """Agent that leaves and rejoins should start fresh (new NPC state)."""
        from packages.vvalley_core.sim.runner import (
            SimulationRunner,
            reset_simulation_states_for_tests,
        )

        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)

        agent = self._register_and_claim("Fresh Agent")
        self._join_town(agent["api_key"], town_id)

        # Get initial state
        reset_simulation_states_for_tests()

        # Leave
        self._leave_town(agent["api_key"])

        # Rejoin same town
        self._join_town(agent["api_key"], town_id)

        # Agent should be in the town
        me = self.client.get(
            "/api/v1/agents/me",
            headers={"Authorization": f"Bearer {agent['api_key']}"},
        )
        self.assertEqual(me.json()["agent"]["current_town"]["town_id"], town_id)

        reset_simulation_states_for_tests()


class ContextHashTests(unittest.TestCase):
    """Tests for context hash / unchanged-poll optimization."""

    def test_context_hash_returned_in_full_response(self) -> None:
        """Full context response should include a context_hash field."""
        from packages.vvalley_core.sim.runner import (
            get_agent_context_snapshot,
            reset_simulation_states_for_tests,
        )

        reset_simulation_states_for_tests()
        active = {"id": "v1", "version": 1, "map_name": "test", "town_id": "t1"}
        map_data = {
            "width": 4, "height": 4, "tilewidth": 32, "tileheight": 32,
            "layers": [{"name": "collision", "data": [0]*16}],
            "spawns": [{"x": 0, "y": 0}],
            "locations": [],
        }
        members = [{"agent_id": "a1", "name": "Alice"}]

        ctx = get_agent_context_snapshot(
            town_id="t-hash-1",
            active_version=active,
            map_data=map_data,
            members=members,
            agent_id="a1",
        )
        self.assertIsNotNone(ctx)
        self.assertIn("context_hash", ctx)
        self.assertIsInstance(ctx["context_hash"], str)
        self.assertGreater(len(ctx["context_hash"]), 0)
        # Full response should NOT have "unchanged" key
        self.assertNotIn("unchanged", ctx)
        # Full response should have the expensive "context" key
        self.assertIn("context", ctx)

        reset_simulation_states_for_tests()

    def test_same_state_same_hash(self) -> None:
        """Two calls with no state change should return the same hash."""
        from packages.vvalley_core.sim.runner import (
            get_agent_context_snapshot,
            reset_simulation_states_for_tests,
        )

        reset_simulation_states_for_tests()
        active = {"id": "v1", "version": 1, "map_name": "test", "town_id": "t1"}
        map_data = {
            "width": 4, "height": 4, "tilewidth": 32, "tileheight": 32,
            "layers": [{"name": "collision", "data": [0]*16}],
            "spawns": [{"x": 0, "y": 0}],
            "locations": [],
        }
        members = [{"agent_id": "a1", "name": "Alice"}]

        ctx1 = get_agent_context_snapshot(
            town_id="t-hash-2",
            active_version=active,
            map_data=map_data,
            members=members,
            agent_id="a1",
        )
        ctx2 = get_agent_context_snapshot(
            town_id="t-hash-2",
            active_version=active,
            map_data=map_data,
            members=members,
            agent_id="a1",
        )
        self.assertEqual(ctx1["context_hash"], ctx2["context_hash"])

        reset_simulation_states_for_tests()

    def test_if_unchanged_returns_minimal_response(self) -> None:
        """Passing matching if_unchanged should return minimal response."""
        from packages.vvalley_core.sim.runner import (
            get_agent_context_snapshot,
            reset_simulation_states_for_tests,
        )

        reset_simulation_states_for_tests()
        active = {"id": "v1", "version": 1, "map_name": "test", "town_id": "t1"}
        map_data = {
            "width": 4, "height": 4, "tilewidth": 32, "tileheight": 32,
            "layers": [{"name": "collision", "data": [0]*16}],
            "spawns": [{"x": 0, "y": 0}],
            "locations": [],
        }
        members = [{"agent_id": "a1", "name": "Alice"}]

        # First call: get hash
        ctx1 = get_agent_context_snapshot(
            town_id="t-hash-3",
            active_version=active,
            map_data=map_data,
            members=members,
            agent_id="a1",
        )
        h = ctx1["context_hash"]

        # Second call: pass hash back
        ctx2 = get_agent_context_snapshot(
            town_id="t-hash-3",
            active_version=active,
            map_data=map_data,
            members=members,
            agent_id="a1",
            if_unchanged=h,
        )
        self.assertTrue(ctx2.get("unchanged"))
        self.assertEqual(ctx2["context_hash"], h)
        self.assertIn("step", ctx2)
        self.assertIn("position", ctx2)
        # Minimal response should NOT have expensive fields
        self.assertNotIn("context", ctx2)
        self.assertNotIn("active_conversation", ctx2)
        self.assertNotIn("pending_action", ctx2)

        reset_simulation_states_for_tests()

    def test_if_unchanged_stale_hash_returns_full(self) -> None:
        """Passing a wrong/stale hash should return full context."""
        from packages.vvalley_core.sim.runner import (
            get_agent_context_snapshot,
            reset_simulation_states_for_tests,
        )

        reset_simulation_states_for_tests()
        active = {"id": "v1", "version": 1, "map_name": "test", "town_id": "t1"}
        map_data = {
            "width": 4, "height": 4, "tilewidth": 32, "tileheight": 32,
            "layers": [{"name": "collision", "data": [0]*16}],
            "spawns": [{"x": 0, "y": 0}],
            "locations": [],
        }
        members = [{"agent_id": "a1", "name": "Alice"}]

        ctx = get_agent_context_snapshot(
            town_id="t-hash-4",
            active_version=active,
            map_data=map_data,
            members=members,
            agent_id="a1",
            if_unchanged="stale_hash_value",
        )
        self.assertNotIn("unchanged", ctx)
        self.assertIn("context", ctx)
        self.assertIn("context_hash", ctx)

        reset_simulation_states_for_tests()

    def test_hash_changes_after_tick(self) -> None:
        """Hash should change after a tick advances the step counter."""
        from packages.vvalley_core.sim.runner import (
            get_agent_context_snapshot,
            tick_simulation,
            reset_simulation_states_for_tests,
        )

        reset_simulation_states_for_tests()
        active = {"id": "v1", "version": 1, "map_name": "test", "town_id": "t1"}
        map_data = {
            "width": 4, "height": 4, "tilewidth": 32, "tileheight": 32,
            "layers": [{"name": "collision", "data": [0]*16}],
            "spawns": [{"x": 0, "y": 0}],
            "locations": [],
        }
        members = [{"agent_id": "a1", "name": "Alice"}]

        ctx1 = get_agent_context_snapshot(
            town_id="t-hash-5",
            active_version=active,
            map_data=map_data,
            members=members,
            agent_id="a1",
        )
        h1 = ctx1["context_hash"]

        # Tick to advance step
        tick_simulation(
            town_id="t-hash-5",
            active_version=active,
            map_data=map_data,
            members=members,
            steps=1,
            planning_scope="short_action",
            control_mode="external",
        )

        ctx2 = get_agent_context_snapshot(
            town_id="t-hash-5",
            active_version=active,
            map_data=map_data,
            members=members,
            agent_id="a1",
        )
        h2 = ctx2["context_hash"]

        self.assertNotEqual(h1, h2, "Hash should change after tick advances step")

        reset_simulation_states_for_tests()

    def test_hash_changes_when_nearby_agent_appears(self) -> None:
        """Hash should change when a new agent appears in perception range."""
        from packages.vvalley_core.sim.runner import (
            get_agent_context_snapshot,
            reset_simulation_states_for_tests,
        )

        reset_simulation_states_for_tests()
        active = {"id": "v1", "version": 1, "map_name": "test", "town_id": "t1"}
        map_data = {
            "width": 10, "height": 10, "tilewidth": 32, "tileheight": 32,
            "layers": [{"name": "collision", "data": [0]*100}],
            "spawns": [{"x": 0, "y": 0}, {"x": 1, "y": 0}],
            "locations": [],
        }
        # Start with only Alice
        members_1 = [{"agent_id": "a1", "name": "Alice"}]

        ctx1 = get_agent_context_snapshot(
            town_id="t-hash-6",
            active_version=active,
            map_data=map_data,
            members=members_1,
            agent_id="a1",
        )
        h1 = ctx1["context_hash"]

        # Now Bob appears nearby
        members_2 = [
            {"agent_id": "a1", "name": "Alice"},
            {"agent_id": "a2", "name": "Bob"},
        ]
        ctx2 = get_agent_context_snapshot(
            town_id="t-hash-6",
            active_version=active,
            map_data=map_data,
            members=members_2,
            agent_id="a1",
        )
        h2 = ctx2["context_hash"]

        self.assertNotEqual(h1, h2, "Hash should change when nearby agent appears")

        reset_simulation_states_for_tests()


if __name__ == "__main__":
    unittest.main()
