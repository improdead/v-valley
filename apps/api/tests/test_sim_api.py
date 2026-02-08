#!/usr/bin/env python3

from __future__ import annotations

import os
import tempfile
import time
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
TEST_DB_DIR = tempfile.TemporaryDirectory(dir=ROOT)
TEST_DB_PATH = Path(TEST_DB_DIR.name) / "test_sim_vvalley.db"
os.environ["VVALLEY_DB_PATH"] = str(TEST_DB_PATH)

from apps.api.vvalley_api.main import app
from apps.api.vvalley_api.services.runtime_scheduler import stop_town_runtime_scheduler
from apps.api.vvalley_api.storage.agents import reset_backend_cache_for_tests as reset_agents_backend
from apps.api.vvalley_api.storage.interaction_hub import reset_backend_cache_for_tests as reset_interaction_backend
from apps.api.vvalley_api.storage.llm_control import reset_backend_cache_for_tests as reset_llm_backend
from apps.api.vvalley_api.storage.map_versions import reset_backend_cache_for_tests as reset_maps_backend
from apps.api.vvalley_api.storage.runtime_control import reset_backend_cache_for_tests as reset_runtime_backend
from packages.vvalley_core.sim.runner import reset_simulation_states_for_tests


class SimApiTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["VVALLEY_DB_PATH"] = str(TEST_DB_PATH)
        stop_town_runtime_scheduler()
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        reset_agents_backend()
        reset_maps_backend()
        reset_llm_backend()
        reset_runtime_backend()
        reset_interaction_backend()
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
        self.assertIn("clock", state_payload)
        self.assertEqual(state_payload["clock"]["step"], 0)
        self.assertEqual(state_payload["clock"]["minute_of_day"], 0)
        self.assertEqual(state_payload["clock"]["step_minutes"], 10)

        tick_resp = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 3},
        )
        self.assertEqual(tick_resp.status_code, 200)
        tick = tick_resp.json()["tick"]
        self.assertEqual(tick["control_mode"], "external")
        self.assertEqual(tick["steps_run"], 3)
        self.assertEqual(tick["step"], 3)
        self.assertEqual(tick["event_count"], 6)
        self.assertIn("social_event_count", tick)
        self.assertIn("social_events", tick)
        self.assertIn("reflection_event_count", tick)
        self.assertIn("reflection_events", tick)
        self.assertGreaterEqual(len(tick["events"]), 1)
        self.assertEqual(tick["events"][0]["decision"]["route"], "awaiting_user_agent")
        self.assertIn("clock", tick)
        self.assertEqual(tick["clock"]["step"], 3)
        self.assertEqual(tick["clock"]["minute_of_day"], 30)

        state_after = self.client.get(f"/api/v1/sim/towns/{town_id}/state")
        self.assertEqual(state_after.status_code, 200)
        after_payload = state_after.json()["state"]
        self.assertEqual(after_payload["step"], 3)
        self.assertEqual(after_payload["npc_count"], 2)
        self.assertEqual(after_payload["clock"]["minute_of_day"], 30)
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

    def test_agent_context_action_and_external_tick(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)
        agent = self._register_and_join(town_id, "ActionPilot", "owner-ap")
        peer = self._register_and_join(town_id, "ActionPeer", "owner-peer")
        api_key = agent["api_key"]

        context_resp = self.client.get(
            f"/api/v1/sim/towns/{town_id}/agents/me/context",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.assertEqual(context_resp.status_code, 200)
        context_payload = context_resp.json()["context"]
        self.assertEqual(context_payload["agent"]["agent_id"], agent["id"])
        self.assertIn("context", context_payload)
        self.assertIn("clock", context_payload)
        self.assertIn("clock", context_payload["context"])
        self.assertIn("schedule", context_payload["context"])

        action_resp = self.client.post(
            f"/api/v1/sim/towns/{town_id}/agents/me/action",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "planning_scope": "short_action",
                "dx": 1,
                "dy": 0,
                "goal_reason": "explore",
                "action_ttl_steps": 2,
                "interrupt_on_social": False,
            },
        )
        self.assertEqual(action_resp.status_code, 200)
        accepted = action_resp.json()["accepted"]
        self.assertEqual(accepted["agent_id"], agent["id"])
        self.assertEqual(accepted["queued_action"]["dx"], 1)
        self.assertEqual(accepted["queued_action"]["dy"], 0)
        self.assertEqual(accepted["queued_action"]["action_ttl_steps"], 2)
        self.assertFalse(accepted["queued_action"]["interrupt_on_social"])
        self.assertEqual(accepted["queued_action"]["social_interrupt_cooldown_steps"], 6)

        tick_resp_1 = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 1, "planning_scope": "short_action", "control_mode": "external"},
        )
        self.assertEqual(tick_resp_1.status_code, 200)
        tick_payload_1 = tick_resp_1.json()["tick"]
        self.assertEqual(tick_payload_1["control_mode"], "external")
        agent_events_1 = [e for e in tick_payload_1["events"] if e["agent_id"] == agent["id"]]
        self.assertEqual(len(agent_events_1), 1)
        self.assertEqual(agent_events_1[0]["decision"]["route"], "external_action")
        self.assertEqual(agent_events_1[0]["decision"]["used_tier"], "external_agent")
        self.assertEqual(agent_events_1[0]["decision"]["action_remaining_steps_before"], 2)
        self.assertEqual(agent_events_1[0]["decision"]["action_remaining_steps_after"], 1)
        self.assertFalse(agent_events_1[0]["decision"]["action_finished"])

        context_after_first_tick = self.client.get(
            f"/api/v1/sim/towns/{town_id}/agents/me/context",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.assertEqual(context_after_first_tick.status_code, 200)
        context_after_payload = context_after_first_tick.json()["context"]
        self.assertIsNone(context_after_payload["pending_action"])
        self.assertIsNotNone(context_after_payload["active_action"])

        tick_resp_2 = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 1, "planning_scope": "short_action", "control_mode": "external"},
        )
        self.assertEqual(tick_resp_2.status_code, 200)
        tick_payload_2 = tick_resp_2.json()["tick"]
        agent_events_2 = [e for e in tick_payload_2["events"] if e["agent_id"] == agent["id"]]
        self.assertEqual(len(agent_events_2), 1)
        self.assertEqual(agent_events_2[0]["decision"]["route"], "external_action_continuation")
        self.assertEqual(agent_events_2[0]["decision"]["action_remaining_steps_before"], 1)
        self.assertEqual(agent_events_2[0]["decision"]["action_remaining_steps_after"], 0)
        self.assertTrue(agent_events_2[0]["decision"]["action_finished"])

        context_after_second_tick = self.client.get(
            f"/api/v1/sim/towns/{town_id}/agents/me/context",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.assertEqual(context_after_second_tick.status_code, 200)
        self.assertIsNone(context_after_second_tick.json()["context"]["active_action"])

        state_for_social = self.client.get(f"/api/v1/sim/towns/{town_id}/state")
        self.assertEqual(state_for_social.status_code, 200)
        npcs_for_social = {npc["agent_id"]: npc for npc in state_for_social.json()["state"]["npcs"]}
        agent_pos = npcs_for_social[agent["id"]]
        peer_pos = npcs_for_social[peer["id"]]
        initial_dist = abs(int(agent_pos["x"]) - int(peer_pos["x"])) + abs(int(agent_pos["y"]) - int(peer_pos["y"]))

        action_with_social = self.client.post(
            f"/api/v1/sim/towns/{town_id}/agents/me/action",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "planning_scope": "short_action",
                "target_x": int(peer_pos["x"]),
                "target_y": int(peer_pos["y"]),
                "interrupt_on_social": False,
                "socials": [{"target_agent_id": peer["id"], "message": "Quick sync"}],
            },
        )
        self.assertEqual(action_with_social.status_code, 200)
        accepted_social_action = action_with_social.json()["accepted"]["queued_action"]
        self.assertEqual(accepted_social_action["action_ttl_steps"], 40)
        self.assertFalse(accepted_social_action["interrupt_on_social"])

        tick_resp_3 = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 1, "planning_scope": "short_action", "control_mode": "external"},
        )
        self.assertEqual(tick_resp_3.status_code, 200)
        tick_payload_3 = tick_resp_3.json()["tick"]
        agent_events_3 = [e for e in tick_payload_3["events"] if e["agent_id"] == agent["id"]]
        self.assertEqual(len(agent_events_3), 1)
        self.assertEqual(agent_events_3[0]["decision"]["route"], "external_action")
        self.assertIn("pending_social_target", agent_events_3[0]["decision"])
        if initial_dist > 1:
            self.assertEqual(tick_payload_3["social_event_count"], 0)

        found_social = any(event.get("source") == "external_action" for event in tick_payload_3["social_events"])
        for _ in range(60):
            if found_social:
                break
            follow_tick = self.client.post(
                f"/api/v1/sim/towns/{town_id}/tick",
                json={"steps": 1, "planning_scope": "short_action", "control_mode": "external"},
            )
            self.assertEqual(follow_tick.status_code, 200)
            follow_payload = follow_tick.json()["tick"]
            if any(event.get("source") == "external_action" for event in follow_payload["social_events"]):
                found_social = True
                break
        self.assertTrue(found_social)

    def test_social_action_without_goal_auto_tracks_target(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)
        agent = self._register_and_join(town_id, "SocialPilot", "owner-sp")
        peer = self._register_and_join(town_id, "SocialPeer", "owner-peer")
        api_key = agent["api_key"]

        action_resp = self.client.post(
            f"/api/v1/sim/towns/{town_id}/agents/me/action",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "planning_scope": "short_action",
                "interrupt_on_social": False,
                "socials": [{"target_agent_id": peer["id"], "message": "checking in"}],
            },
        )
        self.assertEqual(action_resp.status_code, 200)
        queued = action_resp.json()["accepted"]["queued_action"]
        self.assertEqual(queued["action_ttl_steps"], 40)
        self.assertFalse(queued["interrupt_on_social"])
        self.assertIn("socials", queued)
        self.assertEqual(len(queued["socials"]), 1)

        first_tick = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 1, "planning_scope": "short_action", "control_mode": "external"},
        )
        self.assertEqual(first_tick.status_code, 200)
        first_payload = first_tick.json()["tick"]
        agent_events = [event for event in first_payload["events"] if event["agent_id"] == agent["id"]]
        self.assertEqual(len(agent_events), 1)
        decision = agent_events[0]["decision"]
        self.assertEqual(decision["route"], "external_action")
        self.assertIn("pending_social_target", decision)
        self.assertIsNotNone(decision["goal"]["x"])
        self.assertIsNotNone(decision["goal"]["y"])

        payload_with_social = first_payload if any(
            event.get("source") == "external_action" for event in first_payload["social_events"]
        ) else None
        found_social = payload_with_social is not None
        for _ in range(60):
            if found_social:
                break
            tick_resp = self.client.post(
                f"/api/v1/sim/towns/{town_id}/tick",
                json={"steps": 1, "planning_scope": "short_action", "control_mode": "external"},
            )
            self.assertEqual(tick_resp.status_code, 200)
            payload = tick_resp.json()["tick"]
            if any(event.get("source") == "external_action" for event in payload["social_events"]):
                found_social = True
                payload_with_social = payload
                break
        self.assertTrue(found_social)
        self.assertIsNotNone(payload_with_social)
        social_events = [event for event in payload_with_social["events"] if event["agent_id"] == agent["id"]]
        self.assertEqual(len(social_events), 1)
        self.assertTrue(social_events[0]["decision"]["action_finished"])
        self.assertEqual(social_events[0]["decision"]["pending_social_count"], 0)

        context_after_social = self.client.get(
            f"/api/v1/sim/towns/{town_id}/agents/me/context",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.assertEqual(context_after_social.status_code, 200)
        self.assertIsNone(context_after_social.json()["context"]["active_action"])

    def test_conversation_session_pauses_agents_temporarily(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)
        agent = self._register_and_join(town_id, "TalkPilot", "owner-talk")
        peer = self._register_and_join(town_id, "TalkPeer", "owner-peer")
        api_key = agent["api_key"]
        peer_api_key = peer["api_key"]

        state_resp = self.client.get(f"/api/v1/sim/towns/{town_id}/state")
        self.assertEqual(state_resp.status_code, 200)
        npcs = {npc["agent_id"]: npc for npc in state_resp.json()["state"]["npcs"]}
        peer_pos = npcs[peer["id"]]

        approach_resp = self.client.post(
            f"/api/v1/sim/towns/{town_id}/agents/me/action",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "planning_scope": "short_action",
                "target_x": int(peer_pos["x"]),
                "target_y": int(peer_pos["y"]),
                "goal_reason": "meet_peer",
                "action_ttl_steps": 80,
                "interrupt_on_social": False,
            },
        )
        self.assertEqual(approach_resp.status_code, 200)

        adjacent = False
        for _ in range(40):
            tick_resp = self.client.post(
                f"/api/v1/sim/towns/{town_id}/tick",
                json={"steps": 1, "planning_scope": "short_action", "control_mode": "external"},
            )
            self.assertEqual(tick_resp.status_code, 200)
            state_npcs = {npc["agent_id"]: npc for npc in tick_resp.json()["tick"]["state"]["npcs"]}
            a_pos = state_npcs[agent["id"]]
            b_pos = state_npcs[peer["id"]]
            if abs(int(a_pos["x"]) - int(b_pos["x"])) + abs(int(a_pos["y"]) - int(b_pos["y"])) <= 1:
                adjacent = True
                break
        self.assertTrue(adjacent)

        social_action = self.client.post(
            f"/api/v1/sim/towns/{town_id}/agents/me/action",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "planning_scope": "short_action",
                "dx": 0,
                "dy": 0,
                "action_ttl_steps": 4,
                "interrupt_on_social": False,
                "socials": [{"target_agent_id": peer["id"], "message": "let's sync plans"}],
            },
        )
        self.assertEqual(social_action.status_code, 200)

        social_tick = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 1, "planning_scope": "short_action", "control_mode": "external"},
        )
        self.assertEqual(social_tick.status_code, 200)
        social_payload = social_tick.json()["tick"]
        self.assertEqual(social_payload["social_event_count"], 1)
        self.assertEqual(social_payload["social_events"][0]["source"], "external_action")

        context_agent = self.client.get(
            f"/api/v1/sim/towns/{town_id}/agents/me/context",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.assertEqual(context_agent.status_code, 200)
        active_conversation_agent = context_agent.json()["context"]["active_conversation"]
        self.assertIsNotNone(active_conversation_agent)

        context_peer = self.client.get(
            f"/api/v1/sim/towns/{town_id}/agents/me/context",
            headers={"Authorization": f"Bearer {peer_api_key}"},
        )
        self.assertEqual(context_peer.status_code, 200)
        active_conversation_peer = context_peer.json()["context"]["active_conversation"]
        self.assertIsNotNone(active_conversation_peer)
        self.assertEqual(active_conversation_agent["session_id"], active_conversation_peer["session_id"])

        pause_tick = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 1, "planning_scope": "short_action", "control_mode": "external"},
        )
        self.assertEqual(pause_tick.status_code, 200)
        pause_events = {event["agent_id"]: event for event in pause_tick.json()["tick"]["events"]}
        self.assertEqual(pause_events[agent["id"]]["decision"]["route"], "external_action_conversation_pause")
        self.assertEqual(pause_events[peer["id"]]["decision"]["route"], "conversation_session")
        self.assertEqual(pause_events[agent["id"]]["decision"]["movement_mode"], "paused_conversation")
        self.assertEqual(pause_events[peer["id"]]["decision"]["movement_mode"], "paused_conversation")
        self.assertTrue(pause_events[agent["id"]]["decision"]["action_paused"])

    def test_unreachable_external_goal_marks_action_stuck(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)
        agent = self._register_and_join(town_id, "StuckPilot", "owner-stuck")
        api_key = agent["api_key"]

        action_resp = self.client.post(
            f"/api/v1/sim/towns/{town_id}/agents/me/action",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "planning_scope": "short_action",
                "dx": -1,
                "dy": 0,
                "goal_reason": "push_west_until_blocked",
                "action_ttl_steps": 12,
                "interrupt_on_social": False,
            },
        )
        self.assertEqual(action_resp.status_code, 200)

        tick_resp = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 12, "planning_scope": "short_action", "control_mode": "external"},
        )
        self.assertEqual(tick_resp.status_code, 200)
        tick_payload = tick_resp.json()["tick"]
        agent_events = [event for event in tick_payload["events"] if event["agent_id"] == agent["id"]]
        stuck_events = [event for event in agent_events if event["decision"]["route"] == "external_action_stuck"]
        self.assertGreaterEqual(len(stuck_events), 1)
        self.assertTrue(stuck_events[0]["decision"]["stuck"])
        self.assertTrue(stuck_events[0]["decision"]["action_finished"])

        context_after = self.client.get(
            f"/api/v1/sim/towns/{town_id}/agents/me/context",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.assertEqual(context_after.status_code, 200)
        self.assertIsNone(context_after.json()["context"]["active_action"])

    def test_agent_context_requires_auth(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)
        self._register_and_join(town_id, "NoAuthAgent", "owner-na")

        missing_auth = self.client.get(f"/api/v1/sim/towns/{town_id}/agents/me/context")
        self.assertEqual(missing_auth.status_code, 401)

    def test_runtime_persists_across_runner_reset(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)
        agent = self._register_and_join(town_id, "PersistentPilot", "owner-pp")
        api_key = agent["api_key"]

        action_resp = self.client.post(
            f"/api/v1/sim/towns/{town_id}/agents/me/action",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "planning_scope": "short_action",
                "dx": 1,
                "dy": 0,
                "goal_reason": "persist-check",
                "action_ttl_steps": 3,
                "interrupt_on_social": False,
            },
        )
        self.assertEqual(action_resp.status_code, 200)

        tick_resp = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 1, "planning_scope": "short_action", "control_mode": "external"},
        )
        self.assertEqual(tick_resp.status_code, 200)
        tick_payload = tick_resp.json()["tick"]
        self.assertEqual(tick_payload["step"], 1)
        agent_events = [event for event in tick_payload["events"] if event["agent_id"] == agent["id"]]
        self.assertEqual(len(agent_events), 1)
        self.assertEqual(agent_events[0]["decision"]["route"], "external_action")

        reset_simulation_states_for_tests(clear_persisted=False)

        state_after_reset = self.client.get(f"/api/v1/sim/towns/{town_id}/state")
        self.assertEqual(state_after_reset.status_code, 200)
        self.assertEqual(state_after_reset.json()["state"]["step"], 1)

        context_after_reset = self.client.get(
            f"/api/v1/sim/towns/{town_id}/agents/me/context",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.assertEqual(context_after_reset.status_code, 200)
        self.assertIsNotNone(context_after_reset.json()["context"]["active_action"])

        tick_after_reset = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 1, "planning_scope": "short_action", "control_mode": "external"},
        )
        self.assertEqual(tick_after_reset.status_code, 200)
        tick_after_payload = tick_after_reset.json()["tick"]
        resumed_agent_events = [
            event for event in tick_after_payload["events"] if event["agent_id"] == agent["id"]
        ]
        self.assertEqual(len(resumed_agent_events), 1)
        self.assertEqual(resumed_agent_events[0]["decision"]["route"], "external_action_continuation")

    def test_external_mode_uses_delegated_autonomy_without_action(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)
        agent = self._register_and_join(town_id, "DelegatedPilot", "owner-dp")
        api_key = agent["api_key"]

        update_autonomy = self.client.put(
            "/api/v1/agents/me/autonomy",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "mode": "delegated",
                "allowed_scopes": ["short_action"],
                "max_autonomous_ticks": 2,
            },
        )
        self.assertEqual(update_autonomy.status_code, 200)

        first_tick = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 1, "planning_scope": "short_action", "control_mode": "external"},
        )
        self.assertEqual(first_tick.status_code, 200)
        first_event = [
            event for event in first_tick.json()["tick"]["events"] if event["agent_id"] == agent["id"]
        ][0]
        self.assertTrue(bool(first_event["decision"]["delegated_autopilot"]))
        self.assertEqual(first_event["decision"]["autonomy_ticks_after"], 1)

        second_tick = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 1, "planning_scope": "short_action", "control_mode": "external"},
        )
        self.assertEqual(second_tick.status_code, 200)
        second_event = [
            event for event in second_tick.json()["tick"]["events"] if event["agent_id"] == agent["id"]
        ][0]
        self.assertTrue(bool(second_event["decision"]["delegated_autopilot"]))
        self.assertEqual(second_event["decision"]["autonomy_ticks_after"], 2)

        third_tick = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 1, "planning_scope": "short_action", "control_mode": "external"},
        )
        self.assertEqual(third_tick.status_code, 200)
        ingest_payload = third_tick.json()["interaction_ingest"]
        self.assertGreaterEqual(int(ingest_payload["inbox_created"]), 1)
        self.assertGreaterEqual(int(ingest_payload["escalations_created"]), 1)
        third_event = [
            event for event in third_tick.json()["tick"]["events"] if event["agent_id"] == agent["id"]
        ][0]
        self.assertEqual(third_event["decision"]["route"], "awaiting_user_agent_autonomy_limit")
        self.assertFalse(bool(third_event["decision"]["delegated_autopilot"]))

        inbox_resp = self.client.get(
            "/api/v1/agents/me/inbox",
            params={"status": "pending"},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.assertEqual(inbox_resp.status_code, 200)
        inbox_items = inbox_resp.json()["items"]
        self.assertTrue(any(item["kind"] == "autonomy_limit_reached" for item in inbox_items))

        escalations_resp = self.client.get(
            "/api/v1/owners/me/escalations",
            params={"status": "open"},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.assertEqual(escalations_resp.status_code, 200)
        open_escalations = escalations_resp.json()["items"]
        self.assertTrue(any(item["kind"] == "autonomy_limit_reached" for item in open_escalations))

        fourth_tick = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 1, "planning_scope": "short_action", "control_mode": "external"},
        )
        self.assertEqual(fourth_tick.status_code, 200)
        self.assertEqual(int(fourth_tick.json()["interaction_ingest"]["inbox_created"]), 1)
        self.assertEqual(int(fourth_tick.json()["interaction_ingest"]["escalations_created"]), 1)

        escalations_resp_after = self.client.get(
            "/api/v1/owners/me/escalations",
            params={"status": "open"},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.assertEqual(escalations_resp_after.status_code, 200)
        self.assertEqual(len(escalations_resp_after.json()["items"]), len(open_escalations))

    def test_dm_request_approve_and_message_flow(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)
        alice = self._register_and_join(town_id, "AliceDM", "owner-alice")
        bob = self._register_and_join(town_id, "BobDM", "owner-bob")

        create_req = self.client.post(
            "/api/v1/agents/dm/request",
            headers={"Authorization": f"Bearer {alice['api_key']}"},
            json={
                "target_agent_id": bob["id"],
                "message": "Want to sync after lunch?",
            },
        )
        self.assertEqual(create_req.status_code, 200)
        request_id = create_req.json()["request"]["id"]

        check_bob = self.client.get(
            "/api/v1/agents/dm/check",
            headers={"Authorization": f"Bearer {bob['api_key']}"},
        )
        self.assertEqual(check_bob.status_code, 200)
        self.assertGreaterEqual(int(check_bob.json()["pending_requests"]), 1)

        approve_resp = self.client.post(
            f"/api/v1/agents/dm/requests/{request_id}/approve",
            headers={"Authorization": f"Bearer {bob['api_key']}"},
        )
        self.assertEqual(approve_resp.status_code, 200)
        self.assertEqual(approve_resp.json()["request"]["status"], "approved")
        conversation_id = approve_resp.json()["conversation"]["id"]

        send_resp = self.client.post(
            f"/api/v1/agents/dm/conversations/{conversation_id}/send",
            headers={"Authorization": f"Bearer {alice['api_key']}"},
            json={
                "body": "I might need owner input before confirming.",
                "needs_human_input": True,
                "metadata": {"intent": "schedule_change"},
            },
        )
        self.assertEqual(send_resp.status_code, 200)
        self.assertTrue(bool(send_resp.json()["message"]["needs_human_input"]))

        bob_inbox = self.client.get(
            "/api/v1/agents/me/inbox",
            params={"status": "pending"},
            headers={"Authorization": f"Bearer {bob['api_key']}"},
        )
        self.assertEqual(bob_inbox.status_code, 200)
        self.assertTrue(any(item["kind"] == "dm_message" for item in bob_inbox.json()["items"]))

        alice_escalations = self.client.get(
            "/api/v1/owners/me/escalations",
            params={"status": "open"},
            headers={"Authorization": f"Bearer {alice['api_key']}"},
        )
        self.assertEqual(alice_escalations.status_code, 200)
        self.assertTrue(
            any(item["kind"] == "dm_message_needs_human" for item in alice_escalations.json()["items"])
        )

    def test_runtime_start_status_and_stop(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)
        self._register_and_join(town_id, "SchedulerResident", "owner-sr")

        start_resp = self.client.post(
            f"/api/v1/sim/towns/{town_id}/runtime/start",
            json={
                "tick_interval_seconds": 1,
                "steps_per_tick": 1,
                "planning_scope": "short_action",
                "control_mode": "autopilot",
            },
        )
        self.assertEqual(start_resp.status_code, 200)
        self.assertTrue(start_resp.json()["runtime"]["enabled"])
        self.assertTrue(bool(start_resp.json()["scheduler"]["running"]))

        initial_state = self.client.get(f"/api/v1/sim/towns/{town_id}/state")
        self.assertEqual(initial_state.status_code, 200)
        initial_step = int(initial_state.json()["state"]["step"])

        progressed = False
        for _ in range(12):
            time.sleep(0.35)
            state_resp = self.client.get(f"/api/v1/sim/towns/{town_id}/state")
            self.assertEqual(state_resp.status_code, 200)
            current_step = int(state_resp.json()["state"]["step"])
            if current_step > initial_step:
                progressed = True
                break
        self.assertTrue(progressed)

        status_resp = self.client.get(f"/api/v1/sim/towns/{town_id}/runtime/status")
        self.assertEqual(status_resp.status_code, 200)
        self.assertTrue(status_resp.json()["runtime"]["enabled"])
        self.assertIn("running", status_resp.json()["scheduler"])

        stop_resp = self.client.post(f"/api/v1/sim/towns/{town_id}/runtime/stop")
        self.assertEqual(stop_resp.status_code, 200)
        self.assertFalse(stop_resp.json()["runtime"]["enabled"])

    def test_tick_idempotency_key_replays_completed_batch(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)
        self._register_and_join(town_id, "IdemAgent", "owner-idem")

        first_tick = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={
                "steps": 1,
                "planning_scope": "short_action",
                "control_mode": "external",
                "idempotency_key": "manual-batch-1",
            },
        )
        self.assertEqual(first_tick.status_code, 200)
        self.assertFalse(bool(first_tick.json()["idempotent_replay"]))
        first_step = int(first_tick.json()["tick"]["step"])

        second_tick = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={
                "steps": 1,
                "planning_scope": "short_action",
                "control_mode": "external",
                "idempotency_key": "manual-batch-1",
            },
        )
        self.assertEqual(second_tick.status_code, 200)
        self.assertTrue(bool(second_tick.json()["idempotent_replay"]))
        second_step = int(second_tick.json()["tick"]["step"])
        self.assertEqual(second_step, first_step)

        state_resp = self.client.get(f"/api/v1/sim/towns/{town_id}/state")
        self.assertEqual(state_resp.status_code, 200)
        self.assertEqual(int(state_resp.json()["state"]["step"]), first_step)

    def test_runtime_metrics_track_fallback_and_stuck_signals(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)
        agent = self._register_and_join(town_id, "MetricPilot", "owner-metric")

        autopilot_tick = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 1, "planning_scope": "short_action", "control_mode": "autopilot"},
        )
        self.assertEqual(autopilot_tick.status_code, 200)

        action_resp = self.client.post(
            f"/api/v1/sim/towns/{town_id}/agents/me/action",
            headers={"Authorization": f"Bearer {agent['api_key']}"},
            json={
                "planning_scope": "short_action",
                "dx": -1,
                "dy": 0,
                "goal_reason": "stuck-metric-check",
                "action_ttl_steps": 12,
                "interrupt_on_social": False,
            },
        )
        self.assertEqual(action_resp.status_code, 200)

        external_tick = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 12, "planning_scope": "short_action", "control_mode": "external"},
        )
        self.assertEqual(external_tick.status_code, 200)

        status_resp = self.client.get(f"/api/v1/sim/towns/{town_id}/runtime/status")
        self.assertEqual(status_resp.status_code, 200)
        metrics = status_resp.json()["tick_metrics_24h"]
        self.assertGreaterEqual(int(metrics["total_batches"]), 2)
        self.assertGreaterEqual(int(metrics["decision_events"]), 1)
        self.assertGreaterEqual(int(metrics["heuristic_fallback_events"]), 1)
        self.assertGreaterEqual(int(metrics["stuck_action_events"]), 1)
        self.assertIn("autonomous_batches_per_hour", metrics)
        self.assertIn("escalation_rate", metrics)

    def test_conversation_wrap_up_summary_persists_in_external_mode(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)
        agent = self._register_and_join(town_id, "WrapAgent", "owner-wrap-a")
        peer = self._register_and_join(town_id, "WrapPeer", "owner-wrap-b")

        submit = self.client.post(
            f"/api/v1/sim/towns/{town_id}/agents/me/action",
            headers={"Authorization": f"Bearer {agent['api_key']}"},
            json={
                "planning_scope": "short_action",
                "socials": [
                    {
                        "target_agent_id": peer["id"],
                        "message": "Quick sync before we split up.",
                        "transcript": [
                            ["WrapAgent", "Quick sync before we split up."],
                            ["WrapPeer", "Sure, talk soon."],
                        ],
                    }
                ],
                "action_ttl_steps": 40,
                "interrupt_on_social": True,
            },
        )
        self.assertEqual(submit.status_code, 200)

        found_social = False
        for _ in range(60):
            social_tick = self.client.post(
                f"/api/v1/sim/towns/{town_id}/tick",
                json={"steps": 1, "planning_scope": "short_action", "control_mode": "external"},
            )
            self.assertEqual(social_tick.status_code, 200)
            if any(event.get("source") == "external_action" for event in social_tick.json()["tick"]["social_events"]):
                found_social = True
                break
        self.assertTrue(found_social)

        drain_tick = self.client.post(
            f"/api/v1/sim/towns/{town_id}/tick",
            json={"steps": 5, "planning_scope": "short_action", "control_mode": "external"},
        )
        self.assertEqual(drain_tick.status_code, 200)

        memory_resp = self.client.get(
            f"/api/v1/sim/towns/{town_id}/agents/{agent['id']}/memory",
            params={"limit": 200},
        )
        self.assertEqual(memory_resp.status_code, 200)
        snapshot = memory_resp.json()["snapshot"]["agent"]["memory"]
        wrap_nodes = [node for node in snapshot["nodes"] if node.get("predicate") == "conversation_wrap_up"]
        self.assertGreaterEqual(len(wrap_nodes), 1)
        buffer = snapshot["scratch"]["chatting_with_buffer"]
        self.assertIn(peer["name"], buffer)

    def test_dead_letter_endpoint_available(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)
        self._register_and_join(town_id, "OpsAgent", "owner-ops")

        dead_letters = self.client.get(
            f"/api/v1/sim/towns/{town_id}/dead-letters",
            params={"limit": 20},
        )
        self.assertEqual(dead_letters.status_code, 200)
        self.assertEqual(dead_letters.json()["town_id"], town_id)
        self.assertIn("items", dead_letters.json())
        self.assertIn("count", dead_letters.json())


if __name__ == "__main__":
    unittest.main()
