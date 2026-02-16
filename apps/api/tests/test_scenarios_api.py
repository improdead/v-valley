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
TEST_DB_PATH = Path(TEST_DB_DIR.name) / "test_scenarios_vvalley.db"
os.environ["VVALLEY_DB_PATH"] = str(TEST_DB_PATH)

from apps.api.vvalley_api.main import app
from apps.api.vvalley_api.routers.agents import reset_rate_limiter_for_tests as reset_rate_limiter
from apps.api.vvalley_api.routers.sim import reset_sim_rate_limiters_for_tests as reset_sim_limiters
from apps.api.vvalley_api.routers.dm import reset_dm_rate_limiter_for_tests as reset_dm_limiter
from apps.api.vvalley_api.storage.agents import reset_backend_cache_for_tests as reset_agents_backend
from apps.api.vvalley_api.storage.interaction_hub import reset_backend_cache_for_tests as reset_interaction_backend
from apps.api.vvalley_api.storage.llm_control import reset_backend_cache_for_tests as reset_llm_backend
from apps.api.vvalley_api.storage.map_versions import reset_backend_cache_for_tests as reset_maps_backend
from apps.api.vvalley_api.storage.runtime_control import reset_backend_cache_for_tests as reset_runtime_backend
from apps.api.vvalley_api.storage.scenarios import reset_backend_cache_for_tests as reset_scenarios_backend
from packages.vvalley_core.sim.runner import reset_simulation_states_for_tests


class ScenariosApiTests(unittest.TestCase):
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
        reset_sim_limiters()
        reset_dm_limiter()
        reset_simulation_states_for_tests()
        self.client = TestClient(app)

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

    def test_werewolf_queue_match_and_cleanup(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)

        roster = [
            self._register_and_join(town_id, f"Wolf-{i}", f"owner-{i}")
            for i in range(6)
        ]

        defs = self.client.get("/api/v1/scenarios")
        self.assertEqual(defs.status_code, 200)
        keys = {item["scenario_key"] for item in defs.json().get("scenarios", [])}
        self.assertIn("werewolf_6p", keys)

        for agent in roster:
            resp = self.client.post(
                "/api/v1/scenarios/werewolf_6p/queue/join",
                headers={"Authorization": f"Bearer {agent['api_key']}"},
            )
            self.assertEqual(resp.status_code, 200)

        active = self.client.get(f"/api/v1/scenarios/towns/{town_id}/active")
        self.assertEqual(active.status_code, 200)
        self.assertGreaterEqual(active.json()["count"], 1)
        servers = self.client.get("/api/v1/scenarios/servers")
        self.assertEqual(servers.status_code, 200)
        self.assertGreaterEqual(servers.json().get("count", 0), 1)

        state = self.client.get(f"/api/v1/sim/towns/{town_id}/state")
        self.assertEqual(state.status_code, 200)
        npcs = state.json().get("state", {}).get("npcs", [])
        self.assertTrue(any("scenario" in npc for npc in npcs))
        self.assertTrue(any(str(npc.get("status") or "").startswith("scenario_") for npc in npcs if npc.get("scenario")))

        for _ in range(25):
            tick = self.client.post(f"/api/v1/sim/towns/{town_id}/tick", json={"steps": 1})
            self.assertEqual(tick.status_code, 200)
            current = self.client.get(f"/api/v1/scenarios/towns/{town_id}/active")
            self.assertEqual(current.status_code, 200)
            if current.json()["count"] == 0:
                break

        final_active = self.client.get(f"/api/v1/scenarios/towns/{town_id}/active")
        self.assertEqual(final_active.status_code, 200)
        self.assertEqual(final_active.json()["count"], 0)
        servers_after = self.client.get("/api/v1/scenarios/servers")
        self.assertEqual(servers_after.status_code, 200)
        self.assertEqual(servers_after.json().get("count"), 0)

        settled_state = self.client.get(f"/api/v1/sim/towns/{town_id}/state")
        self.assertEqual(settled_state.status_code, 200)
        settled_npcs = settled_state.json().get("state", {}).get("npcs", [])
        self.assertGreaterEqual(len(settled_npcs), 1)
        self.assertTrue(all(not str(npc.get("status") or "").startswith("scenario_") for npc in settled_npcs))
        self.assertTrue(all("scenario" not in npc for npc in settled_npcs))

        me_match = self.client.get(
            "/api/v1/scenarios/me/match",
            headers={"Authorization": f"Bearer {roster[0]['api_key']}"},
        )
        self.assertEqual(me_match.status_code, 200)
        self.assertFalse(me_match.json().get("active"))

        ratings = self.client.get(
            "/api/v1/scenarios/me/ratings",
            headers={"Authorization": f"Bearer {roster[0]['api_key']}"},
        )
        self.assertEqual(ratings.status_code, 200)
        scenario_keys = {item["scenario_key"] for item in ratings.json().get("ratings", [])}
        self.assertIn("werewolf_6p", scenario_keys)

    def test_anaconda_buy_in_and_wallet_updates(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)

        roster = [
            self._register_and_join(town_id, f"Card-{i}", f"card-owner-{i}")
            for i in range(3)
        ]

        before_balances = []
        for agent in roster:
            wallet_before = self.client.get(
                "/api/v1/scenarios/me/wallet",
                headers={"Authorization": f"Bearer {agent['api_key']}"},
            )
            self.assertEqual(wallet_before.status_code, 200)
            before_balances.append(int(wallet_before.json()["wallet"]["balance"]))

        for agent in roster:
            resp = self.client.post(
                "/api/v1/scenarios/anaconda_standard/queue/join",
                headers={"Authorization": f"Bearer {agent['api_key']}"},
            )
            self.assertEqual(resp.status_code, 200)

        active = self.client.get(f"/api/v1/scenarios/towns/{town_id}/active")
        self.assertEqual(active.status_code, 200)
        self.assertGreaterEqual(active.json()["count"], 1)

        for _ in range(22):
            tick = self.client.post(f"/api/v1/sim/towns/{town_id}/tick", json={"steps": 1})
            self.assertEqual(tick.status_code, 200)
            current = self.client.get(f"/api/v1/scenarios/towns/{town_id}/active")
            self.assertEqual(current.status_code, 200)
            if current.json()["count"] == 0:
                break

        after_balances = []
        for agent in roster:
            wallet_after = self.client.get(
                "/api/v1/scenarios/me/wallet",
                headers={"Authorization": f"Bearer {agent['api_key']}"},
            )
            self.assertEqual(wallet_after.status_code, 200)
            after_balances.append(int(wallet_after.json()["wallet"]["balance"]))

        self.assertTrue(any(after > before for before, after in zip(before_balances, after_balances)))
        self.assertTrue(any(after < before for before, after in zip(before_balances, after_balances)))

    def test_anaconda_forfeit_endpoint_resolves_match(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)
        roster = [
            self._register_and_join(town_id, f"Forfeit-{i}", f"forfeit-owner-{i}")
            for i in range(3)
        ]

        for agent in roster:
            resp = self.client.post(
                "/api/v1/scenarios/anaconda_standard/queue/join",
                headers={"Authorization": f"Bearer {agent['api_key']}"},
            )
            self.assertEqual(resp.status_code, 200)

        me_match = self.client.get(
            "/api/v1/scenarios/me/match",
            headers={"Authorization": f"Bearer {roster[0]['api_key']}"},
        )
        self.assertEqual(me_match.status_code, 200)
        match_id = str(me_match.json().get("match", {}).get("match_id") or "")
        self.assertTrue(match_id)

        f1 = self.client.post(
            f"/api/v1/scenarios/matches/{match_id}/forfeit",
            headers={"Authorization": f"Bearer {roster[0]['api_key']}"},
        )
        self.assertEqual(f1.status_code, 200)

        f2 = self.client.post(
            f"/api/v1/scenarios/matches/{match_id}/forfeit",
            headers={"Authorization": f"Bearer {roster[1]['api_key']}"},
        )
        self.assertEqual(f2.status_code, 200)

        detail = self.client.get(f"/api/v1/scenarios/matches/{match_id}")
        self.assertEqual(detail.status_code, 200)
        status = str(detail.json().get("match", {}).get("status") or "")
        self.assertIn(status, {"resolved", "cancelled"})

        active = self.client.get(f"/api/v1/scenarios/towns/{town_id}/active")
        self.assertEqual(active.status_code, 200)
        self.assertEqual(active.json().get("count"), 0)

    def test_anaconda_cancel_endpoint_refunds_buy_in_during_warmup(self) -> None:
        town_id = f"town-{uuid.uuid4().hex[:8]}"
        self._publish_town(town_id)
        roster = [
            self._register_and_join(town_id, f"Cancel-{i}", f"cancel-owner-{i}")
            for i in range(3)
        ]

        before_wallets = []
        for agent in roster:
            wallet_before = self.client.get(
                "/api/v1/scenarios/me/wallet",
                headers={"Authorization": f"Bearer {agent['api_key']}"},
            )
            self.assertEqual(wallet_before.status_code, 200)
            before_wallets.append(int(wallet_before.json()["wallet"]["balance"]))

        for agent in roster:
            resp = self.client.post(
                "/api/v1/scenarios/anaconda_standard/queue/join",
                headers={"Authorization": f"Bearer {agent['api_key']}"},
            )
            self.assertEqual(resp.status_code, 200)

        me_match = self.client.get(
            "/api/v1/scenarios/me/match",
            headers={"Authorization": f"Bearer {roster[0]['api_key']}"},
        )
        self.assertEqual(me_match.status_code, 200)
        match_id = str(me_match.json().get("match", {}).get("match_id") or "")
        self.assertTrue(match_id)

        cancelled = self.client.post(
            f"/api/v1/scenarios/matches/{match_id}/cancel",
            headers={"Authorization": f"Bearer {roster[0]['api_key']}"},
        )
        self.assertEqual(cancelled.status_code, 200)
        self.assertEqual(str(cancelled.json().get("match", {}).get("status") or ""), "cancelled")

        after_wallets = []
        for agent in roster:
            wallet_after = self.client.get(
                "/api/v1/scenarios/me/wallet",
                headers={"Authorization": f"Bearer {agent['api_key']}"},
            )
            self.assertEqual(wallet_after.status_code, 200)
            after_wallets.append(int(wallet_after.json()["wallet"]["balance"]))

        self.assertEqual(before_wallets, after_wallets)


if __name__ == "__main__":
    unittest.main()
