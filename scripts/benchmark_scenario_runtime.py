#!/usr/bin/env python3
"""Benchmark scenario matchmaking/runtime flow end-to-end via FastAPI TestClient."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    points = sorted(values)
    if len(points) == 1:
        return points[0]
    pos = max(0.0, min(1.0, q)) * (len(points) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return points[low]
    frac = pos - low
    return points[low] * (1.0 - frac) + points[high] * frac


def summarize_latencies(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(values),
        "mean_ms": round(statistics.fmean(values), 3),
        "p50_ms": round(percentile(values, 0.5), 3),
        "p95_ms": round(percentile(values, 0.95), 3),
        "max_ms": round(max(values), 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scenario runtime benchmark")
    parser.add_argument("--agents", type=int, default=25, help="Number of agents to create")
    parser.add_argument("--max-ticks", type=int, default=120, help="Tick cap before abort")
    parser.add_argument(
        "--output",
        default="data/perf/scenario_runtime_benchmark.json",
        help="JSON output path",
    )
    parser.add_argument(
        "--db-path",
        default="",
        help="Optional SQLite DB path for this run (defaults to temp file)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Keep API/service logs enabled during the benchmark run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.agents < 6:
        raise SystemExit("--agents must be >= 6")
    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("vvalley_api").setLevel(logging.WARNING)
        logging.getLogger("vvalley_core").setLevel(logging.WARNING)

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.db_path:
        db_path = Path(args.db_path).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory(dir=ROOT)
        db_path = Path(temp_dir.name) / "benchmark_vvalley.db"

    os.environ["VVALLEY_DB_PATH"] = str(db_path)

    from fastapi.testclient import TestClient

    from apps.api.vvalley_api.main import app
    from apps.api.vvalley_api.routers.agents import reset_rate_limiter_for_tests as reset_rate_limiter
    from apps.api.vvalley_api.routers.dm import reset_dm_rate_limiter_for_tests as reset_dm_limiter
    from apps.api.vvalley_api.routers.sim import reset_sim_rate_limiters_for_tests as reset_sim_limiters
    from apps.api.vvalley_api.services.runtime_scheduler import stop_town_runtime_scheduler
    from apps.api.vvalley_api.storage.agents import reset_backend_cache_for_tests as reset_agents_backend
    from apps.api.vvalley_api.storage.interaction_hub import (
        reset_backend_cache_for_tests as reset_interaction_backend,
    )
    from apps.api.vvalley_api.storage.llm_control import reset_backend_cache_for_tests as reset_llm_backend
    from apps.api.vvalley_api.storage.map_versions import reset_backend_cache_for_tests as reset_maps_backend
    from apps.api.vvalley_api.storage.runtime_control import reset_backend_cache_for_tests as reset_runtime_backend
    from apps.api.vvalley_api.storage.scenarios import reset_backend_cache_for_tests as reset_scenarios_backend
    from packages.vvalley_core.sim.runner import reset_simulation_states_for_tests

    stop_town_runtime_scheduler()
    if db_path.exists():
        db_path.unlink()
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

    client = TestClient(app)
    town_id = f"town-bench-{uuid.uuid4().hex[:8]}"

    timings_ms: dict[str, list[float]] = {
        "register_join": [],
        "queue_join": [],
        "spectate": [],
        "tick": [],
        "state": [],
        "active_query": [],
    }

    def timed(method: str, path: str, **kwargs: Any):
        start = time.perf_counter()
        resp = client.request(method=method, url=path, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return resp, elapsed_ms

    publish = client.post(
        "/api/v1/maps/publish-version",
        json={
            "town_id": town_id,
            "map_path": "assets/templates/starter_town/map.json",
            "map_name": "benchmark",
        },
    )
    if publish.status_code != 200:
        raise RuntimeError(f"Failed to publish town map: {publish.status_code} {publish.text[:300]}")

    roster: list[dict[str, str]] = []
    for i in range(args.agents):
        r0, dt0 = timed(
            "POST",
            "/api/v1/agents/register",
            json={"name": f"Bench-{i}", "owner_handle": f"bench-owner-{i}", "auto_claim": True},
        )
        if r0.status_code != 200:
            raise RuntimeError(f"register failed ({r0.status_code}): {r0.text[:240]}")
        agent = r0.json()["agent"]
        r1, dt1 = timed(
            "POST",
            "/api/v1/agents/me/join-town",
            json={"town_id": town_id},
            headers={"Authorization": f"Bearer {agent['api_key']}"},
        )
        if r1.status_code != 200:
            raise RuntimeError(f"join failed ({r1.status_code}): {r1.text[:240]}")
        timings_ms["register_join"].append(dt0 + dt1)
        roster.append(agent)

    werewolf_players = roster[: min(12, len(roster))]
    anaconda_players = roster[min(12, len(roster)) : min(21, len(roster))]

    for agent in werewolf_players:
        resp, dt = timed(
            "POST",
            "/api/v1/scenarios/werewolf_6p/queue/join",
            headers={"Authorization": f"Bearer {agent['api_key']}"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"werewolf queue failed ({resp.status_code}): {resp.text[:240]}")
        timings_ms["queue_join"].append(dt)

    for agent in anaconda_players:
        resp, dt = timed(
            "POST",
            "/api/v1/scenarios/anaconda_standard/queue/join",
            headers={"Authorization": f"Bearer {agent['api_key']}"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"anaconda queue failed ({resp.status_code}): {resp.text[:240]}")
        timings_ms["queue_join"].append(dt)

    active_ids: list[str] = []
    active_resp, dt_active = timed("GET", f"/api/v1/scenarios/towns/{town_id}/active")
    timings_ms["active_query"].append(dt_active)
    if active_resp.status_code == 200:
        active_ids = [str(item.get("match_id") or "") for item in active_resp.json().get("matches", []) if item.get("match_id")]

    for _ in range(3):
        for match_id in active_ids:
            spectate_resp, dt_s = timed("GET", f"/api/v1/scenarios/matches/{match_id}/spectate")
            if spectate_resp.status_code != 200:
                continue
            timings_ms["spectate"].append(dt_s)

    ticks_run = 0
    resolved_early = False
    while ticks_run < int(args.max_ticks):
        tick_resp, dt_tick = timed("POST", f"/api/v1/sim/towns/{town_id}/tick", json={"steps": 1})
        if tick_resp.status_code != 200:
            raise RuntimeError(f"tick failed ({tick_resp.status_code}): {tick_resp.text[:240]}")
        timings_ms["tick"].append(dt_tick)
        ticks_run += 1

        if ticks_run % 3 == 0:
            state_resp, dt_state = timed("GET", f"/api/v1/sim/towns/{town_id}/state")
            if state_resp.status_code == 200:
                timings_ms["state"].append(dt_state)

        active_resp, dt_active = timed("GET", f"/api/v1/scenarios/towns/{town_id}/active")
        timings_ms["active_query"].append(dt_active)
        if active_resp.status_code != 200:
            continue
        if int(active_resp.json().get("count") or 0) == 0:
            resolved_early = True
            break

    match_resp = client.get("/api/v1/scenarios/matches", params={"town_id": town_id, "limit": 200})
    matches = match_resp.json().get("matches", []) if match_resp.status_code == 200 else []
    by_status: dict[str, int] = {}
    for match in matches:
        status = str(match.get("status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1

    servers_resp = client.get("/api/v1/scenarios/servers", params={"town_id": town_id})
    live_servers = int(servers_resp.json().get("count") or 0) if servers_resp.status_code == 200 else -1

    output = {
        "town_id": town_id,
        "db_path": str(db_path),
        "agents": int(args.agents),
        "ticks_run": ticks_run,
        "max_ticks": int(args.max_ticks),
        "resolved_before_cap": bool(resolved_early),
        "match_status_counts": by_status,
        "live_servers_after_run": live_servers,
        "latency_ms": {k: summarize_latencies(v) for k, v in timings_ms.items()},
        "timestamp_utc_epoch": time.time(),
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"\nWrote benchmark report: {output_path}")

    if temp_dir is not None:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
