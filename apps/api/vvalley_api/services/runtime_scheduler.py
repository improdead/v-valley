"""Background scheduler for autonomous town runtime ticking."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import logging
import threading
import time
import uuid

from packages.vvalley_core.maps.map_utils import load_map
from packages.vvalley_core.sim.cognition import CognitionPlanner
from packages.vvalley_core.sim.runner import tick_simulation

from ..storage.agents import list_active_agents_in_town
from ..storage.llm_control import get_policy as get_llm_policy
from ..storage.llm_control import insert_call_log
from ..storage.map_versions import get_active_version
from ..storage.runtime_control import (
    claim_town_lease,
    complete_tick_batch,
    list_agent_autonomy,
    list_enabled_towns,
    record_dead_letter,
    record_town_tick,
    release_town_lease,
    reserve_tick_batch,
)
from .interaction_sink import ingest_tick_outcomes


logger = logging.getLogger("vvalley_api.runtime_scheduler")
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _resolve_workspace_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = (WORKSPACE_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if candidate != WORKSPACE_ROOT and WORKSPACE_ROOT not in candidate.parents:
        raise ValueError("Path must be within workspace")
    return candidate


class TownRuntimeScheduler:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._tick_lock = threading.Lock()
        self._planner = CognitionPlanner(policy_lookup=get_llm_policy, log_sink=insert_call_log)
        self._last_loop_started_at: str | None = None
        self._last_loop_finished_at: str | None = None
        self._last_error: str | None = None
        self._scheduler_instance_id = f"scheduler-{uuid.uuid4().hex[:12]}"

    def start(self) -> bool:
        with self._state_lock:
            if self._thread and self._thread.is_alive():
                return False
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._run_loop,
                name="vvalley-town-runtime-scheduler",
                daemon=True,
            )
            thread.start()
            self._thread = thread
            logger.info("[RUNTIME] Background town scheduler started")
            return True

    def stop(self, *, join_timeout_seconds: float = 3.0) -> bool:
        with self._state_lock:
            thread = self._thread
            if not thread:
                return False
            self._stop_event.set()
        thread.join(timeout=max(0.1, float(join_timeout_seconds)))
        with self._state_lock:
            if self._thread is thread:
                self._thread = None
        logger.info("[RUNTIME] Background town scheduler stopped")
        return True

    def status(self) -> dict[str, object]:
        with self._state_lock:
            running = bool(self._thread and self._thread.is_alive())
            thread_name = self._thread.name if self._thread else None
        return {
            "running": running,
            "thread_name": thread_name,
            "instance_id": self._scheduler_instance_id,
            "last_loop_started_at": self._last_loop_started_at,
            "last_loop_finished_at": self._last_loop_finished_at,
            "last_error": self._last_error,
        }

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            started = _utc_now()
            self._last_loop_started_at = started.isoformat().replace("+00:00", "Z")
            try:
                controls = list_enabled_towns()
                for control in controls:
                    if self._stop_event.is_set():
                        break
                    if not self._town_due(control):
                        continue
                    self._tick_town(control)
            except Exception as exc:
                self._last_error = f"{exc.__class__.__name__}: {exc}"
                logger.exception("[RUNTIME] Scheduler loop error: %s", exc)
            self._last_loop_finished_at = _utc_now().isoformat().replace("+00:00", "Z")
            self._stop_event.wait(1.0)

    @staticmethod
    def _town_due(control: dict[str, object]) -> bool:
        if not bool(control.get("enabled", False)):
            return False
        last_tick = _parse_utc(str(control.get("last_tick_at") or ""))
        if last_tick is None:
            return True
        try:
            interval_seconds = int(control.get("tick_interval_seconds") or 60)
        except Exception:
            interval_seconds = 60
        interval_seconds = max(1, min(3600, interval_seconds))
        elapsed = (_utc_now() - last_tick).total_seconds()
        return elapsed >= float(interval_seconds)

    def _tick_town(self, control: dict[str, object]) -> None:
        town_id = str(control.get("town_id") or "").strip()
        if not town_id:
            return
        with self._tick_lock:
            lease_ttl_seconds = max(10, min(300, int(control.get("tick_interval_seconds") or 60) * 2))
            claimed = claim_town_lease(
                town_id=town_id,
                lease_owner=self._scheduler_instance_id,
                lease_ttl_seconds=lease_ttl_seconds,
            )
            if not claimed:
                return
            batch_key = f"auto:{self._scheduler_instance_id}:{int(time.time())}"
            try:
                batch_status = reserve_tick_batch(town_id=town_id, batch_key=batch_key, step_before=None)
                if batch_status.get("state") == "existing":
                    status = str(batch_status.get("status") or "")
                    if status in {"pending", "completed"}:
                        return
                active = get_active_version(town_id)
                if not active:
                    raise RuntimeError(f"No active map version for town '{town_id}'")
                map_path = _resolve_workspace_path(str(active.get("map_json_path") or ""))
                if not map_path.exists():
                    raise RuntimeError(f"Active map file not found for town '{town_id}': {map_path}")
                map_data = load_map(map_path)
                members = list_active_agents_in_town(town_id)
                autonomy_map = list_agent_autonomy(agent_ids=[str(member["agent_id"]) for member in members])
                planning_scope = str(control.get("planning_scope") or "short_action")
                control_mode = str(control.get("control_mode") or "hybrid")
                try:
                    steps = int(control.get("steps_per_tick") or 1)
                except Exception:
                    steps = 1
                steps = max(1, min(240, steps))

                ticked = tick_simulation(
                    town_id=town_id,
                    active_version=active,
                    map_data=map_data,
                    members=members,
                    steps=steps,
                    planning_scope=planning_scope,
                    control_mode=control_mode,
                    planner=self._planner.plan_move,
                    agent_autonomy=autonomy_map,
                    cognition_planner=self._planner,
                )
                try:
                    ingest_tick_outcomes(
                        town_id=town_id, members=members, tick_result=ticked
                    )
                except Exception as sink_exc:
                    record_dead_letter(
                        town_id=town_id,
                        stage="interaction_sink_runtime_scheduler",
                        payload={"batch_key": batch_key},
                        error=f"{sink_exc.__class__.__name__}: {sink_exc}",
                    )
                complete_tick_batch(
                    town_id=town_id,
                    batch_key=batch_key,
                    success=True,
                    step_after=int(ticked.get("step") or 0),
                    result=ticked,
                    error=None,
                )
                record_town_tick(town_id=town_id, success=True, error=None)
            except Exception as exc:
                message = f"{exc.__class__.__name__}: {exc}"
                logger.warning("[RUNTIME] Tick failed for town '%s': %s", town_id, message)
                try:
                    complete_tick_batch(
                        town_id=town_id,
                        batch_key=batch_key,
                        success=False,
                        step_after=None,
                        result=None,
                        error=message,
                    )
                except Exception:
                    pass
                record_town_tick(town_id=town_id, success=False, error=message)
                record_dead_letter(
                    town_id=town_id,
                    stage="runtime_scheduler_tick",
                    payload={"batch_key": batch_key},
                    error=message,
                )
            finally:
                release_town_lease(town_id=town_id, lease_owner=self._scheduler_instance_id)


_SCHEDULER = TownRuntimeScheduler()


def start_town_runtime_scheduler() -> bool:
    return _SCHEDULER.start()


def stop_town_runtime_scheduler() -> bool:
    return _SCHEDULER.stop()


def town_runtime_scheduler_status() -> dict[str, object]:
    return _SCHEDULER.status()
