"""Persistent runtime control and per-agent autonomy contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
import json
import os
import sqlite3
import threading
import uuid


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
MIGRATIONS_DIR = WORKSPACE_ROOT / "packages" / "vvalley_core" / "db" / "migrations"

ALLOWED_AUTONOMY_MODES = {"manual", "delegated", "autonomous"}
ALLOWED_SCOPES = {"short_action", "daily_plan", "long_term_plan"}
ALLOWED_CONTROL_MODES = {"external", "hybrid", "autopilot"}
DEFAULT_ALLOWED_SCOPES = ["short_action", "daily_plan", "long_term_plan"]
DEFAULT_AUTONOMY_MODE = "manual"
DEFAULT_RUNTIME_CONTROL = {
    "enabled": False,
    "tick_interval_seconds": 60,
    "steps_per_tick": 1,
    "planning_scope": "short_action",
    "control_mode": "hybrid",
    "lease_owner": None,
    "lease_expires_at": None,
    "last_tick_at": None,
    "last_tick_success": None,
    "last_error": None,
}


def _now_utc_sqlite() -> str:
    return "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


def _parse_utc(value: Any) -> datetime | None:
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


def _result_payload(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _decision_metrics_from_batches(
    *,
    rows: list[dict[str, Any]],
    window_hours: int,
) -> dict[str, Any]:
    decision_events = 0
    heuristic_fallback_events = 0
    autonomy_limit_events = 0
    stuck_action_events = 0
    external_action_events = 0
    delegated_autopilot_events = 0
    autonomous_batches = 0

    for row in rows:
        payload = _result_payload(row.get("result_json"))
        if not payload:
            continue
        batch_autonomous = str(payload.get("control_mode") or "").strip().lower() in {"autopilot", "hybrid"}
        events = payload.get("events")
        if not isinstance(events, list):
            if batch_autonomous:
                autonomous_batches += 1
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            decision = event.get("decision")
            if not isinstance(decision, dict):
                continue
            decision_events += 1
            route = str(decision.get("route") or "").strip().lower()
            used_tier = str(decision.get("used_tier") or "").strip().lower()
            if route.startswith("external_action"):
                external_action_events += 1
            if used_tier == "heuristic" or route in {"heuristic", "planner_error", "needs_prefilter"}:
                heuristic_fallback_events += 1
            if route == "awaiting_user_agent_autonomy_limit":
                autonomy_limit_events += 1
            if bool(decision.get("stuck")) or route.startswith("external_action_stuck"):
                stuck_action_events += 1
            if bool(decision.get("delegated_autopilot")):
                delegated_autopilot_events += 1
                batch_autonomous = True
        if batch_autonomous:
            autonomous_batches += 1

    decision_rate_denominator = float(decision_events) if decision_events > 0 else 0.0
    external_rate_denominator = float(external_action_events) if external_action_events > 0 else 0.0
    per_hour = float(autonomous_batches) / float(max(1, int(window_hours)))
    escalation_rate = (
        float(autonomy_limit_events) / decision_rate_denominator
        if decision_rate_denominator > 0.0
        else None
    )

    return {
        "decision_events": decision_events,
        "autonomous_batches": autonomous_batches,
        "autonomous_batches_per_hour": per_hour,
        "delegated_autopilot_events": delegated_autopilot_events,
        "heuristic_fallback_events": heuristic_fallback_events,
        "heuristic_fallback_rate": (
            float(heuristic_fallback_events) / decision_rate_denominator
            if decision_rate_denominator > 0.0
            else None
        ),
        "autonomy_limit_events": autonomy_limit_events,
        "escalation_events": autonomy_limit_events,
        "escalation_rate": escalation_rate,
        "stuck_action_events": stuck_action_events,
        "stuck_action_rate": (
            float(stuck_action_events) / external_rate_denominator
            if external_rate_denominator > 0.0
            else None
        ),
        "external_action_events": external_action_events,
    }


def _normalize_scope(value: str) -> str:
    lowered = str(value or "").strip().lower()
    if lowered in {"short_action", "short"}:
        return "short_action"
    if lowered in {"daily_plan", "daily"}:
        return "daily_plan"
    if lowered in {"long_term_plan", "long_term", "long"}:
        return "long_term_plan"
    return "short_action"


def _normalize_scopes(values: Any) -> list[str]:
    normalized: list[str] = []
    if isinstance(values, list):
        for value in values:
            scope = _normalize_scope(str(value))
            if scope not in ALLOWED_SCOPES:
                continue
            if scope not in normalized:
                normalized.append(scope)
    if not normalized:
        normalized = list(DEFAULT_ALLOWED_SCOPES)
    return normalized


def _normalize_mode(value: str) -> str:
    mode = str(value or DEFAULT_AUTONOMY_MODE).strip().lower()
    if mode not in ALLOWED_AUTONOMY_MODES:
        return DEFAULT_AUTONOMY_MODE
    return mode


def _normalize_control_mode(value: str) -> str:
    mode = str(value or "hybrid").strip().lower()
    if mode not in ALLOWED_CONTROL_MODES:
        return "hybrid"
    return mode


def _normalize_contract_row(agent_id: str, row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "agent_id": str(agent_id),
            "mode": DEFAULT_AUTONOMY_MODE,
            "allowed_scopes": list(DEFAULT_ALLOWED_SCOPES),
            "max_autonomous_ticks": 0,
            "escalation_policy": {},
            "updated_at": None,
        }
    raw_scopes = row.get("allowed_scopes_json")
    scopes: list[str] = []
    if isinstance(raw_scopes, str):
        try:
            scopes = _normalize_scopes(json.loads(raw_scopes))
        except Exception:
            scopes = list(DEFAULT_ALLOWED_SCOPES)
    elif isinstance(raw_scopes, list):
        scopes = _normalize_scopes(raw_scopes)
    else:
        scopes = list(DEFAULT_ALLOWED_SCOPES)

    escalation_raw = row.get("escalation_policy_json")
    escalation_policy: dict[str, Any] = {}
    if isinstance(escalation_raw, str):
        try:
            parsed = json.loads(escalation_raw)
            if isinstance(parsed, dict):
                escalation_policy = parsed
        except Exception:
            escalation_policy = {}
    elif isinstance(escalation_raw, dict):
        escalation_policy = dict(escalation_raw)

    try:
        max_ticks = int(row.get("max_autonomous_ticks") or 0)
    except Exception:
        max_ticks = 0
    if max_ticks < 0:
        max_ticks = 0

    return {
        "agent_id": str(agent_id),
        "mode": _normalize_mode(str(row.get("mode") or DEFAULT_AUTONOMY_MODE)),
        "allowed_scopes": scopes,
        "max_autonomous_ticks": max_ticks,
        "escalation_policy": escalation_policy,
        "updated_at": row.get("updated_at"),
    }


def _normalize_runtime_row(town_id: str, row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "town_id": str(town_id),
            **DEFAULT_RUNTIME_CONTROL,
            "updated_at": None,
        }
    try:
        tick_interval_seconds = int(row.get("tick_interval_seconds") or 60)
    except Exception:
        tick_interval_seconds = 60
    tick_interval_seconds = max(1, min(3600, tick_interval_seconds))

    try:
        steps_per_tick = int(row.get("steps_per_tick") or 1)
    except Exception:
        steps_per_tick = 1
    steps_per_tick = max(1, min(240, steps_per_tick))

    last_success_raw = row.get("last_tick_success")
    if last_success_raw is None:
        last_success = None
    else:
        last_success = bool(last_success_raw)

    return {
        "town_id": str(town_id),
        "enabled": bool(row.get("enabled", False)),
        "tick_interval_seconds": tick_interval_seconds,
        "steps_per_tick": steps_per_tick,
        "planning_scope": _normalize_scope(str(row.get("planning_scope") or "short_action")),
        "control_mode": _normalize_control_mode(str(row.get("control_mode") or "hybrid")),
        "lease_owner": (str(row.get("lease_owner")) if row.get("lease_owner") is not None else None),
        "lease_expires_at": row.get("lease_expires_at"),
        "last_tick_at": row.get("last_tick_at"),
        "last_tick_success": last_success,
        "last_error": row.get("last_error"),
        "updated_at": row.get("updated_at"),
    }


class RuntimeControlStore(ABC):
    @abstractmethod
    def init_db(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_agent_autonomy(self, *, agent_id: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def upsert_agent_autonomy(
        self,
        *,
        agent_id: str,
        mode: str,
        allowed_scopes: list[str],
        max_autonomous_ticks: int,
        escalation_policy: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def list_agent_autonomy(self, *, agent_ids: list[str]) -> dict[str, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_town_runtime(self, *, town_id: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def upsert_town_runtime(
        self,
        *,
        town_id: str,
        enabled: Optional[bool],
        tick_interval_seconds: Optional[int],
        steps_per_tick: Optional[int],
        planning_scope: Optional[str],
        control_mode: Optional[str],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_enabled_towns(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def record_town_tick(self, *, town_id: str, success: bool, error: Optional[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def claim_town_lease(
        self,
        *,
        town_id: str,
        lease_owner: str,
        lease_ttl_seconds: int,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def release_town_lease(self, *, town_id: str, lease_owner: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def reserve_tick_batch(
        self,
        *,
        town_id: str,
        batch_key: str,
        step_before: Optional[int],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def complete_tick_batch(
        self,
        *,
        town_id: str,
        batch_key: str,
        success: bool,
        step_after: Optional[int],
        result: Optional[dict[str, Any]],
        error: Optional[str],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_tick_metrics(self, *, town_id: str, window_hours: int) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def record_dead_letter(
        self,
        *,
        town_id: Optional[str],
        stage: str,
        payload: Optional[dict[str, Any]],
        error: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_dead_letters(
        self,
        *,
        town_id: Optional[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class SQLiteRuntimeControlStore(RuntimeControlStore):
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._initialized = False
        self._local = threading.local()

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        self._local.conn = conn
        return conn

    def init_db(self) -> None:
        if self._initialized:
            return
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS agent_autonomy_contracts (
                  agent_id TEXT PRIMARY KEY,
                  mode TEXT NOT NULL DEFAULT 'manual',
                  allowed_scopes_json TEXT NOT NULL DEFAULT '["short_action","daily_plan","long_term_plan"]',
                  max_autonomous_ticks INTEGER NOT NULL DEFAULT 0,
                  escalation_policy_json TEXT NOT NULL DEFAULT '{{}}',
                  updated_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()}),
                  FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_autonomy_mode ON agent_autonomy_contracts(mode)"
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS town_runtime_controls (
                  town_id TEXT PRIMARY KEY,
                  enabled INTEGER NOT NULL DEFAULT 0,
                  tick_interval_seconds INTEGER NOT NULL DEFAULT 60,
                  steps_per_tick INTEGER NOT NULL DEFAULT 1,
                  planning_scope TEXT NOT NULL DEFAULT 'short_action',
                  control_mode TEXT NOT NULL DEFAULT 'hybrid',
                  lease_owner TEXT,
                  lease_expires_at TEXT,
                  last_tick_at TEXT,
                  last_tick_success INTEGER,
                  last_error TEXT,
                  updated_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()})
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_town_runtime_controls_enabled ON town_runtime_controls(enabled)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_town_runtime_controls_lease "
                "ON town_runtime_controls(lease_owner, lease_expires_at)"
            )
            try:
                conn.execute("ALTER TABLE town_runtime_controls ADD COLUMN lease_owner TEXT")
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE town_runtime_controls ADD COLUMN lease_expires_at TEXT")
            except Exception:
                pass
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS town_tick_batches (
                  town_id TEXT NOT NULL,
                  batch_key TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'pending',
                  step_before INTEGER,
                  step_after INTEGER,
                  result_json TEXT,
                  error TEXT,
                  created_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()}),
                  completed_at TEXT,
                  PRIMARY KEY (town_id, batch_key)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_town_tick_batches_town_created "
                "ON town_tick_batches(town_id, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_town_tick_batches_town_status_created "
                "ON town_tick_batches(town_id, status, created_at DESC)"
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS interaction_dead_letters (
                  id TEXT PRIMARY KEY,
                  town_id TEXT,
                  stage TEXT NOT NULL,
                  payload_json TEXT NOT NULL DEFAULT '{{}}',
                  error TEXT NOT NULL,
                  created_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()})
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interaction_dead_letters_town_created "
                "ON interaction_dead_letters(town_id, created_at DESC)"
            )
        self._initialized = True

    def get_agent_autonomy(self, *, agent_id: str) -> dict[str, Any]:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agent_autonomy_contracts WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
        return _normalize_contract_row(agent_id, dict(row) if row else None)

    def upsert_agent_autonomy(
        self,
        *,
        agent_id: str,
        mode: str,
        allowed_scopes: list[str],
        max_autonomous_ticks: int,
        escalation_policy: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        self.init_db()
        normalized_mode = _normalize_mode(mode)
        normalized_scopes = _normalize_scopes(allowed_scopes)
        max_ticks = max(0, int(max_autonomous_ticks))
        escalation = escalation_policy if isinstance(escalation_policy, dict) else {}

        with self._connect() as conn:
            exists = conn.execute("SELECT 1 FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if not exists:
                return None
            conn.execute(
                f"""
                INSERT INTO agent_autonomy_contracts
                  (agent_id, mode, allowed_scopes_json, max_autonomous_ticks, escalation_policy_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ({_now_utc_sqlite()}))
                ON CONFLICT(agent_id) DO UPDATE SET
                  mode = excluded.mode,
                  allowed_scopes_json = excluded.allowed_scopes_json,
                  max_autonomous_ticks = excluded.max_autonomous_ticks,
                  escalation_policy_json = excluded.escalation_policy_json,
                  updated_at = ({_now_utc_sqlite()})
                """,
                (
                    agent_id,
                    normalized_mode,
                    json.dumps(normalized_scopes, separators=(",", ":")),
                    max_ticks,
                    json.dumps(escalation, separators=(",", ":")),
                ),
            )
            row = conn.execute(
                "SELECT * FROM agent_autonomy_contracts WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
        return _normalize_contract_row(agent_id, dict(row) if row else None)

    def list_agent_autonomy(self, *, agent_ids: list[str]) -> dict[str, dict[str, Any]]:
        self.init_db()
        normalized_ids = [str(value).strip() for value in agent_ids if str(value).strip()]
        if not normalized_ids:
            return {}
        placeholders = ",".join("?" for _ in normalized_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM agent_autonomy_contracts WHERE agent_id IN ({placeholders})",
                tuple(normalized_ids),
            ).fetchall()
        records: dict[str, dict[str, Any]] = {}
        for row in rows:
            row_dict = dict(row)
            agent_id = str(row_dict.get("agent_id") or "")
            if not agent_id:
                continue
            records[agent_id] = _normalize_contract_row(agent_id, row_dict)
        for agent_id in normalized_ids:
            if agent_id not in records:
                records[agent_id] = _normalize_contract_row(agent_id, None)
        return records

    def get_town_runtime(self, *, town_id: str) -> dict[str, Any]:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM town_runtime_controls WHERE town_id = ?",
                (town_id,),
            ).fetchone()
        return _normalize_runtime_row(town_id, dict(row) if row else None)

    def upsert_town_runtime(
        self,
        *,
        town_id: str,
        enabled: Optional[bool],
        tick_interval_seconds: Optional[int],
        steps_per_tick: Optional[int],
        planning_scope: Optional[str],
        control_mode: Optional[str],
    ) -> dict[str, Any]:
        self.init_db()
        current = self.get_town_runtime(town_id=town_id)
        next_enabled = bool(current["enabled"] if enabled is None else enabled)
        next_tick_interval = (
            int(current["tick_interval_seconds"])
            if tick_interval_seconds is None
            else max(1, min(3600, int(tick_interval_seconds)))
        )
        next_steps = (
            int(current["steps_per_tick"])
            if steps_per_tick is None
            else max(1, min(240, int(steps_per_tick)))
        )
        next_scope = _normalize_scope(current["planning_scope"] if planning_scope is None else planning_scope)
        next_control_mode = _normalize_control_mode(current["control_mode"] if control_mode is None else control_mode)

        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO town_runtime_controls
                  (town_id, enabled, tick_interval_seconds, steps_per_tick, planning_scope, control_mode, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ({_now_utc_sqlite()}))
                ON CONFLICT(town_id) DO UPDATE SET
                  enabled = excluded.enabled,
                  tick_interval_seconds = excluded.tick_interval_seconds,
                  steps_per_tick = excluded.steps_per_tick,
                  planning_scope = excluded.planning_scope,
                  control_mode = excluded.control_mode,
                  updated_at = ({_now_utc_sqlite()})
                """,
                (
                    town_id,
                    1 if next_enabled else 0,
                    next_tick_interval,
                    next_steps,
                    next_scope,
                    next_control_mode,
                ),
            )
            row = conn.execute(
                "SELECT * FROM town_runtime_controls WHERE town_id = ?",
                (town_id,),
            ).fetchone()
        return _normalize_runtime_row(town_id, dict(row) if row else None)

    def list_enabled_towns(self) -> list[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM town_runtime_controls
                WHERE enabled = 1
                ORDER BY town_id ASC
                """
            ).fetchall()
        return [_normalize_runtime_row(str(row["town_id"]), dict(row)) for row in rows]

    def record_town_tick(self, *, town_id: str, success: bool, error: Optional[str]) -> None:
        self.init_db()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO town_runtime_controls
                  (town_id, enabled, tick_interval_seconds, steps_per_tick, planning_scope, control_mode,
                   last_tick_at, last_tick_success, last_error, updated_at)
                VALUES (?, 0, 60, 1, 'short_action', 'hybrid', ({_now_utc_sqlite()}), ?, ?, ({_now_utc_sqlite()}))
                ON CONFLICT(town_id) DO UPDATE SET
                  last_tick_at = ({_now_utc_sqlite()}),
                  last_tick_success = excluded.last_tick_success,
                  last_error = excluded.last_error,
                  updated_at = ({_now_utc_sqlite()})
                """,
                (
                    town_id,
                    1 if success else 0,
                    None if success else str(error or "")[:500],
                ),
            )

    def claim_town_lease(
        self,
        *,
        town_id: str,
        lease_owner: str,
        lease_ttl_seconds: int,
    ) -> bool:
        self.init_db()
        owner = str(lease_owner or "").strip()
        if not owner:
            return False
        ttl = max(1, min(300, int(lease_ttl_seconds)))
        now_iso = _utc_now_iso()
        expires_iso = (_utc_now() + timedelta(seconds=ttl)).isoformat().replace("+00:00", "Z")

        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO town_runtime_controls
                  (town_id, enabled, tick_interval_seconds, steps_per_tick, planning_scope, control_mode,
                   lease_owner, lease_expires_at, updated_at)
                VALUES (?, 0, 60, 1, 'short_action', 'hybrid', NULL, NULL, ({_now_utc_sqlite()}))
                ON CONFLICT(town_id) DO NOTHING
                """,
                (town_id,),
            )
            conn.execute(
                """
                UPDATE town_runtime_controls
                SET lease_owner = ?,
                    lease_expires_at = ?,
                    updated_at = ?
                WHERE town_id = ?
                  AND (
                    lease_owner IS NULL
                    OR lease_owner = ?
                    OR lease_expires_at IS NULL
                    OR lease_expires_at <= ?
                  )
                """,
                (
                    owner,
                    expires_iso,
                    now_iso,
                    town_id,
                    owner,
                    now_iso,
                ),
            )
            row = conn.execute(
                "SELECT lease_owner, lease_expires_at FROM town_runtime_controls WHERE town_id = ?",
                (town_id,),
            ).fetchone()
        if not row:
            return False
        if str(row["lease_owner"] or "") != owner:
            return False
        expires_at = _parse_utc(row["lease_expires_at"])
        if expires_at is None:
            return False
        return expires_at > _utc_now()

    def release_town_lease(self, *, town_id: str, lease_owner: str) -> None:
        self.init_db()
        owner = str(lease_owner or "").strip()
        if not owner:
            return
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE town_runtime_controls
                SET lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE town_id = ? AND lease_owner = ?
                """,
                (
                    _utc_now_iso(),
                    town_id,
                    owner,
                ),
            )

    def reserve_tick_batch(
        self,
        *,
        town_id: str,
        batch_key: str,
        step_before: Optional[int],
    ) -> dict[str, Any]:
        self.init_db()
        key = str(batch_key or "").strip()
        if not key:
            return {"state": "invalid"}
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM town_tick_batches WHERE town_id = ? AND batch_key = ?",
                (town_id, key),
            ).fetchone()
            if row:
                row_dict = dict(row)
                status = str(row_dict.get("status") or "pending")
                result_payload = None
                raw_result = row_dict.get("result_json")
                if isinstance(raw_result, str):
                    try:
                        parsed = json.loads(raw_result)
                        if isinstance(parsed, dict):
                            result_payload = parsed
                    except Exception:
                        result_payload = None
                return {
                    "state": "existing",
                    "status": status,
                    "result": result_payload,
                    "error": row_dict.get("error"),
                }
            conn.execute(
                f"""
                INSERT INTO town_tick_batches
                  (town_id, batch_key, status, step_before, created_at)
                VALUES (?, ?, 'pending', ?, ({_now_utc_sqlite()}))
                """,
                (
                    town_id,
                    key,
                    (int(step_before) if step_before is not None else None),
                ),
            )
        return {"state": "new", "status": "pending", "result": None, "error": None}

    def complete_tick_batch(
        self,
        *,
        town_id: str,
        batch_key: str,
        success: bool,
        step_after: Optional[int],
        result: Optional[dict[str, Any]],
        error: Optional[str],
    ) -> None:
        self.init_db()
        key = str(batch_key or "").strip()
        if not key:
            return
        status = "completed" if success else "failed"
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO town_tick_batches
                  (town_id, batch_key, status, step_after, result_json, error, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ({_now_utc_sqlite()}), ({_now_utc_sqlite()}))
                ON CONFLICT(town_id, batch_key) DO UPDATE SET
                  status = excluded.status,
                  step_after = excluded.step_after,
                  result_json = excluded.result_json,
                  error = excluded.error,
                  completed_at = ({_now_utc_sqlite()})
                """,
                (
                    town_id,
                    key,
                    status,
                    (int(step_after) if step_after is not None else None),
                    json.dumps(result or {}, separators=(",", ":")) if success else None,
                    None if success else str(error or "")[:500],
                ),
            )

    def get_tick_metrics(self, *, town_id: str, window_hours: int) -> dict[str, Any]:
        self.init_db()
        hours = max(1, min(168, int(window_hours)))
        cutoff_iso = (_utc_now() - timedelta(hours=hours)).isoformat().replace("+00:00", "Z")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                  COUNT(*) AS total,
                  SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
                  SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed
                FROM town_tick_batches
                WHERE town_id = ?
                  AND created_at >= ?
                """,
                (
                    town_id,
                    cutoff_iso,
                ),
            ).fetchone()
            detail_rows = conn.execute(
                """
                SELECT result_json
                FROM town_tick_batches
                WHERE town_id = ?
                  AND created_at >= ?
                  AND result_json IS NOT NULL
                """,
                (
                    town_id,
                    cutoff_iso,
                ),
            ).fetchall()
        total = int(row["total"] or 0) if row else 0
        completed = int(row["completed"] or 0) if row else 0
        failed = int(row["failed"] or 0) if row else 0
        decision_metrics = _decision_metrics_from_batches(
            rows=[dict(item) for item in detail_rows],
            window_hours=hours,
        )
        return {
            "town_id": town_id,
            "window_hours": hours,
            "total_batches": total,
            "completed_batches": completed,
            "failed_batches": failed,
            "success_rate": (float(completed) / float(total) if total > 0 else None),
            **decision_metrics,
        }

    def record_dead_letter(
        self,
        *,
        town_id: Optional[str],
        stage: str,
        payload: Optional[dict[str, Any]],
        error: str,
    ) -> None:
        self.init_db()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO interaction_dead_letters
                  (id, town_id, stage, payload_json, error)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    (str(town_id).strip() if town_id else None),
                    str(stage or "unknown")[:120],
                    json.dumps(payload or {}, separators=(",", ":")),
                    str(error or "unknown_error")[:800],
                ),
            )

    def list_dead_letters(
        self,
        *,
        town_id: Optional[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        self.init_db()
        max_rows = max(1, min(200, int(limit)))
        with self._connect() as conn:
            if town_id:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM interaction_dead_letters
                    WHERE town_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (str(town_id).strip(), max_rows),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM interaction_dead_letters
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (max_rows,),
                ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            row_dict = dict(row)
            payload = {}
            raw_payload = row_dict.get("payload_json")
            if isinstance(raw_payload, str):
                try:
                    parsed = json.loads(raw_payload)
                    if isinstance(parsed, dict):
                        payload = parsed
                except Exception:
                    payload = {}
            out.append(
                {
                    "id": str(row_dict.get("id") or ""),
                    "town_id": row_dict.get("town_id"),
                    "stage": str(row_dict.get("stage") or ""),
                    "payload": payload,
                    "error": str(row_dict.get("error") or ""),
                    "created_at": row_dict.get("created_at"),
                }
            )
        return out


class PostgresRuntimeControlStore(RuntimeControlStore):
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._ensure_driver()

    @staticmethod
    def _ensure_driver() -> None:
        try:
            import psycopg  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Postgres backend requires `psycopg`. Install it with: pip install psycopg[binary]"
            ) from exc

    def _connect(self):
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self.database_url, row_factory=dict_row)

    def _run_migrations(self) -> None:
        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                      id TEXT PRIMARY KEY,
                      applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute("SELECT id FROM schema_migrations")
                applied = {row["id"] for row in cur.fetchall()}
                for file in migration_files:
                    if file.name in applied:
                        continue
                    cur.execute(file.read_text(encoding="utf-8"))
                    cur.execute("INSERT INTO schema_migrations (id) VALUES (%s)", (file.name,))

    def init_db(self) -> None:
        self._run_migrations()

    def get_agent_autonomy(self, *, agent_id: str) -> dict[str, Any]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM agent_autonomy_contracts WHERE agent_id = %s::uuid", (agent_id,))
                row = cur.fetchone()
        return _normalize_contract_row(agent_id, dict(row) if row else None)

    def upsert_agent_autonomy(
        self,
        *,
        agent_id: str,
        mode: str,
        allowed_scopes: list[str],
        max_autonomous_ticks: int,
        escalation_policy: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        self.init_db()
        normalized_mode = _normalize_mode(mode)
        normalized_scopes = _normalize_scopes(allowed_scopes)
        max_ticks = max(0, int(max_autonomous_ticks))
        escalation = escalation_policy if isinstance(escalation_policy, dict) else {}

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM agents WHERE id = %s::uuid", (agent_id,))
                if not cur.fetchone():
                    return None
                cur.execute(
                    """
                    INSERT INTO agent_autonomy_contracts
                      (agent_id, mode, allowed_scopes_json, max_autonomous_ticks, escalation_policy_json, updated_at)
                    VALUES (%s::uuid, %s, %s::jsonb, %s, %s::jsonb, NOW())
                    ON CONFLICT(agent_id) DO UPDATE SET
                      mode = EXCLUDED.mode,
                      allowed_scopes_json = EXCLUDED.allowed_scopes_json,
                      max_autonomous_ticks = EXCLUDED.max_autonomous_ticks,
                      escalation_policy_json = EXCLUDED.escalation_policy_json,
                      updated_at = NOW()
                    RETURNING *
                    """,
                    (
                        agent_id,
                        normalized_mode,
                        json.dumps(normalized_scopes, separators=(",", ":")),
                        max_ticks,
                        json.dumps(escalation, separators=(",", ":")),
                    ),
                )
                row = cur.fetchone()
        return _normalize_contract_row(agent_id, dict(row) if row else None)

    def list_agent_autonomy(self, *, agent_ids: list[str]) -> dict[str, dict[str, Any]]:
        self.init_db()
        normalized_ids = [str(value).strip() for value in agent_ids if str(value).strip()]
        if not normalized_ids:
            return {}
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM agent_autonomy_contracts WHERE agent_id = ANY(%s::uuid[])",
                    (normalized_ids,),
                )
                rows = cur.fetchall()
        records: dict[str, dict[str, Any]] = {}
        for row in rows:
            row_dict = dict(row)
            agent_id = str(row_dict.get("agent_id") or "")
            if not agent_id:
                continue
            records[agent_id] = _normalize_contract_row(agent_id, row_dict)
        for agent_id in normalized_ids:
            if agent_id not in records:
                records[agent_id] = _normalize_contract_row(agent_id, None)
        return records

    def get_town_runtime(self, *, town_id: str) -> dict[str, Any]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM town_runtime_controls WHERE town_id = %s", (town_id,))
                row = cur.fetchone()
        return _normalize_runtime_row(town_id, dict(row) if row else None)

    def upsert_town_runtime(
        self,
        *,
        town_id: str,
        enabled: Optional[bool],
        tick_interval_seconds: Optional[int],
        steps_per_tick: Optional[int],
        planning_scope: Optional[str],
        control_mode: Optional[str],
    ) -> dict[str, Any]:
        self.init_db()
        current = self.get_town_runtime(town_id=town_id)
        next_enabled = bool(current["enabled"] if enabled is None else enabled)
        next_tick_interval = (
            int(current["tick_interval_seconds"])
            if tick_interval_seconds is None
            else max(1, min(3600, int(tick_interval_seconds)))
        )
        next_steps = (
            int(current["steps_per_tick"])
            if steps_per_tick is None
            else max(1, min(240, int(steps_per_tick)))
        )
        next_scope = _normalize_scope(current["planning_scope"] if planning_scope is None else planning_scope)
        next_control_mode = _normalize_control_mode(current["control_mode"] if control_mode is None else control_mode)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO town_runtime_controls
                      (town_id, enabled, tick_interval_seconds, steps_per_tick, planning_scope, control_mode, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT(town_id) DO UPDATE SET
                      enabled = EXCLUDED.enabled,
                      tick_interval_seconds = EXCLUDED.tick_interval_seconds,
                      steps_per_tick = EXCLUDED.steps_per_tick,
                      planning_scope = EXCLUDED.planning_scope,
                      control_mode = EXCLUDED.control_mode,
                      updated_at = NOW()
                    RETURNING *
                    """,
                    (
                        town_id,
                        next_enabled,
                        next_tick_interval,
                        next_steps,
                        next_scope,
                        next_control_mode,
                    ),
                )
                row = cur.fetchone()
        return _normalize_runtime_row(town_id, dict(row) if row else None)

    def list_enabled_towns(self) -> list[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM town_runtime_controls WHERE enabled = TRUE ORDER BY town_id ASC"
                )
                rows = cur.fetchall()
        return [_normalize_runtime_row(str(row["town_id"]), dict(row)) for row in rows]

    def record_town_tick(self, *, town_id: str, success: bool, error: Optional[str]) -> None:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO town_runtime_controls
                      (town_id, enabled, tick_interval_seconds, steps_per_tick, planning_scope, control_mode,
                       last_tick_at, last_tick_success, last_error, updated_at)
                    VALUES (%s, FALSE, 60, 1, 'short_action', 'hybrid', NOW(), %s, %s, NOW())
                    ON CONFLICT(town_id) DO UPDATE SET
                      last_tick_at = NOW(),
                      last_tick_success = EXCLUDED.last_tick_success,
                      last_error = EXCLUDED.last_error,
                      updated_at = NOW()
                    """,
                    (
                        town_id,
                        bool(success),
                        None if success else str(error or "")[:500],
                    ),
                )

    def claim_town_lease(
        self,
        *,
        town_id: str,
        lease_owner: str,
        lease_ttl_seconds: int,
    ) -> bool:
        self.init_db()
        owner = str(lease_owner or "").strip()
        if not owner:
            return False
        ttl = max(1, min(300, int(lease_ttl_seconds)))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO town_runtime_controls
                      (town_id, enabled, tick_interval_seconds, steps_per_tick, planning_scope, control_mode,
                       lease_owner, lease_expires_at, updated_at)
                    VALUES (%s, FALSE, 60, 1, 'short_action', 'hybrid', NULL, NULL, NOW())
                    ON CONFLICT(town_id) DO NOTHING
                    """,
                    (town_id,),
                )
                cur.execute(
                    """
                    UPDATE town_runtime_controls
                    SET lease_owner = %s,
                        lease_expires_at = NOW() + (%s::text || ' seconds')::interval,
                        updated_at = NOW()
                    WHERE town_id = %s
                      AND (
                        lease_owner IS NULL
                        OR lease_owner = %s
                        OR lease_expires_at IS NULL
                        OR lease_expires_at <= NOW()
                      )
                    RETURNING lease_owner, lease_expires_at
                    """,
                    (owner, str(ttl), town_id, owner),
                )
                row = cur.fetchone()
        if not row:
            return False
        return str(row.get("lease_owner") or "") == owner

    def release_town_lease(self, *, town_id: str, lease_owner: str) -> None:
        self.init_db()
        owner = str(lease_owner or "").strip()
        if not owner:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE town_runtime_controls
                    SET lease_owner = NULL,
                        lease_expires_at = NULL,
                        updated_at = NOW()
                    WHERE town_id = %s
                      AND lease_owner = %s
                    """,
                    (town_id, owner),
                )

    def reserve_tick_batch(
        self,
        *,
        town_id: str,
        batch_key: str,
        step_before: Optional[int],
    ) -> dict[str, Any]:
        self.init_db()
        key = str(batch_key or "").strip()
        if not key:
            return {"state": "invalid"}
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM town_tick_batches WHERE town_id = %s AND batch_key = %s",
                    (town_id, key),
                )
                row = cur.fetchone()
                if row:
                    row_dict = dict(row)
                    status = str(row_dict.get("status") or "pending")
                    result_payload = row_dict.get("result_json")
                    if not isinstance(result_payload, dict):
                        result_payload = None
                    return {
                        "state": "existing",
                        "status": status,
                        "result": result_payload,
                        "error": row_dict.get("error"),
                    }
                cur.execute(
                    """
                    INSERT INTO town_tick_batches
                      (town_id, batch_key, status, step_before, created_at)
                    VALUES (%s, %s, 'pending', %s, NOW())
                    """,
                    (
                        town_id,
                        key,
                        (int(step_before) if step_before is not None else None),
                    ),
                )
        return {"state": "new", "status": "pending", "result": None, "error": None}

    def complete_tick_batch(
        self,
        *,
        town_id: str,
        batch_key: str,
        success: bool,
        step_after: Optional[int],
        result: Optional[dict[str, Any]],
        error: Optional[str],
    ) -> None:
        self.init_db()
        key = str(batch_key or "").strip()
        if not key:
            return
        status = "completed" if success else "failed"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO town_tick_batches
                      (town_id, batch_key, status, step_after, result_json, error, created_at, completed_at)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s, NOW(), NOW())
                    ON CONFLICT(town_id, batch_key) DO UPDATE SET
                      status = EXCLUDED.status,
                      step_after = EXCLUDED.step_after,
                      result_json = EXCLUDED.result_json,
                      error = EXCLUDED.error,
                      completed_at = NOW()
                    """,
                    (
                        town_id,
                        key,
                        status,
                        (int(step_after) if step_after is not None else None),
                        (json.dumps(result or {}, separators=(",", ":")) if success else None),
                        None if success else str(error or "")[:500],
                    ),
                )

    def get_tick_metrics(self, *, town_id: str, window_hours: int) -> dict[str, Any]:
        self.init_db()
        hours = max(1, min(168, int(window_hours)))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      COUNT(*)::int AS total,
                      COALESCE(SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END), 0)::int AS completed,
                      COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0)::int AS failed
                    FROM town_tick_batches
                    WHERE town_id = %s
                      AND created_at >= (NOW() - (%s::text || ' hours')::interval)
                    """,
                    (town_id, str(hours)),
                )
                row = cur.fetchone()
                cur.execute(
                    """
                    SELECT result_json
                    FROM town_tick_batches
                    WHERE town_id = %s
                      AND created_at >= (NOW() - (%s::text || ' hours')::interval)
                      AND result_json IS NOT NULL
                    """,
                    (town_id, str(hours)),
                )
                detail_rows = cur.fetchall() or []
        total = int(row.get("total") or 0) if row else 0
        completed = int(row.get("completed") or 0) if row else 0
        failed = int(row.get("failed") or 0) if row else 0
        decision_metrics = _decision_metrics_from_batches(
            rows=[dict(item) for item in detail_rows],
            window_hours=hours,
        )
        return {
            "town_id": town_id,
            "window_hours": hours,
            "total_batches": total,
            "completed_batches": completed,
            "failed_batches": failed,
            "success_rate": (float(completed) / float(total) if total > 0 else None),
            **decision_metrics,
        }

    def record_dead_letter(
        self,
        *,
        town_id: Optional[str],
        stage: str,
        payload: Optional[dict[str, Any]],
        error: str,
    ) -> None:
        self.init_db()
        dead_letter_id = str(uuid.uuid4())
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO interaction_dead_letters
                      (id, town_id, stage, payload_json, error)
                    VALUES (%s::uuid, %s, %s, %s::jsonb, %s)
                    """,
                    (
                        dead_letter_id,
                        (str(town_id).strip() if town_id else None),
                        str(stage or "unknown")[:120],
                        json.dumps(payload or {}, separators=(",", ":")),
                        str(error or "unknown_error")[:800],
                    ),
                )

    def list_dead_letters(
        self,
        *,
        town_id: Optional[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        self.init_db()
        max_rows = max(1, min(200, int(limit)))
        with self._connect() as conn:
            with conn.cursor() as cur:
                if town_id:
                    cur.execute(
                        """
                        SELECT *
                        FROM interaction_dead_letters
                        WHERE town_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (str(town_id).strip(), max_rows),
                    )
                else:
                    cur.execute(
                        """
                        SELECT *
                        FROM interaction_dead_letters
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (max_rows,),
                    )
                rows = cur.fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            row_dict = dict(row)
            payload = row_dict.get("payload_json")
            if not isinstance(payload, dict):
                payload = {}
            out.append(
                {
                    "id": str(row_dict.get("id") or ""),
                    "town_id": row_dict.get("town_id"),
                    "stage": str(row_dict.get("stage") or ""),
                    "payload": payload,
                    "error": str(row_dict.get("error") or ""),
                    "created_at": row_dict.get("created_at"),
                }
            )
        return out


def _resolve_sqlite_path(database_url: Optional[str]) -> Path:
    if database_url and database_url.startswith("sqlite:///"):
        raw = database_url[len("sqlite:///") :]
        p = Path(raw)
        if not p.is_absolute():
            p = (WORKSPACE_ROOT / p).resolve()
        return p
    raw = os.environ.get("VVALLEY_DB_PATH", str(WORKSPACE_ROOT / "data" / "vvalley.db"))
    p = Path(raw)
    if not p.is_absolute():
        p = (WORKSPACE_ROOT / p).resolve()
    return p


def _database_url() -> Optional[str]:
    return os.environ.get("DATABASE_URL")


@lru_cache(maxsize=1)
def _backend() -> RuntimeControlStore:
    database_url = _database_url()
    if database_url and database_url.startswith(("postgres://", "postgresql://")):
        return PostgresRuntimeControlStore(database_url)
    return SQLiteRuntimeControlStore(_resolve_sqlite_path(database_url))


def reset_backend_cache_for_tests() -> None:
    _backend.cache_clear()


def init_db() -> None:
    _backend().init_db()


def get_agent_autonomy(*, agent_id: str) -> dict[str, Any]:
    return _backend().get_agent_autonomy(agent_id=agent_id)


def upsert_agent_autonomy(
    *,
    agent_id: str,
    mode: str,
    allowed_scopes: list[str],
    max_autonomous_ticks: int = 0,
    escalation_policy: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    return _backend().upsert_agent_autonomy(
        agent_id=agent_id,
        mode=mode,
        allowed_scopes=allowed_scopes,
        max_autonomous_ticks=max_autonomous_ticks,
        escalation_policy=escalation_policy,
    )


def list_agent_autonomy(*, agent_ids: list[str]) -> dict[str, dict[str, Any]]:
    return _backend().list_agent_autonomy(agent_ids=agent_ids)


def get_town_runtime(*, town_id: str) -> dict[str, Any]:
    return _backend().get_town_runtime(town_id=town_id)


def upsert_town_runtime(
    *,
    town_id: str,
    enabled: Optional[bool] = None,
    tick_interval_seconds: Optional[int] = None,
    steps_per_tick: Optional[int] = None,
    planning_scope: Optional[str] = None,
    control_mode: Optional[str] = None,
) -> dict[str, Any]:
    return _backend().upsert_town_runtime(
        town_id=town_id,
        enabled=enabled,
        tick_interval_seconds=tick_interval_seconds,
        steps_per_tick=steps_per_tick,
        planning_scope=planning_scope,
        control_mode=control_mode,
    )


def list_enabled_towns() -> list[dict[str, Any]]:
    return _backend().list_enabled_towns()


def record_town_tick(*, town_id: str, success: bool, error: Optional[str] = None) -> None:
    _backend().record_town_tick(town_id=town_id, success=success, error=error)


def claim_town_lease(
    *,
    town_id: str,
    lease_owner: str,
    lease_ttl_seconds: int = 30,
) -> bool:
    return _backend().claim_town_lease(
        town_id=town_id,
        lease_owner=lease_owner,
        lease_ttl_seconds=lease_ttl_seconds,
    )


def release_town_lease(*, town_id: str, lease_owner: str) -> None:
    _backend().release_town_lease(town_id=town_id, lease_owner=lease_owner)


def reserve_tick_batch(
    *,
    town_id: str,
    batch_key: str,
    step_before: Optional[int] = None,
) -> dict[str, Any]:
    return _backend().reserve_tick_batch(
        town_id=town_id,
        batch_key=batch_key,
        step_before=step_before,
    )


def complete_tick_batch(
    *,
    town_id: str,
    batch_key: str,
    success: bool,
    step_after: Optional[int] = None,
    result: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    _backend().complete_tick_batch(
        town_id=town_id,
        batch_key=batch_key,
        success=success,
        step_after=step_after,
        result=result,
        error=error,
    )


def get_tick_metrics(*, town_id: str, window_hours: int = 24) -> dict[str, Any]:
    return _backend().get_tick_metrics(town_id=town_id, window_hours=window_hours)


def record_dead_letter(
    *,
    town_id: Optional[str],
    stage: str,
    payload: Optional[dict[str, Any]],
    error: str,
) -> None:
    _backend().record_dead_letter(
        town_id=town_id,
        stage=stage,
        payload=payload,
        error=error,
    )


def list_dead_letters(
    *,
    town_id: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    return _backend().list_dead_letters(
        town_id=town_id,
        limit=limit,
    )
