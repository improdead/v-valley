"""Storage for LLM task policies and call telemetry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
import logging
import os
import sqlite3

from packages.vvalley_core.llm.policy import (
    DEFAULT_TASK_POLICIES,
    TaskPolicy,
    default_policy_for_task,
    normalize_policy_row,
)


logger = logging.getLogger("vvalley_api.storage.llm_control")
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
MIGRATIONS_DIR = WORKSPACE_ROOT / "packages" / "vvalley_core" / "db" / "migrations"


class LlmControlStore(ABC):
    @abstractmethod
    def init_db(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert_policy(self, policy: TaskPolicy) -> TaskPolicy:
        raise NotImplementedError

    @abstractmethod
    def get_policy(self, task_name: str) -> Optional[TaskPolicy]:
        raise NotImplementedError

    @abstractmethod
    def list_policies(self) -> list[TaskPolicy]:
        raise NotImplementedError

    @abstractmethod
    def insert_call_log(self, record: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_call_logs(self, *, limit: int = 100, task_name: Optional[str] = None) -> list[dict[str, Any]]:
        raise NotImplementedError


class SQLiteLlmControlStore(LlmControlStore):
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_task_policies (
                  task_name TEXT PRIMARY KEY,
                  model_tier TEXT NOT NULL,
                  max_input_tokens INTEGER NOT NULL,
                  max_output_tokens INTEGER NOT NULL,
                  temperature REAL NOT NULL,
                  timeout_ms INTEGER NOT NULL,
                  retry_limit INTEGER NOT NULL,
                  enable_prompt_cache INTEGER NOT NULL DEFAULT 1,
                  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_call_logs (
                  id TEXT PRIMARY KEY,
                  town_id TEXT,
                  agent_id TEXT,
                  task_name TEXT NOT NULL,
                  model_name TEXT NOT NULL,
                  prompt_tokens INTEGER NOT NULL,
                  completion_tokens INTEGER NOT NULL,
                  cached_tokens INTEGER NOT NULL DEFAULT 0,
                  latency_ms INTEGER,
                  success INTEGER NOT NULL,
                  error_code TEXT,
                  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_logs_task_created ON llm_call_logs(task_name, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_logs_town_created ON llm_call_logs(town_id, created_at DESC)"
            )

    def upsert_policy(self, policy: TaskPolicy) -> TaskPolicy:
        self.init_db()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_task_policies (
                  task_name, model_tier, max_input_tokens, max_output_tokens,
                  temperature, timeout_ms, retry_limit, enable_prompt_cache, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')))
                ON CONFLICT(task_name) DO UPDATE SET
                  model_tier = excluded.model_tier,
                  max_input_tokens = excluded.max_input_tokens,
                  max_output_tokens = excluded.max_output_tokens,
                  temperature = excluded.temperature,
                  timeout_ms = excluded.timeout_ms,
                  retry_limit = excluded.retry_limit,
                  enable_prompt_cache = excluded.enable_prompt_cache,
                  updated_at = (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                """,
                (
                    policy.task_name,
                    policy.model_tier,
                    int(policy.max_input_tokens),
                    int(policy.max_output_tokens),
                    float(policy.temperature),
                    int(policy.timeout_ms),
                    int(policy.retry_limit),
                    1 if policy.enable_prompt_cache else 0,
                ),
            )
            row = conn.execute(
                "SELECT * FROM llm_task_policies WHERE task_name = ?",
                (policy.task_name,),
            ).fetchone()
        return normalize_policy_row(str(row["task_name"]), dict(row))

    def get_policy(self, task_name: str) -> Optional[TaskPolicy]:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM llm_task_policies WHERE task_name = ?",
                (task_name,),
            ).fetchone()
        if not row:
            return None
        return normalize_policy_row(str(row["task_name"]), dict(row))

    def list_policies(self) -> list[TaskPolicy]:
        self.init_db()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM llm_task_policies ORDER BY task_name ASC"
            ).fetchall()
        return [normalize_policy_row(str(r["task_name"]), dict(r)) for r in rows]

    def insert_call_log(self, record: dict[str, Any]) -> None:
        self.init_db()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_call_logs (
                  id, town_id, agent_id, task_name, model_name, prompt_tokens,
                  completion_tokens, cached_tokens, latency_ms, success, error_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(record["id"]),
                    record.get("town_id"),
                    record.get("agent_id"),
                    str(record["task_name"]),
                    str(record["model_name"]),
                    int(record.get("prompt_tokens") or 0),
                    int(record.get("completion_tokens") or 0),
                    int(record.get("cached_tokens") or 0),
                    int(record.get("latency_ms") or 0),
                    1 if bool(record.get("success", False)) else 0,
                    record.get("error_code"),
                ),
            )

    def list_call_logs(self, *, limit: int = 100, task_name: Optional[str] = None) -> list[dict[str, Any]]:
        self.init_db()
        bounded_limit = max(1, min(int(limit), 500))
        with self._connect() as conn:
            if task_name:
                rows = conn.execute(
                    """
                    SELECT * FROM llm_call_logs
                    WHERE task_name = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (task_name, bounded_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM llm_call_logs
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (bounded_limit,),
                ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["success"] = bool(item.get("success", 0))
            out.append(item)
        return out


class PostgresLlmControlStore(LlmControlStore):
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

    def upsert_policy(self, policy: TaskPolicy) -> TaskPolicy:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO llm_task_policies (
                      task_name, model_tier, max_input_tokens, max_output_tokens,
                      temperature, timeout_ms, retry_limit, enable_prompt_cache
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(task_name) DO UPDATE SET
                      model_tier = EXCLUDED.model_tier,
                      max_input_tokens = EXCLUDED.max_input_tokens,
                      max_output_tokens = EXCLUDED.max_output_tokens,
                      temperature = EXCLUDED.temperature,
                      timeout_ms = EXCLUDED.timeout_ms,
                      retry_limit = EXCLUDED.retry_limit,
                      enable_prompt_cache = EXCLUDED.enable_prompt_cache
                    RETURNING *
                    """,
                    (
                        policy.task_name,
                        policy.model_tier,
                        int(policy.max_input_tokens),
                        int(policy.max_output_tokens),
                        float(policy.temperature),
                        int(policy.timeout_ms),
                        int(policy.retry_limit),
                        bool(policy.enable_prompt_cache),
                    ),
                )
                row = cur.fetchone()
        return normalize_policy_row(str(row["task_name"]), row)

    def get_policy(self, task_name: str) -> Optional[TaskPolicy]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM llm_task_policies WHERE task_name = %s", (task_name,))
                row = cur.fetchone()
        if not row:
            return None
        return normalize_policy_row(str(row["task_name"]), row)

    def list_policies(self) -> list[TaskPolicy]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM llm_task_policies ORDER BY task_name ASC")
                rows = cur.fetchall()
        return [normalize_policy_row(str(r["task_name"]), r) for r in rows]

    def insert_call_log(self, record: dict[str, Any]) -> None:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO llm_call_logs (
                      id, town_id, agent_id, task_name, model_name, prompt_tokens,
                      completion_tokens, cached_tokens, latency_ms, success, error_code
                    )
                    VALUES (
                      %s::uuid,
                      NULLIF(%s, '')::uuid,
                      NULLIF(%s, '')::uuid,
                      %s,
                      %s,
                      %s,
                      %s,
                      %s,
                      %s,
                      %s,
                      %s
                    )
                    """,
                    (
                        str(record["id"]),
                        str(record.get("town_id") or ""),
                        str(record.get("agent_id") or ""),
                        str(record["task_name"]),
                        str(record["model_name"]),
                        int(record.get("prompt_tokens") or 0),
                        int(record.get("completion_tokens") or 0),
                        int(record.get("cached_tokens") or 0),
                        int(record.get("latency_ms") or 0),
                        bool(record.get("success", False)),
                        record.get("error_code"),
                    ),
                )

    def list_call_logs(self, *, limit: int = 100, task_name: Optional[str] = None) -> list[dict[str, Any]]:
        self.init_db()
        bounded_limit = max(1, min(int(limit), 500))
        with self._connect() as conn:
            with conn.cursor() as cur:
                if task_name:
                    cur.execute(
                        """
                        SELECT
                          id::text AS id,
                          town_id::text AS town_id,
                          agent_id::text AS agent_id,
                          task_name, model_name, prompt_tokens, completion_tokens,
                          cached_tokens, latency_ms, success, error_code, created_at
                        FROM llm_call_logs
                        WHERE task_name = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (task_name, bounded_limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT
                          id::text AS id,
                          town_id::text AS town_id,
                          agent_id::text AS agent_id,
                          task_name, model_name, prompt_tokens, completion_tokens,
                          cached_tokens, latency_ms, success, error_code, created_at
                        FROM llm_call_logs
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (bounded_limit,),
                    )
                rows = cur.fetchall()
        return [dict(r) for r in rows]


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
def _backend() -> LlmControlStore:
    database_url = _database_url()
    if database_url and database_url.startswith(("postgres://", "postgresql://")):
        return PostgresLlmControlStore(database_url)
    return SQLiteLlmControlStore(_resolve_sqlite_path(database_url))


def reset_backend_cache_for_tests() -> None:
    _backend.cache_clear()


def init_db() -> None:
    _backend().init_db()


def upsert_policy(policy: TaskPolicy) -> TaskPolicy:
    return _backend().upsert_policy(policy)


def get_policy(task_name: str) -> TaskPolicy:
    row = _backend().get_policy(task_name)
    if row:
        return row
    return default_policy_for_task(task_name)


def list_policies() -> list[TaskPolicy]:
    existing = {p.task_name: p for p in _backend().list_policies()}
    defaults = {}
    # include common defaults so operators can inspect without writing first
    for task in sorted(DEFAULT_TASK_POLICIES.keys()):
        defaults[task] = default_policy_for_task(task)
    defaults.update(existing)
    return [defaults[k] for k in sorted(defaults.keys())]


def insert_call_log(record: dict[str, Any]) -> None:
    _backend().insert_call_log(record)


def list_call_logs(*, limit: int = 100, task_name: Optional[str] = None) -> list[dict[str, Any]]:
    return _backend().list_call_logs(limit=limit, task_name=task_name)
