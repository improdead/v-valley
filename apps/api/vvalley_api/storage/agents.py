"""Persistent agent onboarding storage with SQLite fallback and Postgres support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
import hashlib
import json
import logging
import os
import secrets
import sqlite3
import uuid

logger = logging.getLogger("vvalley_api.storage.agents")


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
MIGRATIONS_DIR = WORKSPACE_ROOT / "packages" / "vvalley_core" / "db" / "migrations"


def _now_utc_sqlite() -> str:
    return "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"


def _hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _new_api_key() -> str:
    return f"vvalley_sk_{secrets.token_urlsafe(24)}"


def _new_claim_token() -> str:
    return f"vvalley_claim_{secrets.token_urlsafe(18)}"


def _new_verification_code() -> str:
    return secrets.token_hex(2).upper()


class AgentStore(ABC):
    @abstractmethod
    def init_db(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def register_agent(
        self,
        *,
        name: str,
        description: Optional[str],
        personality: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def claim_agent(
        self,
        *,
        claim_token: str,
        verification_code: str,
        owner_handle: Optional[str],
    ) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_agent_by_api_key(self, api_key: str) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def rotate_api_key(self, api_key: str) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_agent_town(self, *, agent_id: str) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def upsert_agent_town(self, *, agent_id: str, town_id: str) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def clear_agent_town(self, *, agent_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def count_active_agents_in_town(self, town_id: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def list_active_agents_in_town(self, town_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError


class SQLiteAgentStore(AgentStore):
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        out = {k: row[k] for k in row.keys()}
        raw = out.get("personality_json")
        out["personality"] = json.loads(raw) if raw else {}
        out.pop("personality_json", None)
        out["claimed"] = out.get("claim_status") == "active"
        return out

    @staticmethod
    def _membership_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {k: row[k] for k in row.keys()}

    def init_db(self) -> None:
        logger.info("[STORAGE] Initializing SQLite agent database at '%s'", self.db_path)
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS agents (
                  id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  description TEXT,
                  personality_json TEXT,
                  api_key_hash TEXT NOT NULL UNIQUE,
                  api_key_prefix TEXT NOT NULL,
                  claim_token TEXT NOT NULL UNIQUE,
                  verification_code TEXT NOT NULL,
                  claim_status TEXT NOT NULL DEFAULT 'pending',
                  owner_handle TEXT,
                  created_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()}),
                  claimed_at TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agents_claim_token ON agents(claim_token)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agents_api_key_hash ON agents(api_key_hash)"
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS agent_town_memberships (
                  agent_id TEXT PRIMARY KEY,
                  town_id TEXT NOT NULL,
                  joined_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()}),
                  updated_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()}),
                  FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_town_memberships_town_id ON agent_town_memberships(town_id)"
            )
        logger.info("[STORAGE] SQLite agent database initialized successfully")

    def register_agent(
        self,
        *,
        name: str,
        description: Optional[str],
        personality: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        self.init_db()
        agent_id = str(uuid.uuid4())
        api_key = _new_api_key()
        claim_token = _new_claim_token()
        verification_code = _new_verification_code()
        personality_json = json.dumps(personality or {}, separators=(",", ":"))

        logger.info("[STORAGE] Registering agent: id=%s, name='%s'", agent_id, name)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agents
                  (id, name, description, personality_json, api_key_hash, api_key_prefix, claim_token, verification_code)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_id,
                    name,
                    description,
                    personality_json,
                    _hash_api_key(api_key),
                    api_key[:16],
                    claim_token,
                    verification_code,
                ),
            )
            row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()

        out = self._row_to_dict(row)
        out["api_key"] = api_key
        logger.info("[STORAGE] Agent registered: id=%s, name='%s'", agent_id, name)
        return out

    def claim_agent(
        self,
        *,
        claim_token: str,
        verification_code: str,
        owner_handle: Optional[str],
    ) -> Optional[dict[str, Any]]:
        self.init_db()
        logger.debug("[STORAGE] Claiming agent: claim_token='%s...', owner_handle='%s'", 
                    claim_token[:20] if claim_token else "", owner_handle)
        
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agents WHERE claim_token = ?",
                (claim_token,),
            ).fetchone()
            if not row:
                logger.warning("[STORAGE] Claim failed - token not found: claim_token='%s...'", 
                              claim_token[:20] if claim_token else "")
                return None

            if str(row["verification_code"]).upper() != str(verification_code).upper():
                logger.warning("[STORAGE] Claim failed - verification code mismatch: agent_id=%s", row["id"])
                raise ValueError("verification code mismatch")

            if row["claim_status"] != "active":
                conn.execute(
                    f"""
                    UPDATE agents
                    SET claim_status = 'active',
                        owner_handle = ?,
                        claimed_at = ({_now_utc_sqlite()})
                    WHERE id = ?
                    """,
                    (owner_handle, row["id"]),
                )
                logger.info("[STORAGE] Agent claimed: id=%s, name='%s', owner_handle='%s'", 
                           row["id"], row["name"], owner_handle)
            else:
                logger.debug("[STORAGE] Agent already claimed: id=%s", row["id"])

            updated = conn.execute("SELECT * FROM agents WHERE id = ?", (row["id"],)).fetchone()
        return self._row_to_dict(updated)

    def get_agent_by_api_key(self, api_key: str) -> Optional[dict[str, Any]]:
        self.init_db()
        hashed = _hash_api_key(api_key)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agents WHERE api_key_hash = ?",
                (hashed,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def rotate_api_key(self, api_key: str) -> Optional[dict[str, Any]]:
        self.init_db()
        agent = self.get_agent_by_api_key(api_key)
        if not agent:
            logger.warning("[STORAGE] Key rotation failed - invalid API key")
            return None

        logger.info("[STORAGE] Rotating API key: agent_id=%s", agent["id"])
        new_api_key = _new_api_key()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE agents
                SET api_key_hash = ?, api_key_prefix = ?
                WHERE id = ?
                """,
                (_hash_api_key(new_api_key), new_api_key[:16], agent["id"]),
            )
            updated = conn.execute("SELECT * FROM agents WHERE id = ?", (agent["id"],)).fetchone()

        out = self._row_to_dict(updated)
        out["api_key"] = new_api_key
        logger.info("[STORAGE] API key rotated: agent_id=%s", agent["id"])
        return out

    def get_agent_town(self, *, agent_id: str) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT agent_id, town_id, joined_at, updated_at FROM agent_town_memberships WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
        return self._membership_row_to_dict(row) if row else None

    def upsert_agent_town(self, *, agent_id: str, town_id: str) -> Optional[dict[str, Any]]:
        self.init_db()
        logger.debug("[STORAGE] Upserting agent town membership: agent_id=%s, town_id='%s'", agent_id, town_id)
        
        with self._connect() as conn:
            exists = conn.execute("SELECT 1 FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if not exists:
                logger.warning("[STORAGE] Upsert membership failed - agent not found: agent_id=%s", agent_id)
                return None

            conn.execute(
                f"""
                INSERT INTO agent_town_memberships (agent_id, town_id, joined_at, updated_at)
                VALUES (?, ?, ({_now_utc_sqlite()}), ({_now_utc_sqlite()}))
                ON CONFLICT(agent_id) DO UPDATE SET
                  town_id = excluded.town_id,
                  joined_at = CASE
                    WHEN agent_town_memberships.town_id = excluded.town_id THEN agent_town_memberships.joined_at
                    ELSE ({_now_utc_sqlite()})
                  END,
                  updated_at = ({_now_utc_sqlite()})
                """,
                (agent_id, town_id),
            )
            row = conn.execute(
                "SELECT agent_id, town_id, joined_at, updated_at FROM agent_town_memberships WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
        
        if row:
            logger.info("[STORAGE] Agent town membership upserted: agent_id=%s, town_id='%s'", agent_id, town_id)
        return self._membership_row_to_dict(row) if row else None

    def clear_agent_town(self, *, agent_id: str) -> bool:
        self.init_db()
        logger.debug("[STORAGE] Clearing agent town membership: agent_id=%s", agent_id)
        
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM agent_town_memberships WHERE agent_id = ?",
                (agent_id,),
            )
        deleted = int(cur.rowcount or 0) > 0
        if deleted:
            logger.info("[STORAGE] Agent town membership cleared: agent_id=%s", agent_id)
        return deleted

    def count_active_agents_in_town(self, town_id: str) -> int:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM agent_town_memberships WHERE town_id = ?",
                (town_id,),
            ).fetchone()
        count = int(row["c"]) if row else 0
        logger.debug("[STORAGE] Counted active agents in town: town_id='%s', count=%d", town_id, count)
        return count

    def list_active_agents_in_town(self, town_id: str) -> list[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  m.agent_id AS agent_id,
                  a.name AS name,
                  a.claim_status AS claim_status,
                  a.owner_handle AS owner_handle,
                  m.joined_at AS joined_at,
                  m.updated_at AS updated_at
                FROM agent_town_memberships m
                INNER JOIN agents a ON a.id = m.agent_id
                WHERE m.town_id = ?
                ORDER BY m.joined_at ASC, m.agent_id ASC
                """,
                (town_id,),
            ).fetchall()
        return [dict(r) for r in rows]


class PostgresAgentStore(AgentStore):
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._ensure_driver()

    @staticmethod
    def _ensure_driver() -> None:
        try:
            import psycopg  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on environment
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
                    sql = file.read_text(encoding="utf-8")
                    cur.execute(sql)
                    cur.execute("INSERT INTO schema_migrations (id) VALUES (%s)", (file.name,))

    @staticmethod
    def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        raw = out.get("personality_json")
        if isinstance(raw, str):
            out["personality"] = json.loads(raw)
        elif raw is None:
            out["personality"] = {}
        else:
            out["personality"] = dict(raw)
        out.pop("personality_json", None)
        out["claimed"] = out.get("claim_status") == "active"
        return out

    def init_db(self) -> None:
        self._run_migrations()

    def register_agent(
        self,
        *,
        name: str,
        description: Optional[str],
        personality: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        self.init_db()
        agent_id = str(uuid.uuid4())
        api_key = _new_api_key()
        claim_token = _new_claim_token()
        verification_code = _new_verification_code()

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agents
                      (id, name, description, personality_json, api_key_hash, api_key_prefix, claim_token, verification_code)
                    VALUES (%s::uuid, %s, %s, %s::jsonb, %s, %s, %s, %s)
                    """,
                    (
                        agent_id,
                        name,
                        description,
                        json.dumps(personality or {}, separators=(",", ":")),
                        _hash_api_key(api_key),
                        api_key[:16],
                        claim_token,
                        verification_code,
                    ),
                )
                cur.execute("SELECT * FROM agents WHERE id = %s::uuid", (agent_id,))
                row = cur.fetchone()

        out = self._normalize_row(row)
        out["api_key"] = api_key
        return out

    def claim_agent(
        self,
        *,
        claim_token: str,
        verification_code: str,
        owner_handle: Optional[str],
    ) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM agents WHERE claim_token = %s", (claim_token,))
                row = cur.fetchone()
                if not row:
                    return None

                if str(row["verification_code"]).upper() != str(verification_code).upper():
                    raise ValueError("verification code mismatch")

                if row["claim_status"] != "active":
                    cur.execute(
                        """
                        UPDATE agents
                        SET claim_status = 'active', owner_handle = %s, claimed_at = NOW()
                        WHERE id = %s::uuid
                        """,
                        (owner_handle, row["id"]),
                    )

                cur.execute("SELECT * FROM agents WHERE id = %s::uuid", (row["id"],))
                updated = cur.fetchone()
        return self._normalize_row(updated)

    def get_agent_by_api_key(self, api_key: str) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM agents WHERE api_key_hash = %s",
                    (_hash_api_key(api_key),),
                )
                row = cur.fetchone()
        return self._normalize_row(row) if row else None

    def rotate_api_key(self, api_key: str) -> Optional[dict[str, Any]]:
        self.init_db()
        agent = self.get_agent_by_api_key(api_key)
        if not agent:
            return None

        new_api_key = _new_api_key()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE agents
                    SET api_key_hash = %s, api_key_prefix = %s
                    WHERE id = %s::uuid
                    """,
                    (_hash_api_key(new_api_key), new_api_key[:16], agent["id"]),
                )
                cur.execute("SELECT * FROM agents WHERE id = %s::uuid", (agent["id"],))
                updated = cur.fetchone()

        out = self._normalize_row(updated)
        out["api_key"] = new_api_key
        return out

    def get_agent_town(self, *, agent_id: str) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT agent_id::text AS agent_id, town_id, joined_at, updated_at
                    FROM agent_town_memberships
                    WHERE agent_id = %s::uuid
                    """,
                    (agent_id,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def upsert_agent_town(self, *, agent_id: str, town_id: str) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM agents WHERE id = %s::uuid", (agent_id,))
                exists = cur.fetchone()
                if not exists:
                    return None

                cur.execute(
                    """
                    INSERT INTO agent_town_memberships (agent_id, town_id)
                    VALUES (%s::uuid, %s)
                    ON CONFLICT (agent_id) DO UPDATE SET
                      town_id = EXCLUDED.town_id,
                      joined_at = CASE
                        WHEN agent_town_memberships.town_id = EXCLUDED.town_id THEN agent_town_memberships.joined_at
                        ELSE NOW()
                      END,
                      updated_at = NOW()
                    RETURNING agent_id::text AS agent_id, town_id, joined_at, updated_at
                    """,
                    (agent_id, town_id),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def clear_agent_town(self, *, agent_id: str) -> bool:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM agent_town_memberships WHERE agent_id = %s::uuid",
                    (agent_id,),
                )
                deleted = cur.rowcount
        return int(deleted or 0) > 0

    def count_active_agents_in_town(self, town_id: str) -> int:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) AS c FROM agent_town_memberships WHERE town_id = %s",
                    (town_id,),
                )
                row = cur.fetchone()
        return int(row["c"]) if row else 0

    def list_active_agents_in_town(self, town_id: str) -> list[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      m.agent_id::text AS agent_id,
                      a.name AS name,
                      a.claim_status AS claim_status,
                      a.owner_handle AS owner_handle,
                      m.joined_at AS joined_at,
                      m.updated_at AS updated_at
                    FROM agent_town_memberships m
                    INNER JOIN agents a ON a.id = m.agent_id
                    WHERE m.town_id = %s
                    ORDER BY m.joined_at ASC, m.agent_id ASC
                    """,
                    (town_id,),
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
def _backend() -> AgentStore:
    database_url = _database_url()
    if database_url and database_url.startswith(("postgres://", "postgresql://")):
        return PostgresAgentStore(database_url)
    return SQLiteAgentStore(_resolve_sqlite_path(database_url))


def reset_backend_cache_for_tests() -> None:
    _backend.cache_clear()


def init_db() -> None:
    _backend().init_db()


def register_agent(
    *,
    name: str,
    description: Optional[str] = None,
    personality: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return _backend().register_agent(
        name=name,
        description=description,
        personality=personality,
    )


def claim_agent(
    *,
    claim_token: str,
    verification_code: str,
    owner_handle: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    return _backend().claim_agent(
        claim_token=claim_token,
        verification_code=verification_code,
        owner_handle=owner_handle,
    )


def get_agent_by_api_key(api_key: str) -> Optional[dict[str, Any]]:
    return _backend().get_agent_by_api_key(api_key)


def rotate_api_key(api_key: str) -> Optional[dict[str, Any]]:
    return _backend().rotate_api_key(api_key)


def get_agent_town(*, agent_id: str) -> Optional[dict[str, Any]]:
    return _backend().get_agent_town(agent_id=agent_id)


def join_agent_town(*, agent_id: str, town_id: str) -> Optional[dict[str, Any]]:
    return _backend().upsert_agent_town(agent_id=agent_id, town_id=town_id)


def leave_agent_town(*, agent_id: str) -> bool:
    return _backend().clear_agent_town(agent_id=agent_id)


def count_active_agents_in_town(town_id: str) -> int:
    return _backend().count_active_agents_in_town(town_id)


def list_active_agents_in_town(town_id: str) -> list[dict[str, Any]]:
    return _backend().list_active_agents_in_town(town_id)
