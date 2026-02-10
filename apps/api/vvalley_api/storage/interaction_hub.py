"""Durable interaction hub storage: inbox, owner escalations, and DM flows."""

from __future__ import annotations

from abc import ABC, abstractmethod
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

INBOX_STATUSES = {"pending", "read", "resolved", "dismissed"}
ESCALATION_STATUSES = {"open", "resolved", "dismissed"}
DM_REQUEST_STATUSES = {"pending", "approved", "rejected", "expired"}
DM_CONVERSATION_STATUSES = {"active", "closed"}


def _now_utc_sqlite() -> str:
    return "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _normalize_inbox_status(value: str | None) -> str:
    normalized = str(value or "pending").strip().lower()
    if normalized not in INBOX_STATUSES:
        return "pending"
    return normalized


def _normalize_escalation_status(value: str | None) -> str:
    normalized = str(value or "open").strip().lower()
    if normalized not in ESCALATION_STATUSES:
        return "open"
    return normalized


def _normalize_dm_request_status(value: str | None) -> str:
    normalized = str(value or "pending").strip().lower()
    if normalized not in DM_REQUEST_STATUSES:
        return "pending"
    return normalized


def _normalize_dm_conversation_status(value: str | None) -> str:
    normalized = str(value or "active").strip().lower()
    if normalized not in DM_CONVERSATION_STATUSES:
        return "active"
    return normalized


def _normalize_direction(value: str | None) -> str:
    normalized = str(value or "inbox").strip().lower()
    if normalized in {"inbox", "outbox", "all"}:
        return normalized
    return "inbox"


def _normalize_status_filters(values: Optional[list[str]], *, allowed: set[str]) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for raw in values:
        normalized = str(raw or "").strip().lower()
        if normalized in allowed and normalized not in out:
            out.append(normalized)
    return out


def _inbox_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "agent_id": str(row.get("agent_id") or ""),
        "kind": str(row.get("kind") or ""),
        "summary": str(row.get("summary") or ""),
        "payload": _json_object(row.get("payload_json")),
        "status": _normalize_inbox_status(str(row.get("status") or "pending")),
        "dedupe_key": row.get("dedupe_key"),
        "created_at": row.get("created_at"),
        "read_at": row.get("read_at"),
        "resolved_at": row.get("resolved_at"),
        "updated_at": row.get("updated_at"),
    }


def _escalation_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "owner_handle": str(row.get("owner_handle") or ""),
        "agent_id": (str(row.get("agent_id")) if row.get("agent_id") is not None else None),
        "town_id": row.get("town_id"),
        "kind": str(row.get("kind") or ""),
        "summary": str(row.get("summary") or ""),
        "payload": _json_object(row.get("payload_json")),
        "status": _normalize_escalation_status(str(row.get("status") or "open")),
        "dedupe_key": row.get("dedupe_key"),
        "created_at": row.get("created_at"),
        "resolved_at": row.get("resolved_at"),
        "updated_at": row.get("updated_at"),
    }


def _dm_request_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "from_agent_id": str(row.get("from_agent_id") or ""),
        "to_agent_id": str(row.get("to_agent_id") or ""),
        "town_id": row.get("town_id"),
        "message": str(row.get("message") or ""),
        "status": _normalize_dm_request_status(str(row.get("status") or "pending")),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "resolved_at": row.get("resolved_at"),
    }


def _dm_conversation_row(row: dict[str, Any], *, viewer_agent_id: str | None = None) -> dict[str, Any]:
    agent_a = str(row.get("agent_a_id") or "")
    agent_b = str(row.get("agent_b_id") or "")
    viewer = str(viewer_agent_id or "").strip()
    partner = None
    if viewer:
        if viewer == agent_a:
            partner = agent_b
        elif viewer == agent_b:
            partner = agent_a
    return {
        "id": str(row.get("id") or ""),
        "agent_a_id": agent_a,
        "agent_b_id": agent_b,
        "partner_agent_id": partner,
        "town_id": row.get("town_id"),
        "status": _normalize_dm_conversation_status(str(row.get("status") or "active")),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "last_message_at": row.get("last_message_at"),
    }


def _dm_message_row(row: dict[str, Any]) -> dict[str, Any]:
    needs_human = row.get("needs_human_input")
    return {
        "id": str(row.get("id") or ""),
        "conversation_id": str(row.get("conversation_id") or ""),
        "sender_agent_id": str(row.get("sender_agent_id") or ""),
        "body": str(row.get("body") or ""),
        "needs_human_input": bool(needs_human) if needs_human is not None else False,
        "metadata": _json_object(row.get("metadata_json")),
        "created_at": row.get("created_at"),
    }


def _conversation_pair(agent_a_id: str, agent_b_id: str) -> tuple[str, str]:
    first = str(agent_a_id).strip()
    second = str(agent_b_id).strip()
    if first <= second:
        return first, second
    return second, first


class InteractionHubStore(ABC):
    @abstractmethod
    def init_db(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_agent_inbox(
        self,
        *,
        agent_id: str,
        statuses: Optional[list[str]],
        limit: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def mark_inbox_read(self, *, agent_id: str, item_id: str) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def create_inbox_item(
        self,
        *,
        agent_id: str,
        kind: str,
        summary: str,
        payload: Optional[dict[str, Any]],
        dedupe_key: Optional[str],
    ) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def list_owner_escalations(
        self,
        *,
        owner_handle: str,
        statuses: Optional[list[str]],
        limit: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def create_owner_escalation(
        self,
        *,
        owner_handle: str,
        agent_id: Optional[str],
        town_id: Optional[str],
        kind: str,
        summary: str,
        payload: Optional[dict[str, Any]],
        dedupe_key: Optional[str],
    ) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def resolve_owner_escalation(
        self,
        *,
        owner_handle: str,
        escalation_id: str,
        resolution_note: Optional[str],
    ) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def create_dm_request(
        self,
        *,
        from_agent_id: str,
        to_agent_id: str,
        town_id: Optional[str],
        message: str,
    ) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def list_dm_requests(
        self,
        *,
        agent_id: str,
        direction: str,
        statuses: Optional[list[str]],
        limit: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def decide_dm_request(
        self,
        *,
        agent_id: str,
        request_id: str,
        approve: bool,
    ) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def list_dm_conversations(self, *, agent_id: str, limit: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_dm_conversation(
        self,
        *,
        agent_id: str,
        conversation_id: str,
        limit_messages: int,
    ) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def send_dm_message(
        self,
        *,
        sender_agent_id: str,
        conversation_id: str,
        body: str,
        needs_human_input: bool,
        metadata: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def close_conversations_for_agent(
        self,
        *,
        agent_id: str,
        town_id: Optional[str],
    ) -> int:
        raise NotImplementedError


class SQLiteInteractionHubStore(InteractionHubStore):
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
                CREATE TABLE IF NOT EXISTS agent_inbox_items (
                  id TEXT PRIMARY KEY,
                  agent_id TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  summary TEXT NOT NULL,
                  payload_json TEXT NOT NULL DEFAULT '{{}}',
                  status TEXT NOT NULL DEFAULT 'pending',
                  dedupe_key TEXT,
                  created_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()}),
                  read_at TEXT,
                  resolved_at TEXT,
                  updated_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()}),
                  FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_inbox_items_agent_status_created "
                "ON agent_inbox_items(agent_id, status, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_inbox_items_dedupe ON agent_inbox_items(agent_id, dedupe_key)"
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS owner_escalations (
                  id TEXT PRIMARY KEY,
                  owner_handle TEXT NOT NULL,
                  agent_id TEXT,
                  town_id TEXT,
                  kind TEXT NOT NULL,
                  summary TEXT NOT NULL,
                  payload_json TEXT NOT NULL DEFAULT '{{}}',
                  status TEXT NOT NULL DEFAULT 'open',
                  dedupe_key TEXT,
                  created_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()}),
                  resolved_at TEXT,
                  updated_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()}),
                  FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE SET NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_owner_escalations_owner_status_created "
                "ON owner_escalations(owner_handle, status, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_owner_escalations_dedupe "
                "ON owner_escalations(owner_handle, dedupe_key)"
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS dm_requests (
                  id TEXT PRIMARY KEY,
                  from_agent_id TEXT NOT NULL,
                  to_agent_id TEXT NOT NULL,
                  town_id TEXT,
                  message TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'pending',
                  created_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()}),
                  updated_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()}),
                  resolved_at TEXT,
                  FOREIGN KEY (from_agent_id) REFERENCES agents(id) ON DELETE CASCADE,
                  FOREIGN KEY (to_agent_id) REFERENCES agents(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dm_requests_to_status_created "
                "ON dm_requests(to_agent_id, status, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dm_requests_from_status_created "
                "ON dm_requests(from_agent_id, status, created_at DESC)"
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS dm_conversations (
                  id TEXT PRIMARY KEY,
                  agent_a_id TEXT NOT NULL,
                  agent_b_id TEXT NOT NULL,
                  town_id TEXT,
                  status TEXT NOT NULL DEFAULT 'active',
                  created_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()}),
                  updated_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()}),
                  last_message_at TEXT,
                  FOREIGN KEY (agent_a_id) REFERENCES agents(id) ON DELETE CASCADE,
                  FOREIGN KEY (agent_b_id) REFERENCES agents(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dm_conversations_participants "
                "ON dm_conversations(agent_a_id, agent_b_id, status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dm_conversations_town_status "
                "ON dm_conversations(town_id, status, updated_at DESC)"
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS dm_messages (
                  id TEXT PRIMARY KEY,
                  conversation_id TEXT NOT NULL,
                  sender_agent_id TEXT NOT NULL,
                  body TEXT NOT NULL,
                  needs_human_input INTEGER NOT NULL DEFAULT 0,
                  metadata_json TEXT NOT NULL DEFAULT '{{}}',
                  created_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()}),
                  FOREIGN KEY (conversation_id) REFERENCES dm_conversations(id) ON DELETE CASCADE,
                  FOREIGN KEY (sender_agent_id) REFERENCES agents(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dm_messages_conversation_created "
                "ON dm_messages(conversation_id, created_at DESC)"
            )
        self._initialized = True

    def _open_inbox_dedupe(
        self,
        *,
        conn: sqlite3.Connection,
        agent_id: str,
        dedupe_key: str,
    ) -> Optional[dict[str, Any]]:
        row = conn.execute(
            """
            SELECT *
            FROM agent_inbox_items
            WHERE agent_id = ? AND dedupe_key = ? AND status IN ('pending', 'read')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (agent_id, dedupe_key),
        ).fetchone()
        return dict(row) if row else None

    def _open_escalation_dedupe(
        self,
        *,
        conn: sqlite3.Connection,
        owner_handle: str,
        dedupe_key: str,
    ) -> Optional[dict[str, Any]]:
        row = conn.execute(
            """
            SELECT *
            FROM owner_escalations
            WHERE owner_handle = ? AND dedupe_key = ? AND status = 'open'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (owner_handle, dedupe_key),
        ).fetchone()
        return dict(row) if row else None

    def _insert_inbox_item(
        self,
        *,
        conn: sqlite3.Connection,
        agent_id: str,
        kind: str,
        summary: str,
        payload: Optional[dict[str, Any]],
        dedupe_key: Optional[str],
    ) -> Optional[dict[str, Any]]:
        existing_agent = conn.execute("SELECT 1 FROM agents WHERE id = ?", (agent_id,)).fetchone()
        if not existing_agent:
            return None

        dedupe = str(dedupe_key or "").strip() or None
        if dedupe:
            existing = self._open_inbox_dedupe(conn=conn, agent_id=agent_id, dedupe_key=dedupe)
            if existing:
                return _inbox_row(existing)

        item_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO agent_inbox_items
              (id, agent_id, kind, summary, payload_json, status, dedupe_key)
            VALUES (?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                item_id,
                agent_id,
                str(kind or "notice"),
                str(summary or "").strip()[:500],
                json.dumps(payload or {}, separators=(",", ":")),
                dedupe,
            ),
        )
        row = conn.execute("SELECT * FROM agent_inbox_items WHERE id = ?", (item_id,)).fetchone()
        return _inbox_row(dict(row)) if row else None

    def _insert_owner_escalation(
        self,
        *,
        conn: sqlite3.Connection,
        owner_handle: str,
        agent_id: Optional[str],
        town_id: Optional[str],
        kind: str,
        summary: str,
        payload: Optional[dict[str, Any]],
        dedupe_key: Optional[str],
    ) -> Optional[dict[str, Any]]:
        normalized_owner = str(owner_handle or "").strip()
        if not normalized_owner:
            return None
        normalized_agent = str(agent_id or "").strip() or None
        if normalized_agent:
            exists = conn.execute("SELECT 1 FROM agents WHERE id = ?", (normalized_agent,)).fetchone()
            if not exists:
                normalized_agent = None
        dedupe = str(dedupe_key or "").strip() or None
        if dedupe:
            existing = self._open_escalation_dedupe(conn=conn, owner_handle=normalized_owner, dedupe_key=dedupe)
            if existing:
                return _escalation_row(existing)
        escalation_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO owner_escalations
              (id, owner_handle, agent_id, town_id, kind, summary, payload_json, status, dedupe_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?)
            """,
            (
                escalation_id,
                normalized_owner,
                normalized_agent,
                (str(town_id).strip() if town_id else None),
                str(kind or "needs_review"),
                str(summary or "").strip()[:500],
                json.dumps(payload or {}, separators=(",", ":")),
                dedupe,
            ),
        )
        row = conn.execute("SELECT * FROM owner_escalations WHERE id = ?", (escalation_id,)).fetchone()
        return _escalation_row(dict(row)) if row else None

    def _agent_owner(self, *, conn: sqlite3.Connection, agent_id: str) -> Optional[str]:
        row = conn.execute("SELECT owner_handle FROM agents WHERE id = ?", (agent_id,)).fetchone()
        if not row:
            return None
        owner = str(row["owner_handle"] or "").strip()
        return owner or None

    def _ensure_conversation(
        self,
        *,
        conn: sqlite3.Connection,
        first_agent_id: str,
        second_agent_id: str,
        town_id: Optional[str],
    ) -> Optional[dict[str, Any]]:
        agent_a, agent_b = _conversation_pair(first_agent_id, second_agent_id)
        row = conn.execute(
            """
            SELECT *
            FROM dm_conversations
            WHERE agent_a_id = ? AND agent_b_id = ? AND COALESCE(town_id, '') = COALESCE(?, '')
              AND status = 'active'
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (agent_a, agent_b, town_id),
        ).fetchone()
        if row:
            return _dm_conversation_row(dict(row))

        conv_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO dm_conversations
              (id, agent_a_id, agent_b_id, town_id, status)
            VALUES (?, ?, ?, ?, 'active')
            """,
            (conv_id, agent_a, agent_b, town_id),
        )
        created = conn.execute("SELECT * FROM dm_conversations WHERE id = ?", (conv_id,)).fetchone()
        if not created:
            return None
        return _dm_conversation_row(dict(created))

    def list_agent_inbox(
        self,
        *,
        agent_id: str,
        statuses: Optional[list[str]],
        limit: int,
    ) -> list[dict[str, Any]]:
        self.init_db()
        status_filters = _normalize_status_filters(statuses, allowed=INBOX_STATUSES)
        with self._connect() as conn:
            params: list[Any] = [agent_id]
            query = "SELECT * FROM agent_inbox_items WHERE agent_id = ?"
            if status_filters:
                placeholders = ",".join("?" for _ in status_filters)
                query += f" AND status IN ({placeholders})"
                params.extend(status_filters)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(max(1, min(200, int(limit))))
            rows = conn.execute(query, tuple(params)).fetchall()
        return [_inbox_row(dict(row)) for row in rows]

    def mark_inbox_read(self, *, agent_id: str, item_id: str) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE agent_inbox_items
                SET status = CASE WHEN status = 'pending' THEN 'read' ELSE status END,
                    read_at = COALESCE(read_at, ({_now_utc_sqlite()})),
                    updated_at = ({_now_utc_sqlite()})
                WHERE id = ? AND agent_id = ?
                """,
                (item_id, agent_id),
            )
            row = conn.execute(
                "SELECT * FROM agent_inbox_items WHERE id = ? AND agent_id = ?",
                (item_id, agent_id),
            ).fetchone()
        return _inbox_row(dict(row)) if row else None

    def create_inbox_item(
        self,
        *,
        agent_id: str,
        kind: str,
        summary: str,
        payload: Optional[dict[str, Any]],
        dedupe_key: Optional[str],
    ) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            return self._insert_inbox_item(
                conn=conn,
                agent_id=agent_id,
                kind=kind,
                summary=summary,
                payload=payload,
                dedupe_key=dedupe_key,
            )

    def list_owner_escalations(
        self,
        *,
        owner_handle: str,
        statuses: Optional[list[str]],
        limit: int,
    ) -> list[dict[str, Any]]:
        self.init_db()
        status_filters = _normalize_status_filters(statuses, allowed=ESCALATION_STATUSES)
        with self._connect() as conn:
            params: list[Any] = [owner_handle]
            query = "SELECT * FROM owner_escalations WHERE owner_handle = ?"
            if status_filters:
                placeholders = ",".join("?" for _ in status_filters)
                query += f" AND status IN ({placeholders})"
                params.extend(status_filters)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(max(1, min(200, int(limit))))
            rows = conn.execute(query, tuple(params)).fetchall()
        return [_escalation_row(dict(row)) for row in rows]

    def create_owner_escalation(
        self,
        *,
        owner_handle: str,
        agent_id: Optional[str],
        town_id: Optional[str],
        kind: str,
        summary: str,
        payload: Optional[dict[str, Any]],
        dedupe_key: Optional[str],
    ) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            return self._insert_owner_escalation(
                conn=conn,
                owner_handle=owner_handle,
                agent_id=agent_id,
                town_id=town_id,
                kind=kind,
                summary=summary,
                payload=payload,
                dedupe_key=dedupe_key,
            )

    def resolve_owner_escalation(
        self,
        *,
        owner_handle: str,
        escalation_id: str,
        resolution_note: Optional[str],
    ) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM owner_escalations WHERE id = ? AND owner_handle = ?",
                (escalation_id, owner_handle),
            ).fetchone()
            if not row:
                return None
            current_payload = _json_object(row["payload_json"])
            if resolution_note:
                current_payload["resolution_note"] = str(resolution_note)[:500]
            conn.execute(
                f"""
                UPDATE owner_escalations
                SET status = 'resolved',
                    payload_json = ?,
                    resolved_at = COALESCE(resolved_at, ({_now_utc_sqlite()})),
                    updated_at = ({_now_utc_sqlite()})
                WHERE id = ? AND owner_handle = ?
                """,
                (
                    json.dumps(current_payload, separators=(",", ":")),
                    escalation_id,
                    owner_handle,
                ),
            )
            updated = conn.execute(
                "SELECT * FROM owner_escalations WHERE id = ? AND owner_handle = ?",
                (escalation_id, owner_handle),
            ).fetchone()
        return _escalation_row(dict(updated)) if updated else None

    def create_dm_request(
        self,
        *,
        from_agent_id: str,
        to_agent_id: str,
        town_id: Optional[str],
        message: str,
    ) -> Optional[dict[str, Any]]:
        self.init_db()
        if str(from_agent_id).strip() == str(to_agent_id).strip():
            return None
        with self._connect() as conn:
            for agent_id in (from_agent_id, to_agent_id):
                exists = conn.execute("SELECT 1 FROM agents WHERE id = ?", (agent_id,)).fetchone()
                if not exists:
                    return None
            existing = conn.execute(
                """
                SELECT *
                FROM dm_requests
                WHERE status = 'pending'
                  AND COALESCE(town_id, '') = COALESCE(?, '')
                  AND (
                    (from_agent_id = ? AND to_agent_id = ?)
                    OR (from_agent_id = ? AND to_agent_id = ?)
                  )
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (town_id, from_agent_id, to_agent_id, to_agent_id, from_agent_id),
            ).fetchone()
            if existing:
                return _dm_request_row(dict(existing))

            request_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO dm_requests
                  (id, from_agent_id, to_agent_id, town_id, message, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
                """,
                (
                    request_id,
                    from_agent_id,
                    to_agent_id,
                    (str(town_id).strip() if town_id else None),
                    str(message or "").strip()[:1000],
                ),
            )
            self._insert_inbox_item(
                conn=conn,
                agent_id=to_agent_id,
                kind="dm_request",
                summary=f"DM request from {from_agent_id[:8]}",
                payload={
                    "request_id": request_id,
                    "from_agent_id": from_agent_id,
                    "town_id": town_id,
                    "message": str(message or "").strip()[:1000],
                },
                dedupe_key=f"dm_request:{request_id}",
            )
            row = conn.execute("SELECT * FROM dm_requests WHERE id = ?", (request_id,)).fetchone()
        return _dm_request_row(dict(row)) if row else None

    def list_dm_requests(
        self,
        *,
        agent_id: str,
        direction: str,
        statuses: Optional[list[str]],
        limit: int,
    ) -> list[dict[str, Any]]:
        self.init_db()
        normalized_direction = _normalize_direction(direction)
        status_filters = _normalize_status_filters(statuses, allowed=DM_REQUEST_STATUSES)
        with self._connect() as conn:
            params: list[Any] = []
            query = "SELECT * FROM dm_requests WHERE "
            if normalized_direction == "inbox":
                query += "to_agent_id = ?"
                params.append(agent_id)
            elif normalized_direction == "outbox":
                query += "from_agent_id = ?"
                params.append(agent_id)
            else:
                query += "(from_agent_id = ? OR to_agent_id = ?)"
                params.extend((agent_id, agent_id))
            if status_filters:
                placeholders = ",".join("?" for _ in status_filters)
                query += f" AND status IN ({placeholders})"
                params.extend(status_filters)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(max(1, min(200, int(limit))))
            rows = conn.execute(query, tuple(params)).fetchall()
        return [_dm_request_row(dict(row)) for row in rows]

    def decide_dm_request(
        self,
        *,
        agent_id: str,
        request_id: str,
        approve: bool,
    ) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM dm_requests WHERE id = ?",
                (request_id,),
            ).fetchone()
            if not row:
                return None
            request_row = dict(row)
            if str(request_row.get("to_agent_id") or "") != str(agent_id):
                return None
            if _normalize_dm_request_status(str(request_row.get("status"))) != "pending":
                return {
                    "request": _dm_request_row(request_row),
                    "conversation": None,
                }

            status = "approved" if approve else "rejected"
            conn.execute(
                f"""
                UPDATE dm_requests
                SET status = ?,
                    updated_at = ({_now_utc_sqlite()}),
                    resolved_at = ({_now_utc_sqlite()})
                WHERE id = ?
                """,
                (status, request_id),
            )
            updated = conn.execute("SELECT * FROM dm_requests WHERE id = ?", (request_id,)).fetchone()
            if updated is None:
                return None
            updated_request = _dm_request_row(dict(updated))

            conversation: dict[str, Any] | None = None
            if approve:
                conversation = self._ensure_conversation(
                    conn=conn,
                    first_agent_id=str(updated_request["from_agent_id"]),
                    second_agent_id=str(updated_request["to_agent_id"]),
                    town_id=(str(updated_request["town_id"]) if updated_request.get("town_id") is not None else None),
                )
            self._insert_inbox_item(
                conn=conn,
                agent_id=str(updated_request["from_agent_id"]),
                kind="dm_request_result",
                summary=f"DM request {status}",
                payload={
                    "request_id": str(updated_request["id"]),
                    "status": status,
                    "conversation_id": (conversation or {}).get("id"),
                },
                dedupe_key=f"dm_request_result:{updated_request['id']}",
            )
        return {
            "request": updated_request,
            "conversation": conversation,
        }

    def list_dm_conversations(self, *, agent_id: str, limit: int) -> list[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM dm_conversations
                WHERE agent_a_id = ? OR agent_b_id = ?
                ORDER BY COALESCE(last_message_at, updated_at) DESC
                LIMIT ?
                """,
                (agent_id, agent_id, max(1, min(200, int(limit)))),
            ).fetchall()
        return [_dm_conversation_row(dict(row), viewer_agent_id=agent_id) for row in rows]

    def get_dm_conversation(
        self,
        *,
        agent_id: str,
        conversation_id: str,
        limit_messages: int,
    ) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM dm_conversations
                WHERE id = ? AND (agent_a_id = ? OR agent_b_id = ?)
                """,
                (conversation_id, agent_id, agent_id),
            ).fetchone()
            if not row:
                return None
            conv = _dm_conversation_row(dict(row), viewer_agent_id=agent_id)
            message_rows = conn.execute(
                """
                SELECT *
                FROM dm_messages
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (conversation_id, max(1, min(500, int(limit_messages)))),
            ).fetchall()
        messages = [_dm_message_row(dict(message_row)) for message_row in reversed(message_rows)]
        return {
            "conversation": conv,
            "messages": messages,
        }

    def send_dm_message(
        self,
        *,
        sender_agent_id: str,
        conversation_id: str,
        body: str,
        needs_human_input: bool,
        metadata: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM dm_conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if not row:
                return None
            conv = dict(row)
            if _normalize_dm_conversation_status(str(conv.get("status"))) != "active":
                return None
            agent_a = str(conv.get("agent_a_id") or "")
            agent_b = str(conv.get("agent_b_id") or "")
            sender = str(sender_agent_id).strip()
            if sender not in {agent_a, agent_b}:
                return None
            receiver = agent_b if sender == agent_a else agent_a

            message_id = str(uuid.uuid4())
            normalized_body = str(body or "").strip()
            if not normalized_body:
                return None

            conn.execute(
                """
                INSERT INTO dm_messages
                  (id, conversation_id, sender_agent_id, body, needs_human_input, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    sender,
                    normalized_body[:2000],
                    1 if needs_human_input else 0,
                    json.dumps(metadata or {}, separators=(",", ":")),
                ),
            )
            conn.execute(
                f"""
                UPDATE dm_conversations
                SET updated_at = ({_now_utc_sqlite()}),
                    last_message_at = ({_now_utc_sqlite()})
                WHERE id = ?
                """,
                (conversation_id,),
            )
            inserted = conn.execute("SELECT * FROM dm_messages WHERE id = ?", (message_id,)).fetchone()
            if not inserted:
                return None
            message = _dm_message_row(dict(inserted))

            self._insert_inbox_item(
                conn=conn,
                agent_id=receiver,
                kind="dm_message",
                summary=f"New DM from {sender[:8]}",
                payload={
                    "conversation_id": conversation_id,
                    "message_id": message_id,
                    "sender_agent_id": sender,
                    "body": normalized_body[:300],
                },
                dedupe_key=f"dm_message:{message_id}",
            )

            if needs_human_input:
                owner = self._agent_owner(conn=conn, agent_id=sender)
                if owner:
                    self._insert_owner_escalation(
                        conn=conn,
                        owner_handle=owner,
                        agent_id=sender,
                        town_id=(str(conv.get("town_id")) if conv.get("town_id") is not None else None),
                        kind="dm_message_needs_human",
                        summary="DM message flagged for human review",
                        payload={
                            "conversation_id": conversation_id,
                            "message_id": message_id,
                            "sender_agent_id": sender,
                            "receiver_agent_id": receiver,
                        },
                        dedupe_key=f"dm_message_human:{message_id}",
                    )

        return message

    def close_conversations_for_agent(
        self,
        *,
        agent_id: str,
        town_id: Optional[str],
    ) -> int:
        agent_id = str(agent_id).strip()
        if not agent_id:
            return 0
        self.init_db()
        closed = 0
        with self._connect() as conn:
            if town_id:
                cur = conn.execute(
                    "UPDATE dm_conversations SET status = 'closed', "
                    f"updated_at = {_now_utc_sqlite()} "
                    "WHERE status = 'active' AND town_id = ? "
                    "AND (agent_a_id = ? OR agent_b_id = ?)",
                    (str(town_id), agent_id, agent_id),
                )
            else:
                cur = conn.execute(
                    "UPDATE dm_conversations SET status = 'closed', "
                    f"updated_at = {_now_utc_sqlite()} "
                    "WHERE status = 'active' "
                    "AND (agent_a_id = ? OR agent_b_id = ?)",
                    (agent_id, agent_id),
                )
            closed = cur.rowcount
            conn.execute(
                "UPDATE dm_requests SET status = 'expired', "
                f"updated_at = {_now_utc_sqlite()} "
                "WHERE status = 'pending' "
                "AND (from_agent_id = ? OR to_agent_id = ?)",
                (agent_id, agent_id),
            )
        return closed


class PostgresInteractionHubStore(InteractionHubStore):
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

    def _open_inbox_dedupe(
        self,
        *,
        cur: Any,
        agent_id: str,
        dedupe_key: str,
    ) -> Optional[dict[str, Any]]:
        cur.execute(
            """
            SELECT *
            FROM agent_inbox_items
            WHERE agent_id = %s::uuid AND dedupe_key = %s AND status IN ('pending', 'read')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (agent_id, dedupe_key),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def _open_escalation_dedupe(
        self,
        *,
        cur: Any,
        owner_handle: str,
        dedupe_key: str,
    ) -> Optional[dict[str, Any]]:
        cur.execute(
            """
            SELECT *
            FROM owner_escalations
            WHERE owner_handle = %s AND dedupe_key = %s AND status = 'open'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (owner_handle, dedupe_key),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def _insert_inbox_item(
        self,
        *,
        cur: Any,
        agent_id: str,
        kind: str,
        summary: str,
        payload: Optional[dict[str, Any]],
        dedupe_key: Optional[str],
    ) -> Optional[dict[str, Any]]:
        cur.execute("SELECT 1 FROM agents WHERE id = %s::uuid", (agent_id,))
        if not cur.fetchone():
            return None
        dedupe = str(dedupe_key or "").strip() or None
        if dedupe:
            existing = self._open_inbox_dedupe(cur=cur, agent_id=agent_id, dedupe_key=dedupe)
            if existing:
                return _inbox_row(existing)
        item_id = str(uuid.uuid4())
        cur.execute(
            """
            INSERT INTO agent_inbox_items
              (id, agent_id, kind, summary, payload_json, status, dedupe_key, updated_at)
            VALUES (%s::uuid, %s::uuid, %s, %s, %s::jsonb, 'pending', %s, NOW())
            RETURNING *
            """,
            (
                item_id,
                agent_id,
                str(kind or "notice"),
                str(summary or "").strip()[:500],
                json.dumps(payload or {}, separators=(",", ":")),
                dedupe,
            ),
        )
        row = cur.fetchone()
        return _inbox_row(dict(row)) if row else None

    def _insert_owner_escalation(
        self,
        *,
        cur: Any,
        owner_handle: str,
        agent_id: Optional[str],
        town_id: Optional[str],
        kind: str,
        summary: str,
        payload: Optional[dict[str, Any]],
        dedupe_key: Optional[str],
    ) -> Optional[dict[str, Any]]:
        normalized_owner = str(owner_handle or "").strip()
        if not normalized_owner:
            return None
        normalized_agent = str(agent_id or "").strip() or None
        if normalized_agent:
            cur.execute("SELECT 1 FROM agents WHERE id = %s::uuid", (normalized_agent,))
            if not cur.fetchone():
                normalized_agent = None
        dedupe = str(dedupe_key or "").strip() or None
        if dedupe:
            existing = self._open_escalation_dedupe(cur=cur, owner_handle=normalized_owner, dedupe_key=dedupe)
            if existing:
                return _escalation_row(existing)
        escalation_id = str(uuid.uuid4())
        cur.execute(
            """
            INSERT INTO owner_escalations
              (id, owner_handle, agent_id, town_id, kind, summary, payload_json, status, dedupe_key, updated_at)
            VALUES (%s::uuid, %s, %s::uuid, %s, %s, %s, %s::jsonb, 'open', %s, NOW())
            RETURNING *
            """,
            (
                escalation_id,
                normalized_owner,
                normalized_agent,
                (str(town_id).strip() if town_id else None),
                str(kind or "needs_review"),
                str(summary or "").strip()[:500],
                json.dumps(payload or {}, separators=(",", ":")),
                dedupe,
            ),
        )
        row = cur.fetchone()
        return _escalation_row(dict(row)) if row else None

    def _agent_owner(self, *, cur: Any, agent_id: str) -> Optional[str]:
        cur.execute("SELECT owner_handle FROM agents WHERE id = %s::uuid", (agent_id,))
        row = cur.fetchone()
        if not row:
            return None
        owner = str(row.get("owner_handle") or "").strip()
        return owner or None

    def _ensure_conversation(
        self,
        *,
        cur: Any,
        first_agent_id: str,
        second_agent_id: str,
        town_id: Optional[str],
    ) -> Optional[dict[str, Any]]:
        agent_a, agent_b = _conversation_pair(first_agent_id, second_agent_id)
        cur.execute(
            """
            SELECT *
            FROM dm_conversations
            WHERE agent_a_id = %s::uuid AND agent_b_id = %s::uuid
              AND COALESCE(town_id, '') = COALESCE(%s, '')
              AND status = 'active'
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (agent_a, agent_b, town_id),
        )
        row = cur.fetchone()
        if row:
            return _dm_conversation_row(dict(row))

        conv_id = str(uuid.uuid4())
        cur.execute(
            """
            INSERT INTO dm_conversations
              (id, agent_a_id, agent_b_id, town_id, status, updated_at)
            VALUES (%s::uuid, %s::uuid, %s::uuid, %s, 'active', NOW())
            RETURNING *
            """,
            (conv_id, agent_a, agent_b, town_id),
        )
        created = cur.fetchone()
        return _dm_conversation_row(dict(created)) if created else None

    def list_agent_inbox(
        self,
        *,
        agent_id: str,
        statuses: Optional[list[str]],
        limit: int,
    ) -> list[dict[str, Any]]:
        self.init_db()
        status_filters = _normalize_status_filters(statuses, allowed=INBOX_STATUSES)
        with self._connect() as conn:
            with conn.cursor() as cur:
                if status_filters:
                    cur.execute(
                        """
                        SELECT *
                        FROM agent_inbox_items
                        WHERE agent_id = %s::uuid AND status = ANY(%s::text[])
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (agent_id, status_filters, max(1, min(200, int(limit)))),
                    )
                else:
                    cur.execute(
                        """
                        SELECT *
                        FROM agent_inbox_items
                        WHERE agent_id = %s::uuid
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (agent_id, max(1, min(200, int(limit)))),
                    )
                rows = cur.fetchall()
        return [_inbox_row(dict(row)) for row in rows]

    def mark_inbox_read(self, *, agent_id: str, item_id: str) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE agent_inbox_items
                    SET status = CASE WHEN status = 'pending' THEN 'read' ELSE status END,
                        read_at = COALESCE(read_at, NOW()),
                        updated_at = NOW()
                    WHERE id = %s::uuid AND agent_id = %s::uuid
                    RETURNING *
                    """,
                    (item_id, agent_id),
                )
                row = cur.fetchone()
        return _inbox_row(dict(row)) if row else None

    def create_inbox_item(
        self,
        *,
        agent_id: str,
        kind: str,
        summary: str,
        payload: Optional[dict[str, Any]],
        dedupe_key: Optional[str],
    ) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                return self._insert_inbox_item(
                    cur=cur,
                    agent_id=agent_id,
                    kind=kind,
                    summary=summary,
                    payload=payload,
                    dedupe_key=dedupe_key,
                )

    def list_owner_escalations(
        self,
        *,
        owner_handle: str,
        statuses: Optional[list[str]],
        limit: int,
    ) -> list[dict[str, Any]]:
        self.init_db()
        status_filters = _normalize_status_filters(statuses, allowed=ESCALATION_STATUSES)
        with self._connect() as conn:
            with conn.cursor() as cur:
                if status_filters:
                    cur.execute(
                        """
                        SELECT *
                        FROM owner_escalations
                        WHERE owner_handle = %s AND status = ANY(%s::text[])
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (owner_handle, status_filters, max(1, min(200, int(limit)))),
                    )
                else:
                    cur.execute(
                        """
                        SELECT *
                        FROM owner_escalations
                        WHERE owner_handle = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (owner_handle, max(1, min(200, int(limit)))),
                    )
                rows = cur.fetchall()
        return [_escalation_row(dict(row)) for row in rows]

    def create_owner_escalation(
        self,
        *,
        owner_handle: str,
        agent_id: Optional[str],
        town_id: Optional[str],
        kind: str,
        summary: str,
        payload: Optional[dict[str, Any]],
        dedupe_key: Optional[str],
    ) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                return self._insert_owner_escalation(
                    cur=cur,
                    owner_handle=owner_handle,
                    agent_id=agent_id,
                    town_id=town_id,
                    kind=kind,
                    summary=summary,
                    payload=payload,
                    dedupe_key=dedupe_key,
                )

    def resolve_owner_escalation(
        self,
        *,
        owner_handle: str,
        escalation_id: str,
        resolution_note: Optional[str],
    ) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM owner_escalations WHERE id = %s::uuid AND owner_handle = %s",
                    (escalation_id, owner_handle),
                )
                row = cur.fetchone()
                if not row:
                    return None
                payload = _json_object(row.get("payload_json"))
                if resolution_note:
                    payload["resolution_note"] = str(resolution_note)[:500]
                cur.execute(
                    """
                    UPDATE owner_escalations
                    SET status = 'resolved',
                        payload_json = %s::jsonb,
                        resolved_at = COALESCE(resolved_at, NOW()),
                        updated_at = NOW()
                    WHERE id = %s::uuid AND owner_handle = %s
                    RETURNING *
                    """,
                    (json.dumps(payload, separators=(",", ":")), escalation_id, owner_handle),
                )
                updated = cur.fetchone()
        return _escalation_row(dict(updated)) if updated else None

    def create_dm_request(
        self,
        *,
        from_agent_id: str,
        to_agent_id: str,
        town_id: Optional[str],
        message: str,
    ) -> Optional[dict[str, Any]]:
        self.init_db()
        if str(from_agent_id).strip() == str(to_agent_id).strip():
            return None
        with self._connect() as conn:
            with conn.cursor() as cur:
                for agent_id in (from_agent_id, to_agent_id):
                    cur.execute("SELECT 1 FROM agents WHERE id = %s::uuid", (agent_id,))
                    if not cur.fetchone():
                        return None
                cur.execute(
                    """
                    SELECT *
                    FROM dm_requests
                    WHERE status = 'pending'
                      AND COALESCE(town_id, '') = COALESCE(%s, '')
                      AND (
                        (from_agent_id = %s::uuid AND to_agent_id = %s::uuid)
                        OR (from_agent_id = %s::uuid AND to_agent_id = %s::uuid)
                      )
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (town_id, from_agent_id, to_agent_id, to_agent_id, from_agent_id),
                )
                existing = cur.fetchone()
                if existing:
                    return _dm_request_row(dict(existing))
                request_id = str(uuid.uuid4())
                cur.execute(
                    """
                    INSERT INTO dm_requests
                      (id, from_agent_id, to_agent_id, town_id, message, status, updated_at)
                    VALUES (%s::uuid, %s::uuid, %s::uuid, %s, %s, 'pending', NOW())
                    RETURNING *
                    """,
                    (
                        request_id,
                        from_agent_id,
                        to_agent_id,
                        town_id,
                        str(message or "").strip()[:1000],
                    ),
                )
                row = cur.fetchone()
                self._insert_inbox_item(
                    cur=cur,
                    agent_id=to_agent_id,
                    kind="dm_request",
                    summary=f"DM request from {from_agent_id[:8]}",
                    payload={
                        "request_id": request_id,
                        "from_agent_id": from_agent_id,
                        "town_id": town_id,
                        "message": str(message or "").strip()[:1000],
                    },
                    dedupe_key=f"dm_request:{request_id}",
                )
        return _dm_request_row(dict(row)) if row else None

    def list_dm_requests(
        self,
        *,
        agent_id: str,
        direction: str,
        statuses: Optional[list[str]],
        limit: int,
    ) -> list[dict[str, Any]]:
        self.init_db()
        normalized_direction = _normalize_direction(direction)
        status_filters = _normalize_status_filters(statuses, allowed=DM_REQUEST_STATUSES)
        with self._connect() as conn:
            with conn.cursor() as cur:
                if normalized_direction == "inbox":
                    where_clause = "to_agent_id = %s::uuid"
                    params: list[Any] = [agent_id]
                elif normalized_direction == "outbox":
                    where_clause = "from_agent_id = %s::uuid"
                    params = [agent_id]
                else:
                    where_clause = "(from_agent_id = %s::uuid OR to_agent_id = %s::uuid)"
                    params = [agent_id, agent_id]
                if status_filters:
                    where_clause += " AND status = ANY(%s::text[])"
                    params.append(status_filters)
                params.append(max(1, min(200, int(limit))))
                cur.execute(
                    f"""
                    SELECT *
                    FROM dm_requests
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
        return [_dm_request_row(dict(row)) for row in rows]

    def decide_dm_request(
        self,
        *,
        agent_id: str,
        request_id: str,
        approve: bool,
    ) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM dm_requests WHERE id = %s::uuid", (request_id,))
                row = cur.fetchone()
                if not row:
                    return None
                request_row = dict(row)
                if str(request_row.get("to_agent_id") or "") != str(agent_id):
                    return None
                if _normalize_dm_request_status(str(request_row.get("status"))) != "pending":
                    return {
                        "request": _dm_request_row(request_row),
                        "conversation": None,
                    }
                status = "approved" if approve else "rejected"
                cur.execute(
                    """
                    UPDATE dm_requests
                    SET status = %s,
                        updated_at = NOW(),
                        resolved_at = NOW()
                    WHERE id = %s::uuid
                    RETURNING *
                    """,
                    (status, request_id),
                )
                updated = cur.fetchone()
                if not updated:
                    return None
                updated_request = _dm_request_row(dict(updated))
                conversation: dict[str, Any] | None = None
                if approve:
                    conversation = self._ensure_conversation(
                        cur=cur,
                        first_agent_id=str(updated_request["from_agent_id"]),
                        second_agent_id=str(updated_request["to_agent_id"]),
                        town_id=(str(updated_request["town_id"]) if updated_request.get("town_id") is not None else None),
                    )
                self._insert_inbox_item(
                    cur=cur,
                    agent_id=str(updated_request["from_agent_id"]),
                    kind="dm_request_result",
                    summary=f"DM request {status}",
                    payload={
                        "request_id": str(updated_request["id"]),
                        "status": status,
                        "conversation_id": (conversation or {}).get("id"),
                    },
                    dedupe_key=f"dm_request_result:{updated_request['id']}",
                )
        return {
            "request": updated_request,
            "conversation": conversation,
        }

    def list_dm_conversations(self, *, agent_id: str, limit: int) -> list[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM dm_conversations
                    WHERE agent_a_id = %s::uuid OR agent_b_id = %s::uuid
                    ORDER BY COALESCE(last_message_at, updated_at) DESC
                    LIMIT %s
                    """,
                    (agent_id, agent_id, max(1, min(200, int(limit)))),
                )
                rows = cur.fetchall()
        return [_dm_conversation_row(dict(row), viewer_agent_id=agent_id) for row in rows]

    def get_dm_conversation(
        self,
        *,
        agent_id: str,
        conversation_id: str,
        limit_messages: int,
    ) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM dm_conversations
                    WHERE id = %s::uuid AND (agent_a_id = %s::uuid OR agent_b_id = %s::uuid)
                    """,
                    (conversation_id, agent_id, agent_id),
                )
                row = cur.fetchone()
                if not row:
                    return None
                conv = _dm_conversation_row(dict(row), viewer_agent_id=agent_id)
                cur.execute(
                    """
                    SELECT *
                    FROM dm_messages
                    WHERE conversation_id = %s::uuid
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (conversation_id, max(1, min(500, int(limit_messages)))),
                )
                message_rows = cur.fetchall()
        messages = [_dm_message_row(dict(message_row)) for message_row in reversed(message_rows)]
        return {
            "conversation": conv,
            "messages": messages,
        }

    def send_dm_message(
        self,
        *,
        sender_agent_id: str,
        conversation_id: str,
        body: str,
        needs_human_input: bool,
        metadata: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        self.init_db()
        normalized_body = str(body or "").strip()
        if not normalized_body:
            return None
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM dm_conversations WHERE id = %s::uuid", (conversation_id,))
                row = cur.fetchone()
                if not row:
                    return None
                conv = dict(row)
                if _normalize_dm_conversation_status(str(conv.get("status"))) != "active":
                    return None
                agent_a = str(conv.get("agent_a_id") or "")
                agent_b = str(conv.get("agent_b_id") or "")
                sender = str(sender_agent_id).strip()
                if sender not in {agent_a, agent_b}:
                    return None
                receiver = agent_b if sender == agent_a else agent_a
                message_id = str(uuid.uuid4())
                cur.execute(
                    """
                    INSERT INTO dm_messages
                      (id, conversation_id, sender_agent_id, body, needs_human_input, metadata_json)
                    VALUES (%s::uuid, %s::uuid, %s::uuid, %s, %s, %s::jsonb)
                    RETURNING *
                    """,
                    (
                        message_id,
                        conversation_id,
                        sender,
                        normalized_body[:2000],
                        bool(needs_human_input),
                        json.dumps(metadata or {}, separators=(",", ":")),
                    ),
                )
                inserted = cur.fetchone()
                if not inserted:
                    return None
                cur.execute(
                    """
                    UPDATE dm_conversations
                    SET updated_at = NOW(),
                        last_message_at = NOW()
                    WHERE id = %s::uuid
                    """,
                    (conversation_id,),
                )
                message = _dm_message_row(dict(inserted))
                self._insert_inbox_item(
                    cur=cur,
                    agent_id=receiver,
                    kind="dm_message",
                    summary=f"New DM from {sender[:8]}",
                    payload={
                        "conversation_id": conversation_id,
                        "message_id": message_id,
                        "sender_agent_id": sender,
                        "body": normalized_body[:300],
                    },
                    dedupe_key=f"dm_message:{message_id}",
                )
                if needs_human_input:
                    owner = self._agent_owner(cur=cur, agent_id=sender)
                    if owner:
                        self._insert_owner_escalation(
                            cur=cur,
                            owner_handle=owner,
                            agent_id=sender,
                            town_id=(str(conv.get("town_id")) if conv.get("town_id") is not None else None),
                            kind="dm_message_needs_human",
                            summary="DM message flagged for human review",
                            payload={
                                "conversation_id": conversation_id,
                                "message_id": message_id,
                                "sender_agent_id": sender,
                                "receiver_agent_id": receiver,
                            },
                            dedupe_key=f"dm_message_human:{message_id}",
                        )
        return message

    def close_conversations_for_agent(
        self,
        *,
        agent_id: str,
        town_id: Optional[str],
    ) -> int:
        agent_id = str(agent_id).strip()
        if not agent_id:
            return 0
        self.init_db()
        closed = 0
        with self._connect() as conn:
            with conn.cursor() as cur:
                if town_id:
                    cur.execute(
                        "UPDATE dm_conversations SET status = 'closed', "
                        "updated_at = NOW() "
                        "WHERE status = 'active' AND town_id = %s "
                        "AND (agent_a_id = %s OR agent_b_id = %s)",
                        (str(town_id), agent_id, agent_id),
                    )
                else:
                    cur.execute(
                        "UPDATE dm_conversations SET status = 'closed', "
                        "updated_at = NOW() "
                        "WHERE status = 'active' "
                        "AND (agent_a_id = %s OR agent_b_id = %s)",
                        (agent_id, agent_id),
                    )
                closed = cur.rowcount
                cur.execute(
                    "UPDATE dm_requests SET status = 'expired', "
                    "updated_at = NOW() "
                    "WHERE status = 'pending' "
                    "AND (from_agent_id = %s OR to_agent_id = %s)",
                    (agent_id, agent_id),
                )
        return closed


def _resolve_sqlite_path(database_url: Optional[str]) -> Path:
    if database_url and database_url.startswith("sqlite:///"):
        raw = database_url[len("sqlite:///") :]
        path = Path(raw)
        if not path.is_absolute():
            path = (WORKSPACE_ROOT / path).resolve()
        return path
    raw = os.environ.get("VVALLEY_DB_PATH", str(WORKSPACE_ROOT / "data" / "vvalley.db"))
    path = Path(raw)
    if not path.is_absolute():
        path = (WORKSPACE_ROOT / path).resolve()
    return path


def _database_url() -> Optional[str]:
    return os.environ.get("DATABASE_URL")


@lru_cache(maxsize=1)
def _backend() -> InteractionHubStore:
    database_url = _database_url()
    if database_url and database_url.startswith(("postgres://", "postgresql://")):
        return PostgresInteractionHubStore(database_url)
    return SQLiteInteractionHubStore(_resolve_sqlite_path(database_url))


def reset_backend_cache_for_tests() -> None:
    _backend.cache_clear()


def init_db() -> None:
    _backend().init_db()


def list_agent_inbox(
    *,
    agent_id: str,
    statuses: Optional[list[str]] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    return _backend().list_agent_inbox(agent_id=agent_id, statuses=statuses, limit=limit)


def mark_inbox_read(*, agent_id: str, item_id: str) -> Optional[dict[str, Any]]:
    return _backend().mark_inbox_read(agent_id=agent_id, item_id=item_id)


def create_inbox_item(
    *,
    agent_id: str,
    kind: str,
    summary: str,
    payload: Optional[dict[str, Any]] = None,
    dedupe_key: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    return _backend().create_inbox_item(
        agent_id=agent_id,
        kind=kind,
        summary=summary,
        payload=payload,
        dedupe_key=dedupe_key,
    )


def list_owner_escalations(
    *,
    owner_handle: str,
    statuses: Optional[list[str]] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    return _backend().list_owner_escalations(owner_handle=owner_handle, statuses=statuses, limit=limit)


def create_owner_escalation(
    *,
    owner_handle: str,
    agent_id: Optional[str],
    town_id: Optional[str],
    kind: str,
    summary: str,
    payload: Optional[dict[str, Any]] = None,
    dedupe_key: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    return _backend().create_owner_escalation(
        owner_handle=owner_handle,
        agent_id=agent_id,
        town_id=town_id,
        kind=kind,
        summary=summary,
        payload=payload,
        dedupe_key=dedupe_key,
    )


def resolve_owner_escalation(
    *,
    owner_handle: str,
    escalation_id: str,
    resolution_note: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    return _backend().resolve_owner_escalation(
        owner_handle=owner_handle,
        escalation_id=escalation_id,
        resolution_note=resolution_note,
    )


def create_dm_request(
    *,
    from_agent_id: str,
    to_agent_id: str,
    town_id: Optional[str],
    message: str,
) -> Optional[dict[str, Any]]:
    return _backend().create_dm_request(
        from_agent_id=from_agent_id,
        to_agent_id=to_agent_id,
        town_id=town_id,
        message=message,
    )


def list_dm_requests(
    *,
    agent_id: str,
    direction: str = "inbox",
    statuses: Optional[list[str]] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    return _backend().list_dm_requests(
        agent_id=agent_id,
        direction=direction,
        statuses=statuses,
        limit=limit,
    )


def approve_dm_request(*, agent_id: str, request_id: str) -> Optional[dict[str, Any]]:
    return _backend().decide_dm_request(agent_id=agent_id, request_id=request_id, approve=True)


def reject_dm_request(*, agent_id: str, request_id: str) -> Optional[dict[str, Any]]:
    return _backend().decide_dm_request(agent_id=agent_id, request_id=request_id, approve=False)


def list_dm_conversations(*, agent_id: str, limit: int = 50) -> list[dict[str, Any]]:
    return _backend().list_dm_conversations(agent_id=agent_id, limit=limit)


def get_dm_conversation(
    *,
    agent_id: str,
    conversation_id: str,
    limit_messages: int = 100,
) -> Optional[dict[str, Any]]:
    return _backend().get_dm_conversation(
        agent_id=agent_id,
        conversation_id=conversation_id,
        limit_messages=limit_messages,
    )


def send_dm_message(
    *,
    sender_agent_id: str,
    conversation_id: str,
    body: str,
    needs_human_input: bool = False,
    metadata: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    return _backend().send_dm_message(
        sender_agent_id=sender_agent_id,
        conversation_id=conversation_id,
        body=body,
        needs_human_input=needs_human_input,
        metadata=metadata,
    )


def close_dm_conversations_for_agent(
    *,
    agent_id: str,
    town_id: Optional[str] = None,
) -> int:
    return _backend().close_conversations_for_agent(agent_id=agent_id, town_id=town_id)
