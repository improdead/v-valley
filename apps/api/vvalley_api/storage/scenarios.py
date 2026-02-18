"""Storage for competitive scenario definitions, queues, matches, ratings, and wallets."""

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


def _now_utc_sqlite() -> str:
    return "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        return "{}"


def _json_loads(raw: Any, default: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    text = str(raw or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _uuid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


DEFAULT_SCENARIO_DEFINITIONS: list[dict[str, Any]] = [
    {
        "scenario_key": "werewolf_6p",
        "name": "Werewolf (6-10 Players)",
        "description": "Social deduction with hidden roles and day/night phases.",
        "category": "social_deduction",
        "min_players": 6,
        "max_players": 10,
        "team_size": None,
        "warmup_steps": 2,
        "max_duration_steps": 30,
        "rules_json": {
            "engine": "werewolf",
            "ui_group": "social",
            "spectator_kind": "werewolf",
            "phase_sequence": ["night", "day"],
            "max_rounds": 6,
            "arena_id": "park_circle",
        },
        "rating_mode": "elo",
        "buy_in": 0,
        "enabled": 1,
    },
    {
        "scenario_key": "anaconda_standard",
        "name": "Anaconda Poker (3-7 Players)",
        "description": "Pass-the-trash poker with reveal rounds and betting.",
        "category": "card_game",
        "min_players": 3,
        "max_players": 7,
        "team_size": None,
        "warmup_steps": 2,
        "max_duration_steps": 20,
        "rules_json": {
            "engine": "anaconda",
            "ui_group": "casino",
            "spectator_kind": "anaconda",
            "phase_sequence": [
                "deal",
                "pass1",
                "pass2",
                "pass3",
                "discard",
                "reveal1",
                "reveal2",
                "reveal3",
                "reveal4",
                "reveal5",
                "showdown",
            ],
            "arena_id": "hobbs_cafe_table",
            "small_blind": 5,
            "big_blind": 10,
        },
        "rating_mode": "elo",
        "buy_in": 100,
        "enabled": 1,
    },
    {
        "scenario_key": "blackjack_tournament",
        "name": "Blackjack Tournament (2-6 Players)",
        "description": "Multiplayer tournament blackjack versus a shared dealer.",
        "category": "card_game",
        "min_players": 2,
        "max_players": 6,
        "team_size": None,
        "warmup_steps": 2,
        "max_duration_steps": 24,
        "rules_json": {
            "engine": "blackjack",
            "ui_group": "casino",
            "spectator_kind": "blackjack",
            "phase_sequence": ["deal", "player_turns", "dealer_turn", "settle"],
            "arena_id": "hobbs_cafe_table",
            "max_hands": 8,
            "min_bet": 5,
            "max_bet": 25,
            "dealer_soft17_stand": True,
            "allow_double": True,
            "allow_split": False,
            "allow_insurance": False,
        },
        "rating_mode": "elo",
        "buy_in": 50,
        "enabled": 1,
    },
    {
        "scenario_key": "holdem_fixed_limit",
        "name": "Texas Hold'em (Fixed Limit, 2-6 Players)",
        "description": "Single-hand fixed-limit hold'em with blinds and showdown.",
        "category": "card_game",
        "min_players": 2,
        "max_players": 6,
        "team_size": None,
        "warmup_steps": 2,
        "max_duration_steps": 18,
        "rules_json": {
            "engine": "holdem",
            "ui_group": "casino",
            "spectator_kind": "holdem",
            "phase_sequence": ["preflop", "flop", "turn", "river", "showdown"],
            "arena_id": "hobbs_cafe_table",
            "small_blind": 5,
            "big_blind": 10,
            "bet_units": [10, 10, 20, 20],
            "max_raises_per_street": 3,
            "hands_per_match": 1,
        },
        "rating_mode": "elo",
        "buy_in": 100,
        "enabled": 1,
    },
]


class ScenarioStore(ABC):
    @abstractmethod
    def init_db(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_scenarios(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_scenario(self, *, scenario_key: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def list_agent_queue(self, *, agent_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def list_queue(self, *, scenario_key: str, town_id: str | None = None, status: str = "queued") -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def enqueue(self, *, scenario_key: str, agent_id: str, town_id: str, rating_snapshot: int, party_id: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def update_queue_status(
        self,
        *,
        entry_id: str,
        status: str,
        match_id: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_agent_active_match(self, *, agent_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def list_active_matches(self, *, town_id: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def list_matches(self, *, town_id: str | None = None, scenario_key: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_match(self, *, match_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def create_match(
        self,
        *,
        scenario_key: str,
        town_id: str,
        arena_id: str,
        status: str,
        phase: str,
        phase_step: int,
        round_number: int,
        created_step: int,
        warmup_start_step: int | None,
        start_step: int | None,
        state_json: dict[str, Any] | None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def update_match(
        self,
        *,
        match_id: str,
        status: str | None = None,
        phase: str | None = None,
        phase_step: int | None = None,
        round_number: int | None = None,
        warmup_start_step: int | None = None,
        start_step: int | None = None,
        end_step: int | None = None,
        state_json: dict[str, Any] | None = None,
        result_json: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def insert_match_participants(self, *, participants: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_match_participants(self, *, match_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def update_match_participant(self, *, match_id: str, agent_id: str, updates: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def insert_match_event(
        self,
        *,
        match_id: str,
        step: int,
        phase: str,
        event_type: str,
        agent_id: str | None,
        target_id: str | None,
        data_json: dict[str, Any] | None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_match_events(self, *, match_id: str, limit: int = 200) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_wallet(self, *, agent_id: str, create: bool = True) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def update_wallet(
        self,
        *,
        agent_id: str,
        delta: int,
        stipend_iso: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_rating(self, *, agent_id: str, scenario_key: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_agent_ratings(self, *, agent_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def upsert_rating(
        self,
        *,
        agent_id: str,
        scenario_key: str,
        rating: int,
        games_played: int,
        wins: int,
        losses: int,
        draws: int,
        peak_rating: int,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_leaderboard(self, *, scenario_key: str, limit: int = 100) -> list[dict[str, Any]]:
        raise NotImplementedError


class SQLiteScenarioStore(ScenarioStore):
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

    @staticmethod
    def _scenario_row(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        out["enabled"] = bool(out.get("enabled", 0))
        out["min_players"] = int(out.get("min_players") or 0)
        out["max_players"] = int(out.get("max_players") or 0)
        out["warmup_steps"] = int(out.get("warmup_steps") or 0)
        out["max_duration_steps"] = int(out.get("max_duration_steps") or 0)
        out["buy_in"] = int(out.get("buy_in") or 0)
        out["rules_json"] = _json_loads(out.get("rules_json"), {})
        return out

    @staticmethod
    def _match_row(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        for key in (
            "phase_step",
            "round_number",
            "created_step",
            "warmup_start_step",
            "start_step",
            "end_step",
        ):
            if out.get(key) is not None:
                out[key] = int(out.get(key) or 0)
        out["state_json"] = _json_loads(out.get("state_json"), {})
        out["result_json"] = _json_loads(out.get("result_json"), {})
        return out

    @staticmethod
    def _queue_row(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        out["rating_snapshot"] = int(out.get("rating_snapshot") or 1500)
        return out

    @staticmethod
    def _rating_row(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        for key in ("rating", "games_played", "wins", "losses", "draws", "peak_rating"):
            out[key] = int(out.get(key) or 0)
        return out

    @staticmethod
    def _wallet_row(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        for key in ("balance", "total_earned", "total_spent"):
            out[key] = int(out.get(key) or 0)
        return out

    def init_db(self) -> None:
        if self._initialized:
            return
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS scenario_definitions (
                  scenario_key TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  description TEXT NOT NULL DEFAULT '',
                  category TEXT NOT NULL DEFAULT 'social',
                  min_players INTEGER NOT NULL DEFAULT 2,
                  max_players INTEGER NOT NULL DEFAULT 10,
                  team_size INTEGER,
                  warmup_steps INTEGER NOT NULL DEFAULT 2,
                  max_duration_steps INTEGER NOT NULL DEFAULT 30,
                  rules_json TEXT NOT NULL DEFAULT '{{}}',
                  rating_mode TEXT NOT NULL DEFAULT 'elo',
                  buy_in INTEGER NOT NULL DEFAULT 0,
                  enabled INTEGER NOT NULL DEFAULT 1,
                  created_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()}),
                  updated_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()})
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS agent_scenario_ratings (
                  agent_id TEXT NOT NULL,
                  scenario_key TEXT NOT NULL,
                  rating INTEGER NOT NULL DEFAULT 1500,
                  games_played INTEGER NOT NULL DEFAULT 0,
                  wins INTEGER NOT NULL DEFAULT 0,
                  losses INTEGER NOT NULL DEFAULT 0,
                  draws INTEGER NOT NULL DEFAULT 0,
                  peak_rating INTEGER NOT NULL DEFAULT 1500,
                  updated_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()}),
                  PRIMARY KEY (agent_id, scenario_key)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_scenario_ratings_scenario_rating "
                "ON agent_scenario_ratings(scenario_key, rating DESC)"
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS agent_wallets (
                  agent_id TEXT PRIMARY KEY,
                  balance INTEGER NOT NULL DEFAULT 1000,
                  total_earned INTEGER NOT NULL DEFAULT 0,
                  total_spent INTEGER NOT NULL DEFAULT 0,
                  last_stipend TEXT,
                  updated_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()})
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS scenario_queue_entries (
                  entry_id TEXT PRIMARY KEY,
                  scenario_key TEXT NOT NULL,
                  agent_id TEXT NOT NULL,
                  town_id TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'queued',
                  queued_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()}),
                  rating_snapshot INTEGER NOT NULL DEFAULT 1500,
                  party_id TEXT,
                  match_id TEXT,
                  updated_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()})
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scenario_queue_lookup "
                "ON scenario_queue_entries(scenario_key, town_id, status, queued_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scenario_queue_agent "
                "ON scenario_queue_entries(agent_id, status, queued_at)"
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS scenario_matches (
                  match_id TEXT PRIMARY KEY,
                  scenario_key TEXT NOT NULL,
                  town_id TEXT NOT NULL,
                  arena_id TEXT NOT NULL DEFAULT '',
                  status TEXT NOT NULL DEFAULT 'forming',
                  phase TEXT NOT NULL DEFAULT 'warmup',
                  phase_step INTEGER NOT NULL DEFAULT 0,
                  round_number INTEGER NOT NULL DEFAULT 0,
                  created_step INTEGER NOT NULL DEFAULT 0,
                  warmup_start_step INTEGER,
                  start_step INTEGER,
                  end_step INTEGER,
                  state_json TEXT NOT NULL DEFAULT '{{}}',
                  result_json TEXT NOT NULL DEFAULT '{{}}',
                  created_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()}),
                  updated_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()})
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scenario_matches_town_status "
                "ON scenario_matches(town_id, status, updated_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scenario_matches_scenario_status "
                "ON scenario_matches(scenario_key, status, updated_at DESC)"
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS scenario_match_participants (
                  match_id TEXT NOT NULL,
                  agent_id TEXT NOT NULL,
                  role TEXT,
                  status TEXT NOT NULL DEFAULT 'assigned',
                  score REAL,
                  chips_start INTEGER,
                  chips_end INTEGER,
                  rating_before INTEGER,
                  rating_after INTEGER,
                  rating_delta INTEGER,
                  PRIMARY KEY (match_id, agent_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scenario_participants_agent ON scenario_match_participants(agent_id, match_id)"
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS scenario_match_events (
                  event_id TEXT PRIMARY KEY,
                  match_id TEXT NOT NULL,
                  step INTEGER NOT NULL,
                  phase TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  agent_id TEXT,
                  target_id TEXT,
                  data_json TEXT NOT NULL DEFAULT '{{}}',
                  created_at TEXT NOT NULL DEFAULT ({_now_utc_sqlite()})
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scenario_match_events_match_step "
                "ON scenario_match_events(match_id, step DESC, created_at DESC)"
            )

            for item in DEFAULT_SCENARIO_DEFINITIONS:
                key = str(item.get("scenario_key") or "").strip()
                if not key:
                    continue
                existing = conn.execute(
                    "SELECT scenario_key, rules_json FROM scenario_definitions WHERE scenario_key = ?",
                    (key,),
                ).fetchone()
                if not existing:
                    conn.execute(
                        f"""
                        INSERT INTO scenario_definitions (
                          scenario_key, name, description, category, min_players, max_players,
                          team_size, warmup_steps, max_duration_steps, rules_json,
                          rating_mode, buy_in, enabled, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ({_now_utc_sqlite()}), ({_now_utc_sqlite()}))
                        """,
                        (
                            key,
                            item["name"],
                            item["description"],
                            item["category"],
                            int(item["min_players"]),
                            int(item["max_players"]),
                            item.get("team_size"),
                            int(item["warmup_steps"]),
                            int(item["max_duration_steps"]),
                            _json_dumps(item.get("rules_json") or {}),
                            item.get("rating_mode") or "elo",
                            int(item.get("buy_in") or 0),
                            1 if bool(item.get("enabled", True)) else 0,
                        ),
                    )
                    continue
                existing_rules = _json_loads(dict(existing).get("rules_json"), {})
                if not isinstance(existing_rules, dict):
                    existing_rules = {}
                default_rules = item.get("rules_json") or {}
                if not isinstance(default_rules, dict):
                    default_rules = {}
                merged_rules = dict(default_rules)
                merged_rules.update(existing_rules)
                if merged_rules != existing_rules:
                    conn.execute(
                        f"""
                        UPDATE scenario_definitions
                        SET rules_json = ?, updated_at = ({_now_utc_sqlite()})
                        WHERE scenario_key = ?
                        """,
                        (_json_dumps(merged_rules), key),
                    )

        self._initialized = True

    def list_scenarios(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            if enabled_only:
                rows = conn.execute(
                    "SELECT * FROM scenario_definitions WHERE enabled = 1 ORDER BY scenario_key ASC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM scenario_definitions ORDER BY scenario_key ASC"
                ).fetchall()
        return [self._scenario_row(dict(row)) for row in rows]

    def get_scenario(self, *, scenario_key: str) -> dict[str, Any] | None:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scenario_definitions WHERE scenario_key = ?",
                (str(scenario_key),),
            ).fetchone()
        return self._scenario_row(dict(row)) if row else None

    def list_agent_queue(self, *, agent_id: str) -> list[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM scenario_queue_entries
                WHERE agent_id = ? AND status = 'queued'
                ORDER BY queued_at ASC
                """,
                (str(agent_id),),
            ).fetchall()
        return [self._queue_row(dict(row)) for row in rows]

    def list_queue(self, *, scenario_key: str, town_id: str | None = None, status: str = "queued") -> list[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            if town_id:
                rows = conn.execute(
                    """
                    SELECT * FROM scenario_queue_entries
                    WHERE scenario_key = ? AND town_id = ? AND status = ?
                    ORDER BY queued_at ASC
                    """,
                    (str(scenario_key), str(town_id), str(status)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM scenario_queue_entries
                    WHERE scenario_key = ? AND status = ?
                    ORDER BY queued_at ASC
                    """,
                    (str(scenario_key), str(status)),
                ).fetchall()
        return [self._queue_row(dict(row)) for row in rows]

    def enqueue(
        self,
        *,
        scenario_key: str,
        agent_id: str,
        town_id: str,
        rating_snapshot: int,
        party_id: str | None = None,
    ) -> dict[str, Any]:
        self.init_db()
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT * FROM scenario_queue_entries
                WHERE scenario_key = ? AND town_id = ? AND agent_id = ? AND status = 'queued'
                ORDER BY queued_at DESC
                LIMIT 1
                """,
                (str(scenario_key), str(town_id), str(agent_id)),
            ).fetchone()
            if existing:
                return self._queue_row(dict(existing))

            entry_id = _uuid("sq")
            conn.execute(
                f"""
                INSERT INTO scenario_queue_entries (
                  entry_id, scenario_key, agent_id, town_id, status, queued_at, rating_snapshot, party_id, match_id, updated_at
                ) VALUES (?, ?, ?, ?, 'queued', ({_now_utc_sqlite()}), ?, ?, NULL, ({_now_utc_sqlite()}))
                """,
                (entry_id, str(scenario_key), str(agent_id), str(town_id), int(rating_snapshot), party_id),
            )
            row = conn.execute(
                "SELECT * FROM scenario_queue_entries WHERE entry_id = ?",
                (entry_id,),
            ).fetchone()
        return self._queue_row(dict(row))

    def update_queue_status(
        self,
        *,
        entry_id: str,
        status: str,
        match_id: str | None = None,
    ) -> None:
        self.init_db()
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE scenario_queue_entries
                SET status = ?,
                    match_id = ?,
                    updated_at = ({_now_utc_sqlite()})
                WHERE entry_id = ?
                """,
                (str(status), match_id, str(entry_id)),
            )

    def get_agent_active_match(self, *, agent_id: str) -> dict[str, Any] | None:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT m.*
                FROM scenario_matches m
                JOIN scenario_match_participants p ON p.match_id = m.match_id
                WHERE p.agent_id = ?
                  AND m.status IN ('forming', 'warmup', 'active')
                  AND LOWER(COALESCE(p.status, '')) NOT IN ('done', 'forfeit', 'cancelled')
                ORDER BY m.updated_at DESC
                LIMIT 1
                """,
                (str(agent_id),),
            ).fetchone()
        return self._match_row(dict(row)) if row else None

    def list_active_matches(self, *, town_id: str | None = None) -> list[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            if town_id:
                rows = conn.execute(
                    """
                    SELECT * FROM scenario_matches
                    WHERE town_id = ? AND status IN ('forming', 'warmup', 'active')
                    ORDER BY updated_at DESC
                    """,
                    (str(town_id),),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM scenario_matches
                    WHERE status IN ('forming', 'warmup', 'active')
                    ORDER BY updated_at DESC
                    """
                ).fetchall()
        return [self._match_row(dict(row)) for row in rows]

    def list_matches(self, *, town_id: str | None = None, scenario_key: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        self.init_db()
        bounded_limit = max(1, min(500, int(limit)))
        where: list[str] = []
        args: list[Any] = []
        if town_id:
            where.append("town_id = ?")
            args.append(str(town_id))
        if scenario_key:
            where.append("scenario_key = ?")
            args.append(str(scenario_key))
        where_clause = f"WHERE {' AND '.join(where)}" if where else ""
        query = f"SELECT * FROM scenario_matches {where_clause} ORDER BY created_at DESC LIMIT ?"
        args.append(bounded_limit)
        with self._connect() as conn:
            rows = conn.execute(query, tuple(args)).fetchall()
        return [self._match_row(dict(row)) for row in rows]

    def get_match(self, *, match_id: str) -> dict[str, Any] | None:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scenario_matches WHERE match_id = ?",
                (str(match_id),),
            ).fetchone()
        return self._match_row(dict(row)) if row else None

    def create_match(
        self,
        *,
        scenario_key: str,
        town_id: str,
        arena_id: str,
        status: str,
        phase: str,
        phase_step: int,
        round_number: int,
        created_step: int,
        warmup_start_step: int | None,
        start_step: int | None,
        state_json: dict[str, Any] | None,
    ) -> dict[str, Any]:
        self.init_db()
        match_id = _uuid("match")
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO scenario_matches (
                  match_id, scenario_key, town_id, arena_id, status,
                  phase, phase_step, round_number, created_step,
                  warmup_start_step, start_step, end_step,
                  state_json, result_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, '{{}}', ({_now_utc_sqlite()}), ({_now_utc_sqlite()}))
                """,
                (
                    match_id,
                    str(scenario_key),
                    str(town_id),
                    str(arena_id or ""),
                    str(status),
                    str(phase),
                    int(phase_step),
                    int(round_number),
                    int(created_step),
                    warmup_start_step,
                    start_step,
                    _json_dumps(state_json or {}),
                ),
            )
            row = conn.execute("SELECT * FROM scenario_matches WHERE match_id = ?", (match_id,)).fetchone()
        return self._match_row(dict(row))

    def update_match(
        self,
        *,
        match_id: str,
        status: str | None = None,
        phase: str | None = None,
        phase_step: int | None = None,
        round_number: int | None = None,
        warmup_start_step: int | None = None,
        start_step: int | None = None,
        end_step: int | None = None,
        state_json: dict[str, Any] | None = None,
        result_json: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        self.init_db()
        updates: list[str] = []
        args: list[Any] = []
        if status is not None:
            updates.append("status = ?")
            args.append(str(status))
        if phase is not None:
            updates.append("phase = ?")
            args.append(str(phase))
        if phase_step is not None:
            updates.append("phase_step = ?")
            args.append(int(phase_step))
        if round_number is not None:
            updates.append("round_number = ?")
            args.append(int(round_number))
        if warmup_start_step is not None:
            updates.append("warmup_start_step = ?")
            args.append(int(warmup_start_step))
        if start_step is not None:
            updates.append("start_step = ?")
            args.append(int(start_step))
        if end_step is not None:
            updates.append("end_step = ?")
            args.append(int(end_step))
        if state_json is not None:
            updates.append("state_json = ?")
            args.append(_json_dumps(state_json))
        if result_json is not None:
            updates.append("result_json = ?")
            args.append(_json_dumps(result_json))
        if not updates:
            return self.get_match(match_id=match_id)

        updates.append(f"updated_at = ({_now_utc_sqlite()})")
        args.append(str(match_id))
        with self._connect() as conn:
            conn.execute(
                f"UPDATE scenario_matches SET {', '.join(updates)} WHERE match_id = ?",
                tuple(args),
            )
            row = conn.execute("SELECT * FROM scenario_matches WHERE match_id = ?", (str(match_id),)).fetchone()
        return self._match_row(dict(row)) if row else None

    def insert_match_participants(self, *, participants: list[dict[str, Any]]) -> None:
        self.init_db()
        if not participants:
            return
        with self._connect() as conn:
            for item in participants:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO scenario_match_participants (
                      match_id, agent_id, role, status, score,
                      chips_start, chips_end, rating_before, rating_after, rating_delta
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(item["match_id"]),
                        str(item["agent_id"]),
                        item.get("role"),
                        str(item.get("status") or "assigned"),
                        item.get("score"),
                        item.get("chips_start"),
                        item.get("chips_end"),
                        item.get("rating_before"),
                        item.get("rating_after"),
                        item.get("rating_delta"),
                    ),
                )

    def list_match_participants(self, *, match_id: str) -> list[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scenario_match_participants WHERE match_id = ? ORDER BY agent_id ASC",
                (str(match_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_match_participant(self, *, match_id: str, agent_id: str, updates: dict[str, Any]) -> None:
        self.init_db()
        if not updates:
            return
        allowed = {
            "role",
            "status",
            "score",
            "chips_start",
            "chips_end",
            "rating_before",
            "rating_after",
            "rating_delta",
        }
        parts: list[str] = []
        args: list[Any] = []
        for key, value in updates.items():
            if key not in allowed:
                continue
            parts.append(f"{key} = ?")
            args.append(value)
        if not parts:
            return
        args.append(str(match_id))
        args.append(str(agent_id))
        with self._connect() as conn:
            conn.execute(
                f"UPDATE scenario_match_participants SET {', '.join(parts)} WHERE match_id = ? AND agent_id = ?",
                tuple(args),
            )

    def insert_match_event(
        self,
        *,
        match_id: str,
        step: int,
        phase: str,
        event_type: str,
        agent_id: str | None,
        target_id: str | None,
        data_json: dict[str, Any] | None,
    ) -> dict[str, Any]:
        self.init_db()
        event_id = _uuid("evt")
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO scenario_match_events (
                  event_id, match_id, step, phase, event_type, agent_id, target_id, data_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ({_now_utc_sqlite()}))
                """,
                (
                    event_id,
                    str(match_id),
                    int(step),
                    str(phase),
                    str(event_type),
                    agent_id,
                    target_id,
                    _json_dumps(data_json or {}),
                ),
            )
            row = conn.execute(
                "SELECT * FROM scenario_match_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        out = dict(row)
        out["step"] = int(out.get("step") or 0)
        out["data_json"] = _json_loads(out.get("data_json"), {})
        return out

    def list_match_events(self, *, match_id: str, limit: int = 200) -> list[dict[str, Any]]:
        self.init_db()
        bounded = max(1, min(1000, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM scenario_match_events
                WHERE match_id = ?
                ORDER BY step DESC, created_at DESC
                LIMIT ?
                """,
                (str(match_id), bounded),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["step"] = int(item.get("step") or 0)
            item["data_json"] = _json_loads(item.get("data_json"), {})
            out.append(item)
        return out

    def get_wallet(self, *, agent_id: str, create: bool = True) -> dict[str, Any]:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM agent_wallets WHERE agent_id = ?", (str(agent_id),)).fetchone()
            if row is None and create:
                conn.execute(
                    f"""
                    INSERT INTO agent_wallets (agent_id, balance, total_earned, total_spent, last_stipend, updated_at)
                    VALUES (?, 1000, 0, 0, NULL, ({_now_utc_sqlite()}))
                    """,
                    (str(agent_id),),
                )
                row = conn.execute("SELECT * FROM agent_wallets WHERE agent_id = ?", (str(agent_id),)).fetchone()
        if row is None:
            return {
                "agent_id": str(agent_id),
                "balance": 0,
                "total_earned": 0,
                "total_spent": 0,
                "last_stipend": None,
                "updated_at": None,
            }
        return self._wallet_row(dict(row))

    def update_wallet(
        self,
        *,
        agent_id: str,
        delta: int,
        stipend_iso: str | None = None,
    ) -> dict[str, Any]:
        wallet = self.get_wallet(agent_id=agent_id, create=True)
        next_balance = int(wallet["balance"]) + int(delta)
        next_earned = int(wallet["total_earned"]) + (int(delta) if int(delta) > 0 else 0)
        next_spent = int(wallet["total_spent"]) + (abs(int(delta)) if int(delta) < 0 else 0)
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE agent_wallets
                SET balance = ?,
                    total_earned = ?,
                    total_spent = ?,
                    last_stipend = COALESCE(?, last_stipend),
                    updated_at = ({_now_utc_sqlite()})
                WHERE agent_id = ?
                """,
                (
                    int(next_balance),
                    int(next_earned),
                    int(next_spent),
                    stipend_iso,
                    str(agent_id),
                ),
            )
            row = conn.execute("SELECT * FROM agent_wallets WHERE agent_id = ?", (str(agent_id),)).fetchone()
        return self._wallet_row(dict(row))

    def get_rating(self, *, agent_id: str, scenario_key: str) -> dict[str, Any]:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM agent_scenario_ratings
                WHERE agent_id = ? AND scenario_key = ?
                """,
                (str(agent_id), str(scenario_key)),
            ).fetchone()
            if row is None:
                conn.execute(
                    f"""
                    INSERT INTO agent_scenario_ratings (
                      agent_id, scenario_key, rating, games_played, wins, losses, draws, peak_rating, updated_at
                    ) VALUES (?, ?, 1500, 0, 0, 0, 0, 1500, ({_now_utc_sqlite()}))
                    """,
                    (str(agent_id), str(scenario_key)),
                )
                row = conn.execute(
                    """
                    SELECT * FROM agent_scenario_ratings
                    WHERE agent_id = ? AND scenario_key = ?
                    """,
                    (str(agent_id), str(scenario_key)),
                ).fetchone()
        return self._rating_row(dict(row))

    def list_agent_ratings(self, *, agent_id: str) -> list[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_scenario_ratings WHERE agent_id = ? ORDER BY scenario_key ASC",
                (str(agent_id),),
            ).fetchall()
        return [self._rating_row(dict(row)) for row in rows]

    def upsert_rating(
        self,
        *,
        agent_id: str,
        scenario_key: str,
        rating: int,
        games_played: int,
        wins: int,
        losses: int,
        draws: int,
        peak_rating: int,
    ) -> dict[str, Any]:
        self.init_db()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO agent_scenario_ratings (
                  agent_id, scenario_key, rating, games_played, wins, losses, draws, peak_rating, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ({_now_utc_sqlite()}))
                ON CONFLICT(agent_id, scenario_key) DO UPDATE SET
                  rating = excluded.rating,
                  games_played = excluded.games_played,
                  wins = excluded.wins,
                  losses = excluded.losses,
                  draws = excluded.draws,
                  peak_rating = excluded.peak_rating,
                  updated_at = ({_now_utc_sqlite()})
                """,
                (
                    str(agent_id),
                    str(scenario_key),
                    int(rating),
                    int(games_played),
                    int(wins),
                    int(losses),
                    int(draws),
                    int(peak_rating),
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM agent_scenario_ratings
                WHERE agent_id = ? AND scenario_key = ?
                """,
                (str(agent_id), str(scenario_key)),
            ).fetchone()
        return self._rating_row(dict(row))

    def list_leaderboard(self, *, scenario_key: str, limit: int = 100) -> list[dict[str, Any]]:
        self.init_db()
        bounded = max(1, min(500, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM agent_scenario_ratings
                WHERE scenario_key = ?
                ORDER BY rating DESC, peak_rating DESC, wins DESC, updated_at ASC
                LIMIT ?
                """,
                (str(scenario_key), bounded),
            ).fetchall()
        return [self._rating_row(dict(row)) for row in rows]


def _sqlite_path() -> Path:
    configured = str(os.environ.get("VVALLEY_DB_PATH") or "").strip()
    if configured:
        return Path(configured)
    return WORKSPACE_ROOT / "data" / "vvalley.db"


@lru_cache(maxsize=1)
def _backend() -> ScenarioStore:
    # SQLite-first for now; keeps parity with existing local/test deployments.
    return SQLiteScenarioStore(_sqlite_path())


def reset_backend_cache_for_tests() -> None:
    _backend.cache_clear()


def init_db() -> None:
    _backend().init_db()


def list_scenarios(*, enabled_only: bool = True) -> list[dict[str, Any]]:
    return _backend().list_scenarios(enabled_only=enabled_only)


def get_scenario(*, scenario_key: str) -> dict[str, Any] | None:
    return _backend().get_scenario(scenario_key=scenario_key)


def list_agent_queue(*, agent_id: str) -> list[dict[str, Any]]:
    return _backend().list_agent_queue(agent_id=agent_id)


def list_queue(*, scenario_key: str, town_id: str | None = None, status: str = "queued") -> list[dict[str, Any]]:
    return _backend().list_queue(scenario_key=scenario_key, town_id=town_id, status=status)


def enqueue(*, scenario_key: str, agent_id: str, town_id: str, rating_snapshot: int, party_id: str | None = None) -> dict[str, Any]:
    return _backend().enqueue(
        scenario_key=scenario_key,
        agent_id=agent_id,
        town_id=town_id,
        rating_snapshot=rating_snapshot,
        party_id=party_id,
    )


def update_queue_status(*, entry_id: str, status: str, match_id: str | None = None) -> None:
    _backend().update_queue_status(entry_id=entry_id, status=status, match_id=match_id)


def get_agent_active_match(*, agent_id: str) -> dict[str, Any] | None:
    return _backend().get_agent_active_match(agent_id=agent_id)


def list_active_matches(*, town_id: str | None = None) -> list[dict[str, Any]]:
    return _backend().list_active_matches(town_id=town_id)


def list_matches(*, town_id: str | None = None, scenario_key: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    return _backend().list_matches(town_id=town_id, scenario_key=scenario_key, limit=limit)


def get_match(*, match_id: str) -> dict[str, Any] | None:
    return _backend().get_match(match_id=match_id)


def create_match(
    *,
    scenario_key: str,
    town_id: str,
    arena_id: str,
    status: str,
    phase: str,
    phase_step: int,
    round_number: int,
    created_step: int,
    warmup_start_step: int | None,
    start_step: int | None,
    state_json: dict[str, Any] | None,
) -> dict[str, Any]:
    return _backend().create_match(
        scenario_key=scenario_key,
        town_id=town_id,
        arena_id=arena_id,
        status=status,
        phase=phase,
        phase_step=phase_step,
        round_number=round_number,
        created_step=created_step,
        warmup_start_step=warmup_start_step,
        start_step=start_step,
        state_json=state_json,
    )


def update_match(
    *,
    match_id: str,
    status: str | None = None,
    phase: str | None = None,
    phase_step: int | None = None,
    round_number: int | None = None,
    warmup_start_step: int | None = None,
    start_step: int | None = None,
    end_step: int | None = None,
    state_json: dict[str, Any] | None = None,
    result_json: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    return _backend().update_match(
        match_id=match_id,
        status=status,
        phase=phase,
        phase_step=phase_step,
        round_number=round_number,
        warmup_start_step=warmup_start_step,
        start_step=start_step,
        end_step=end_step,
        state_json=state_json,
        result_json=result_json,
    )


def insert_match_participants(*, participants: list[dict[str, Any]]) -> None:
    _backend().insert_match_participants(participants=participants)


def list_match_participants(*, match_id: str) -> list[dict[str, Any]]:
    return _backend().list_match_participants(match_id=match_id)


def update_match_participant(*, match_id: str, agent_id: str, updates: dict[str, Any]) -> None:
    _backend().update_match_participant(match_id=match_id, agent_id=agent_id, updates=updates)


def insert_match_event(
    *,
    match_id: str,
    step: int,
    phase: str,
    event_type: str,
    agent_id: str | None,
    target_id: str | None,
    data_json: dict[str, Any] | None,
) -> dict[str, Any]:
    return _backend().insert_match_event(
        match_id=match_id,
        step=step,
        phase=phase,
        event_type=event_type,
        agent_id=agent_id,
        target_id=target_id,
        data_json=data_json,
    )


def list_match_events(*, match_id: str, limit: int = 200) -> list[dict[str, Any]]:
    return _backend().list_match_events(match_id=match_id, limit=limit)


def get_wallet(*, agent_id: str, create: bool = True) -> dict[str, Any]:
    return _backend().get_wallet(agent_id=agent_id, create=create)


def update_wallet(*, agent_id: str, delta: int, stipend_iso: str | None = None) -> dict[str, Any]:
    return _backend().update_wallet(agent_id=agent_id, delta=delta, stipend_iso=stipend_iso)


def get_rating(*, agent_id: str, scenario_key: str) -> dict[str, Any]:
    return _backend().get_rating(agent_id=agent_id, scenario_key=scenario_key)


def list_agent_ratings(*, agent_id: str) -> list[dict[str, Any]]:
    return _backend().list_agent_ratings(agent_id=agent_id)


def upsert_rating(
    *,
    agent_id: str,
    scenario_key: str,
    rating: int,
    games_played: int,
    wins: int,
    losses: int,
    draws: int,
    peak_rating: int,
) -> dict[str, Any]:
    return _backend().upsert_rating(
        agent_id=agent_id,
        scenario_key=scenario_key,
        rating=rating,
        games_played=games_played,
        wins=wins,
        losses=losses,
        draws=draws,
        peak_rating=peak_rating,
    )


def list_leaderboard(*, scenario_key: str, limit: int = 100) -> list[dict[str, Any]]:
    return _backend().list_leaderboard(scenario_key=scenario_key, limit=limit)
