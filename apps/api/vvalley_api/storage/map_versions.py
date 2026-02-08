"""Persistent map version store with SQLite fallback and Postgres support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from logging import getLogger
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse
import os
import sqlite3
import uuid

logger = getLogger("vvalley_api.storage.map_versions")


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
MIGRATIONS_DIR = WORKSPACE_ROOT / "packages" / "vvalley_core" / "db" / "migrations"


class MapVersionStore(ABC):
    @abstractmethod
    def init_db(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def next_version(self, town_id: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def publish_version(
        self,
        *,
        town_id: str,
        version: int,
        map_name: str,
        map_json_path: str,
        nav_data_path: str,
        source_sha256: str,
        affordances: Iterable[dict[str, Any]],
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_town_ids(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def list_versions(self, town_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_version(self, town_id: str, version: int) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_active_version(self, town_id: str) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_affordances(self, map_version_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def activate_version(self, town_id: str, version: int) -> Optional[dict[str, Any]]:
        raise NotImplementedError


class SQLiteMapVersionStore(MapVersionStore):
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
        return {k: row[k] for k in row.keys()}

    def init_db(self) -> None:
        logger.info("[STORAGE] Initializing SQLite map versions database at '%s'", self.db_path)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS town_map_versions (
                  id TEXT PRIMARY KEY,
                  town_id TEXT NOT NULL,
                  version INTEGER NOT NULL,
                  map_name TEXT NOT NULL,
                  map_json_path TEXT NOT NULL,
                  nav_data_path TEXT NOT NULL,
                  source_sha256 TEXT NOT NULL,
                  is_active INTEGER NOT NULL DEFAULT 0,
                  notes TEXT,
                  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                  UNIQUE (town_id, version)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS location_affordances (
                  map_version_id TEXT NOT NULL,
                  town_id TEXT NOT NULL,
                  location_name TEXT NOT NULL,
                  tile_x INTEGER NOT NULL,
                  tile_y INTEGER NOT NULL,
                  affordance TEXT NOT NULL,
                  metadata_json TEXT,
                  PRIMARY KEY (map_version_id, location_name, affordance),
                  FOREIGN KEY (map_version_id) REFERENCES town_map_versions(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tmv_town_active ON town_map_versions(town_id, is_active)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tmv_town_version ON town_map_versions(town_id, version)"
            )
        logger.info("[STORAGE] SQLite map versions database initialized successfully")

    def next_version(self, town_id: str) -> int:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS max_version FROM town_map_versions WHERE town_id = ?",
                (town_id,),
            ).fetchone()
        next_ver = int(row["max_version"]) + 1
        logger.debug("[STORAGE] Next version for town '%s': %d", town_id, next_ver)
        return next_ver

    def publish_version(
        self,
        *,
        town_id: str,
        version: int,
        map_name: str,
        map_json_path: str,
        nav_data_path: str,
        source_sha256: str,
        affordances: Iterable[dict[str, Any]],
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        self.init_db()
        version_id = str(uuid.uuid4())
        affordances_list = list(affordances)

        logger.info("[STORAGE] Publishing version: town_id='%s', version=%d, map_name='%s', affordances=%d", 
                   town_id, version, map_name, len(affordances_list))

        with self._connect() as conn:
            has_existing = conn.execute(
                "SELECT 1 FROM town_map_versions WHERE town_id = ? LIMIT 1",
                (town_id,),
            ).fetchone()

            is_active = 1 if has_existing is None else 0

            conn.execute(
                """
                INSERT INTO town_map_versions
                  (id, town_id, version, map_name, map_json_path, nav_data_path, source_sha256, is_active, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    town_id,
                    int(version),
                    map_name,
                    map_json_path,
                    nav_data_path,
                    source_sha256,
                    is_active,
                    notes,
                ),
            )

            for item in affordances_list:
                conn.execute(
                    """
                    INSERT INTO location_affordances
                      (map_version_id, town_id, location_name, tile_x, tile_y, affordance, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        version_id,
                        town_id,
                        item["location_name"],
                        int(item["tile_x"]),
                        int(item["tile_y"]),
                        item["affordance"],
                        item.get("metadata_json"),
                    ),
                )

            row = conn.execute(
                "SELECT * FROM town_map_versions WHERE id = ?",
                (version_id,),
            ).fetchone()

        logger.info("[STORAGE] Version published: town_id='%s', version=%d, version_id=%s, is_active=%d", 
                   town_id, version, version_id, is_active)
        return self._row_to_dict(row)

    def list_town_ids(self) -> list[str]:
        self.init_db()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT town_id
                FROM town_map_versions
                ORDER BY town_id ASC
                """
            ).fetchall()
        return [str(r["town_id"]) for r in rows]

    def list_versions(self, town_id: str) -> list[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM town_map_versions
                WHERE town_id = ?
                ORDER BY version DESC
                """,
                (town_id,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_version(self, town_id: str, version: int) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM town_map_versions WHERE town_id = ? AND version = ?",
                (town_id, int(version)),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_active_version(self, town_id: str) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM town_map_versions WHERE town_id = ? AND is_active = 1",
                (town_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_affordances(self, map_version_id: str) -> list[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT location_name, tile_x, tile_y, affordance, metadata_json
                FROM location_affordances
                WHERE map_version_id = ?
                ORDER BY location_name, affordance
                """,
                (map_version_id,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def activate_version(self, town_id: str, version: int) -> Optional[dict[str, Any]]:
        self.init_db()
        logger.info("[STORAGE] Activating version: town_id='%s', version=%d", town_id, version)
        
        target = self.get_version(town_id, version)
        if not target:
            logger.warning("[STORAGE] Activate version failed - version not found: town_id='%s', version=%d", 
                          town_id, version)
            return None

        with self._connect() as conn:
            conn.execute(
                "UPDATE town_map_versions SET is_active = 0 WHERE town_id = ?",
                (town_id,),
            )
            conn.execute(
                "UPDATE town_map_versions SET is_active = 1 WHERE town_id = ? AND version = ?",
                (town_id, int(version)),
            )

        logger.info("[STORAGE] Version activated: town_id='%s', version=%d", town_id, version)
        return self.get_version(town_id, version)


class PostgresMapVersionStore(MapVersionStore):
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

    def init_db(self) -> None:
        self._run_migrations()

    def next_version(self, town_id: str) -> int:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COALESCE(MAX(version), 0) AS max_version FROM town_map_versions WHERE town_id = %s",
                    (town_id,),
                )
                row = cur.fetchone()
        return int(row["max_version"]) + 1

    def publish_version(
        self,
        *,
        town_id: str,
        version: int,
        map_name: str,
        map_json_path: str,
        nav_data_path: str,
        source_sha256: str,
        affordances: Iterable[dict[str, Any]],
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        self.init_db()
        version_id = str(uuid.uuid4())

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM town_map_versions WHERE town_id = %s LIMIT 1",
                    (town_id,),
                )
                has_existing = cur.fetchone() is not None
                is_active = not has_existing

                cur.execute(
                    """
                    INSERT INTO town_map_versions
                      (id, town_id, version, map_name, map_json_path, nav_data_path, source_sha256, is_active, notes)
                    VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        version_id,
                        town_id,
                        int(version),
                        map_name,
                        map_json_path,
                        nav_data_path,
                        source_sha256,
                        is_active,
                        notes,
                    ),
                )

                for item in affordances:
                    cur.execute(
                        """
                        INSERT INTO location_affordances
                          (map_version_id, town_id, location_name, tile_x, tile_y, affordance, metadata_json)
                        VALUES (%s::uuid, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            version_id,
                            town_id,
                            item["location_name"],
                            int(item["tile_x"]),
                            int(item["tile_y"]),
                            item["affordance"],
                            item.get("metadata_json"),
                        ),
                    )

                cur.execute("SELECT * FROM town_map_versions WHERE id = %s::uuid", (version_id,))
                row = cur.fetchone()

        return dict(row)

    def list_town_ids(self) -> list[str]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT town_id
                    FROM town_map_versions
                    ORDER BY town_id ASC
                    """
                )
                rows = cur.fetchall()
        return [str(r["town_id"]) for r in rows]

    def list_versions(self, town_id: str) -> list[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM town_map_versions
                    WHERE town_id = %s
                    ORDER BY version DESC
                    """,
                    (town_id,),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def get_version(self, town_id: str, version: int) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM town_map_versions WHERE town_id = %s AND version = %s",
                    (town_id, int(version)),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def get_active_version(self, town_id: str) -> Optional[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM town_map_versions WHERE town_id = %s AND is_active = TRUE",
                    (town_id,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def get_affordances(self, map_version_id: str) -> list[dict[str, Any]]:
        self.init_db()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT location_name, tile_x, tile_y, affordance, metadata_json
                    FROM location_affordances
                    WHERE map_version_id = %s::uuid
                    ORDER BY location_name, affordance
                    """,
                    (map_version_id,),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def activate_version(self, town_id: str, version: int) -> Optional[dict[str, Any]]:
        self.init_db()
        target = self.get_version(town_id, version)
        if not target:
            return None

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE town_map_versions SET is_active = FALSE WHERE town_id = %s",
                    (town_id,),
                )
                cur.execute(
                    "UPDATE town_map_versions SET is_active = TRUE WHERE town_id = %s AND version = %s",
                    (town_id, int(version)),
                )

        return self.get_version(town_id, version)


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
def _backend() -> MapVersionStore:
    database_url = _database_url()
    if database_url and database_url.startswith(("postgres://", "postgresql://")):
        return PostgresMapVersionStore(database_url)
    return SQLiteMapVersionStore(_resolve_sqlite_path(database_url))


def reset_backend_cache_for_tests() -> None:
    _backend.cache_clear()


def init_db() -> None:
    _backend().init_db()


def next_version(town_id: str) -> int:
    return _backend().next_version(town_id)


def publish_version(
    *,
    town_id: str,
    version: int,
    map_name: str,
    map_json_path: str,
    nav_data_path: str,
    source_sha256: str,
    affordances: Iterable[dict[str, Any]],
    notes: Optional[str] = None,
) -> dict[str, Any]:
    return _backend().publish_version(
        town_id=town_id,
        version=version,
        map_name=map_name,
        map_json_path=map_json_path,
        nav_data_path=nav_data_path,
        source_sha256=source_sha256,
        affordances=affordances,
        notes=notes,
    )


def list_versions(town_id: str) -> list[dict[str, Any]]:
    return _backend().list_versions(town_id)


def list_town_ids() -> list[str]:
    return _backend().list_town_ids()


def get_version(town_id: str, version: int) -> Optional[dict[str, Any]]:
    return _backend().get_version(town_id, version)


def get_active_version(town_id: str) -> Optional[dict[str, Any]]:
    return _backend().get_active_version(town_id)


def get_affordances(map_version_id: str) -> list[dict[str, Any]]:
    return _backend().get_affordances(map_version_id)


def activate_version(town_id: str, version: int) -> Optional[dict[str, Any]]:
    return _backend().activate_version(town_id, version)
