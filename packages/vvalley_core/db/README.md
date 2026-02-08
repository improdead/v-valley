# V-Valley DB Migrations

SQL migrations for Postgres-backed deployments.

## Migration files

- `migrations/0001_map_versioning.sql`
- `migrations/0002_llm_control_plane.sql`
- `migrations/0003_agents_identity.sql`

## Runtime behavior

When `DATABASE_URL` points to Postgres and `psycopg` is installed, storage modules apply migrations automatically at startup.

Primary storage modules:
- `apps/api/vvalley_api/storage/map_versions.py`
- `apps/api/vvalley_api/storage/llm_control.py`
- `apps/api/vvalley_api/storage/agents.py`

## Requirements

```bash
pip install psycopg[binary]
```

## Environment

Postgres:

- `DATABASE_URL=postgresql://user:pass@host:5432/dbname`

SQLite fallback:

- Leave `DATABASE_URL` unset
- Optional path override: `VVALLEY_DB_PATH`
