-- PostgreSQL migration: map versioning metadata

CREATE TABLE IF NOT EXISTS town_map_versions (
  id UUID PRIMARY KEY,
  town_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  map_name TEXT NOT NULL,
  map_json_path TEXT NOT NULL,
  nav_data_path TEXT NOT NULL,
  source_sha256 TEXT NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT FALSE,
  notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (town_id, version)
);

CREATE TABLE IF NOT EXISTS location_affordances (
  map_version_id UUID NOT NULL REFERENCES town_map_versions(id) ON DELETE CASCADE,
  town_id TEXT NOT NULL,
  location_name TEXT NOT NULL,
  tile_x INTEGER NOT NULL,
  tile_y INTEGER NOT NULL,
  affordance TEXT NOT NULL,
  metadata_json TEXT,
  PRIMARY KEY (map_version_id, location_name, affordance)
);

CREATE INDEX IF NOT EXISTS idx_tmv_town_active ON town_map_versions(town_id, is_active);
CREATE INDEX IF NOT EXISTS idx_tmv_town_version ON town_map_versions(town_id, version);
