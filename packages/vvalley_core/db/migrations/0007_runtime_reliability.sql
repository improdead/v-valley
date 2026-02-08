-- PostgreSQL migration: runtime reliability primitives (leases, idempotency batches)

ALTER TABLE town_runtime_controls
  ADD COLUMN IF NOT EXISTS lease_owner TEXT NULL,
  ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ NULL;

CREATE INDEX IF NOT EXISTS idx_town_runtime_controls_lease
  ON town_runtime_controls(lease_owner, lease_expires_at);

CREATE TABLE IF NOT EXISTS town_tick_batches (
  town_id TEXT NOT NULL,
  batch_key TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  step_before INTEGER NULL,
  step_after INTEGER NULL,
  result_json JSONB NULL,
  error TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at TIMESTAMPTZ NULL,
  PRIMARY KEY (town_id, batch_key)
);

CREATE INDEX IF NOT EXISTS idx_town_tick_batches_town_created
  ON town_tick_batches(town_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_town_tick_batches_town_status_created
  ON town_tick_batches(town_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS interaction_dead_letters (
  id UUID PRIMARY KEY,
  town_id TEXT NULL,
  stage TEXT NOT NULL,
  payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  error TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_interaction_dead_letters_town_created
  ON interaction_dead_letters(town_id, created_at DESC);
