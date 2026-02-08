-- PostgreSQL migration: runtime scheduler controls and per-agent autonomy contracts

CREATE TABLE IF NOT EXISTS agent_autonomy_contracts (
  agent_id UUID PRIMARY KEY REFERENCES agents(id) ON DELETE CASCADE,
  mode TEXT NOT NULL DEFAULT 'manual',
  allowed_scopes_json JSONB NOT NULL DEFAULT '["short_action","daily_plan","long_term_plan"]'::jsonb,
  max_autonomous_ticks INTEGER NOT NULL DEFAULT 0,
  escalation_policy_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_autonomy_mode ON agent_autonomy_contracts(mode);

CREATE TABLE IF NOT EXISTS town_runtime_controls (
  town_id TEXT PRIMARY KEY,
  enabled BOOLEAN NOT NULL DEFAULT FALSE,
  tick_interval_seconds INTEGER NOT NULL DEFAULT 60,
  steps_per_tick INTEGER NOT NULL DEFAULT 1,
  planning_scope TEXT NOT NULL DEFAULT 'short_action',
  control_mode TEXT NOT NULL DEFAULT 'hybrid',
  last_tick_at TIMESTAMPTZ NULL,
  last_tick_success BOOLEAN NULL,
  last_error TEXT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_town_runtime_controls_enabled ON town_runtime_controls(enabled);
