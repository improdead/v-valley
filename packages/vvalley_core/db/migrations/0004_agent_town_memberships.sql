-- PostgreSQL migration: active town membership per agent

CREATE TABLE IF NOT EXISTS agent_town_memberships (
  agent_id UUID PRIMARY KEY REFERENCES agents(id) ON DELETE CASCADE,
  town_id TEXT NOT NULL,
  joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_town_memberships_town_id
  ON agent_town_memberships(town_id);
