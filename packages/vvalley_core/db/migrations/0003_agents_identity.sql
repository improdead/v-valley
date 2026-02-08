-- PostgreSQL migration: agent onboarding and ownership

CREATE TABLE IF NOT EXISTS agents (
  id UUID PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT NULL,
  personality_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  api_key_hash TEXT NOT NULL UNIQUE,
  api_key_prefix TEXT NOT NULL,
  claim_token TEXT NOT NULL UNIQUE,
  verification_code TEXT NOT NULL,
  claim_status TEXT NOT NULL DEFAULT 'pending',
  owner_handle TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  claimed_at TIMESTAMPTZ NULL
);

CREATE INDEX IF NOT EXISTS idx_agents_claim_token ON agents(claim_token);
CREATE INDEX IF NOT EXISTS idx_agents_api_key_hash ON agents(api_key_hash);
