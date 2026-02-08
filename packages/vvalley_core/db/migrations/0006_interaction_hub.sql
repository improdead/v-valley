-- PostgreSQL migration: durable interaction hub (inbox, escalations, dm flow)

CREATE TABLE IF NOT EXISTS agent_inbox_items (
  id UUID PRIMARY KEY,
  agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  summary TEXT NOT NULL,
  payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  status TEXT NOT NULL DEFAULT 'pending',
  dedupe_key TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  read_at TIMESTAMPTZ NULL,
  resolved_at TIMESTAMPTZ NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_inbox_items_agent_status_created
  ON agent_inbox_items(agent_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_inbox_items_dedupe
  ON agent_inbox_items(agent_id, dedupe_key);

CREATE TABLE IF NOT EXISTS owner_escalations (
  id UUID PRIMARY KEY,
  owner_handle TEXT NOT NULL,
  agent_id UUID NULL REFERENCES agents(id) ON DELETE SET NULL,
  town_id TEXT NULL,
  kind TEXT NOT NULL,
  summary TEXT NOT NULL,
  payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  status TEXT NOT NULL DEFAULT 'open',
  dedupe_key TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  resolved_at TIMESTAMPTZ NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_owner_escalations_owner_status_created
  ON owner_escalations(owner_handle, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_owner_escalations_dedupe
  ON owner_escalations(owner_handle, dedupe_key);

CREATE TABLE IF NOT EXISTS dm_requests (
  id UUID PRIMARY KEY,
  from_agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  to_agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  town_id TEXT NULL,
  message TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  resolved_at TIMESTAMPTZ NULL
);

CREATE INDEX IF NOT EXISTS idx_dm_requests_to_status_created
  ON dm_requests(to_agent_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dm_requests_from_status_created
  ON dm_requests(from_agent_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS dm_conversations (
  id UUID PRIMARY KEY,
  agent_a_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  agent_b_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  town_id TEXT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_message_at TIMESTAMPTZ NULL
);

CREATE INDEX IF NOT EXISTS idx_dm_conversations_participants
  ON dm_conversations(agent_a_id, agent_b_id, status);
CREATE INDEX IF NOT EXISTS idx_dm_conversations_town_status
  ON dm_conversations(town_id, status, updated_at DESC);

CREATE TABLE IF NOT EXISTS dm_messages (
  id UUID PRIMARY KEY,
  conversation_id UUID NOT NULL REFERENCES dm_conversations(id) ON DELETE CASCADE,
  sender_agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  body TEXT NOT NULL,
  needs_human_input BOOLEAN NOT NULL DEFAULT FALSE,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dm_messages_conversation_created
  ON dm_messages(conversation_id, created_at DESC);
