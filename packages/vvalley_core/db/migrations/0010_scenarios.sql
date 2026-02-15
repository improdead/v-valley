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
  rules_json TEXT NOT NULL DEFAULT '{}',
  rating_mode TEXT NOT NULL DEFAULT 'elo',
  buy_in INTEGER NOT NULL DEFAULT 0,
  enabled INTEGER NOT NULL DEFAULT 1,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_scenario_ratings (
  agent_id TEXT NOT NULL,
  scenario_key TEXT NOT NULL,
  rating INTEGER NOT NULL DEFAULT 1500,
  games_played INTEGER NOT NULL DEFAULT 0,
  wins INTEGER NOT NULL DEFAULT 0,
  losses INTEGER NOT NULL DEFAULT 0,
  draws INTEGER NOT NULL DEFAULT 0,
  peak_rating INTEGER NOT NULL DEFAULT 1500,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (agent_id, scenario_key)
);
CREATE INDEX IF NOT EXISTS idx_agent_scenario_ratings_scenario_rating
  ON agent_scenario_ratings(scenario_key, rating DESC);

CREATE TABLE IF NOT EXISTS agent_wallets (
  agent_id TEXT PRIMARY KEY,
  balance INTEGER NOT NULL DEFAULT 1000,
  total_earned INTEGER NOT NULL DEFAULT 0,
  total_spent INTEGER NOT NULL DEFAULT 0,
  last_stipend TIMESTAMPTZ,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS scenario_queue_entries (
  entry_id TEXT PRIMARY KEY,
  scenario_key TEXT NOT NULL,
  agent_id TEXT NOT NULL,
  town_id TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  queued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  rating_snapshot INTEGER NOT NULL DEFAULT 1500,
  party_id TEXT,
  match_id TEXT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_scenario_queue_lookup
  ON scenario_queue_entries(scenario_key, town_id, status, queued_at);
CREATE INDEX IF NOT EXISTS idx_scenario_queue_agent
  ON scenario_queue_entries(agent_id, status, queued_at);

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
  state_json TEXT NOT NULL DEFAULT '{}',
  result_json TEXT NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_scenario_matches_town_status
  ON scenario_matches(town_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_scenario_matches_scenario_status
  ON scenario_matches(scenario_key, status, updated_at DESC);

CREATE TABLE IF NOT EXISTS scenario_match_participants (
  match_id TEXT NOT NULL,
  agent_id TEXT NOT NULL,
  role TEXT,
  status TEXT NOT NULL DEFAULT 'assigned',
  score DOUBLE PRECISION,
  chips_start INTEGER,
  chips_end INTEGER,
  rating_before INTEGER,
  rating_after INTEGER,
  rating_delta INTEGER,
  PRIMARY KEY (match_id, agent_id)
);
CREATE INDEX IF NOT EXISTS idx_scenario_participants_agent
  ON scenario_match_participants(agent_id, match_id);

CREATE TABLE IF NOT EXISTS scenario_match_events (
  event_id TEXT PRIMARY KEY,
  match_id TEXT NOT NULL,
  step INTEGER NOT NULL,
  phase TEXT NOT NULL,
  event_type TEXT NOT NULL,
  agent_id TEXT,
  target_id TEXT,
  data_json TEXT NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_scenario_match_events_match_step
  ON scenario_match_events(match_id, step DESC, created_at DESC);
