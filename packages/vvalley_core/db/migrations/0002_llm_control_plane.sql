-- PostgreSQL migration: LLM task policies and call telemetry

CREATE TABLE IF NOT EXISTS llm_task_policies (
  task_name TEXT PRIMARY KEY,
  model_tier TEXT NOT NULL,
  max_input_tokens INTEGER NOT NULL,
  max_output_tokens INTEGER NOT NULL,
  temperature DOUBLE PRECISION NOT NULL,
  timeout_ms INTEGER NOT NULL,
  retry_limit INTEGER NOT NULL,
  enable_prompt_cache BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS llm_call_logs (
  id UUID PRIMARY KEY,
  town_id UUID NULL,
  agent_id UUID NULL,
  task_name TEXT NOT NULL,
  model_name TEXT NOT NULL,
  prompt_tokens INTEGER NOT NULL,
  completion_tokens INTEGER NOT NULL,
  cached_tokens INTEGER NOT NULL DEFAULT 0,
  latency_ms INTEGER NULL,
  success BOOLEAN NOT NULL,
  error_code TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_llm_logs_task_created ON llm_call_logs(task_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_logs_town_created ON llm_call_logs(town_id, created_at DESC);
