-- Track which town an agent was last in (for rejoin preference)
ALTER TABLE agents ADD COLUMN IF NOT EXISTS last_town_id TEXT NULL;
