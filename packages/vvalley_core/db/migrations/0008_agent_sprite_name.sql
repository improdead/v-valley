-- Add sprite_name column to agents table for character appearance
ALTER TABLE agents ADD COLUMN IF NOT EXISTS sprite_name TEXT NULL;
