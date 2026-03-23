-- ═══════════════════════════════════════════════════════════
-- Migration: Fix feedback table schema
-- Adds message_id column, fixes rating CHECK constraint.
-- Safe to run multiple times.
-- ═══════════════════════════════════════════════════════════

-- Add message_id column if missing
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = '{app_schema}' AND table_name = 'feedback' AND column_name = 'message_id'
    ) THEN
        ALTER TABLE {app_schema}.feedback ADD COLUMN message_id UUID;
    END IF;
END $$;

-- Drop old message_index column if it exists (replaced by message_id)
DO $$ BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = '{app_schema}' AND table_name = 'feedback' AND column_name = 'message_index'
    ) THEN
        ALTER TABLE {app_schema}.feedback DROP COLUMN message_index;
    END IF;
END $$;

-- Fix rating CHECK constraint: allow -1 (dislike) and 1 (like) instead of 1 and 5
DO $$ BEGIN
    -- Drop old constraint if exists
    IF EXISTS (
        SELECT 1 FROM information_schema.constraint_column_usage
        WHERE table_schema = '{app_schema}' AND table_name = 'feedback' AND column_name = 'rating'
    ) THEN
        ALTER TABLE {app_schema}.feedback DROP CONSTRAINT IF EXISTS feedback_rating_check;
    END IF;
    -- Add corrected constraint
    ALTER TABLE {app_schema}.feedback ADD CONSTRAINT feedback_rating_check CHECK (rating IN (-1, 1));
EXCEPTION WHEN duplicate_object THEN
    NULL; -- constraint already correct
END $$;
