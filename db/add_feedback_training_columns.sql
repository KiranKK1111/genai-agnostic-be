-- ═══════════════════════════════════════════════════════════════
-- Migration: Add training-signal columns to feedback table
-- Enables online learning from user interactions.
-- Safe to run multiple times (idempotent).
-- ═══════════════════════════════════════════════════════════════

-- query_text: the original natural-language query the user typed
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = '{app_schema}'
          AND table_name   = 'feedback'
          AND column_name  = 'query_text'
    ) THEN
        ALTER TABLE {app_schema}.feedback ADD COLUMN query_text TEXT;
    END IF;
END $$;

-- resolved_table: the primary DB table used to answer the query
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = '{app_schema}'
          AND table_name   = 'feedback'
          AND column_name  = 'resolved_table'
    ) THEN
        ALTER TABLE {app_schema}.feedback ADD COLUMN resolved_table TEXT;
    END IF;
END $$;

-- resolved_columns: columns returned / filtered on (used to build richer positive text)
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = '{app_schema}'
          AND table_name   = 'feedback'
          AND column_name  = 'resolved_columns'
    ) THEN
        ALTER TABLE {app_schema}.feedback ADD COLUMN resolved_columns TEXT[] DEFAULT '{}';
    END IF;
END $$;

-- plan_json: the full QueryPlan dict (kept for audit / future use)
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = '{app_schema}'
          AND table_name   = 'feedback'
          AND column_name  = 'plan_json'
    ) THEN
        ALTER TABLE {app_schema}.feedback ADD COLUMN plan_json JSONB;
    END IF;
END $$;

-- trained_on: whether this row has already been used in a fine-tuning run
--   false → new signal, include in next training cycle
--   true  → already learned, skip to avoid re-weighting old data
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = '{app_schema}'
          AND table_name   = 'feedback'
          AND column_name  = 'trained_on'
    ) THEN
        ALTER TABLE {app_schema}.feedback
            ADD COLUMN trained_on BOOLEAN NOT NULL DEFAULT false;
    END IF;
END $$;

-- Partial index: fast lookup of untrained positive pairs for the training loop
CREATE INDEX IF NOT EXISTS idx_feedback_training
    ON {app_schema}.feedback (rating, created_at DESC)
    WHERE trained_on = false AND rating = 1
      AND query_text IS NOT NULL AND resolved_table IS NOT NULL;
