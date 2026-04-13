-- Migration: Add "name" column to users for display purposes.
-- The existing "username" column stays as the login identifier; "name" is the
-- friendly display name shown in the sidebar avatar / profile area.
-- Safe to run multiple times (idempotent).

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = '{app_schema}'
          AND table_name   = 'users'
          AND column_name  = 'name'
    ) THEN
        ALTER TABLE {app_schema}.users ADD COLUMN name TEXT;
        -- Backfill: use the username as a reasonable default so existing rows
        -- still have something to display after this migration.
        UPDATE {app_schema}.users SET name = username WHERE name IS NULL;
    END IF;
END $$;
