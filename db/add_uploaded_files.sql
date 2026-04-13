-- ═══════════════════════════════════════════════════════════════
-- Migration: Persist uploaded files in PostgreSQL (BYTEA)
-- Replaces filesystem storage under uploads/{session_id}/
-- Safe to run multiple times (idempotent).
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS {app_schema}.uploaded_files (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID,                          -- nullable until session is established
    user_id     TEXT,
    file_name   TEXT         NOT NULL,
    file_size   INTEGER      NOT NULL DEFAULT 0,
    mime_type   TEXT,
    file_data   BYTEA        NOT NULL,
    uploaded_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_uploaded_files_session
    ON {app_schema}.uploaded_files (session_id)
    WHERE session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_uploaded_files_user
    ON {app_schema}.uploaded_files (user_id, uploaded_at DESC)
    WHERE user_id IS NOT NULL;
