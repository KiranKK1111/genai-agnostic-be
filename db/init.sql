-- ═══════════════════════════════════════════════════════════
-- GenAI Dashboard — Application Schema DDL
-- Run once on first startup. Idempotent (IF NOT EXISTS).
-- ═══════════════════════════════════════════════════════════

-- Create app schema
CREATE SCHEMA IF NOT EXISTS {app_schema};

-- ── Users ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS {app_schema}.users (
    id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    username        TEXT UNIQUE NOT NULL,
    email           TEXT UNIQUE,
    hashed_password TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    is_active       BOOLEAN NOT NULL DEFAULT true,
    preferences     JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── KV Store (Redis replacement) ──────────────────────────
CREATE TABLE IF NOT EXISTS {app_schema}.kv_store (
    key         TEXT PRIMARY KEY,
    value       JSONB NOT NULL,
    namespace   TEXT NOT NULL DEFAULT 'default',
    expires_at  TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_kv_namespace ON {app_schema}.kv_store(namespace);
CREATE INDEX IF NOT EXISTS idx_kv_expires ON {app_schema}.kv_store(expires_at) WHERE expires_at IS NOT NULL;

-- ── Chat Sessions ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS {app_schema}.chat_sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT REFERENCES {app_schema}.users(id) ON DELETE CASCADE,
    title       TEXT DEFAULT 'New Chat',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON {app_schema}.chat_sessions(user_id, updated_at DESC);

-- ── Chat Messages ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS {app_schema}.chat_messages (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          UUID NOT NULL REFERENCES {app_schema}.chat_sessions(id) ON DELETE CASCADE,
    role                TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content             TEXT NOT NULL,
    content_sql         TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}',
    follow_ups          JSONB NOT NULL DEFAULT '[]',
    reaction            TEXT CHECK (reaction IN ('like', 'dislike')),
    reaction_comment    TEXT,
    version             INTEGER NOT NULL DEFAULT 1,
    parent_message_id   UUID REFERENCES {app_schema}.chat_messages(id),
    is_active           BOOLEAN NOT NULL DEFAULT true,
    session_snapshot    JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON {app_schema}.chat_messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_parent ON {app_schema}.chat_messages(parent_message_id) WHERE parent_message_id IS NOT NULL;

-- ── File Chunks (RAG) ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS {app_schema}.file_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL,
    file_name       TEXT NOT NULL,
    file_path       TEXT,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    node_id         TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chunks_session ON {app_schema}.file_chunks(session_id, file_name);

-- ── Schema Index (metadata cache + description cache) ─────
CREATE TABLE IF NOT EXISTS {app_schema}.schema_index (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name      TEXT NOT NULL,
    column_name     TEXT,
    data_type       TEXT,
    description     TEXT,
    sample_values   JSONB,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_schema_index_table ON {app_schema}.schema_index(table_name);

-- ── Embedding Cache ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS {app_schema}.embedding_cache (
    text_hash   TEXT PRIMARY KEY,
    embedding   real[] NOT NULL,
    model       TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Audit Log ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS {app_schema}.audit_log (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             TEXT,
    session_id          UUID,
    user_prompt         TEXT,
    classified_intent   TEXT,
    generated_sql       TEXT,
    query_plan_json     JSONB,
    execution_time_ms   INTEGER,
    row_count           INTEGER,
    status              TEXT NOT NULL DEFAULT 'SUCCESS',
    error_message       TEXT,
    llm_tokens_used     INTEGER DEFAULT 0,
    input_quality_score FLOAT,
    injection_flags     TEXT[],
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_audit_user_time ON {app_schema}.audit_log(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_status ON {app_schema}.audit_log(status);

-- ── Feedback ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS {app_schema}.feedback (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT,
    session_id      UUID,
    message_id      UUID,
    rating          SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    feedback_text   TEXT,
    intent_at_time  TEXT,
    generated_sql_at_time TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_feedback_session ON {app_schema}.feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON {app_schema}.feedback(rating, created_at DESC);

-- ── Saved Queries ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS {app_schema}.saved_queries (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT REFERENCES {app_schema}.users(id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    original_prompt TEXT NOT NULL,
    generated_sql   TEXT,
    viz_config      JSONB,
    tags            TEXT[] DEFAULT '{}',
    run_count       INTEGER DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_run_at     TIMESTAMPTZ
);

-- ── FAISS Persistence Tables ────────────────────────────────
-- Embedding Metadata: FAISS vector index → PostgreSQL row mapping
CREATE TABLE IF NOT EXISTS {app_schema}.embedding_metadata (
    id              SERIAL PRIMARY KEY,
    index_name      TEXT NOT NULL,
    embedding       real[] NOT NULL,
    payload         JSONB NOT NULL DEFAULT '{}',
    content         TEXT DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_embedding_meta_index ON {app_schema}.embedding_metadata(index_name);

-- Query Logs: audit trail for all FAISS searches
CREATE TABLE IF NOT EXISTS {app_schema}.query_logs (
    id                  SERIAL PRIMARY KEY,
    user_query          TEXT,
    index_name          TEXT,
    matched_ids         INTEGER[],
    similarity_scores   FLOAT[],
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_query_logs_time ON {app_schema}.query_logs(created_at DESC);

-- Retrieval Cache: speeds up repeated searches (TTL-aware)
CREATE TABLE IF NOT EXISTS {app_schema}.retrieval_cache (
    query_hash      TEXT PRIMARY KEY,
    matched_ids     INTEGER[] NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_retrieval_cache_age ON {app_schema}.retrieval_cache(created_at);

-- ── Retrieval Trace (explainability + debugging) ──────────
CREATE TABLE IF NOT EXISTS {app_schema}.retrieval_trace (
    id              SERIAL PRIMARY KEY,
    user_query      TEXT NOT NULL,
    index_name      TEXT NOT NULL,
    matched_nodes   JSONB NOT NULL DEFAULT '[]',
    similarity_scores FLOAT[] NOT NULL DEFAULT '{}',
    response        TEXT,
    latency_ms      INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_retrieval_trace_time ON {app_schema}.retrieval_trace(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_retrieval_trace_index ON {app_schema}.retrieval_trace(index_name);

-- ── Auto-update timestamp trigger ─────────────────────────
CREATE OR REPLACE FUNCTION {app_schema}.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_users_updated_at') THEN
        CREATE TRIGGER trg_users_updated_at BEFORE UPDATE ON {app_schema}.users
            FOR EACH ROW EXECUTE FUNCTION {app_schema}.update_updated_at();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_sessions_updated_at') THEN
        CREATE TRIGGER trg_sessions_updated_at BEFORE UPDATE ON {app_schema}.chat_sessions
            FOR EACH ROW EXECUTE FUNCTION {app_schema}.update_updated_at();
    END IF;
END $$;
