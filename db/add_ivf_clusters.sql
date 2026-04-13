-- ═══════════════════════════════════════════════════════════════
-- Migration: Add IVF cluster support to embedding_metadata
-- Enables approximate nearest-neighbour search entirely in
-- PostgreSQL with zero in-memory state in the application.
--
-- Strategy: IVF (Inverted File Index) — the same algorithm used
-- inside pgvector's IVFFlat, implemented in pure PostgreSQL.
--   1. K-means clusters → centroids stored in vector_centroids
--   2. Each vector row gets a cluster_id (FK to centroid)
--   3. Search = probe k nearest centroids → scan those clusters
--
-- Safe to run multiple times (idempotent).
-- ═══════════════════════════════════════════════════════════════

-- ── cluster_id column on embedding_metadata ───────────────────
-- -1 = not yet assigned (before first cluster build)
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = '{app_schema}'
          AND table_name   = 'embedding_metadata'
          AND column_name  = 'cluster_id'
    ) THEN
        ALTER TABLE {app_schema}.embedding_metadata
            ADD COLUMN cluster_id INTEGER NOT NULL DEFAULT -1;
    END IF;
END $$;

-- Index: fast cluster-filtered scans at query time
CREATE INDEX IF NOT EXISTS idx_embedding_cluster
    ON {app_schema}.embedding_metadata (index_name, cluster_id);

-- ── Vector centroids table ────────────────────────────────────
CREATE TABLE IF NOT EXISTS {app_schema}.vector_centroids (
    id            SERIAL       PRIMARY KEY,
    index_name    TEXT         NOT NULL,
    cluster_id    INTEGER      NOT NULL,
    centroid      real[]       NOT NULL,
    vector_count  INTEGER      NOT NULL DEFAULT 0,
    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (index_name, cluster_id)
);
CREATE INDEX IF NOT EXISTS idx_centroids_index
    ON {app_schema}.vector_centroids (index_name);
