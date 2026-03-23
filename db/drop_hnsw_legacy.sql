-- ═══════════════════════════════════════════════════════════
-- Migration: Drop legacy PL/pgSQL HNSW objects
-- Run ONCE on existing deployments that had the old HNSW architecture.
-- Safe to run multiple times (IF EXISTS / CASCADE).
-- ═══════════════════════════════════════════════════════════

-- Drop legacy functions
DROP FUNCTION IF EXISTS {app_schema}.hnsw_search(TEXT, real[], INTEGER, TEXT, TEXT) CASCADE;
DROP FUNCTION IF EXISTS {app_schema}.cosine_similarity(real[], real[]) CASCADE;

-- Drop legacy tables (order matters: edges → nodes → config due to FK)
DROP TABLE IF EXISTS {app_schema}.hnsw_edges CASCADE;
DROP TABLE IF EXISTS {app_schema}.hnsw_nodes CASCADE;
DROP TABLE IF EXISTS {app_schema}.hnsw_config CASCADE;
