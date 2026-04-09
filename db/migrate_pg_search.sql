-- Migration: Add tsvector column for PostgreSQL full-text search (replaces in-memory BM25)
-- Also adds a function for dot-product similarity on real[] columns (replaces FAISS)

-- 1. Add tsvector column for keyword search
ALTER TABLE {app_schema}.embedding_metadata
    ADD COLUMN IF NOT EXISTS tsv tsvector;

-- 2. Populate tsvector from existing content
UPDATE {app_schema}.embedding_metadata
    SET tsv = to_tsvector('simple', coalesce(content, ''));

-- 3. GIN index for fast full-text search
CREATE INDEX IF NOT EXISTS idx_embedding_tsv
    ON {app_schema}.embedding_metadata USING GIN (tsv);

-- 4. Composite index for filtered searches (index_name + id)
CREATE INDEX IF NOT EXISTS idx_embedding_meta_name_id
    ON {app_schema}.embedding_metadata (index_name, id);

-- 5. Auto-update tsvector on insert/update via trigger
CREATE OR REPLACE FUNCTION {app_schema}.embedding_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('simple', coalesce(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_embedding_tsv ON {app_schema}.embedding_metadata;
CREATE TRIGGER trg_embedding_tsv
    BEFORE INSERT OR UPDATE OF content ON {app_schema}.embedding_metadata
    FOR EACH ROW EXECUTE FUNCTION {app_schema}.embedding_tsv_trigger();
