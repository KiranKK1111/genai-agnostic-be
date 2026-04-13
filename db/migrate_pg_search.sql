-- Migration: Add tsvector column for PostgreSQL full-text search (replaces in-memory BM25)
-- Safe to run multiple times (idempotent). Only does work on first run.

-- Wrap everything in a DO block: only perform work if the tsv column doesn't exist yet
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = '{app_schema}'
          AND table_name   = 'embedding_metadata'
          AND column_name  = 'tsv'
    ) THEN
        -- 1. Add tsvector column
        ALTER TABLE {app_schema}.embedding_metadata ADD COLUMN tsv tsvector;

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
        CREATE OR REPLACE FUNCTION {app_schema}.embedding_tsv_trigger() RETURNS trigger AS $fn$
        BEGIN
            NEW.tsv := to_tsvector('simple', coalesce(NEW.content, ''));
            RETURN NEW;
        END;
        $fn$ LANGUAGE plpgsql;

        DROP TRIGGER IF EXISTS trg_embedding_tsv ON {app_schema}.embedding_metadata;
        CREATE TRIGGER trg_embedding_tsv
            BEFORE INSERT OR UPDATE OF content ON {app_schema}.embedding_metadata
            FOR EACH ROW EXECUTE FUNCTION {app_schema}.embedding_tsv_trigger();
    END IF;
END $$;
