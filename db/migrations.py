"""Simple migration runner — applies init.sql idempotently."""
import asyncio
import os
import asyncpg
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import get_settings


async def run_migrations():
    settings = get_settings()
    conn = await asyncpg.connect(dsn=settings.dsn)

    base_dir = os.path.dirname(__file__)

    try:
        # Step 1: Drop legacy HNSW objects (safe to run multiple times)
        drop_path = os.path.join(base_dir, "drop_hnsw_legacy.sql")
        if os.path.exists(drop_path):
            with open(drop_path) as f:
                drop_sql = f.read().replace("{app_schema}", settings.APP_SCHEMA)
            await conn.execute(drop_sql)
            print(f"✓ Legacy HNSW objects dropped (if they existed)")

        # Step 2: Apply main init.sql (creates FAISS + PostgreSQL tables)
        init_path = os.path.join(base_dir, "init.sql")
        with open(init_path) as f:
            sql = f.read().replace("{app_schema}", settings.APP_SCHEMA)
        await conn.execute(sql)
        print(f"✓ Migrations applied to {settings.POSTGRES_DB} (schema: {settings.APP_SCHEMA})")
    except Exception as e:
        print(f"✗ Migration error: {e}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run_migrations())
