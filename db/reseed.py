"""Re-seed schema index — run after schema changes."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import get_settings
from app.database import create_pool, close_pool
from app.services.schema_inspector import inspect_schema
from app.services.schema_seeder import seed_all


async def reseed():
    await create_pool()
    print("Inspecting schema...")
    graph = await inspect_schema()
    print(f"Found {len(graph.tables)} tables")
    print("Seeding FAISS + BM25 indexes...")
    await seed_all(graph)
    print("✓ Schema indexes re-seeded")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(reseed())
