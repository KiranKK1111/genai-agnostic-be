"""Schema seeder — embed table/column names and distinct values into FAISS + PostgreSQL.

Architecture:
    1. Generate embeddings via Ollama (concurrent batch)
    2. Batch insert into PostgreSQL (single transaction)
    3. Build FAISS in-memory indexes
"""
import logging
from app.services.embedder import embed_texts
from app.services.faiss_manager import batch_add_to_index, clear_index
from app.services.schema_inspector import SchemaGraph
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)

# Max distinct values per column to index
MAX_DISTINCT_VALUES = 200


async def seed_schema_index(graph: SchemaGraph):
    """Embed all table and column names into schema_idx (FAISS + PostgreSQL)."""
    await clear_index("schema_idx")

    texts = []
    payloads = []
    for tname, tmeta in graph.tables.items():
        col_names = ", ".join(tmeta.columns.keys())
        texts.append(f"table {tname}: {col_names}")
        payloads.append({"type": "table", "table": tname, "description": col_names,
                         "columns": list(tmeta.columns.keys())})

        for cname, cinfo in tmeta.columns.items():
            texts.append(f"column {cname} in table {tname}, type {cinfo['data_type']}")
            payloads.append({"type": "column", "table": tname, "column": cname,
                            "description": f"{cname} ({cinfo['data_type']})",
                            "data_type": cinfo["data_type"]})

    if not texts:
        logger.warning("No schema entries to seed")
        return

    logger.info(f"Seeding schema_idx with {len(texts)} entries (batch)...")

    # Concurrent embedding (8 parallel requests)
    embeddings = await embed_texts(texts)

    # Single batch insert (1 transaction instead of 93 individual inserts)
    await batch_add_to_index("schema_idx", embeddings, payloads, contents=texts)

    logger.info(f"  schema_idx seeded: {len(texts)} entries")


async def seed_values_index(graph: SchemaGraph):
    """Embed distinct string column values into values_idx (FAISS + PostgreSQL)."""
    settings = get_settings()
    pool = get_pool()
    db_schema = settings.POSTGRES_SCHEMA

    await clear_index("values_idx")

    texts = []
    payloads = []

    async with pool.acquire() as conn:
        for tname, tmeta in graph.tables.items():
            for cname, cinfo in tmeta.columns.items():
                if cinfo["data_type"] not in ("character varying", "text", "varchar"):
                    continue

                try:
                    count_row = await conn.fetchrow(
                        f"SELECT COUNT(DISTINCT {cname}) as cnt FROM {db_schema}.{tname}"
                    )
                    distinct_count = count_row["cnt"] if count_row else 0

                    if distinct_count == 0 or distinct_count > MAX_DISTINCT_VALUES:
                        continue

                    rows = await conn.fetch(
                        f"SELECT DISTINCT {cname} FROM {db_schema}.{tname} WHERE {cname} IS NOT NULL ORDER BY {cname} LIMIT {MAX_DISTINCT_VALUES}"
                    )

                    for row in rows:
                        val = row[cname]
                        if val and isinstance(val, str) and len(val.strip()) > 0:
                            texts.append(val)
                            payloads.append({
                                "value": val,
                                "table": tname,
                                "column": cname,
                            })
                except Exception as e:
                    logger.debug(f"Skipping {tname}.{cname} for values_idx: {e}")
                    continue

    if not texts:
        logger.info("  values_idx: no indexable values found")
        return

    logger.info(f"Seeding values_idx with {len(texts)} values (batch)...")

    # Concurrent embedding (8 parallel requests)
    embeddings = await embed_texts(texts)

    # Single batch insert
    await batch_add_to_index("values_idx", embeddings, payloads, contents=texts)

    logger.info(f"  values_idx seeded: {len(texts)} values across {len(set(p['table'] for p in payloads))} tables")


async def seed_all(graph: SchemaGraph):
    """Seed both schema_idx and values_idx (FAISS + PostgreSQL)."""
    await seed_schema_index(graph)
    await seed_values_index(graph)
