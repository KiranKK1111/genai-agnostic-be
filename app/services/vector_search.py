"""Vector search — PostgreSQL-native hybrid search (no in-memory indexes).

Architecture:
    PostgreSQL → stores embeddings (real[]) + tsvector for keyword search
    Dense       → dot product on normalized real[] arrays (cosine similarity)
    Sparse      → tsvector/tsquery full-text search with ts_rank
    RRF         → Reciprocal Rank Fusion combines both result sets

Flow:
    Search:  Query → Embedding + Text → PostgreSQL(dense + sparse) → RRF → Top-K
    Insert:  Embedding → PostgreSQL (L2-normalized)
"""
import json
import hashlib
import logging
from app.services.faiss_manager import (
    search_index, add_to_index,
    log_search, get_cached_retrieval, cache_retrieval,
    trace_retrieval,
)
from app.services.hybrid_search import hybrid_search
from app.config import get_settings

logger = logging.getLogger(__name__)


async def search(index_name: str, query_embedding: list[float], k: int = 5,
                 filter_key: str = None, filter_value: str = None,
                 query_text: str = None) -> list[dict]:
    """Search using hybrid dense+sparse → fetch metadata from PostgreSQL.
    Returns [{node_id, similarity, payload}].

    If query_text is provided, uses hybrid search (dense + sparse).
    Otherwise falls back to dense-only search.
    """
    if query_text and not filter_key:
        # Hybrid search: combine dense + sparse via RRF
        fused_results = await hybrid_search(
            index_name, query_text, query_embedding, k=k * 3 if filter_key else k
        )
        if fused_results:
            # Fetch metadata from PostgreSQL for fused results
            metadata_ids = [r[0] for r in fused_results]
            scores = {r[0]: r[1] for r in fused_results}

            from app.database import get_pool
            settings = get_settings()
            pool = get_pool()
            schema = settings.APP_SCHEMA

            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""SELECT id, payload, content FROM {schema}.embedding_metadata
                        WHERE id = ANY($1::int[])""",
                    metadata_ids
                )

            results = []
            for row in rows:
                payload = row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"])
                if filter_key and filter_value:
                    if payload.get(filter_key) != filter_value:
                        continue
                results.append({
                    "node_id": str(row["id"]),
                    "similarity": scores.get(row["id"], 0.0),
                    "payload": payload,
                })

            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:k]

    # Fallback: dense-only search (already returns full metadata)
    results = await search_index(index_name, query_embedding, k=k,
                                 filter_key=filter_key, filter_value=filter_value)
    return results


async def insert_node(index_name: str, embedding: list[float], payload: dict) -> int:
    """Insert embedding + metadata into PostgreSQL.
    Returns the PostgreSQL metadata ID."""
    content = payload.get("value", "") or payload.get("description", "") or payload.get("table", "")
    metadata_id = await add_to_index(index_name, embedding, payload, content=content)
    return metadata_id


async def schema_search(query: str, k: int = 5) -> list[dict]:
    """Search schema_idx for matching tables/columns by semantic similarity.
    Returns [{table, column, description, similarity}]."""
    from app.services.embedder import embed_single

    # Check retrieval cache first
    query_hash = hashlib.sha256(f"schema:{query}".encode()).hexdigest()
    cached_ids = await get_cached_retrieval(query_hash)
    if cached_ids:
        settings = get_settings()
        from app.database import get_pool
        pool = get_pool()
        schema = settings.APP_SCHEMA
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT payload FROM {schema}.embedding_metadata WHERE id = ANY($1::int[])",
                cached_ids
            )
            return [
                {
                    "table": (r["payload"] if isinstance(r["payload"], dict) else json.loads(r["payload"])).get("table"),
                    "column": (r["payload"] if isinstance(r["payload"], dict) else json.loads(r["payload"])).get("column"),
                    "description": (r["payload"] if isinstance(r["payload"], dict) else json.loads(r["payload"])).get("description", ""),
                    "similarity": 1.0,
                }
                for r in rows
            ]

    embedding = await embed_single(query)
    results = await search("schema_idx", embedding, k=k, query_text=query)

    # Log search + cache results + trace retrieval
    matched_ids = [int(r["node_id"]) for r in results]
    scores = [r["similarity"] for r in results]
    await log_search(query, "schema_idx", matched_ids, scores)
    if matched_ids:
        await cache_retrieval(query_hash, matched_ids)

    # Trace for explainability
    matched_nodes = [{"id": r["node_id"], "table": r["payload"].get("table"),
                      "column": r["payload"].get("column")} for r in results]
    await trace_retrieval(query, "schema_idx", matched_nodes, scores)

    return [
        {
            "table": r["payload"].get("table"),
            "column": r["payload"].get("column"),
            "description": r["payload"].get("description", ""),
            "similarity": r["similarity"],
        }
        for r in results
    ]


async def ground_value(value: str, k: int = 5) -> list[dict]:
    """Search values_idx for matching database values by semantic similarity.
    Used for value grounding: 'Vizag' -> 'Visakhapatnam'.
    Returns [{value, table, column, similarity}]."""
    from app.services.embedder import embed_single
    embedding = await embed_single(value)
    results = await search("values_idx", embedding, k=k, query_text=value)

    # Log the grounding search
    matched_ids = [int(r["node_id"]) for r in results]
    scores = [r["similarity"] for r in results]
    await log_search(value, "values_idx", matched_ids, scores)

    # Trace for explainability
    matched_nodes = [{"id": r["node_id"], "value": r["payload"].get("value"),
                      "table": r["payload"].get("table"),
                      "column": r["payload"].get("column")} for r in results]
    await trace_retrieval(value, "values_idx", matched_nodes, scores)

    return [
        {
            "value": r["payload"].get("value"),
            "table": r["payload"].get("table"),
            "column": r["payload"].get("column"),
            "similarity": r["similarity"],
        }
        for r in results
    ]


async def retrieve_chunks(query: str, session_id: str, k: int = 5) -> list[dict]:
    """Search chunks_idx for matching file chunks filtered by session.
    Used for RAG retrieval on uploaded files.
    Returns [{chunk_text, file_name, chunk_index, similarity}]."""
    from app.services.embedder import embed_single
    embedding = await embed_single(query)
    results = await search("chunks_idx", embedding, k=k,
                           filter_key="session_id", filter_value=session_id,
                           query_text=query)

    # Log the retrieval search
    matched_ids = [int(r["node_id"]) for r in results]
    scores = [r["similarity"] for r in results]
    await log_search(query, "chunks_idx", matched_ids, scores)

    # Trace for explainability
    matched_nodes = [{"id": r["node_id"], "file_name": r["payload"].get("file_name"),
                      "chunk_index": r["payload"].get("chunk_index")} for r in results]
    await trace_retrieval(query, "chunks_idx", matched_nodes, scores)

    return [
        {
            "chunk_text": r["payload"].get("chunk_text", ""),
            "file_name": r["payload"].get("file_name", ""),
            "chunk_index": r["payload"].get("chunk_index", 0),
            "similarity": r["similarity"],
        }
        for r in results
    ]
