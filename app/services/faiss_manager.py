"""Vector Index Manager — PostgreSQL-native vector search (no in-memory FAISS).

Architecture:
    PostgreSQL → stores embeddings (real[]) + tsvector for keyword search
    Dense search  → dot product on normalized real[] arrays (cosine similarity)
    Sparse search → tsvector/tsquery full-text search with ts_rank
    No FAISS, no in-memory indexes — everything fetched dynamically from PostgreSQL.

Flow:
    User query → Embedding model → PostgreSQL dot product → Top-K results
    User query → tsquery → PostgreSQL ts_rank → Top-K results
    → Reciprocal Rank Fusion → Final ranked results
"""
import json
import time
import hashlib
import logging
import numpy as np
from typing import Optional
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)


# ── Dense Search (replaces FAISS) ─────────────────────────────
async def dense_search(index_name: str, query_embedding: list[float], k: int = 5,
                       filter_key: str = None, filter_value: str = None) -> list[tuple[int, float]]:
    """Cosine similarity search via PostgreSQL dot product on normalized real[] vectors.

    Since vectors are L2-normalized at insert time, dot product = cosine similarity.
    Returns [(metadata_id, similarity_score)] sorted by similarity descending.
    """
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    # Build filter clause
    filter_clause = ""
    params = [query_embedding, index_name, k * 3 if filter_key else k]
    if filter_key and filter_value:
        filter_clause = f"AND payload->>$4 = $5"
        params.extend([filter_key, filter_value])

    query_sql = f"""
        SELECT id, payload,
               (SELECT COALESCE(sum(a * b), 0)
                FROM unnest(embedding, $1::real[]) AS t(a, b)) AS similarity
        FROM {schema}.embedding_metadata
        WHERE index_name = $2 {filter_clause}
        ORDER BY similarity DESC
        LIMIT $3
    """

    start = time.time()
    async with pool.acquire() as conn:
        rows = await conn.fetch(query_sql, *params)

    results = [(row["id"], float(row["similarity"])) for row in rows]
    elapsed_ms = int((time.time() - start) * 1000)
    logger.debug(f"Dense search '{index_name}': {len(results)} results in {elapsed_ms}ms")
    return results


# ── Sparse Search (replaces BM25) ─────────────────────────────
async def sparse_search(index_name: str, query_text: str, k: int = 5) -> list[tuple[int, float]]:
    """Full-text keyword search via PostgreSQL tsvector/tsquery.

    Uses 'simple' config (no stemming) to match exact tokens — mirrors BM25 behavior.
    Returns [(metadata_id, ts_rank_score)] sorted by rank descending.
    """
    if not query_text or not query_text.strip():
        return []

    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    # Build tsquery: split words and OR them for broad recall
    tokens = [t.strip() for t in query_text.lower().split() if t.strip()]
    if not tokens:
        return []
    tsquery_str = " | ".join(tokens)

    query_sql = f"""
        SELECT id, ts_rank(tsv, to_tsquery('simple', $1)) AS rank
        FROM {schema}.embedding_metadata
        WHERE index_name = $2 AND tsv @@ to_tsquery('simple', $1)
        ORDER BY rank DESC
        LIMIT $3
    """

    start = time.time()
    async with pool.acquire() as conn:
        rows = await conn.fetch(query_sql, tsquery_str, index_name, k)

    results = [(row["id"], float(row["rank"])) for row in rows]
    elapsed_ms = int((time.time() - start) * 1000)
    logger.debug(f"Sparse search '{index_name}': {len(results)} results in {elapsed_ms}ms")
    return results


# ── Index Operations (PostgreSQL-only) ────────────────────────

async def add_to_index(index_name: str, embedding: list[float], payload: dict,
                       content: str = "") -> int:
    """Add a vector to PostgreSQL. No in-memory state.

    Vectors are L2-normalized before storage so dot product = cosine similarity.
    The tsvector column is auto-populated by the database trigger.
    """
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    # L2-normalize before storing
    vec = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    normalized = vec.tolist()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""INSERT INTO {schema}.embedding_metadata
                (index_name, embedding, payload, content)
                VALUES ($1, $2, $3::jsonb, $4)
                RETURNING id""",
            index_name, normalized, json.dumps(payload), content
        )
    return row["id"]


async def batch_add_to_index(index_name: str, embeddings: list[list[float]],
                              payloads: list[dict], contents: list[str] = None) -> list[int]:
    """Batch-add vectors to PostgreSQL. No in-memory state.

    Normalizes vectors and inserts in a single transaction.
    """
    if not embeddings:
        return []

    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    if contents is None:
        contents = [""] * len(embeddings)

    # L2-normalize all vectors
    arr = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    normalized_list = arr.tolist()

    metadata_ids = []
    async with pool.acquire() as conn:
        async with conn.transaction():
            for emb, payload, content in zip(normalized_list, payloads, contents):
                row = await conn.fetchrow(
                    f"""INSERT INTO {schema}.embedding_metadata
                        (index_name, embedding, payload, content)
                        VALUES ($1, $2, $3::jsonb, $4)
                        RETURNING id""",
                    index_name, emb, json.dumps(payload), content
                )
                metadata_ids.append(row["id"])

    logger.info(f"Batch added {len(metadata_ids)} vectors to '{index_name}' (PostgreSQL)")
    return metadata_ids


async def search_index(index_name: str, query_embedding: list[float], k: int = 5,
                       filter_key: str = None, filter_value: str = None) -> list[dict]:
    """Search PostgreSQL for similar vectors, fetch metadata inline.

    Replaces the old FAISS search → PostgreSQL metadata lookup flow.
    Now everything is a single SQL query.
    """
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    start_time = time.time()

    # Build filter clause
    filter_clause = ""
    params = [query_embedding, index_name, k * 3 if filter_key else k]
    if filter_key and filter_value:
        filter_clause = f"AND payload->>$4 = $5"
        params.extend([filter_key, filter_value])

    query_sql = f"""
        SELECT id, payload, content,
               (SELECT COALESCE(sum(a * b), 0)
                FROM unnest(embedding, $1::real[]) AS t(a, b)) AS similarity
        FROM {schema}.embedding_metadata
        WHERE index_name = $2 {filter_clause}
        ORDER BY similarity DESC
        LIMIT $3
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query_sql, *params)

    results = []
    for row in rows:
        payload = row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"])
        results.append({
            "node_id": str(row["id"]),
            "similarity": float(row["similarity"]),
            "payload": payload,
        })

    elapsed_ms = int((time.time() - start_time) * 1000)
    logger.debug(f"PG search '{index_name}': {len(results)} results in {elapsed_ms}ms")
    return results


# ── Audit & Cache (unchanged — already PostgreSQL) ────────────

async def log_search(user_query: str, index_name: str, matched_ids: list[int],
                     similarity_scores: list[float] = None):
    """Persist search request to query_logs for audit/analytics."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {schema}.query_logs
                    (user_query, index_name, matched_ids, similarity_scores)
                    VALUES ($1, $2, $3, $4)""",
                user_query, index_name, matched_ids,
                similarity_scores or []
            )
    except Exception as e:
        logger.debug(f"Query log write skipped: {e}")


async def get_cached_retrieval(query_hash: str) -> list[int] | None:
    """Check retrieval cache for repeated queries. Respects TTL."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""SELECT matched_ids FROM {schema}.retrieval_cache
                    WHERE query_hash = $1
                      AND created_at > now() - interval '{settings.RETRIEVAL_CACHE_TTL_MIN} minutes'""",
                query_hash
            )
            return list(row["matched_ids"]) if row else None
    except Exception:
        return None


async def cache_retrieval(query_hash: str, matched_ids: list[int]):
    """Cache retrieval results for repeated queries."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {schema}.retrieval_cache (query_hash, matched_ids)
                    VALUES ($1, $2)
                    ON CONFLICT (query_hash) DO UPDATE
                        SET matched_ids = $2, created_at = now()""",
                query_hash, matched_ids
            )
    except Exception as e:
        logger.debug(f"Retrieval cache write skipped: {e}")


async def invalidate_retrieval_cache():
    """Clear all entries from retrieval_cache. Called during schema rebuild."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    try:
        async with pool.acquire() as conn:
            deleted = await conn.execute(f"DELETE FROM {schema}.retrieval_cache")
            logger.info(f"Retrieval cache invalidated: {deleted}")
    except Exception as e:
        logger.debug(f"Retrieval cache invalidation skipped: {e}")


async def cleanup_stale_cache():
    """Remove expired cache entries (older than TTL)."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                f"""DELETE FROM {schema}.retrieval_cache
                    WHERE created_at < now() - interval '{settings.RETRIEVAL_CACHE_TTL_MIN} minutes'"""
            )
    except Exception as e:
        logger.debug(f"Cache cleanup skipped: {e}")


async def trace_retrieval(user_query: str, index_name: str, matched_nodes: list[dict],
                          similarity_scores: list[float] = None, response: str = None,
                          latency_ms: int = None):
    """Persist retrieval trace for explainability and debugging."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {schema}.retrieval_trace
                    (user_query, index_name, matched_nodes, similarity_scores, response, latency_ms)
                    VALUES ($1, $2, $3::jsonb, $4, $5, $6)""",
                user_query, index_name, json.dumps(matched_nodes),
                similarity_scores or [], response, latency_ms
            )
    except Exception as e:
        logger.debug(f"Retrieval trace write skipped: {e}")


async def clear_index(index_name: str):
    """Clear all vectors for an index from PostgreSQL."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        await conn.execute(
            f"DELETE FROM {schema}.embedding_metadata WHERE index_name = $1",
            index_name
        )
    logger.info(f"Index '{index_name}' cleared (PostgreSQL)")
