"""FAISS Index Manager — in-memory vector search with PostgreSQL persistence.

Architecture:
    FAISS  → stores embeddings (vectors) in RAM for fast similarity search
    PostgreSQL → stores metadata + mappings + search logs (source of truth)

Flow:
    User query → Embedding model → FAISS similarity search → Top-K vector IDs
    → Fetch metadata from PostgreSQL → Return structured response
"""
import asyncio
import json
import time
import hashlib
import logging
import numpy as np
import faiss
from typing import Optional
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)


class FAISSIndex:
    """Single FAISS index with ID mapping back to PostgreSQL.

    Thread-safe via asyncio.Lock — all mutations (build/add/clear)
    are serialized while searches can proceed concurrently.
    """

    def __init__(self, index_name: str, dimensions: int = 768):
        self.index_name = index_name
        self.dimensions = dimensions
        self.index: Optional[faiss.IndexFlatIP] = None
        self.id_map: list[int] = []  # FAISS position → PostgreSQL metadata ID
        self._initialized = False
        self._lock = asyncio.Lock()

    def build(self, embeddings: np.ndarray, metadata_ids: list[int]):
        """Build FAISS index from embeddings. Normalizes vectors for cosine similarity.

        For large indexes (>10K vectors), uses IVF for faster search.
        """
        if len(embeddings) == 0:
            self.index = faiss.IndexFlatIP(self.dimensions)
            self.id_map = []
            self._initialized = True
            return

        # Normalize for cosine similarity (inner product on normalized = cosine)
        faiss.normalize_L2(embeddings)

        settings = get_settings()
        n_vectors = len(embeddings)

        # Use IVF for large indexes (configurable threshold)
        if n_vectors >= settings.FAISS_IVF_THRESHOLD:
            n_centroids = min(settings.FAISS_IVF_NCENTROIDS, n_vectors // 10)
            n_centroids = max(n_centroids, 1)
            quantizer = faiss.IndexFlatIP(self.dimensions)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimensions, n_centroids, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.nprobe = settings.FAISS_IVF_NPROBE
            self.index.add(embeddings)
            logger.info(f"FAISS index '{self.index_name}' built (IVF): {n_vectors} vectors, {n_centroids} centroids")
        else:
            self.index = faiss.IndexFlatIP(self.dimensions)
            self.index.add(embeddings)
            logger.info(f"FAISS index '{self.index_name}' built (Flat): {n_vectors} vectors")

        self.id_map = list(metadata_ids)
        self._initialized = True

        # Log memory usage
        mem_mb = (n_vectors * self.dimensions * 4) / (1024 * 1024)
        logger.info(f"  Memory estimate: {mem_mb:.1f} MB ({n_vectors} x {self.dimensions} x float32)")

    def add(self, embedding: np.ndarray, metadata_id: int):
        """Add a single vector to the index."""
        if not self._initialized:
            self.index = faiss.IndexFlatIP(self.dimensions)
            self._initialized = True

        vec = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self.id_map.append(metadata_id)

    def search(self, query_embedding: list[float], k: int = 5) -> list[tuple[int, float]]:
        """Search for top-k similar vectors. Returns [(metadata_id, similarity_score)]."""
        if not self._initialized or self.index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        actual_k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.id_map):
                results.append((self.id_map[idx], float(score)))
        return results

    @property
    def count(self) -> int:
        return self.index.ntotal if self._initialized and self.index else 0


# ── Global index registry (protected by lock) ─────────
_indexes: dict[str, FAISSIndex] = {}
_registry_lock = asyncio.Lock()


async def get_faiss_index(index_name: str) -> FAISSIndex:
    """Get or create a FAISS index by name. Thread-safe."""
    async with _registry_lock:
        if index_name not in _indexes:
            settings = get_settings()
            _indexes[index_name] = FAISSIndex(index_name, settings.EMBEDDING_DIMENSIONS)
        return _indexes[index_name]


async def load_index_from_db(index_name: str):
    """Load a FAISS index from PostgreSQL embedding_metadata table."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""SELECT id, embedding FROM {schema}.embedding_metadata
                WHERE index_name = $1 ORDER BY id""",
            index_name
        )

    idx = await get_faiss_index(index_name)
    async with idx._lock:
        if not rows:
            idx.build(np.array([], dtype=np.float32).reshape(0, settings.EMBEDDING_DIMENSIONS), [])
            logger.info(f"FAISS index '{index_name}': empty (no data in PostgreSQL)")
            return

        embeddings = np.array([list(r["embedding"]) for r in rows], dtype=np.float32)
        metadata_ids = [r["id"] for r in rows]
        idx.build(embeddings, metadata_ids)


async def load_all_indexes():
    """Load all 3 indexes from PostgreSQL at startup."""
    for name in ("schema_idx", "values_idx", "chunks_idx"):
        await load_index_from_db(name)
    total = 0
    for name in ("schema_idx", "values_idx", "chunks_idx"):
        idx = await get_faiss_index(name)
        total += idx.count
    logger.info(f"FAISS indexes loaded: {total} total vectors")


async def add_to_index(index_name: str, embedding: list[float], payload: dict,
                       content: str = "") -> int:
    """Add a vector to both PostgreSQL (persistence) and FAISS (memory).

    PostgreSQL is written FIRST (source of truth). FAISS is updated only
    after the DB write succeeds, preventing desync on DB failures.
    Returns the PostgreSQL metadata ID.
    """
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    # Step 1: Persist to PostgreSQL FIRST (source of truth)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""INSERT INTO {schema}.embedding_metadata
                (index_name, embedding, payload, content)
                VALUES ($1, $2, $3::jsonb, $4)
                RETURNING id""",
            index_name, embedding, json.dumps(payload), content
        )
        metadata_id = row["id"]

    # Step 2: Add to FAISS (in-memory) only after DB success
    idx = await get_faiss_index(index_name)
    async with idx._lock:
        vec = np.array(embedding, dtype=np.float32)
        idx.add(vec, metadata_id)

    return metadata_id


async def batch_add_to_index(index_name: str, embeddings: list[list[float]],
                              payloads: list[dict], contents: list[str] = None) -> list[int]:
    """Batch-add vectors to both PostgreSQL and FAISS.

    Much faster than calling add_to_index() in a loop:
    - Single PostgreSQL transaction for all inserts
    - Single FAISS index rebuild from all vectors
    """
    if not embeddings:
        return []

    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    if contents is None:
        contents = [""] * len(embeddings)

    # Step 1: Batch insert into PostgreSQL
    metadata_ids = []
    async with pool.acquire() as conn:
        async with conn.transaction():
            for emb, payload, content in zip(embeddings, payloads, contents):
                row = await conn.fetchrow(
                    f"""INSERT INTO {schema}.embedding_metadata
                        (index_name, embedding, payload, content)
                        VALUES ($1, $2, $3::jsonb, $4)
                        RETURNING id""",
                    index_name, emb, json.dumps(payload), content
                )
                metadata_ids.append(row["id"])

    # Step 2: Batch add to FAISS
    idx = await get_faiss_index(index_name)
    async with idx._lock:
        for emb, mid in zip(embeddings, metadata_ids):
            vec = np.array(emb, dtype=np.float32)
            idx.add(vec, mid)

    logger.info(f"Batch added {len(metadata_ids)} vectors to '{index_name}'")
    return metadata_ids


async def search_index(index_name: str, query_embedding: list[float], k: int = 5,
                       filter_key: str = None, filter_value: str = None) -> list[dict]:
    """Search FAISS index, then fetch metadata from PostgreSQL.

    Flow: FAISS search → top-K IDs → PostgreSQL metadata lookup → structured results.
    """
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    start_time = time.time()

    idx = await get_faiss_index(index_name)

    # If filter needed, fetch more candidates and filter post-search
    search_k = k * 3 if filter_key else k
    faiss_results = idx.search(query_embedding, k=search_k)

    if not faiss_results:
        return []

    # Fetch metadata from PostgreSQL
    metadata_ids = [r[0] for r in faiss_results]
    scores = {r[0]: r[1] for r in faiss_results}

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""SELECT id, payload, content FROM {schema}.embedding_metadata
                WHERE id = ANY($1::int[])""",
            metadata_ids
        )

    # Build results with metadata + similarity scores
    results = []
    for row in rows:
        payload = row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"])

        # Apply filter if specified
        if filter_key and filter_value:
            if payload.get(filter_key) != filter_value:
                continue

        results.append({
            "node_id": str(row["id"]),
            "similarity": scores.get(row["id"], 0.0),
            "payload": payload,
        })

    # Sort by similarity descending, limit to k
    results.sort(key=lambda x: x["similarity"], reverse=True)
    results = results[:k]

    # Log the search for audit
    elapsed_ms = int((time.time() - start_time) * 1000)
    logger.debug(f"FAISS search '{index_name}': {len(results)} results in {elapsed_ms}ms")

    return results


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
    """Clear a FAISS index and its PostgreSQL metadata. Thread-safe."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    idx = await get_faiss_index(index_name)
    async with idx._lock:
        # Clear PostgreSQL
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {schema}.embedding_metadata WHERE index_name = $1",
                index_name
            )

        # Rebuild empty FAISS index
        idx.build(np.array([], dtype=np.float32).reshape(0, settings.EMBEDDING_DIMENSIONS), [])

    logger.info(f"FAISS index '{index_name}' cleared")
