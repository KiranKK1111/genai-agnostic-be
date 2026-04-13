"""Vector Index Manager — Pure PostgreSQL IVF (Inverted File Index).

Architecture — zero in-memory state, everything persisted in the database:

    ┌──────────────────────────────────────────────────────────────┐
    │  {schema}.embedding_metadata                                 │
    │    id, index_name, embedding real[], payload, cluster_id     │
    │                                                              │
    │  {schema}.vector_centroids                                   │
    │    index_name, cluster_id, centroid real[], vector_count     │
    └──────────────────────────────────────────────────────────────┘

Search (IVF two-phase, O(n_probe * N/k)):
    1. Dot-product against k centroids (tiny scan, ~100 rows)
    2. Scan only the top n_probe clusters (fraction of total vectors)

Search (direct, O(N)) — automatic fallback when:
    • Index has fewer than FAISS_IVF_THRESHOLD vectors, OR
    • Centroids have not been built yet for this index

Clustering (mini-batch k-means, cursor-paginated):
    • Reads vectors in batches — peak RAM ≈ batch_size × dim × 4 bytes
    • Assigns cluster_id via batch SQL unnest UPDATE
    • build_all_ivf_clusters() called once at startup and on watcher refresh

No FAISS library used at runtime. No in-memory index structures.
No state is lost on restart — everything lives in PostgreSQL.
"""
import json
import time
import logging
import numpy as np
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)

# ── IVF clustering ────────────────────────────────────────────

_KMEANS_BATCH = 5_000   # rows loaded per iteration (keeps peak RAM low)
_ASSIGN_BATCH = 10_000  # rows per bulk cluster-assignment UPDATE


async def _kmeans_cluster(
    pool, schema: str, index_name: str,
    n_clusters: int, max_iter: int = 20,
) -> int:
    """Mini-batch k-means on a single index. Stores centroids + assigns cluster_id.

    All intermediate state is kept in numpy arrays (temporary, not persisted).
    Final result — centroids and per-row cluster_id — is written to PostgreSQL.
    Returns the actual number of clusters built.
    """
    dim = get_settings().EMBEDDING_DIMENSIONS

    # Count vectors
    async with pool.acquire() as conn:
        total: int = await conn.fetchval(
            f"SELECT COUNT(*) FROM {schema}.embedding_metadata WHERE index_name=$1",
            index_name,
        ) or 0

    if total == 0:
        return 0

    n_clusters = min(n_clusters, total)

    # ── Step 1: Seed centroids with random sample ─────────────
    async with pool.acquire() as conn:
        seed_rows = await conn.fetch(
            f"""SELECT embedding FROM {schema}.embedding_metadata
                WHERE index_name = $1
                ORDER BY random()
                LIMIT $2""",
            index_name, n_clusters,
        )
    centroids = np.array([list(r["embedding"]) for r in seed_rows], dtype=np.float32)

    # ── Step 2: Lloyd's iterations (cursor-paged) ─────────────
    for _it in range(max_iter):
        sum_vecs = np.zeros_like(centroids)
        counts   = np.zeros(n_clusters, dtype=np.int64)
        offset   = 0

        while True:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""SELECT embedding FROM {schema}.embedding_metadata
                        WHERE index_name = $1
                        ORDER BY id
                        LIMIT $2 OFFSET $3""",
                    index_name, _KMEANS_BATCH, offset,
                )
            if not rows:
                break

            vecs   = np.array([list(r["embedding"]) for r in rows], dtype=np.float32)
            assign = np.argmax(vecs @ centroids.T, axis=1)

            for i, a in enumerate(assign):
                sum_vecs[a] += vecs[i]
                counts[a]   += 1

            offset += len(rows)
            if len(rows) < _KMEANS_BATCH:
                break

        # Update + re-normalise centroids
        for i in range(n_clusters):
            if counts[i] > 0:
                c = sum_vecs[i] / counts[i]
                nrm = np.linalg.norm(c)
                centroids[i] = c / nrm if nrm > 0 else c

    # ── Step 3: Persist centroids ─────────────────────────────
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                f"DELETE FROM {schema}.vector_centroids WHERE index_name=$1",
                index_name,
            )
            for i, (cen, cnt) in enumerate(zip(centroids, counts)):
                await conn.execute(
                    f"""INSERT INTO {schema}.vector_centroids
                            (index_name, cluster_id, centroid, vector_count)
                        VALUES ($1, $2, $3, $4)""",
                    index_name, i, cen.tolist(), int(cnt),
                )

    # ── Step 4: Assign cluster_id to every row (batched UPDATE) ─
    offset = 0
    while True:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT id, embedding FROM {schema}.embedding_metadata
                    WHERE index_name = $1
                    ORDER BY id
                    LIMIT $2 OFFSET $3""",
                index_name, _ASSIGN_BATCH, offset,
            )
        if not rows:
            break

        ids   = [r["id"] for r in rows]
        vecs  = np.array([list(r["embedding"]) for r in rows], dtype=np.float32)
        asgn  = np.argmax(vecs @ centroids.T, axis=1).tolist()

        async with pool.acquire() as conn:
            await conn.execute(
                f"""UPDATE {schema}.embedding_metadata m
                    SET cluster_id = u.cid
                    FROM unnest($1::int[], $2::int[]) AS u(id, cid)
                    WHERE m.id = u.id""",
                ids, asgn,
            )

        offset += len(rows)
        if len(rows) < _ASSIGN_BATCH:
            break

    logger.info(
        f"IVF '{index_name}': {n_clusters} clusters built for {total} vectors"
    )
    return n_clusters


async def build_ivf_clusters(index_name: str, force: bool = False) -> int:
    """Build IVF clusters for one index.

    Skips the full k-means rebuild when the persisted clusters are still valid:
      - centroids already exist in vector_centroids, AND
      - their stored vector_count total matches the current row count in
        embedding_metadata (meaning no inserts/deletes since last build), AND
      - no rows are unassigned (cluster_id = -1).

    Pass force=True to always rebuild regardless (e.g. after a forced full reindex).
    Returns the number of clusters (built or reused).
    """
    settings = get_settings()
    pool     = get_pool()
    schema   = settings.APP_SCHEMA
    k        = settings.FAISS_IVF_NCENTROIDS

    if not force:
        async with pool.acquire() as conn:
            existing_clusters: int = await conn.fetchval(
                f"SELECT COUNT(*) FROM {schema}.vector_centroids WHERE index_name=$1",
                index_name,
            ) or 0
            if existing_clusters > 0:
                total_vectors: int = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {schema}.embedding_metadata WHERE index_name=$1",
                    index_name,
                ) or 0
                built_for: int = await conn.fetchval(
                    f"SELECT COALESCE(SUM(vector_count), 0) FROM {schema}.vector_centroids WHERE index_name=$1",
                    index_name,
                ) or 0
                unassigned: int = await conn.fetchval(
                    f"""SELECT COUNT(*) FROM {schema}.embedding_metadata
                        WHERE index_name=$1 AND (cluster_id IS NULL OR cluster_id = -1)""",
                    index_name,
                ) or 0
                if total_vectors == built_for and unassigned == 0:
                    logger.info(
                        f"IVF '{index_name}': clusters valid "
                        f"({existing_clusters} clusters, {total_vectors} vectors) — skipping rebuild"
                    )
                    return int(existing_clusters)

    return await _kmeans_cluster(pool, schema, index_name, n_clusters=k)


async def build_all_ivf_clusters(force: bool = False) -> dict[str, int]:
    """Build IVF clusters for every index_name present in embedding_metadata.

    Each index is checked individually — only indexes whose vector count has
    changed since the last build (or that have unassigned rows) are rebuilt.
    Pass force=True to unconditionally rebuild all indexes.
    Returns {index_name: n_clusters}.
    """
    settings = get_settings()
    pool     = get_pool()
    schema   = settings.APP_SCHEMA

    async with pool.acquire() as conn:
        names = await conn.fetch(
            f"SELECT DISTINCT index_name FROM {schema}.embedding_metadata"
        )

    result: dict[str, int] = {}
    for row in names:
        name = row["index_name"]
        n = await build_ivf_clusters(name, force=force)
        result[name] = n

    logger.info(f"IVF cluster status: {result}")
    return result


# ── Cluster assignment helper (used on insert) ────────────────

async def _assign_cluster(
    pool, schema: str, index_name: str, vec: list[float]
) -> int:
    """Find the nearest centroid for a single vector. Returns cluster_id or -1."""
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""SELECT cluster_id
                    FROM {schema}.vector_centroids
                    WHERE index_name = $1
                    ORDER BY (
                        SELECT COALESCE(sum(a * b), 0)
                        FROM unnest(centroid, $2::real[]) AS t(a, b)
                    ) DESC
                    LIMIT 1""",
                index_name, vec,
            )
        return row["cluster_id"] if row else -1
    except Exception:
        return -1


async def _batch_assign_clusters(
    pool, schema: str, index_name: str, vecs: np.ndarray
) -> list[int]:
    """Assign cluster_ids for a batch of vectors.

    Loads centroids once, then does the dot-product in numpy (fast).
    Falls back to -1 if no centroids exist yet.
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT cluster_id, centroid
                    FROM {schema}.vector_centroids
                    WHERE index_name = $1
                    ORDER BY cluster_id""",
                index_name,
            )
        if not rows:
            return [-1] * len(vecs)

        centroids = np.array([list(r["centroid"]) for r in rows], dtype=np.float32)
        cids      = [r["cluster_id"] for r in rows]
        sims      = vecs @ centroids.T             # (N, k)
        best      = np.argmax(sims, axis=1)        # (N,) — index into cids list
        return [cids[i] for i in best.tolist()]
    except Exception:
        return [-1] * len(vecs)


# ── Dense Search ──────────────────────────────────────────────

async def dense_search(
    index_name: str,
    query_embedding: list[float],
    k: int = 5,
    filter_key: str = None,
    filter_value: str = None,
) -> list[tuple[int, float]]:
    """IVF-accelerated cosine similarity search — pure PostgreSQL, no RAM state.

    Path A (IVF):  centroids scan → cluster scan — O(n_probe × N/k)
    Path B (flat): full table scan — O(N)  — used when IVF not available

    Returns [(postgres_id, similarity_score)] sorted descending.
    """
    settings = get_settings()
    pool     = get_pool()
    schema   = settings.APP_SCHEMA
    n_probe  = settings.FAISS_IVF_NPROBE
    threshold = settings.FAISS_IVF_THRESHOLD

    # Build filter clause
    filter_clause = ""
    base_params: list = [query_embedding, index_name]
    if filter_key and filter_value:
        filter_clause = "AND payload->>$3 = $4"
        base_params  += [filter_key, filter_value]

    fetch_k = k * 4 if filter_key else k   # over-fetch before filter

    start = time.time()

    # Check if IVF centroids exist and index is large enough
    use_ivf = False
    async with pool.acquire() as conn:
        n_centroids: int = await conn.fetchval(
            f"SELECT COUNT(*) FROM {schema}.vector_centroids WHERE index_name=$1",
            index_name,
        ) or 0
        n_vectors: int = await conn.fetchval(
            f"SELECT COUNT(*) FROM {schema}.embedding_metadata WHERE index_name=$1",
            index_name,
        ) or 0

    use_ivf = n_centroids > 0 and n_vectors >= threshold

    if use_ivf:
        # ── Path A: IVF two-phase search ───────────────────────
        # Phase 1: find top n_probe cluster IDs via centroid dot-product
        # Phase 2: scan only vectors in those clusters
        p_idx = len(base_params) + 1   # next param slot
        ivf_sql = f"""
            WITH top_clusters AS (
                SELECT cluster_id
                FROM {schema}.vector_centroids
                WHERE index_name = $2
                ORDER BY (
                    SELECT COALESCE(sum(a * b), 0)
                    FROM unnest(centroid, $1::real[]) AS t(a, b)
                ) DESC
                LIMIT {n_probe}
            )
            SELECT m.id, m.payload,
                   (SELECT COALESCE(sum(a * b), 0)
                    FROM unnest(m.embedding, $1::real[]) AS t(a, b)) AS similarity
            FROM {schema}.embedding_metadata m
            JOIN top_clusters tc ON m.cluster_id = tc.cluster_id
            WHERE m.index_name = $2
              {filter_clause}
            ORDER BY similarity DESC
            LIMIT {fetch_k}
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(ivf_sql, *base_params)
    else:
        # ── Path B: Direct full-table scan (fallback for small indexes) ─
        flat_sql = f"""
            SELECT id, payload,
                   (SELECT COALESCE(sum(a * b), 0)
                    FROM unnest(embedding, $1::real[]) AS t(a, b)) AS similarity
            FROM {schema}.embedding_metadata
            WHERE index_name = $2
              {filter_clause}
            ORDER BY similarity DESC
            LIMIT {fetch_k}
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(flat_sql, *base_params)

    results = [(row["id"], float(row["similarity"])) for row in rows]
    elapsed = int((time.time() - start) * 1000)
    path    = "IVF" if use_ivf else "flat"
    logger.debug(
        f"Dense search '{index_name}' ({path}): {len(results)} results in {elapsed}ms"
    )
    return results[:k]


# ── Sparse Search (PostgreSQL tsvector — stateless) ──────────

async def sparse_search(
    index_name: str, query_text: str, k: int = 5
) -> list[tuple[int, float]]:
    """Full-text keyword search via PostgreSQL tsvector/tsquery.

    Returns [(metadata_id, ts_rank_score)] sorted descending.
    """
    if not query_text or not query_text.strip():
        return []

    settings = get_settings()
    pool     = get_pool()
    schema   = settings.APP_SCHEMA

    tokens = [t.strip() for t in query_text.lower().split() if t.strip()]
    if not tokens:
        return []

    start = time.time()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""SELECT id, ts_rank(tsv, to_tsquery('simple', $1)) AS rank
                FROM {schema}.embedding_metadata
                WHERE index_name = $2
                  AND tsv @@ to_tsquery('simple', $1)
                ORDER BY rank DESC
                LIMIT $3""",
            " | ".join(tokens), index_name, k,
        )

    results = [(row["id"], float(row["rank"])) for row in rows]
    elapsed = int((time.time() - start) * 1000)
    logger.debug(f"Sparse search '{index_name}': {len(results)} results in {elapsed}ms")
    return results


# ── Index Writes ──────────────────────────────────────────────

async def add_to_index(
    index_name: str,
    embedding: list[float],
    payload: dict,
    content: str = "",
) -> int:
    """Add one vector to PostgreSQL + assign it to the nearest IVF cluster.

    The cluster assignment is a single SQL query against vector_centroids
    (O(k) ≈ O(100 rows)) — negligible overhead.
    """
    settings = get_settings()
    pool     = get_pool()
    schema   = settings.APP_SCHEMA

    # L2-normalise (dot product on normalised vectors = cosine similarity)
    vec  = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    normalized = vec.tolist()

    # Assign cluster
    cluster_id = await _assign_cluster(pool, schema, index_name, normalized)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""INSERT INTO {schema}.embedding_metadata
                    (index_name, embedding, payload, content, cluster_id)
                VALUES ($1, $2, $3::jsonb, $4, $5)
                RETURNING id""",
            index_name, normalized, json.dumps(payload), content, cluster_id,
        )
    return row["id"]


async def batch_add_to_index(
    index_name: str,
    embeddings: list[list[float]],
    payloads: list[dict],
    contents: list[str] = None,
) -> list[int]:
    """Batch-insert vectors. Cluster assignment done in-process via numpy dot-product.

    Centroids are loaded once from the DB, then all assignments are computed
    in numpy — one round-trip regardless of batch size.
    """
    if not embeddings:
        return []

    settings = get_settings()
    pool     = get_pool()
    schema   = settings.APP_SCHEMA
    if contents is None:
        contents = [""] * len(embeddings)

    # Normalise
    arr   = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr  /= norms

    # Batch cluster assignment (one DB round-trip for centroids)
    cluster_ids = await _batch_assign_clusters(pool, schema, index_name, arr)

    # Bulk insert (single transaction)
    metadata_ids: list[int] = []
    async with pool.acquire() as conn:
        async with conn.transaction():
            for emb, payload, content, cid in zip(
                arr.tolist(), payloads, contents, cluster_ids
            ):
                row = await conn.fetchrow(
                    f"""INSERT INTO {schema}.embedding_metadata
                            (index_name, embedding, payload, content, cluster_id)
                        VALUES ($1, $2, $3::jsonb, $4, $5)
                        RETURNING id""",
                    index_name, emb, json.dumps(payload), content, cid,
                )
                metadata_ids.append(row["id"])

    logger.info(
        f"Batch added {len(metadata_ids)} vectors to '{index_name}' (PostgreSQL IVF)"
    )
    return metadata_ids


# ── Combined Search (with payload fetch) ─────────────────────

async def search_index(
    index_name: str,
    query_embedding: list[float],
    k: int = 5,
    filter_key: str = None,
    filter_value: str = None,
) -> list[dict]:
    """IVF search that returns full payload dicts.

    Returns [{node_id, similarity, payload}] sorted descending.
    """
    start_time = time.time()

    candidates = await dense_search(
        index_name, query_embedding,
        k * 3 if filter_key else k,
        filter_key, filter_value,
    )
    if not candidates:
        return []

    pool   = get_pool()
    schema = get_settings().APP_SCHEMA
    ids    = [c[0] for c in candidates]
    id_to_score = {c[0]: c[1] for c in candidates}

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT id, payload, content FROM {schema}.embedding_metadata "
            f"WHERE id = ANY($1::int[])",
            ids,
        )

    results = []
    for row in rows:
        payload = (
            row["payload"]
            if isinstance(row["payload"], dict)
            else json.loads(row["payload"])
        )
        results.append({
            "node_id":    str(row["id"]),
            "similarity": id_to_score.get(row["id"], 0.0),
            "payload":    payload,
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    elapsed = int((time.time() - start_time) * 1000)
    logger.debug(f"search_index '{index_name}': {len(results)} results in {elapsed}ms")
    return results[:k]


# ── Audit & Cache (stateless PostgreSQL — unchanged) ─────────

async def log_search(user_query: str, index_name: str, matched_ids: list[int],
                     similarity_scores: list[float] = None):
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {schema}.query_logs
                    (user_query, index_name, matched_ids, similarity_scores)
                    VALUES ($1, $2, $3, $4)""",
                user_query, index_name, matched_ids, similarity_scores or [],
            )
    except Exception as e:
        logger.debug(f"Query log write skipped: {e}")


async def get_cached_retrieval(query_hash: str) -> list[int] | None:
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""SELECT matched_ids FROM {schema}.retrieval_cache
                    WHERE query_hash = $1
                      AND created_at > now() - interval '{settings.RETRIEVAL_CACHE_TTL_MIN} minutes'""",
                query_hash,
            )
            return list(row["matched_ids"]) if row else None
    except Exception:
        return None


async def cache_retrieval(query_hash: str, matched_ids: list[int]):
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
                query_hash, matched_ids,
            )
    except Exception as e:
        logger.debug(f"Retrieval cache write skipped: {e}")


async def invalidate_retrieval_cache():
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    try:
        async with pool.acquire() as conn:
            deleted = await conn.execute(
                f"DELETE FROM {schema}.retrieval_cache"
            )
            logger.info(f"Retrieval cache invalidated: {deleted}")
    except Exception as e:
        logger.debug(f"Retrieval cache invalidation skipped: {e}")


async def cleanup_stale_cache():
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
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {schema}.retrieval_trace
                    (user_query, index_name, matched_nodes, similarity_scores,
                     response, latency_ms)
                    VALUES ($1, $2, $3::jsonb, $4, $5, $6)""",
                user_query, index_name, json.dumps(matched_nodes),
                similarity_scores or [], response, latency_ms,
            )
    except Exception as e:
        logger.debug(f"Retrieval trace write skipped: {e}")


async def clear_index(index_name: str):
    """Delete all vectors and centroids for an index from PostgreSQL."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        await conn.execute(
            f"DELETE FROM {schema}.embedding_metadata WHERE index_name=$1",
            index_name,
        )
        await conn.execute(
            f"DELETE FROM {schema}.vector_centroids WHERE index_name=$1",
            index_name,
        )
    logger.info(f"Index '{index_name}' cleared (PostgreSQL IVF)")
