"""Embedding cache — L1 in-process LRU + L2 PostgreSQL table."""
import hashlib
import json
import logging
from functools import lru_cache
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)

# ── L1: In-process LRU cache (up to 1000 entries) ─────
_L1_CACHE: dict[str, list[float]] = {}


def _l1_max_size() -> int:
    from app.config import get_settings
    return get_settings().EMBED_CACHE_L1_MAX_SIZE


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _l1_get(text_hash: str) -> list[float] | None:
    """Check L1 in-memory cache."""
    return _L1_CACHE.get(text_hash)


def _l1_set(text_hash: str, embedding: list[float]):
    """Store in L1 in-memory cache with LRU eviction."""
    if len(_L1_CACHE) >= _l1_max_size():
        # Evict oldest entry (FIFO approximation)
        oldest_key = next(iter(_L1_CACHE))
        del _L1_CACHE[oldest_key]
    _L1_CACHE[text_hash] = embedding


# ── Public API ─────────────────────────────────────────

async def get_cached_embedding(text: str) -> list[float] | None:
    """Look up a cached embedding. Checks L1 (memory) first, then L2 (PostgreSQL)."""
    th = _text_hash(text)

    # L1: in-process check
    result = _l1_get(th)
    if result is not None:
        return result

    # L2: PostgreSQL check
    settings = get_settings()
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT embedding FROM {settings.APP_SCHEMA}.embedding_cache WHERE text_hash=$1",
            th
        )
        if row:
            embedding = list(row["embedding"])
            _l1_set(th, embedding)  # Promote to L1
            return embedding

    return None


async def cache_embedding(text: str, embedding: list[float]):
    """Store an embedding in both L1 (memory) and L2 (PostgreSQL)."""
    th = _text_hash(text)

    # L1: store in memory
    _l1_set(th, embedding)

    # L2: store in PostgreSQL
    settings = get_settings()
    pool = get_pool()
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {settings.APP_SCHEMA}.embedding_cache (text_hash, embedding, model)
                    VALUES ($1, $2, $3) ON CONFLICT (text_hash) DO NOTHING""",
                th, embedding, settings.EMBEDDING_MODEL
            )
    except Exception as e:
        logger.warning(f"L2 cache write failed: {e}")
