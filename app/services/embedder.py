"""Embedding client — batch embed via Ollama with concurrent requests."""
import asyncio
import httpx
import logging
from app.config import get_settings

logger = logging.getLogger(__name__)

# Reusable HTTP client (created once, reused across all calls)
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=30)
    return _client


async def _embed_one(client: httpx.AsyncClient, text: str, settings) -> list[float]:
    """Embed a single text using an existing client connection."""
    try:
        resp = await client.post(
            settings.EMBEDDINGS_API,
            json={"model": settings.EMBEDDING_MODEL, "prompt": text}
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception as e:
        logger.error(f"Embedding failed: {text[:50]}... Error: {e}")
        return [0.0] * settings.EMBEDDING_DIMENSIONS


async def embed_texts(texts: list[str], concurrency: int = 8) -> list[list[float]]:
    """Embed a batch of texts with concurrent requests.

    Uses a semaphore to limit concurrent Ollama calls (default: 8).
    Reuses a single HTTP client for connection pooling.
    """
    if not texts:
        return []

    settings = get_settings()
    client = _get_client()
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded_embed(text: str) -> list[float]:
        async with semaphore:
            return await _embed_one(client, text, settings)

    # Fire all requests concurrently (bounded by semaphore)
    results = await asyncio.gather(*[_bounded_embed(t) for t in texts])
    return list(results)


async def embed_single(text: str) -> list[float]:
    """Embed a single text. Returns one embedding vector."""
    settings = get_settings()
    client = _get_client()
    return await _embed_one(client, text, settings)
