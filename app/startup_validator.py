"""Startup validation — check PostgreSQL, Ollama, config, and required models.

PostgreSQL is mandatory (exits on failure).
Ollama/LLM is optional at startup — the server starts in degraded mode and
chat endpoints return a clear error until the LLM becomes reachable.
"""
import httpx
import asyncpg
import logging
import sys
from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Runtime LLM availability flag ─────────────────────────
_llm_available: bool = False


def is_llm_available() -> bool:
    """Check whether the LLM backend was reachable at startup."""
    return _llm_available


def set_llm_available(value: bool):
    global _llm_available
    _llm_available = value


async def validate_startup():
    """Run all startup checks.
    PostgreSQL failure → hard exit (required).
    Ollama/LLM failure → warning only (server starts in degraded mode).
    """
    global _llm_available
    settings = get_settings()

    # 1. PostgreSQL connectivity (REQUIRED — exit on failure)
    logger.info("Checking PostgreSQL connectivity...")
    try:
        conn = await asyncpg.connect(dsn=settings.dsn, timeout=5)
        await conn.fetchval("SELECT 1")
        await conn.close()
        logger.info(f"  ✓ PostgreSQL connected: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}")
    except Exception as e:
        logger.error(f"  ✗ Cannot connect to PostgreSQL at {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")
        logger.error(f"    Error: {e}")
        logger.error(f"    Fix: Ensure PostgreSQL is running and check POSTGRES_HOST/PORT/USER/PASSWORD in .env")
        sys.exit(1)

    # 2. Ollama availability (OPTIONAL — warn and continue)
    logger.info("Checking Ollama availability...")
    ollama_url = settings.ollama_base_url
    models: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            logger.info(f"  ✓ Ollama running at {ollama_url}. Available models: {', '.join(models[:5])}")
    except Exception as e:
        logger.warning(f"  ⚠ Cannot reach Ollama at {ollama_url}")
        logger.warning(f"    Error: {e}")
        logger.warning(f"    LLM chat will be unavailable until Ollama is started: ollama serve")
        logger.warning(f"    Embeddings are unaffected (local SentenceEncoder).")
        _llm_available = False
        # Skip model check — no Ollama means no model list
        logger.info(f"  ✓ Embeddings: local SentenceEncoder ({settings.EMBEDDING_DIMENSIONS}d) — no external model")
        return  # Continue startup without LLM

    # 3. Required LLM model (warn if missing, don't exit)
    logger.info("Checking required models...")
    llm_model = settings.AI_FACTORY_MODEL.split(":")[0]
    available = [m.split(":")[0] for m in models]
    if llm_model not in available:
        logger.warning(f"  ⚠ Required model '{llm_model}' not found in Ollama")
        logger.warning(f"    Fix: Run: ollama pull {settings.AI_FACTORY_MODEL}")
        logger.warning(f"    LLM chat will fail until the model is pulled.")
        _llm_available = False
    else:
        logger.info(f"  ✓ LLM model available: {llm_model}")
        _llm_available = True
    logger.info(f"  ✓ Embeddings: local SentenceEncoder ({settings.EMBEDDING_DIMENSIONS}d) — no external model")

    # 4. Configuration validation (Issue #10, #15, #16, #17)
    logger.info("Validating configuration...")
    warnings = []

    # JWT secret must not be default in production
    if settings.AUTH_ENABLED and settings.JWT_SECRET == "local-dev-secret-change-in-production":
        warnings.append("JWT_SECRET is the default value — change it for production!")

    # CORS wildcard check
    if settings.CORS_ORIGINS == "*" and settings.AUTH_ENABLED:
        warnings.append("CORS_ORIGINS=* with AUTH_ENABLED=true is insecure — restrict to specific origins")

    # Embedding dimensions sanity check
    if settings.EMBEDDING_DIMENSIONS not in (128, 256, 384, 512, 768, 1024, 2048, 4096):
        warnings.append(f"EMBEDDING_DIMENSIONS={settings.EMBEDDING_DIMENSIONS} is unusual — common values: 384, 768")

    # Pool size check
    if settings.DB_POOL_MAX < 2:
        warnings.append(f"DB_POOL_MAX={settings.DB_POOL_MAX} is very low — may cause connection starvation")

    for w in warnings:
        logger.warning(f"  ⚠ {w}")

    if not warnings:
        logger.info("  ✓ Configuration validated")

    logger.info("All startup checks passed!")
