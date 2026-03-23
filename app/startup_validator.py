"""Startup validation — check PostgreSQL, Ollama, and required models."""
import httpx
import asyncpg
import logging
import sys
from app.config import get_settings

logger = logging.getLogger(__name__)


async def validate_startup():
    """Run all startup checks. Exits with clear error if anything fails."""
    settings = get_settings()

    # 1. PostgreSQL connectivity
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

    # 2. Ollama availability
    logger.info("Checking Ollama availability...")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            logger.info(f"  ✓ Ollama running. Available models: {', '.join(models[:5])}")
    except Exception as e:
        logger.error(f"  ✗ Cannot reach Ollama at http://localhost:11434")
        logger.error(f"    Error: {e}")
        logger.error(f"    Fix: Start Ollama with: ollama serve")
        sys.exit(1)

    # 3. Required models
    logger.info("Checking required models...")
    required = [settings.AI_FACTORY_MODEL.split(":")[0], settings.EMBEDDING_MODEL.split(":")[0]]
    available = [m.split(":")[0] for m in models]
    for model in required:
        if model not in available:
            logger.error(f"  ✗ Required model '{model}' not found")
            logger.error(f"    Fix: Run: ollama pull {model}")
            sys.exit(1)
        logger.info(f"  ✓ Model available: {model}")

    logger.info("All startup checks passed!")
