"""Autocomplete / suggestion service for the search/input bar."""
import logging
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)


async def get_suggestions(query: str, limit: int = 5) -> list[dict]:
    """Get autocomplete suggestions based on partial input."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    suggestions = []

    async with pool.acquire() as conn:
        # Search recent user prompts
        rows = await conn.fetch(
            f"""SELECT DISTINCT content, session_id, created_at
                FROM {schema}.chat_messages
                WHERE role='user' AND content ILIKE $1 AND is_active=true
                ORDER BY created_at DESC LIMIT $2""",
            f"%{query}%", limit
        )
        for r in rows:
            suggestions.append({
                "text": r["content"][:100],
                "type": "history",
                "session_id": str(r["session_id"]),
            })

        # Search saved query names
        saved = await conn.fetch(
            f"""SELECT name, original_prompt
                FROM {schema}.saved_queries
                WHERE name ILIKE $1 OR original_prompt ILIKE $1
                ORDER BY run_count DESC LIMIT $2""",
            f"%{query}%", limit
        )
        for r in saved:
            suggestions.append({
                "text": r["name"],
                "type": "saved_query",
                "prompt": r["original_prompt"],
            })

    return suggestions[:limit]
