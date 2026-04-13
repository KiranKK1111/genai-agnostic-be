"""Autocomplete / suggestion service for the search/input bar."""
import logging
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)


async def get_suggestions(query: str, user_id: str = None, limit: int = 5) -> list[dict]:
    """Get autocomplete suggestions based on partial input.
    Only returns suggestions from the requesting user's own history."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    suggestions = []

    async with pool.acquire() as conn:
        # Search recent user prompts — scoped to this user's sessions only
        if user_id:
            rows = await conn.fetch(
                f"""SELECT DISTINCT cm.content, cm.session_id, cm.created_at
                    FROM {schema}.chat_messages cm
                    JOIN {schema}.chat_sessions cs ON cs.id = cm.session_id
                    WHERE cm.role='user' AND cm.content ILIKE $1
                      AND cm.is_active=true AND cs.user_id=$2
                    ORDER BY cm.created_at DESC LIMIT $3""",
                f"%{query}%", user_id, limit
            )
        else:
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

        # Search saved query names — scoped to this user
        if user_id:
            saved = await conn.fetch(
                f"""SELECT name, original_prompt
                    FROM {schema}.saved_queries
                    WHERE (name ILIKE $1 OR original_prompt ILIKE $1)
                      AND user_id=$2
                    ORDER BY run_count DESC LIMIT $3""",
                f"%{query}%", user_id, limit
            )
        else:
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
