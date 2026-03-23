"""Full-text search across messages and sessions."""
import logging
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)


async def search_conversations(query: str, user_id: str = None, limit: int = 20) -> list[dict]:
    """Search across all messages. Returns matching messages with session context."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    async with pool.acquire() as conn:
        if user_id:
            rows = await conn.fetch(
                f"""SELECT m.id, m.session_id, m.content, m.role, m.created_at,
                           s.title as session_title
                    FROM {schema}.chat_messages m
                    JOIN {schema}.chat_sessions s ON m.session_id = s.id
                    WHERE m.content ILIKE $1 AND s.user_id=$2 AND m.is_active=true
                    ORDER BY m.created_at DESC LIMIT $3""",
                f"%{query}%", user_id, limit
            )
        else:
            rows = await conn.fetch(
                f"""SELECT m.id, m.session_id, m.content, m.role, m.created_at,
                           s.title as session_title
                    FROM {schema}.chat_messages m
                    JOIN {schema}.chat_sessions s ON m.session_id = s.id
                    WHERE m.content ILIKE $1 AND m.is_active=true
                    ORDER BY m.created_at DESC LIMIT $2""",
                f"%{query}%", limit
            )
        return [dict(r) for r in rows]
