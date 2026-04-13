"""Full-text search across messages and sessions."""
import logging
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)


async def search_conversations(query: str, user_id: str = None, limit: int = 20) -> list[dict]:
    """Search across messages owned by user_id. Returns matching messages with session context.
    user_id is required for data isolation — omitting it returns an empty list."""
    if not user_id:
        logger.warning("search_conversations called without user_id — returning empty to prevent data leak")
        return []

    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""SELECT m.id, m.session_id, m.content, m.role, m.created_at,
                       s.title as session_title
                FROM {schema}.chat_messages m
                JOIN {schema}.chat_sessions s ON m.session_id = s.id
                WHERE m.content ILIKE $1 AND s.user_id=$2 AND m.is_active=true
                ORDER BY m.created_at DESC LIMIT $3""",
            f"%{query}%", user_id, limit
        )
        return [dict(r) for r in rows]
