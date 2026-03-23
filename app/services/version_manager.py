"""Message version management — for retry/regenerate tracking."""
import logging
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)


async def get_message_versions(message_id: str) -> list[dict]:
    """Get all versions of a message (for retry history)."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    async with pool.acquire() as conn:
        # Find the parent user message
        msg = await conn.fetchrow(
            f"SELECT * FROM {schema}.chat_messages WHERE id=$1::uuid", message_id
        )
        if not msg:
            return []

        # Get all assistant responses to the same parent
        parent_id = msg["parent_message_id"] if msg["role"] == "assistant" else msg["id"]
        versions = await conn.fetch(
            f"""SELECT id, content, content_sql, version, is_active, created_at
                FROM {schema}.chat_messages
                WHERE parent_message_id=$1::uuid AND role='assistant'
                ORDER BY version ASC""",
            str(parent_id)
        )
        return [dict(v) for v in versions]


async def get_next_version(parent_message_id: str) -> int:
    """Get the next version number for a message."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""SELECT COALESCE(MAX(version), 0) + 1 as next_version
                FROM {schema}.chat_messages WHERE parent_message_id=$1::uuid""",
            parent_message_id
        )
        return row["next_version"]
