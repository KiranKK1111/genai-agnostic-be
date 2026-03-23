"""User preferences persistence."""
import json
import logging
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)


async def get_preferences(user_id: str) -> dict:
    """Get user preferences."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT preferences FROM {schema}.users WHERE id=$1", user_id
        )
        if row and row["preferences"]:
            return row["preferences"] if isinstance(row["preferences"], dict) else json.loads(row["preferences"])
    return {}


async def update_preferences(user_id: str, prefs: dict):
    """Update user preferences (merge with existing)."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    current = await get_preferences(user_id)
    current.update(prefs)
    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE {schema}.users SET preferences=$1::jsonb WHERE id=$2",
            json.dumps(current), user_id
        )
