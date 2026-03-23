"""Saved queries — CRUD for user-bookmarked queries."""
import json
import logging
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)


async def save_query(user_id: str, name: str, prompt: str,
                     sql: str = None, viz_config: dict = None, tags: list = None) -> str:
    """Save a query. Returns the saved query ID."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""INSERT INTO {schema}.saved_queries (user_id, name, original_prompt, generated_sql, viz_config, tags)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6)
                RETURNING id""",
            user_id, name, prompt, sql, json.dumps(viz_config) if viz_config else None, tags or []
        )
        return str(row["id"])


async def list_saved_queries(user_id: str) -> list[dict]:
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""SELECT id, name, original_prompt, generated_sql, tags, run_count, created_at, last_run_at
                FROM {schema}.saved_queries WHERE user_id=$1 ORDER BY created_at DESC""",
            user_id
        )
        return [dict(r) for r in rows]


async def delete_saved_query(query_id: str, user_id: str):
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        await conn.execute(
            f"DELETE FROM {schema}.saved_queries WHERE id=$1::uuid AND user_id=$2",
            query_id, user_id
        )


async def get_saved_query(query_id: str, user_id: str) -> dict | None:
    """Get a single saved query by ID."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT * FROM {schema}.saved_queries WHERE id=$1::uuid AND user_id=$2",
            query_id, user_id
        )
        return dict(row) if row else None


async def increment_run_count(query_id: str):
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE {schema}.saved_queries SET run_count=run_count+1, last_run_at=now() WHERE id=$1::uuid",
            query_id
        )
