"""Redis-like KV store over PostgreSQL."""
import json
import logging
from datetime import datetime, timedelta, timezone
from app.database import get_pool

logger = logging.getLogger(__name__)

class KVStore:
    def __init__(self, schema: str = "genai_app"):
        self.schema = schema

    async def get(self, key: str) -> dict | None:
        pool = get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT value FROM {self.schema}.kv_store WHERE key=$1 AND (expires_at IS NULL OR expires_at > now())", key
            )
            if row:
                return json.loads(row["value"]) if isinstance(row["value"], str) else row["value"]
            return None

    async def set(self, key: str, value: dict, namespace: str = "default", ttl_seconds: int = None):
        pool = get_pool()
        expires = None
        if ttl_seconds:
            expires = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        async with pool.acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {self.schema}.kv_store (key, value, namespace, expires_at)
                    VALUES ($1, $2::jsonb, $3, $4)
                    ON CONFLICT (key) DO UPDATE SET value=$2::jsonb, namespace=$3, expires_at=$4""",
                key, json.dumps(value), namespace, expires
            )

    async def delete(self, key: str):
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self.schema}.kv_store WHERE key=$1", key)

    async def expire(self, key: str, ttl_seconds: int):
        """Set TTL on an existing key."""
        pool = get_pool()
        expires = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        async with pool.acquire() as conn:
            await conn.execute(
                f"UPDATE {self.schema}.kv_store SET expires_at=$1 WHERE key=$2",
                expires, key
            )

    async def keys(self, pattern: str = "*", namespace: str = None) -> list[str]:
        """List keys matching a pattern. Uses SQL LIKE (% for *, ? for _)."""
        pool = get_pool()
        sql_pattern = pattern.replace("*", "%").replace("?", "_")
        async with pool.acquire() as conn:
            if namespace:
                rows = await conn.fetch(
                    f"SELECT key FROM {self.schema}.kv_store WHERE key LIKE $1 AND namespace=$2 AND (expires_at IS NULL OR expires_at > now())",
                    sql_pattern, namespace
                )
            else:
                rows = await conn.fetch(
                    f"SELECT key FROM {self.schema}.kv_store WHERE key LIKE $1 AND (expires_at IS NULL OR expires_at > now())",
                    sql_pattern
                )
            return [r["key"] for r in rows]

    async def cleanup_expired(self):
        """Purge all expired entries."""
        pool = get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self.schema}.kv_store WHERE expires_at IS NOT NULL AND expires_at < now()"
            )
            logger.info(f"KV cleanup: {result}")
