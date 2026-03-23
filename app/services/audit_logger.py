"""Async audit log writer."""
import logging
import json
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)


async def log_query(user_id: str = None, session_id: str = None, user_prompt: str = None,
                    intent: str = None, sql: str = None, plan: dict = None,
                    execution_ms: int = None, row_count: int = None,
                    status: str = "SUCCESS", error: str = None, tokens: int = 0,
                    quality_score: float = None, injection_flags: list = None):
    """Write an entry to the audit log."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {schema}.audit_log
                    (user_id, session_id, user_prompt, classified_intent, generated_sql,
                     query_plan_json, execution_time_ms, row_count, status, error_message,
                     llm_tokens_used, input_quality_score, injection_flags)
                    VALUES ($1, $2::uuid, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11, $12, $13)""",
                user_id, session_id, user_prompt, intent, sql,
                json.dumps(plan) if plan else None, execution_ms, row_count,
                status, error, tokens, quality_score, injection_flags
            )
    except Exception as e:
        logger.error(f"Audit log write failed: {e}")
