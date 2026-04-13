"""Message action handlers — edit, retry, resend, copy content, like, dislike."""
import json
import logging
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)


async def react_to_message(message_id: str, reaction: str | None, comment: str = None, user_id: str = None) -> dict:
    """Like/dislike a message. Pass None to clear reaction (toggle).
    Returns {status, reaction}."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    async with pool.acquire() as conn:
        # Check current reaction (for toggle behavior)
        current = await conn.fetchrow(
            f"SELECT reaction FROM {schema}.chat_messages WHERE id=$1::uuid", message_id
        )
        if not current:
            raise ValueError("Message not found")

        # Toggle: same reaction again -> clear it
        final_reaction = None if current["reaction"] == reaction else reaction

        await conn.execute(
            f"UPDATE {schema}.chat_messages SET reaction=$1, reaction_comment=$2 WHERE id=$3::uuid",
            final_reaction, comment, message_id
        )

        # Also write to feedback table for admin analysis + online learning
        if final_reaction:
            try:
                msg_full = await conn.fetchrow(
                    f"""SELECT session_id, content_sql, metadata, created_at,
                               parent_message_id,
                               metadata->'intent' as intent
                        FROM {schema}.chat_messages WHERE id=$1::uuid""",
                    message_id
                )
                rating = 1 if final_reaction == "like" else -1

                # ── Enrich with training signal fields ──────────────────
                query_text = None
                resolved_table = None
                resolved_columns: list = []

                if msg_full:
                    # 1. Get the original user query from the parent message
                    if msg_full["parent_message_id"]:
                        parent = await conn.fetchrow(
                            f"SELECT content FROM {schema}.chat_messages WHERE id=$1::uuid",
                            str(msg_full["parent_message_id"]),
                        )
                        if parent:
                            query_text = parent["content"]

                    # 2. Extract resolved_table from the generated SQL (FROM clause)
                    sql = msg_full.get("content_sql") or ""
                    if sql:
                        import re as _re_tbl
                        m = _re_tbl.search(
                            r'FROM\s+\w+\.(\w+)', sql, _re_tbl.IGNORECASE
                        )
                        if m:
                            resolved_table = m.group(1)

                    # 3. Get resolved_columns from message metadata
                    meta = msg_full.get("metadata") or {}
                    if isinstance(meta, str):
                        import json as _jm
                        try:
                            meta = _jm.loads(meta)
                        except Exception:
                            meta = {}
                    if isinstance(meta, dict):
                        resolved_columns = (meta.get("data") or {}).get("columns", [])

                await conn.execute(
                    f"""INSERT INTO {schema}.feedback
                        (user_id, session_id, message_id, rating, feedback_text,
                         intent_at_time, generated_sql_at_time,
                         query_text, resolved_table, resolved_columns, plan_json,
                         trained_on)
                        VALUES ($1, $2, $3::uuid, $4, $5, $6, $7, $8, $9, $10, '{{}}', false)""",
                    user_id or "unknown",
                    str(msg_full["session_id"]) if msg_full else None,
                    message_id,
                    rating,
                    comment,
                    str(msg_full["intent"]) if msg_full and msg_full["intent"] else None,
                    msg_full["content_sql"] if msg_full else None,
                    query_text,
                    resolved_table,
                    resolved_columns or [],
                )
            except Exception as e:
                logger.debug(f"Feedback write skipped: {e}")

    return {"status": "ok", "reaction": final_reaction}


async def edit_user_message(message_id: str, new_content: str) -> dict:
    """Edit a user message: deactivate all downstream, prepare for re-processing.
    Returns {session_id, original_content, deactivated_count}."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    async with pool.acquire() as conn:
        # Get the message
        msg = await conn.fetchrow(
            f"SELECT * FROM {schema}.chat_messages WHERE id=$1::uuid", message_id
        )
        if not msg:
            raise ValueError("Message not found")
        if msg["role"] != "user":
            raise ValueError("Can only edit user messages")

        # Update the message content
        await conn.execute(
            f"UPDATE {schema}.chat_messages SET content=$1 WHERE id=$2::uuid",
            new_content, message_id
        )

        # Deactivate all messages after this one in the same session
        # Also clean up orphaned feedback for deactivated messages (Issue #13)
        deactivated_ids = await conn.fetch(
            f"""SELECT id FROM {schema}.chat_messages
                WHERE session_id=$1 AND created_at > $2 AND is_active=true""",
            msg["session_id"], msg["created_at"]
        )
        result = await conn.execute(
            f"""UPDATE {schema}.chat_messages SET is_active=false
                WHERE session_id=$1 AND created_at > $2 AND is_active=true""",
            msg["session_id"], msg["created_at"]
        )
        if deactivated_ids:
            ids = [str(r["id"]) for r in deactivated_ids]
            await conn.execute(
                f"DELETE FROM {schema}.feedback WHERE message_id = ANY($1::uuid[])",
                ids
            )

        # Count deactivated
        count_str = result.split()[-1] if result else "0"
        try:
            deactivated = int(count_str)
        except ValueError:
            deactivated = 0

        # Rollback session state using snapshot — selective update only.
        # Do NOT bulk-overwrite with state.update(snapshot): that would restore
        # transient flags (cancel_requested, viz_suggestion_pending) from the
        # snapshot, bypassing the safety resets added in get_or_create.
        if msg.get("session_snapshot"):
            from app.services.session import SessionManager
            sm = SessionManager()
            state = await sm.get_or_create(str(msg["session_id"]))
            snapshot = json.loads(msg["session_snapshot"]) if isinstance(msg["session_snapshot"], str) else msg["session_snapshot"]
            for key in (
                "last_sql", "last_plan", "last_data", "last_columns", "last_table",
                "last_intent", "intent_chain", "file_context",
                "history", "history_summary", "total_turns", "clarification_pending",
            ):
                if key in snapshot:
                    state[key] = snapshot[key]
            # Always reset transient flags regardless of snapshot content
            state["viz_suggestion_pending"] = False
            state["cancel_requested"] = False
            await sm.save(state)

    return {
        "session_id": str(msg["session_id"]),
        "original_content": msg["content"],
        "deactivated_count": deactivated,
    }


async def get_message_content(message_id: str, format: str = "raw") -> str:
    """Get message content in a specific format for copy."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    async with pool.acquire() as conn:
        msg = await conn.fetchrow(
            f"SELECT content, content_sql, metadata FROM {schema}.chat_messages WHERE id=$1::uuid",
            message_id
        )
        if not msg:
            raise ValueError("Message not found")

    if format == "sql" and msg["content_sql"]:
        return msg["content_sql"]
    elif format == "markdown":
        return msg["content"]
    elif format == "json":
        meta = msg["metadata"] if isinstance(msg["metadata"], dict) else json.loads(msg["metadata"] or "{}")
        return json.dumps({"content": msg["content"], "sql": msg["content_sql"], "metadata": meta}, indent=2, default=str)
    else:
        return msg["content"]
