"""Session management built on KV store."""
import json
import uuid
import logging
from datetime import datetime, timezone
from app.services.kv_store import KVStore
from app.config import get_settings
from app.database import get_pool

logger = logging.getLogger(__name__)

def new_session_state(session_id: str, user_id: str = None) -> dict:
    settings = get_settings()
    ttl_hours = settings.SESSION_TTL_HOURS
    expires = datetime.now(timezone.utc).replace(microsecond=0) + __import__("datetime").timedelta(hours=ttl_hours)
    return {
        "session_id": session_id,
        "user_id": user_id,
        "title": "New Chat",
        "history": [],
        "history_summary": "",
        "last_sql": None,
        "last_plan": None,
        "last_data": None,
        "last_columns": None,
        "last_table": None,
        "last_intent": None,
        "intent_chain": [],
        "file_context": None,
        "clarification_pending": None,
        "clarification_history": [],   # Accumulated Q&A pairs across chained clarifications
        "cancel_requested": False,
        "expires_at": expires.isoformat(),
        "total_turns": 0,
    }

class SessionManager:
    def __init__(self):
        self.kv = KVStore()
        self.settings = get_settings()

    async def get_or_create(self, session_id: str = None, user_id: str = None) -> dict:
        if session_id:
            state = await self.kv.get(f"session:{session_id}")
            if state:
                return state
        # Create new
        sid = session_id or str(uuid.uuid4())
        state = new_session_state(sid, user_id)
        await self.save(state)
        # Also create in chat_sessions table
        pool = get_pool()
        schema = self.settings.APP_SCHEMA
        async with pool.acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {schema}.chat_sessions (id, user_id, title)
                    VALUES ($1::uuid, $2, $3) ON CONFLICT DO NOTHING""",
                sid, user_id, "New Chat"
            )
        return state

    async def save(self, state: dict):
        ttl = self.settings.SESSION_TTL_HOURS * 3600
        # Reset expires_at to +TTL on every save (per spec: reset on every message)
        import datetime as _dt
        state["expires_at"] = (datetime.now(timezone.utc) + _dt.timedelta(hours=self.settings.SESSION_TTL_HOURS)).isoformat()
        # Increment version counter to detect concurrent modification (Issue #4)
        state["_version"] = state.get("_version", 0) + 1
        await self.kv.set(f"session:{state['session_id']}", state, namespace="session", ttl_seconds=ttl)

    async def append_history(self, state: dict, role: str, content: str):
        state["history"].append({"role": role, "content": content})
        # Keep last 20 turns
        if len(state["history"]) > 20:
            state["history"] = state["history"][-20:]
        state["total_turns"] += 1

    async def update_db_context(self, state: dict, sql: str, plan: dict, data: list, columns: list, table: str):
        state["last_sql"] = sql
        state["last_plan"] = plan
        state["last_data"] = data[:200] if data else None
        state["last_columns"] = columns
        state["last_table"] = table

    async def save_message(self, session_id: str, role: str, content: str, **kwargs) -> str:
        pool = get_pool()
        schema = self.settings.APP_SCHEMA
        msg_id = str(uuid.uuid4())
        metadata = kwargs.get("metadata", {})
        follow_ups = kwargs.get("follow_ups", [])
        content_sql = kwargs.get("content_sql")
        parent_id = kwargs.get("parent_message_id")
        snapshot = kwargs.get("session_snapshot")

        async with pool.acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {schema}.chat_messages
                    (id, session_id, role, content, content_sql, metadata, follow_ups, parent_message_id, session_snapshot)
                    VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6::jsonb, $7::jsonb, $8, $9::jsonb)""",
                msg_id, session_id, role, content, content_sql,
                json.dumps(metadata), json.dumps(follow_ups),
                parent_id, json.dumps(snapshot) if snapshot else None
            )
        return msg_id

    async def get_session_messages(self, session_id: str) -> list[dict]:
        pool = get_pool()
        schema = self.settings.APP_SCHEMA
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT id, role, content, content_sql, metadata, follow_ups, reaction,
                           version, parent_message_id, is_active, created_at
                    FROM {schema}.chat_messages
                    WHERE session_id=$1::uuid AND is_active=true
                    ORDER BY created_at ASC""",
                session_id
            )
            messages = []
            for r in rows:
                msg = dict(r)
                # Reconstruct a full response object from persisted metadata
                # so the frontend can render tables, charts, and clarifications on reload
                if msg["role"] == "assistant":
                    meta = msg.get("metadata") or {}
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    response = {
                        "message": msg["content"],
                        "sql": msg.get("content_sql"),
                    }
                    # Reconstruct data payload
                    data_meta = meta.get("data")
                    if data_meta and data_meta.get("rows"):
                        response["data"] = {
                            "columns": data_meta.get("columns", []),
                            "rows": data_meta["rows"],
                            "row_count": data_meta.get("row_count", len(data_meta["rows"])),
                            "truncated": data_meta.get("truncated", False),
                        }
                    # Reconstruct visualizations
                    viz_meta = meta.get("viz_config")
                    if data_meta and data_meta.get("rows"):
                        viz_type = (viz_meta or {}).get("viz_type", "table")
                        available_views = (viz_meta or {}).get("available_views", ["table", "bar", "pie", "line"])
                        response["visualizations"] = [{
                            "chart_id": "auto-viz",
                            "type": viz_type,
                            "title": "Data",
                            "data": data_meta["rows"],
                            "config": {
                                "available_views": available_views,
                                "primary_view": viz_type,
                            },
                        }]
                    # Reconstruct clarification
                    clar_meta = meta.get("clarification")
                    if clar_meta:
                        response["clarifying_question"] = clar_meta
                    # Reconstruct clarification Q&A pairs
                    qa_meta = meta.get("clarification_qa")
                    if qa_meta:
                        response["clarification_qa"] = qa_meta
                    msg["response"] = response
                messages.append(msg)
            return messages
