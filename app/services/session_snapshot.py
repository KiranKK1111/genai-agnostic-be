"""Session snapshot management — capture and restore full session state."""
import json
import logging
from app.services.session import SessionManager

logger = logging.getLogger(__name__)

# All session state fields that should be captured for rollback
SNAPSHOT_FIELDS = [
    "last_sql",
    "last_plan",
    "last_data",
    "last_columns",
    "last_table",
    "last_intent",
    "intent_chain",
    "file_context",
    "clarification_pending",
    "history_summary",
    "total_turns",
]


async def capture_snapshot(session_id: str) -> dict:
    """Capture current session state for rollback purposes.
    Captures all fields needed to fully restore session context."""
    sm = SessionManager()
    state = await sm.get_or_create(session_id)
    snapshot = {}
    for field in SNAPSHOT_FIELDS:
        snapshot[field] = state.get(field)
    return snapshot


async def restore_snapshot(session_id: str, snapshot: dict):
    """Restore session state from a snapshot."""
    sm = SessionManager()
    state = await sm.get_or_create(session_id)
    for key, value in snapshot.items():
        if key in state:
            state[key] = value
    await sm.save(state)
    logger.info(f"Session {session_id} restored to snapshot")
