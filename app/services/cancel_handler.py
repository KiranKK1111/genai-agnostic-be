"""Cancellation handler — cooperative cancellation for streaming responses."""
import logging
from app.services.session import SessionManager

logger = logging.getLogger(__name__)


async def is_cancelled(session_id: str) -> bool:
    """Check if the current session has a pending cancellation request."""
    sm = SessionManager()
    state = await sm.get_or_create(session_id)
    return state.get("cancel_requested", False)


async def clear_cancel(session_id: str):
    """Clear the cancellation flag after handling."""
    sm = SessionManager()
    state = await sm.get_or_create(session_id)
    state["cancel_requested"] = False
    await sm.save(state)


async def handle_cancel(session_id: str, partial_content: str, message_id: str) -> dict:
    """Process a cancellation — save partial content and return cancel event."""
    await clear_cancel(session_id)
    return {
        "type": "cancelled",
        "message_id": message_id,
        "partial_content": partial_content,
    }
