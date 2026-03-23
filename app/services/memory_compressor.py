"""Compress conversation history when it exceeds 10 turns."""
import logging
from app.services.llm_client import chat

logger = logging.getLogger(__name__)

COMPRESS_THRESHOLD = 10  # Compress when history exceeds this many turns


async def maybe_compress(session_state: dict) -> dict:
    """Compress old history turns into a summary if threshold exceeded.
    Returns the (potentially modified) session state."""
    history = session_state.get("history", [])
    if len(history) <= COMPRESS_THRESHOLD:
        return session_state

    # Compress turns 0..(N-5) into a summary, keep last 5 verbatim
    old_turns = history[:-5]
    recent_turns = history[-5:]

    old_text = "\n".join(f"{t['role']}: {t['content'][:200]}" for t in old_turns)

    try:
        summary = await chat(
            [{"role": "user", "content": f"Summarize this conversation history in 3-4 sentences. Focus on what data was queried, what tables/filters were used, and key results:\n\n{old_text}"}],
            temperature=0.2
        )
        session_state["history_summary"] = summary.strip()
        session_state["history"] = recent_turns
        logger.info(f"Compressed {len(old_turns)} turns into summary ({len(summary)} chars)")
    except Exception as e:
        logger.warning(f"History compression failed: {e}")

    return session_state
