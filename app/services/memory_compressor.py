"""Compress conversation history when it exceeds 10 turns."""
import logging
from app.services.llm_client import chat

logger = logging.getLogger(__name__)

async def maybe_compress(session_state: dict) -> dict:
    """Compress old history turns into a summary if threshold exceeded.
    Returns the (potentially modified) session state."""
    from app.config import get_settings
    compress_threshold = getattr(get_settings(), "HISTORY_COMPRESS_THRESHOLD", 10)
    history = session_state.get("history", [])
    if len(history) <= compress_threshold:
        return session_state

    # Compress turns 0..(N-5) into a summary, keep last 5 verbatim
    old_turns = history[:-5]
    recent_turns = history[-5:]

    old_text = "\n".join(f"{t['role']}: {t['content'][:200]}" for t in old_turns)

    try:
        summary = await chat(
            [{"role": "user", "content":
                f"Summarize this conversation history in 3-4 sentences.\n"
                f"Focus on: what the user asked for, what data was returned (counts, categories, key numbers), "
                f"any filters applied, and what follow-up actions were taken.\n"
                f"Write in plain language — do NOT use heading labels, bullet lists, or markdown tables.\n\n"
                f"{old_text}"}],
            system="You are a concise conversation summarizer for a data analytics assistant. "
                   "Produce plain-text summaries that capture user intent and key data findings.",
            temperature=0.2
        )
        session_state["history_summary"] = summary.strip()
        session_state["history"] = recent_turns
        logger.info(f"Compressed {len(old_turns)} turns into summary ({len(summary)} chars)")
    except Exception as e:
        logger.warning(f"History compression failed: {e}")

    return session_state
