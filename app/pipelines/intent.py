"""Intent classification — 11-step deterministic + LLM fallback."""
import re
import logging
from app.services.llm_client import chat_json

logger = logging.getLogger(__name__)

GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy|greetings|what'?s\s+up|sup)\b",
    re.IGNORECASE
)
VIZ_KEYWORDS = re.compile(
    r"\b(bar\s*(chart|graph)|pie\s*(chart)?|line\s*(chart|graph)|chart|graph|visuali[sz]e|plot|heatmap|table|show\s+as)\b",
    re.IGNORECASE
)
DB_KEYWORDS = re.compile(
    r"\b(get|show|list|find|fetch|give|display|retrieve|select|count|how\s+many|total|average|sum|max|min|customers?|clients?|orders?|products?|records?|data|table|rows?)\b",
    re.IGNORECASE
)
COMPARE_KEYWORDS = re.compile(r"\b(compare|vs|versus|both|difference|match|cross)\b", re.IGNORECASE)


async def classify_intent(message: str, session_state: dict, has_file: bool = False) -> dict:
    """Classify user intent. Returns {intent, confidence, method}."""

    # Step 1: Clarification pending — treat as reply UNLESS the message is clearly a new query
    if session_state.get("clarification_pending"):
        pending_type = session_state["clarification_pending"].get("type", "")
        msg_lower = message.strip().lower()

        # Fast path: if /clarify was already called, clarification_response is set.
        # No need for LLM — the user explicitly answered via the UI.
        if session_state.get("clarification_response"):
            return {"intent": "CLARIFICATION_REPLY", "confidence": 1.0, "method": "clarify_endpoint"}

        # Use LLM to dynamically determine if the user's message is answering the
        # pending clarification or is an entirely new/unrelated query.
        # This only runs when the user typed a free-text reply in the chat input
        # instead of using the clarification popup.
        display = session_state.get("clarification_display") or {}
        question_text = display.get("question") or session_state["clarification_pending"].get("question", "")
        options = display.get("options") or session_state["clarification_pending"].get("options", [])
        options_str = ", ".join(
            o.get("label", o.get("value", "")) for o in options
        ) if options else "free-text response"

        try:
            result = await chat_json([{"role": "user", "content":
                f'A system asked the user this clarification question:\n'
                f'  Question: "{question_text}"\n'
                f'  Options: [{options_str}]\n\n'
                f'The user then replied: "{message}"\n\n'
                f'Is the user\'s reply an answer to the clarification question above, '
                f'or is it a completely new/unrelated query that ignores the question?\n\n'
                f'Return JSON: {{"is_reply": true/false}}\n'
                f'- true = the message answers or relates to the clarification (even indirectly)\n'
                f'- false = the message is a brand new query unrelated to the question'}])
            is_reply = result.get("is_reply", True)
        except Exception:
            # On LLM failure, default to treating as a clarification reply (safer)
            is_reply = True

        if is_reply:
            return {"intent": "CLARIFICATION_REPLY", "confidence": 1.0, "method": "llm_contextual"}
        else:
            # User ignored the clarification and asked something new — clear it
            logger.info(f"Clarification '{pending_type}' dismissed by new query: {message[:50]}")
            session_state["clarification_pending"] = None
            session_state["clarification_display"] = None
            session_state["clarification_history"] = []

    # Step 2: File attached — always prioritize file analysis (like ChatGPT behavior).
    # Even if the message is just "Hi", the user attached a file so they want it analyzed.
    if has_file:
        return {"intent": "FILE_ANALYSIS", "confidence": 1.0, "method": "file_attached"}

    # Step 3: Greeting (only when no file is attached)
    if GREETING_PATTERNS.match(message.strip()):
        return {"intent": "CHAT", "confidence": 1.0, "method": "greeting_regex"}

    # Step 4: Viz keywords + last_data
    if VIZ_KEYWORDS.search(message) and session_state.get("last_data"):
        return {"intent": "VIZ_FOLLOW_UP", "confidence": 0.9, "method": "viz_keywords"}

    # Step 5: Follow-up vs new query — use LLM to decide dynamically.
    # When the user has an active query context (last_sql/last_table), determine if the
    # new message is a follow-up to that context or an entirely new, independent query.
    if session_state.get("last_sql"):
        # Build recent conversation context (last few user messages)
        history = session_state.get("history", [])
        recent_user_msgs = []
        for h in reversed(history):
            if h.get("role") == "user" and h.get("content"):
                recent_user_msgs.insert(0, h["content"])
                if len(recent_user_msgs) >= 3:
                    break

        conversation_trail = ""
        if recent_user_msgs:
            numbered = [f"  {i+1}. \"{m}\"" for i, m in enumerate(recent_user_msgs)]
            conversation_trail = "Recent conversation (oldest to newest):\n" + "\n".join(numbered)

        try:
            result = await chat_json([{"role": "user", "content":
                f'The user is interacting with a data query tool.\n'
                f'{conversation_trail}\n\n'
                f'New message: "{message}"\n\n'
                f'Determine if this new message is:\n'
                f'A) A FOLLOW-UP — the user wants to operate on, refine, or explore the previous '
                f'results further. The message explicitly or implicitly references the data already retrieved.\n\n'
                f'B) A NEW QUERY — the user wants to start fresh and retrieve a new dataset '
                f'independently. The message does not depend on or reference any previous results.\n\n'
                f'Think carefully: does the new message make sense on its own as a standalone request, '
                f'or does it only make sense in the context of the previous conversation?\n'
                f'If it makes sense as a standalone request, it is a NEW QUERY.\n\n'
                f'Return JSON: {{"is_follow_up": true/false, "reason": "brief explanation"}}'}])
            is_follow_up = result.get("is_follow_up", False)
            logger.info(f"Follow-up check: is_follow_up={is_follow_up}, reason={result.get('reason', '')}")
        except Exception:
            is_follow_up = False

        if is_follow_up:
            return {"intent": "DB_FOLLOW_UP", "confidence": 0.9, "method": "llm_follow_up"}
        else:
            return {"intent": "DB_QUERY", "confidence": 0.85, "method": "llm_new_query"}

    # Step 6: File context + no DB keywords
    if session_state.get("file_context") and not DB_KEYWORDS.search(message):
        return {"intent": "FILE_FOLLOW_UP", "confidence": 0.8, "method": "file_context"}

    # Step 9: File context + compare keywords
    if session_state.get("file_context") and session_state.get("last_data") and COMPARE_KEYWORDS.search(message):
        return {"intent": "HYBRID", "confidence": 0.85, "method": "compare_keywords"}

    # Step 10: DB keywords
    if DB_KEYWORDS.search(message):
        return {"intent": "DB_QUERY", "confidence": 0.8, "method": "db_keywords"}

    # Step 11: LLM fallback
    try:
        result = await chat_json([{"role": "user", "content": f"""Classify this message into one of these intents:
- CHAT (greeting, small talk, capability question)
- DB_QUERY (asking about data, records, tables)
- FILE_ANALYSIS (about an uploaded file)
- DB_FOLLOW_UP (follow-up on previous query)

Message: "{message}"
Previous intent: {session_state.get("last_intent", "none")}
Has file context: {bool(session_state.get("file_context"))}
Has previous SQL: {bool(session_state.get("last_sql"))}

Return JSON: {{"intent": "...", "confidence": 0.0-1.0}}"""}])
        return {
            "intent": result.get("intent", "CHAT"),
            "confidence": result.get("confidence", 0.5),
            "method": "llm_fallback"
        }
    except Exception:
        return {"intent": "CHAT", "confidence": 0.5, "method": "llm_fallback_error"}
