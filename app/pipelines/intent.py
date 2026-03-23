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
PRONOUN_SIGNALS = re.compile(r"\b(them|those|that|above|previous|it|these|this\s+data)\b", re.IGNORECASE)
GROUP_BY_SIGNALS = re.compile(
    r"\b(summarize|group|break\s*down|split|categorize|distribute)\b.+?\b(by|with\s+respect\s+to|per|for\s+each)\b"
    r"|\b(for|per)\s+each\b"
    r"|\bcount\s+(of\s+records?\s+)?(by|per|for\s+each)\b",
    re.IGNORECASE
)
DB_KEYWORDS = re.compile(
    r"\b(get|show|list|find|fetch|give|display|retrieve|select|count|how\s+many|total|average|sum|max|min|customers?|clients?|orders?|products?|records?|data|table|rows?)\b",
    re.IGNORECASE
)
COMPARE_KEYWORDS = re.compile(r"\b(compare|vs|versus|both|difference|match|cross)\b", re.IGNORECASE)


async def classify_intent(message: str, session_state: dict, has_file: bool = False) -> dict:
    """Classify user intent. Returns {intent, confidence, method}."""

    # Step 1: Clarification pending — but only if the message looks like a reply
    if session_state.get("clarification_pending"):
        pending_type = session_state["clarification_pending"].get("type", "")
        msg_lower = message.strip().lower()
        # Known clarification reply values (from button clicks or short typed answers)
        clar_replies = {
            "record_limit": {"limited", "all", "populate", "yes", "no"},
            "viz_type": {"table", "bar", "pie", "line", "bar graph", "pie chart", "line graph"},
            "axis_mode": {"on the fly", "specific", "auto"},
        }
        known_replies = clar_replies.get(pending_type, set())
        # Treat as clarification reply if: message matches known reply values,
        # or message is very short (likely a button click), or clarification_response was set
        is_reply = (
            msg_lower in known_replies
            or len(message.split()) <= 2
            or session_state.get("clarification_response")
        )
        if is_reply:
            return {"intent": "CLARIFICATION_REPLY", "confidence": 1.0, "method": "session_state"}
        else:
            # User ignored the clarification and asked something new — clear it
            logger.info(f"Clarification '{pending_type}' dismissed by new query: {message[:50]}")
            session_state["clarification_pending"] = None

    # Step 2: Greeting
    if GREETING_PATTERNS.match(message.strip()):
        return {"intent": "CHAT", "confidence": 1.0, "method": "greeting_regex"}

    # Step 3: File attached
    if has_file:
        return {"intent": "FILE_ANALYSIS", "confidence": 1.0, "method": "file_attached"}

    # Step 4: Viz keywords + last_data
    if VIZ_KEYWORDS.search(message) and session_state.get("last_data"):
        return {"intent": "VIZ_FOLLOW_UP", "confidence": 0.9, "method": "viz_keywords"}

    # Step 5: Topic break — message mentions a new entity different from last query
    # Must run BEFORE follow-up detection so "get me all buyers" isn't treated as a follow-up
    if DB_KEYWORDS.search(message) and session_state.get("last_table"):
        last_table = session_state["last_table"]
        msg_lower = message.lower()
        # Check if message references a different table/entity via synonyms or direct name
        from app.services.schema_inspector import DOMAIN_SYNONYMS
        mentioned_tables = set()
        for word in re.findall(r"\w+", msg_lower):
            if word in DOMAIN_SYNONYMS:
                mentioned_tables.add(DOMAIN_SYNONYMS[word])
            # Direct table name match (e.g. "customers", "accounts")
            if word == last_table or word + "s" == last_table or word == last_table.rstrip("s"):
                mentioned_tables.add(last_table)
        if mentioned_tables and last_table not in mentioned_tables:
            return {"intent": "DB_QUERY", "confidence": 0.85, "method": "topic_break"}
        # Also check by simple name mismatch (original logic)
        if last_table.lower() not in msg_lower:
            # Check if any DB keyword suggests a fresh query (get/show/list/find + entity)
            if re.search(r"\b(get|show|list|find|fetch|give|display|retrieve)\b.*\b(all|every)\b", msg_lower):
                return {"intent": "DB_QUERY", "confidence": 0.85, "method": "topic_break"}

    # Step 6a: GROUP BY / summarize signals + last_sql
    if GROUP_BY_SIGNALS.search(message) and session_state.get("last_sql"):
        return {"intent": "DB_FOLLOW_UP", "confidence": 0.9, "method": "group_by_signals"}

    # Step 6b: Pronoun signals + last_sql (follow-up refinements like "filter those by X")
    if PRONOUN_SIGNALS.search(message) and session_state.get("last_sql"):
        # Only treat as follow-up if the message does NOT also contain action verbs
        # suggesting a fresh query (e.g. "get me all those buyers" = new query)
        if not re.search(r"\b(get|show|list|find|fetch|give|display|retrieve)\s+(me\s+)?(all|every)\b", message, re.IGNORECASE):
            return {"intent": "DB_FOLLOW_UP", "confidence": 0.85, "method": "pronoun_signals"}

    # Step 7: Short additive filter + last_sql (e.g. "in AP", "status active")
    word_count = len(message.split())
    if word_count <= 6 and session_state.get("last_sql") and not GREETING_PATTERNS.match(message):
        # Very short messages without action verbs are likely follow-up filters
        if not re.search(r"\b(get|show|list|find|fetch|give|display|retrieve)\b", message, re.IGNORECASE):
            return {"intent": "DB_FOLLOW_UP", "confidence": 0.8, "method": "short_additive"}

    # Step 8: File context + no DB keywords
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
