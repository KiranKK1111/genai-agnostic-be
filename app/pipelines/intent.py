"""Intent classification — 11-step deterministic + LLM fallback + dual-search probe."""
import re
import logging
from app.services.llm_client import chat_json
from app.config import get_settings

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

# Minimum similarity to consider a search hit "strong"
_SOURCE_PROBE_THRESHOLD = 0.35


async def _probe_sources(message: str, session_state: dict) -> dict:
    """Probe both file (chunks_idx) and DB (schema_idx) vectors for the query.

    Returns:
        {
            "file_score": float,   # best similarity in chunks_idx
            "db_score": float,     # best similarity in schema_idx
            "file_hits": int,      # results above threshold
            "db_hits": int,
        }
    """
    from app.services.embedder import embed_single
    from app.services.vector_search import search

    try:
        emb = await embed_single(message)
    except Exception as e:
        logger.debug(f"Dual probe embed failed: {e}")
        return {"file_score": 0.0, "db_score": 0.0, "file_hits": 0, "db_hits": 0}

    file_score, db_score = 0.0, 0.0
    file_hits, db_hits = 0, 0

    # Probe file chunks (filtered by session)
    has_file = bool(session_state.get("file_context"))
    if has_file:
        try:
            file_results = await search(
                "chunks_idx", emb, k=3,
                filter_key="session_id", filter_value=session_state["session_id"],
                query_text=message
            )
            if file_results:
                file_score = max(r.get("similarity", 0.0) for r in file_results)
                file_hits = sum(1 for r in file_results if r.get("similarity", 0) >= _SOURCE_PROBE_THRESHOLD)
        except Exception as e:
            logger.debug(f"File probe failed: {e}")

    # Probe schema index
    try:
        db_results = await search("schema_idx", emb, k=3, query_text=message)
        if db_results:
            db_score = max(r.get("similarity", 0.0) for r in db_results)
            db_hits = sum(1 for r in db_results if r.get("similarity", 0) >= _SOURCE_PROBE_THRESHOLD)
    except Exception as e:
        logger.debug(f"DB probe failed: {e}")

    return {
        "file_score": file_score, "db_score": db_score,
        "file_hits": file_hits, "db_hits": db_hits,
    }


def source_clarification(file_names: list[str]) -> dict:
    """Build a source clarification asking user whether they mean file(s) or DB."""
    if len(file_names) == 1:
        file_label = f"Uploaded file ({file_names[0]})"
        question = (
            f"Your question could relate to the uploaded file (**{file_names[0]}**) "
            f"or the connected database. Which source did you mean?"
        )
    else:
        names_str = ", ".join(f"**{n}**" for n in file_names)
        file_label = f"Uploaded files ({', '.join(file_names)})"
        question = (
            f"Your question could relate to your uploaded files ({names_str}) "
            f"or the connected database. Which source did you mean?"
        )
    return {
        "type": "source_clarification",
        "mode": "single_select",
        "support_for_custom_replies": True,
        "question": question,
        "options": [
            {"value": "file", "label": file_label, "icon": "\U0001f4c4"},
            {"value": "database", "label": "Connected database", "icon": "\U0001f5c3\ufe0f"},
        ],
    }


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

        # Also tell the LLM about file context so it can detect file follow-ups
        file_ctx = session_state.get("file_context")
        file_history = session_state.get("file_history", [])
        file_hint = ""
        if file_history:
            file_names = [f.get("file_name", "?") for f in file_history]
            file_hint = f"\nThe user has uploaded file(s): {', '.join(file_names)}\n"
        elif file_ctx:
            file_hint = f"\nThe user has an uploaded file: {file_ctx.get('file_name', 'unknown')}\n"

        # Include query history so the LLM knows ALL previous queries, not just the last
        sql_history = session_state.get("sql_history", [])
        query_history_hint = ""
        if sql_history:
            lines = [f"  Q{i+1}. \"{e.get('message', '')}\"" for i, e in enumerate(sql_history[-5:])]
            query_history_hint = "\nPrevious database queries:\n" + "\n".join(lines) + "\n"

        try:
            result = await chat_json([{"role": "user", "content":
                f'The user is interacting with a data query tool.\n'
                f'{conversation_trail}\n'
                f'{file_hint}'
                f'{query_history_hint}\n'
                f'New message: "{message}"\n\n'
                f'Determine if this new message is:\n'
                f'A) A DB_FOLLOW_UP — the user wants to operate on, refine, or explore ANY of the previous '
                f'database results. The message explicitly or implicitly references data already retrieved '
                f'(could be the most recent query OR an earlier one).\n\n'
                f'B) A FILE_FOLLOW_UP — the user wants to ask about or explore the uploaded file. '
                f'The message references the file content, the file itself, or topics from the file.\n\n'
                f'C) A NEW_QUERY — the user wants to start fresh and retrieve a completely new dataset '
                f'that has no relation to any previous query or file.\n\n'
                f'Think carefully: does the new message relate to ANY previous query context, '
                f'or is it entirely independent?\n'
                f'If it refines, narrows, or builds on ANY previous query, it is DB_FOLLOW_UP.\n'
                f'If it makes sense only as a standalone request with no relation to history, it is NEW_QUERY.\n\n'
                f'Return JSON: {{"type": "DB_FOLLOW_UP" or "FILE_FOLLOW_UP" or "NEW_QUERY", "reason": "brief explanation"}}'}])
            detected_type = result.get("type", "NEW_QUERY")
            logger.info(f"Follow-up check: type={detected_type}, reason={result.get('reason', '')}")
        except Exception:
            detected_type = "NEW_QUERY"

        if detected_type == "DB_FOLLOW_UP":
            return {"intent": "DB_FOLLOW_UP", "confidence": 0.9, "method": "llm_follow_up"}
        elif detected_type == "FILE_FOLLOW_UP" and file_ctx:
            return {"intent": "FILE_FOLLOW_UP", "confidence": 0.9, "method": "llm_file_follow_up"}
        # else: fall through to dual-search probe or new query logic

    # Step 6: Dual-search probe — when user has BOTH file context AND DB,
    # search both indexes and route based on scores. If both are strong,
    # return DUAL_SEARCH so the orchestrator can ask a source clarification.
    has_file_ctx = bool(session_state.get("file_context")) or bool(session_state.get("file_history"))
    if has_file_ctx:
        probe = await _probe_sources(message, session_state)
        logger.info(
            f"Dual probe: file_score={probe['file_score']:.3f} ({probe['file_hits']} hits), "
            f"db_score={probe['db_score']:.3f} ({probe['db_hits']} hits)"
        )

        both_strong = probe["file_hits"] > 0 and probe["db_hits"] > 0
        file_only = probe["file_hits"] > 0 and probe["db_hits"] == 0
        db_only = probe["db_hits"] > 0 and probe["file_hits"] == 0

        if both_strong:
            # Both sources have relevant results — ask user
            return {"intent": "DUAL_SEARCH", "confidence": 0.85, "method": "dual_probe",
                    "probe": probe}
        elif file_only:
            return {"intent": "FILE_FOLLOW_UP", "confidence": 0.8, "method": "dual_probe_file"}
        elif db_only:
            return {"intent": "DB_QUERY", "confidence": 0.8, "method": "dual_probe_db"}
        # Neither strong — fall through to keyword/LLM

    # Step 6b: File context + no DB keywords (legacy fallback when no dual probe)
    if has_file_ctx and not DB_KEYWORDS.search(message):
        return {"intent": "FILE_FOLLOW_UP", "confidence": 0.8, "method": "file_context"}

    # Step 9: File context(s) + compare keywords
    if has_file_ctx and session_state.get("last_data") and COMPARE_KEYWORDS.search(message):
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
