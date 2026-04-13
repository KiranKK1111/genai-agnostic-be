"""Intent classification — unified LLM-driven classification."""
import logging
from app.services.llm_client import chat_json
from app.config import get_settings

logger = logging.getLogger(__name__)


# No hardcoded keyword regexes — all classification is LLM-driven.

# Minimum similarity to consider a search hit "strong"
_SOURCE_PROBE_THRESHOLD = 0.35


def _build_session_context(session_state: dict) -> str:
    """Build a comprehensive, structured session context for LLM intent classification.

    Includes:
    - Compressed summary of older turns (if memory compressor has run)
    - Full recent conversation — BOTH user and assistant turns
    - Every SQL query run this session with tables and columns returned
    - What is currently on screen (columns, row count)
    - Previous intent and viz state
    """
    parts: list[str] = []

    # 1. Compressed summary of older turns
    history_summary = (session_state.get("history_summary") or "").strip()
    if history_summary:
        parts.append(f"[Older context summary: {history_summary[:500]}]")

    # 2. Full conversation history — user AND assistant turns
    history = session_state.get("history", [])
    if history:
        parts.append("\nFull conversation so far:")
        for turn in history:
            role = (turn.get("role") or "").strip()
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            # Truncate long assistant messages but keep enough for context
            display = content[:300] + "…" if len(content) > 300 else content
            label = "User" if role == "user" else "Assistant"
            parts.append(f"  {label}: \"{display}\"")

    # 3. All SQL queries run this session
    sql_history = session_state.get("sql_history", [])
    if sql_history:
        parts.append("\nDatabase queries this session:")
        for i, entry in enumerate(sql_history):
            msg = (entry.get("message") or "").strip()
            tables = ", ".join(entry.get("tables", []))
            cols = entry.get("columns", [])
            col_str = ", ".join(c.replace("_", " ").title() for c in cols[:5])
            intent_label = entry.get("intent", "")
            line = f"  Q{i + 1}. \"{msg}\""
            if intent_label:
                line += f"  [intent: {intent_label}]"
            if tables:
                line += f"  [tables: {tables}]"
            if col_str:
                line += f"  [columns returned: {col_str}]"
            parts.append(line)

    # 4. Current screen state
    last_cols = session_state.get("last_columns", [])
    if last_cols:
        friendly = [c.replace("_", " ").title() for c in last_cols[:6]]
        row_count = session_state.get("last_row_count", "?")
        parts.append(f"\nCurrently displayed: {', '.join(friendly)} ({row_count} rows)")

    # 5. Uploaded files
    file_history = session_state.get("file_history", [])
    file_ctx = session_state.get("file_context")
    if file_history:
        names = [f.get("file_name", "?") for f in file_history]
        parts.append(f"\nUploaded file(s): {', '.join(names)}")
    elif file_ctx:
        parts.append(f"\nUploaded file: {file_ctx.get('file_name', 'unknown')}")

    # 6. Previous intent and viz state
    prev_intent = session_state.get("last_intent", "")
    if prev_intent:
        parts.append(f"\nPrevious intent: {prev_intent}")
    if session_state.get("viz_suggestion_pending"):
        parts.append("System just asked the user if they want to visualize the data.")

    return "\n".join(parts)


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

    # ── Unified LLM classification — zero hardcoded keywords ─────────────
    # One LLM call handles ALL intent classification: greetings, viz requests,
    # follow-ups, new queries, file analysis, hybrid comparisons, chat.
    # The LLM also extracts viz details (types, colors) in the same pass
    # so the orchestrator can render directly without a second LLM call.

    session_ctx = _build_session_context(session_state)
    has_file_ctx = bool(session_state.get("file_context")) or bool(session_state.get("file_history"))
    has_data_on_screen = bool(session_state.get("last_data"))
    has_prior_query = bool(session_state.get("last_sql"))

    # Derive available viz types dynamically
    available_viz_types = []
    if has_data_on_screen:
        from app.builders.viz_config import viz_type_clarification
        _available = viz_type_clarification()
        available_viz_types = [o["value"] for o in _available.get("options", [])]

    # Dual-search probe — provides vector similarity signals for the LLM
    probe_info = ""
    if has_file_ctx:
        probe = await _probe_sources(message, session_state)
        logger.info(
            f"Dual probe: file_score={probe['file_score']:.3f} ({probe['file_hits']} hits), "
            f"db_score={probe['db_score']:.3f} ({probe['db_hits']} hits)"
        )
        probe_info = (
            f"\nVector similarity probe results:\n"
            f"  File index: score={probe['file_score']:.3f}, hits={probe['file_hits']}\n"
            f"  DB index: score={probe['db_score']:.3f}, hits={probe['db_hits']}\n"
        )

    # Build the unified classification prompt
    context_flags = []
    if has_data_on_screen:
        context_flags.append(f"Data is currently displayed on screen. Available viz types: {available_viz_types}")
    if has_file_ctx:
        context_flags.append("User has uploaded file(s) in this session.")
    if has_prior_query:
        context_flags.append("User has run previous database queries this session.")
    context_flags_str = "\n".join(context_flags) if context_flags else "Fresh session — no prior context."

    try:
        result = await chat_json([{"role": "user", "content":
            f'You are classifying a user message for a data analytics assistant.\n\n'
            f'=== SESSION CONTEXT ===\n{session_ctx}\n========================\n\n'
            f'=== ACTIVE STATE ===\n{context_flags_str}\n{probe_info}====================\n\n'
            f'User message: "{message}"\n\n'
            f'Classify as EXACTLY ONE of these intents:\n\n'
            f'CHAT — greeting, small talk, thanks, goodbye, capability question, or general conversation\n\n'
            f'DB_QUERY — a self-contained, standalone request to fetch data from the database.\n'
            f'  Could be the very first message in a fresh session.\n\n'
            f'DB_FOLLOW_UP — REFINES or BUILDS ON a specific previous database result.\n'
            f'  Must DEPEND ON prior context (pronouns like "those"/"them", added filters, references to previous values).\n'
            f'  KEY TEST: Could this be the first message in a brand-new session? If YES → NOT DB_FOLLOW_UP.\n\n'
            f'VIZ_FOLLOW_UP — wants to visualize, view, display, or change the format of EXISTING on-screen data.\n'
            f'  KEY SIGNAL: any reference to "this data", "the data", "these results", "the results" combined with a\n'
            f'  display format (table, chart, graph, pie, bar, line, visualize, view, display, export) = VIZ_FOLLOW_UP.\n'
            f'  Examples: "get me this data in table", "show this as pie chart", "visualize the results",\n'
            f'  "I want to view this data", "display as bar graph", "yes", "sure", "ok"\n'
            f'  Does NOT include requests for NEW/DIFFERENT data even if they mention a chart type\n'
            f'  (e.g. "show me a bar chart of customers from Mumbai" = DB_QUERY, not VIZ_FOLLOW_UP).\n'
            f'  Only valid when data is currently on screen.\n\n'
            f'FILE_ANALYSIS — asks about content of an uploaded file (only when file context exists)\n\n'
            f'FILE_FOLLOW_UP — follow-up question about an already-analysed file\n\n'
            f'HYBRID — asks to compare or cross-reference BOTH uploaded file data AND database data\n\n'
            f'DUAL_SEARCH — the message is ambiguous about whether it refers to file data or database data '
            f'(only when BOTH file and DB context exist and vector probe shows strong matches in both)\n\n'
            f'DEFAULT RULES:\n'
            f'- When in doubt between DB_QUERY and DB_FOLLOW_UP → choose DB_QUERY\n'
            f'- When in doubt between DB_QUERY and CHAT → choose DB_QUERY\n'
            f'- VIZ_FOLLOW_UP only when data is on screen AND user wants to change its display format\n\n'
            f'If intent is VIZ_FOLLOW_UP, also extract:\n'
            f'- types_specified: true if user named ANY specific format/type from {available_viz_types}\n'
            f'  ("in table" → types_specified=true, types=["table"]; "as pie chart" → types_specified=true, types=["pie"])\n'
            f'  false ONLY for completely vague requests like "visualize this" or "I want to see this" with NO type mentioned\n'
            f'- types: ALL matching types from {available_viz_types} the user mentioned (can be multiple)\n'
            f'- color_mode: "varied" for multi-color request, null otherwise\n'
            f'- bar_color: "#hex" for single color request, null otherwise\n\n'
            f'Return JSON:\n'
            f'{{"intent": "...", "confidence": 0.0-1.0, "reason": "one line",\n'
            f'  "types_specified": false, "types": [], "color_mode": null, "bar_color": null}}'}])

        intent = result.get("intent", "CHAT")
        confidence = result.get("confidence", 0.7)
        logger.info(f"Unified LLM classification: intent={intent}, confidence={confidence}, reason={result.get('reason', '')}")

        # Store viz extraction in session so orchestrator doesn't need another LLM call
        if intent == "VIZ_FOLLOW_UP" and has_data_on_screen:
            detected_types = [t for t in result.get("types", []) if t in available_viz_types]
            color_hints = {}
            if result.get("color_mode") == "varied":
                color_hints = {"color_mode": "varied"}
            elif result.get("bar_color"):
                color_hints = {"bar_color": result["bar_color"]}
            session_state["_viz_detected"] = {
                "types_specified": bool(result.get("types_specified") and detected_types),
                "types": detected_types,
                "color_hints": color_hints,
            }
            session_state["viz_suggestion_pending"] = False
        elif intent == "VIZ_FOLLOW_UP" and not has_data_on_screen:
            # No data on screen — can't visualize; reclassify as DB_QUERY
            intent = "DB_QUERY"
            confidence = 0.7

        # DUAL_SEARCH validation — only valid when both sources have strong matches
        if intent == "DUAL_SEARCH" and not has_file_ctx:
            intent = "DB_QUERY"

        return {"intent": intent, "confidence": confidence, "method": "llm_unified"}
    except Exception as e:
        logger.warning(f"Unified LLM classification failed: {e}")
        # Graceful fallback: if file context and no prior queries, assume file
        if has_file_ctx and not has_prior_query:
            return {"intent": "FILE_FOLLOW_UP", "confidence": 0.5, "method": "llm_unified_fallback"}
        return {"intent": "DB_QUERY" if has_prior_query else "CHAT", "confidence": 0.5, "method": "llm_unified_fallback"}
