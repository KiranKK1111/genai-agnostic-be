"""Main orchestrator — routes every message through input gate → scope → intent → pipeline."""
import json
import logging
from app.pipelines.intent import classify_intent
from app.pipelines.scope_validator import validate_scope
from app.pipelines.chat import execute_chat
from app.pipelines.db_pipeline import execute_db_query
from app.pipelines.file_pipeline import execute_file_upload, execute_file_followup
from app.pipelines.hybrid_pipeline import execute_hybrid
from app.services.input_quality import detect_gibberish
from app.services.prompt_guard import check_injection, scan_output
from app.services.error_registry import get_error
from app.services.session import SessionManager
from app.services.audit_logger import log_query
from app.services.schema_inspector import SchemaGraph
from app.services.query_logger import (
    log_query_arrival, log_intent, log_scope_rejected, log_pipeline_done,
)
from app.config import get_settings
from typing import AsyncIterator

logger = logging.getLogger(__name__)

# ── Dynamic viz type resolution ──────────────────────────────────────────────
def _get_available_viz_types() -> list[str]:
    """Derive available viz types dynamically from the viz config builder."""
    from app.builders.viz_config import viz_type_clarification
    return [o["value"] for o in viz_type_clarification().get("options", [])]

# ── Viz color helpers ─────────────────────────────────────────────────────────
async def _detect_viz_color_llm(msg: str) -> dict:
    """Use the LLM to detect chart color preferences — no hardcoded keywords."""
    from app.services.llm_client import chat_json as _cj
    try:
        result = await _cj([{"role": "user", "content":
            f'Analyze this message for chart color preferences: "{msg}"\n'
            f'Return JSON: {{"color_mode": "varied" or null, "bar_color": "#hex" or null}}\n'
            f'- color_mode: set to "varied" when the user wants multiple distinct colors across bars '
            f'(e.g. "colorful", "variety of colors", "rainbow", "different colors", "multicolor")\n'
            f'- bar_color: set to a valid CSS hex color code when the user names a single specific color '
            f'(e.g. "red" → "#ef4444", "green" → "#22c55e", "blue" → "#3b82f6", "purple" → "#8b5cf6")\n'
            f'- Set both to null if no color preference is expressed\n'
            f'- color_mode and bar_color are mutually exclusive; prefer color_mode if both apply'}])
        if result.get("color_mode") == "varied":
            return {"color_mode": "varied"}
        if result.get("bar_color"):
            return {"bar_color": result["bar_color"]}
    except Exception:
        pass
    return {}

# Global schema graph (loaded at startup)
_schema_graph: SchemaGraph | None = None

def set_schema_graph(graph: SchemaGraph):
    global _schema_graph
    _schema_graph = graph

def get_schema_graph() -> SchemaGraph:
    return _schema_graph


async def process_message(message: str, session_id: str = None, user_id: str = None,
                          user_name: str = "User", file_id: str = None,
                          file_name: str = None, file_size: int = None,
                          file_mime: str = None) -> AsyncIterator[dict]:
    """Main entry point. Processes a user message and yields SSE events."""
    settings = get_settings()
    session_mgr = SessionManager()
    state = await session_mgr.get_or_create(session_id, user_id)
    session_id = state["session_id"]

    log_query_arrival(message, session_id=session_id, user_name=user_name)

    yield {"type": "typing_start"}

    # ── Step 1: Input Quality Gate ─────────────────────────
    quality = detect_gibberish(message)
    if quality.get("tier") == "reject" or quality["score"] > settings.GIBBERISH_HARD_THRESHOLD:
        err = get_error("E020", "I couldn't understand that. Please try again with a clear message.")
        err["type"] = "error"
        err["recoverable"] = False
        yield err
        yield {"type": "typing_end"}
        yield {"type": "done", "message_id": ""}
        return
    if quality.get("tier") == "rephrase" or quality["score"] > settings.GIBBERISH_THRESHOLD:
        err = get_error("E020", "Your message seems unclear. Could you rephrase it?")
        err["type"] = "error"
        yield err
        yield {"type": "typing_end"}
        yield {"type": "done", "message_id": ""}
        return

    # ── Step 2: Prompt Injection Guard ─────────────────────
    injection = check_injection(message)
    if injection["is_injection"]:
        err = get_error("E021", "Your message contains patterns that I can't process for security reasons.")
        err["type"] = "error"
        yield err
        await log_query(user_id=user_id, session_id=session_id, user_prompt=message,
                       intent="BLOCKED", status="REJECTED", injection_flags=injection["matched_patterns"])
        yield {"type": "typing_end"}
        yield {"type": "done", "message_id": ""}
        return

    # ── Step 3: Intent Classification (runs BEFORE scope validation) ─────
    # Intent must be classified first so we know if the user is replying to
    # a clarification (which should bypass scope) or asking something new.
    has_file = file_id is not None
    intent_result = await classify_intent(message, state, has_file=has_file)
    intent = intent_result["intent"]
    logger.info(f"Intent: {intent} (confidence={intent_result['confidence']}, method={intent_result['method']})")
    log_intent(intent, intent_result["confidence"], intent_result["method"], session_id=session_id)

    # ── Step 4: Scope Validation ───────────────────────────
    # Skip scope validation for clarification replies — the user is answering our
    # question, not asking something new. Custom replies are combined with the
    # original query and sent through the full LLM pipeline.
    if intent != "CLARIFICATION_REPLY":
        scope = validate_scope(message)
        if not scope["in_scope"]:
            log_scope_rejected(message, session_id=session_id)
            yield {"type": "step", "step_number": 1, "label": "Understanding your question"}
            from app.services.llm_client import chat
            rejection = await chat([{"role": "user", "content": f"The user asked: \"{message}\"\nPolitely explain that you are designed only for database queries, file analysis, and conversation. You cannot: write code, generate images, browse the web. Keep it brief and friendly (2 sentences). Use an emoji."}])
            yield {"type": "text_done", "content": rejection}
            yield {"type": "typing_end"}
            msg_id = await session_mgr.save_message(session_id, "user", message)
            await session_mgr.save_message(session_id, "assistant", rejection, parent_message_id=msg_id)
            yield {"type": "done", "message_id": msg_id}
            return

    state["last_intent"] = intent
    state["intent_chain"] = (state.get("intent_chain", []) + [intent])[-settings.INTENT_CHAIN_MAX_LENGTH:]

    # Save user message — skip clarification replies (they are transient UI interactions)
    is_clarification_reply = intent == "CLARIFICATION_REPLY"
    is_first_message = state.get("total_turns", 0) == 0  # capture BEFORE incrementing
    if not is_clarification_reply:
        # If the user uploaded a file with this message, persist the attachment
        # so the chip + download button can be reconstructed on history reload.
        user_metadata = {}
        if file_id and file_name:
            user_metadata["attachments"] = [file_name]
            user_metadata["attachment_ids"] = {file_name: file_id}
            # attachment_meta mirrors the frontend type: { fileName: { size, type } }
            # so the file-size label on the chip survives page reloads.
            meta_entry: dict = {}
            if file_size is not None:
                meta_entry["size"] = file_size
            if file_mime:
                meta_entry["type"] = file_mime
            if meta_entry:
                user_metadata["attachment_meta"] = {file_name: meta_entry}
        user_msg_id = await session_mgr.save_message(
            session_id, "user", message, metadata=user_metadata
        )
        await session_mgr.append_history(state, "user", message)
    else:
        user_msg_id = None

    # Rename session to the user's prompt immediately — this is the ONLY place
    # the session title is set. Pipelines no longer override it with LLM summaries
    # (those produced misleading titles like "New Message from Unknown User").
    if is_first_message and not is_clarification_reply:
        raw = (message or "").strip()
        # File uploads with an empty / generic auto-message: fall back to the filename
        if file_name and (not raw or raw.lower().startswith(("analyze this file", "uploaded"))):
            raw = f"File: {file_name}"
        prompt_title = raw[:settings.SESSION_TITLE_MAX_LENGTH] or settings.DEFAULT_SESSION_TITLE
        state["title"] = prompt_title
        try:
            from app.database import get_pool as _get_pool
            _pool = _get_pool()
            async with _pool.acquire() as _conn:
                await _conn.execute(
                    f"UPDATE {settings.APP_SCHEMA}.chat_sessions SET title=$1 WHERE id=$2::uuid",
                    prompt_title, session_id,
                )
        except Exception as e:
            logger.warning(f"Failed to set initial session title: {e}")
        yield {"type": "session_meta", "session_title": prompt_title}

    # ── Step 5: Pipeline Dispatch ──────────────────────────
    # NOTE: The SSE consumer (chat.py) does event.pop("type") which mutates the dict.
    # We must capture the event type BEFORE yielding, and store (etype, event) tuples
    # in all_events so the metadata extraction loop can still identify event types.
    # Set flag so pipelines know whether to generate a title
    state["_is_first_message"] = is_first_message
    full_text = ""
    all_events = []  # list of (event_type_str, event_dict) tuples

    if intent == "CHAT":
        async for event in execute_chat(message, state, user_name):
            etype = event.get("type")
            all_events.append((etype, event))
            yield event
            if etype == "text_done":
                full_text = event.get("content", "")

    elif intent in ("DB_QUERY", "DB_FOLLOW_UP"):
        is_follow = intent == "DB_FOLLOW_UP"
        async for event in execute_db_query(message, state, get_schema_graph(), is_follow_up=is_follow, user_name=user_name):
            etype = event.get("type")
            all_events.append((etype, event))
            yield event
            if etype == "text_done":
                full_text = event.get("content", "")
            if etype == "error":
                await log_query(user_id=user_id, session_id=session_id, user_prompt=message,
                               intent=intent, status="ERROR", error=event.get("message"))

    elif intent == "FILE_ANALYSIS" and file_id:
        async for event in execute_file_upload(file_id, file_name, message, state, user_name):
            etype = event.get("type")
            all_events.append((etype, event))
            yield event
            if etype == "text_done":
                full_text = event.get("content", "")

    elif intent == "FILE_FOLLOW_UP":
        async for event in execute_file_followup(message, state):
            etype = event.get("type")
            all_events.append((etype, event))
            yield event
            if etype == "text_done":
                full_text = event.get("content", "")

    elif intent == "VIZ_FOLLOW_UP":
        # Reuse cached data
        if state.get("last_data") and state.get("last_columns"):
            # Read pre-extracted viz details from intent classifier (single LLM pass)
            viz_detected = state.pop("_viz_detected", {})
            user_specified = viz_detected.get("types_specified", False)
            detected_types = viz_detected.get("types", [])
            color_hints = viz_detected.get("color_hints", {})

            # Build dynamic tool capability string for follow-up questions
            from app.builders.viz_config import viz_type_clarification as _vtc_orch
            _viz_opts = _vtc_orch()
            _viz_labels = ", ".join(o.get("label", o["value"]) for o in _viz_opts.get("options", []))
            _tool_caps = f"filtering by specific values, drilling down into subsets, sorting, exporting, and visualizing as: {_viz_labels}"

            if user_specified and detected_types:
                # User already specified — skip clarification, render directly
                primary_viz = detected_types[0]
                from app.pipelines.viz_engine import build_viz_config
                from app.services.llm_client import chat_stream
                from app.services.response_beautifier import beautify
                from app.services.sql_executor import execute_sql as _exec_sql

                columns = state["last_columns"]
                sql = state.get("last_sql", "")

                # Re-execute SQL to get full dataset (session only stores 200 rows)
                if sql:
                    yield {"type": "step", "step_number": 1, "label": "Loading full dataset"}
                    _result = await _exec_sql(sql, uncapped=True)
                    if "error" not in _result and _result.get("rows"):
                        rows = []
                        for row in _result["rows"]:
                            sr = {}
                            for k, v in row.items():
                                sr[k] = v.isoformat() if hasattr(v, 'isoformat') else (
                                    str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v)
                            rows.append(sr)
                    else:
                        rows = state.get("last_data", [])
                else:
                    rows = state.get("last_data", [])
                row_count = len(rows)

                yield {"type": "step", "step_number": 2, "label": "Preparing response"}
                friendly_cols = ', '.join(c.replace('_', ' ').title() for c in columns)
                response_prompt = (
                    f"The user asked: \"{message}\"\n"
                    f"Found {row_count} records with fields: {friendly_cols}.\n"
                    f"Write 2-3 natural sentences summarizing the results. Use **bold** for key numbers. "
                    f"Do NOT use pipe-symbol tables or markdown tables — they render as broken symbols. Do NOT enumerate individual rows. Do NOT start with any heading label.\n"
                    f"IMPORTANT: Do NOT mention database internals (schema names, table names, column names, SQL). Use natural business language only.\n"
                    f"MANDATORY: End with a follow-up question suggesting ACTIONABLE next steps. The tool supports: {_tool_caps}. Reference specific data from the response. Do NOT ask hypothetical/analytical questions — only suggest things the tool can actually do. Never skip this."
                )
                full_text = ""
                async for token in chat_stream([{"role": "user", "content": response_prompt}]):
                    full_text += token
                    yield {"type": "text_delta", "delta": token}

                full_text = beautify(full_text)
                if "FOLLOW_UPS:" in full_text:
                    full_text = full_text.split("FOLLOW_UPS:")[0].strip()
                yield {"type": "text_done", "content": full_text}

                viz_cfg = build_viz_config(primary_viz, columns)
                # Show all types the user asked for (e.g. ["table", "pie"])
                viz_cfg["available_views"] = detected_types
                viz_cfg.update(color_hints)
                viz_event = {"type": "viz_config", "config": viz_cfg}
                all_events.append(("viz_config", viz_event))
                yield viz_event

                data_event = {"type": "data", "rows": rows, "columns": columns,
                              "row_count": row_count, "sql": sql, "truncated": False}
                all_events.append(("data", data_event))
                yield data_event
            else:
                # User didn't specify — ask via clarification
                from app.pipelines.viz_engine import build_viz_clarification
                clar = build_viz_clarification(state["last_columns"])
                clar_event = {"type": "clarification", "clarification": clar}
                all_events.append(("clarification", clar_event))
                yield clar_event
                state["clarification_pending"] = {"type": "viz_type", "message": message}
        else:
            yield {"type": "text_done", "content": "I don't have previous data to visualize. Please run a query first."}
            full_text = "I don't have previous data to visualize."

    elif intent == "DUAL_SEARCH":
        # Both file and DB indexes matched — ask user which source they mean
        from app.pipelines.intent import source_clarification
        file_history = state.get("file_history", [])
        file_names = [f.get("file_name", "uploaded file") for f in file_history] if file_history else [state.get("file_context", {}).get("file_name", "uploaded file")]
        clar = source_clarification(file_names)
        clar_event = {"type": "clarification", "clarification": clar}
        all_events.append(("clarification", clar_event))
        yield clar_event
        state["clarification_pending"] = {
            "type": "source_clarification", "message": message,
        }

    elif intent == "HYBRID":
        async for event in execute_hybrid(message, state):
            etype = event.get("type")
            all_events.append((etype, event))
            yield event
            if etype == "text_done":
                full_text = event.get("content", "")

    elif intent == "CLARIFICATION_REPLY":
        # Re-run the pipeline with the user's clarification applied
        pending = state.get("clarification_pending") or {}
        response = state.get("clarification_response") or {}
        selected = response.get("selected_values")
        clar_type = pending.get("type", "")
        original_message = pending.get("message", message)
        logger.info(f"CLARIFICATION_REPLY: clar_type={clar_type}, selected={selected!r}, "
                    f"original_message={original_message!r}, stored_sql={'yes' if pending.get('sql') else 'no'}")

        # Skip = use defaults. Map __default to the appropriate value per clarification type.
        if selected == "__default":
            defaults = {
                "record_limit": "limited",
                "viz_type": "table",
                "axis_mode": "on_the_fly",
                "source_clarification": "database",  # default to DB when skipped
                "filter_criteria": None,   # no sensible default — user must type criteria
            }
            # For entity/table ambiguity: pick the first option from the pending clarification
            if clar_type == "ambiguous_entity":
                options = pending.get("options") or []
                first_option = options[0].get("value") if options else None
                if first_option:
                    defaults["ambiguous_entity"] = first_option
                    logger.info(f"Skip ambiguous_entity: defaulting to first option '{first_option}'")
            elif clar_type in ("ambiguous_table", "ambiguous_column", "ambiguous_value"):
                options = pending.get("options") or []
                first_option = options[0].get("value") if options else None
                if first_option:
                    defaults[clar_type] = first_option
                    logger.info(f"Skip {clar_type}: defaulting to first option '{first_option}'")

            default_val = defaults.get(clar_type)
            if default_val:
                selected = default_val
                logger.info(f"Skip with default: clar_type={clar_type}, default={selected}")
                # Update the response so downstream handlers see the resolved value
                response["selected_values"] = selected
                state["clarification_response"] = response
            else:
                # No default available — cancel the flow
                state["clarification_pending"] = None
                state["clarification_response"] = None
                state["clarification_display"] = None
                state["clarification_history"] = []
                yield {"type": "text_done", "content": "Got it, standing by! Let me know what you need next."}
                full_text = "Got it, standing by! Let me know what you need next."

        # Frontend may send clarification as a regular chat message instead of /clarify.
        # Extract the selection from the message text if clarification_response wasn't set.
        if not selected and clar_type == "record_limit":
            msg_lower = message.strip().lower()
            if msg_lower in ("limited", "all"):
                selected = msg_lower
                logger.info(f"Record limit clarification: selected='{selected}' from chat message")
            # else: custom text reply — falls through to re-run as DB_QUERY with full LLM
        if not selected and clar_type == "viz_type":
            msg_lower = message.strip().lower()
            _avail = set(_get_available_viz_types())
            if msg_lower in _avail:
                selected = msg_lower
                logger.info(f"Viz type clarification: selected='{selected}' from chat message")
            # else: custom text reply — falls through to re-run as DB_QUERY with full LLM

        # If the user typed the answer as a regular chat message (not via /clarify),
        # record the Q&A pair in clarification_history now.
        if not response.get("selected_values"):
            display = state.get("clarification_display") or {}
            question_text = display.get("question") or pending.get("question", "")
            answer_text = message.strip()
            history = state.get("clarification_history", [])
            history.append({"question": question_text, "answer": answer_text, "type": clar_type})
            state["clarification_history"] = history

        # Clear clarification state FIRST (prevents infinite loop)
        state["clarification_pending"] = None
        state["clarification_response"] = None
        state["clarification_display"] = None

        # Determine the effective message to re-run
        clarified_message = None
        handled = False  # True when record_limit already executed the query

        if clar_type.startswith("ambiguous_") and selected and _schema_graph:
            if clar_type == "ambiguous_table":
                resolved_table = selected if isinstance(selected, str) else (selected[0] if isinstance(selected, list) else str(selected))
                # Direct table reference — don't hint, just use the user's original query
                # The query planner will resolve it since we cleared the ambiguity
                clarified_message = original_message
                # Force the resolved table into session state so planner picks it
                state["resolved_table"] = resolved_table
            elif clar_type == "ambiguous_value":
                resolved = selected if isinstance(selected, str) else (selected[0] if isinstance(selected, list) else str(selected))
                # resolved is "table.column" — extract parts for a natural hint
                token = pending.get("token", "")
                if "." in resolved:
                    res_table, res_column = resolved.split(".", 1)
                    friendly_col = res_column.replace('_', ' ')
                    clarified_message = f"{original_message} (where {friendly_col} is {token})"
                else:
                    clarified_message = f"{original_message} (filter on {resolved})"
                # Store resolved ambiguity so the detector skips it on re-run
                resolved_ambiguities = state.get("resolved_ambiguities", [])
                resolved_ambiguities.append({"token": token, "resolved": resolved, "type": clar_type})
                state["resolved_ambiguities"] = resolved_ambiguities
            elif clar_type == "ambiguous_entity":
                # User picked which table/report to use for the entity
                resolved_table = selected if isinstance(selected, str) else (selected[0] if isinstance(selected, list) else str(selected))
                token = pending.get("token", "")
                clarified_message = original_message
                # Store so detector skips re-asking AND planner picks the right table
                resolved_ambiguities = state.get("resolved_ambiguities", [])
                resolved_ambiguities.append({
                    "token": token,
                    "resolved_table": resolved_table,
                    "value": resolved_table,
                    "type": clar_type,
                })
                state["resolved_ambiguities"] = resolved_ambiguities
                # Also set resolved_table so Stage 2a in the planner uses it directly
                state["resolved_table"] = resolved_table
                logger.info(
                    f"Clarification resolved: entity='{token}' → table='{resolved_table}'"
                )
            else:
                clarified_message = original_message

        elif clar_type == "viz_type":
            # User chose visualization type(s) — render data with summary.
            # For custom text replies (e.g. "show me a pie chart"), use LLM to extract viz type and color.
            color_hints: dict = {}
            if isinstance(selected, list):
                viz_types = selected
                color_hints = await _detect_viz_color_llm(original_message)
            elif isinstance(selected, str) and all(s.strip() in set(_get_available_viz_types()) for s in selected.split(",")):
                viz_types = [s.strip() for s in selected.split(",")]
                color_hints = await _detect_viz_color_llm(original_message)
            elif selected:
                # Custom text — ask LLM to extract viz type and color preferences
                from app.services.llm_client import chat_json as _cj2
                _avail_types = _get_available_viz_types()
                extract = await _cj2([{"role": "user", "content":
                    f'The user was asked how they want to view data. They replied: "{selected}"\n'
                    f'Return JSON:\n'
                    f'{{"types": ["table"], "color_mode": "varied" or null, "bar_color": "#hex" or null}}\n'
                    f'- types: visualization type(s) chosen from options: {_avail_types}\n'
                    f'- color_mode: "varied" if the user wants multiple distinct colors across bars; null otherwise\n'
                    f'- bar_color: a valid CSS hex color code if the user names a single specific color; null otherwise'}])
                viz_types = extract.get("types", ["table"])
                viz_types = [v for v in viz_types if v in set(_avail_types)] or ["table"]
                if extract.get("color_mode") == "varied":
                    color_hints = {"color_mode": "varied"}
                elif extract.get("bar_color"):
                    color_hints = {"bar_color": extract["bar_color"]}
            else:
                viz_types = ["table"]
            primary_viz = viz_types[0] if viz_types else "table"

            if state.get("last_columns") and (state.get("last_data") or state.get("last_sql")):
                from app.pipelines.viz_engine import build_viz_config
                from app.services.llm_client import chat_stream
                from app.services.response_beautifier import beautify
                from app.services.sql_executor import execute_sql
                from app.builders.viz_config import viz_type_clarification as _vtc_clar
                _viz_opts_clar = _vtc_clar()
                _viz_labels_clar = ", ".join(o.get("label", o["value"]) for o in _viz_opts_clar.get("options", []))
                _tool_caps = f"filtering by specific values, drilling down into subsets, sorting, exporting, and visualizing as: {_viz_labels_clar}"

                columns = state["last_columns"]
                sql = state.get("last_sql", "")

                # Re-execute SQL to get the full dataset (session only stores 200 rows)
                if sql:
                    yield {"type": "step", "step_number": 1, "label": "Loading full dataset"}
                    result = await execute_sql(sql, uncapped=True)
                    if "error" not in result and result.get("rows"):
                        rows = result["rows"]
                        # Serialize rows for JSON (handle datetime etc)
                        serializable_rows = []
                        for row in rows:
                            sr = {}
                            for k, v in row.items():
                                sr[k] = v.isoformat() if hasattr(v, 'isoformat') else (
                                    str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v)
                            serializable_rows.append(sr)
                        rows = serializable_rows
                    else:
                        rows = state.get("last_data", [])
                else:
                    rows = state.get("last_data", [])

                row_count = len(rows)

                # Generate summary via LLM
                yield {"type": "step", "step_number": 2, "label": "Preparing response"}
                friendly_cols = ', '.join(c.replace('_', ' ').title() for c in columns)
                response_prompt = (
                    f"The user asked: \"{original_message}\"\n"
                    f"Found {row_count} records with fields: {friendly_cols}.\n"
                    f"Write 2-3 natural sentences summarizing the results. Use **bold** for key numbers. "
                    f"Do NOT use pipe-symbol tables or markdown tables — they render as broken symbols. Do NOT enumerate individual rows. Do NOT start with any heading label.\n"
                    f"IMPORTANT: Do NOT mention database internals (schema names, table names, column names, SQL). Use natural business language only.\n"
                    f"MANDATORY: End with a follow-up question suggesting ACTIONABLE next steps. The tool supports: {_tool_caps}. Reference specific data from the response. Do NOT ask hypothetical/analytical questions — only suggest things the tool can actually do. Never skip this."
                )
                full_text = ""
                async for token in chat_stream([{"role": "user", "content": response_prompt}]):
                    full_text += token
                    yield {"type": "text_delta", "delta": token}

                full_text = beautify(full_text)
                if "FOLLOW_UPS:" in full_text:
                    full_text = full_text.split("FOLLOW_UPS:")[0].strip()
                yield {"type": "text_done", "content": full_text}

                # Send viz config
                viz_cfg = build_viz_config(primary_viz, columns)
                viz_cfg["available_views"] = viz_types
                viz_cfg.update(color_hints)
                viz_event = {"type": "viz_config", "config": viz_cfg}
                all_events.append(("viz_config", viz_event))
                yield viz_event

                # Send data
                data_event = {"type": "data", "rows": rows, "columns": columns,
                              "row_count": row_count, "sql": sql, "truncated": False}
                all_events.append(("data", data_event))
                yield data_event
            else:
                yield {"type": "text_done", "content": "I don't have previous data to visualize. Please run a query first."}
                full_text = "I don't have previous data to visualize."
            handled = True

        elif clar_type == "record_limit":
            stored_sql = pending.get("sql")
            stored_plan = pending.get("plan", {})
            logger.info(f"record_limit handler: stored_sql={'yes' if stored_sql else 'NO'}, selected={selected!r}")
            if stored_sql:
                if selected == "all":
                    max_populate = get_settings().MAX_POPULATE_ROWS
                    final_sql = stored_sql.rstrip().rstrip(";") + f" LIMIT {max_populate};"
                elif selected == "limited":
                    limit_val = get_settings().RECORD_WARN_THRESHOLD
                    final_sql = stored_sql.rstrip().rstrip(";") + f" LIMIT {limit_val};"
                else:
                    # Custom text reply (e.g. "populate first 1000 records", "just 200")
                    # Use LLM to extract the intended row count from the user's reply
                    from app.services.llm_client import chat_json as _cj
                    extract = await _cj([{"role": "user", "content":
                        f'The user was asked how many records to return. They replied: "{message}"\n'
                        f'Extract the number of rows they want. Return JSON: {{"limit": <number or null>}}\n'
                        f'If they want all records, return {{"limit": null}}. If unclear, return {{"limit": null}}.'}])
                    extracted_limit = extract.get("limit")
                    if extracted_limit and isinstance(extracted_limit, (int, float)) and extracted_limit > 0:
                        final_sql = stored_sql.rstrip().rstrip(";") + f" LIMIT {int(extracted_limit)};"
                    else:
                        final_sql = stored_sql  # Fallback to all records

                state["last_sql"] = final_sql
                state["last_plan"] = stored_plan
                from app.pipelines.db_pipeline import execute_db_query_with_sql
                async for event in execute_db_query_with_sql(
                    final_sql, stored_plan, original_message, state, _schema_graph, user_name=user_name
                ):
                    etype = event.get("type")
                    all_events.append((etype, event))
                    yield event
                    if etype == "text_done":
                        full_text = event.get("content", "")
                    if etype == "error":
                        await log_query(user_id=user_id, session_id=session_id,
                                       user_prompt=original_message, intent=intent,
                                       status="ERROR", error=event.get("message"))
                handled = True

        if not handled and clar_type == "filter_criteria":
            # User typed filter criteria (e.g. "country is India, balance > 50000")
            # Amend the stored SQL directly — don't go through _select_base_query
            filter_text = selected if isinstance(selected, str) and selected else message
            stored_sql = pending.get("sql", "")
            if stored_sql and _schema_graph:
                # Pin last_sql to the stored SQL so the follow-up path uses it directly
                # and doesn't pick a different query from history
                state["last_sql"] = stored_sql
                # Temporarily set sql_history to just this entry so _select_base_query
                # returns it without an LLM call
                saved_history = state.get("sql_history", [])
                state["sql_history"] = [{"sql": stored_sql, "message": original_message, "tables": [], "has_join": False, "columns": []}]
                clarified_message = f"filter where {filter_text}"
                async for event in execute_db_query(clarified_message, state, _schema_graph,
                                                    is_follow_up=True, user_name=user_name):
                    etype = event.get("type")
                    all_events.append((etype, event))
                    yield event
                    if etype == "text_done":
                        full_text = event.get("content", "")
                # Restore full history (new query was appended by _update_session)
                state["sql_history"] = saved_history + [e for e in state.get("sql_history", []) if e not in saved_history]
            else:
                yield {"type": "text_done", "content": "I couldn't process the filter. Could you try rephrasing?"}
                full_text = "I couldn't process the filter."
            handled = True

        if not handled and clar_type == "source_clarification":
            # User chose file or database — route the original message to the right pipeline
            source = selected if isinstance(selected, str) else "database"
            if not source:
                # Use LLM to detect whether the user means a file or the database
                from app.services.llm_client import chat_json as _cj_src
                try:
                    _src = await _cj_src([{"role": "user", "content":
                        f'The user said: "{message}"\n'
                        f'Are they referring to an uploaded file/document or a database table?\n'
                        f'Return JSON: {{"source": "file" or "database"}}'}])
                    source = _src.get("source", "database")
                    if source not in ("file", "database"):
                        source = "database"
                except Exception:
                    source = "database"
            if source == "file":
                async for event in execute_file_followup(original_message, state):
                    etype = event.get("type")
                    all_events.append((etype, event))
                    yield event
                    if etype == "text_done":
                        full_text = event.get("content", "")
            else:
                if _schema_graph:
                    async for event in execute_db_query(original_message, state, _schema_graph,
                                                        is_follow_up=False, user_name=user_name):
                        etype = event.get("type")
                        all_events.append((etype, event))
                        yield event
                        if etype == "text_done":
                            full_text = event.get("content", "")
                else:
                    yield {"type": "text_done", "content": "I couldn't process that. Could you try rephrasing your question?"}
                    full_text = "I couldn't process that."
            handled = True

        if not handled:
            # Custom text reply — combine original query with user's clarification
            # so the LLM has full context (e.g. "get me all clients" + "give me first 200 records")
            if not clarified_message:
                if original_message and original_message != message:
                    clarified_message = f"{original_message} ({message})"
                else:
                    clarified_message = message

            # Re-run DB pipeline
            if _schema_graph:
                async for event in execute_db_query(clarified_message, state, _schema_graph,
                                                    is_follow_up=False, user_name=user_name):
                    etype = event.get("type")
                    all_events.append((etype, event))
                    yield event
                    if etype == "text_done":
                        full_text = event.get("content", "")
                    if etype == "error":
                        await log_query(user_id=user_id, session_id=session_id,
                                       user_prompt=clarified_message, intent=intent,
                                       status="ERROR", error=event.get("message"))
            else:
                yield {"type": "text_done", "content": "I couldn't process that. Could you try rephrasing your question?"}
                full_text = "I couldn't process that."

    else:
        # Default to chat
        async for event in execute_chat(message, state, user_name):
            etype = event.get("type")
            all_events.append((etype, event))
            yield event
            if etype == "text_done":
                full_text = event.get("content", "")

    # ── Emit accumulated clarification Q&A pairs ───────────
    # When the pipeline completes with a final response (not another clarification),
    # emit all accumulated Q&A pairs as a single event so the frontend can display
    # them together. The pairs are also persisted in the message metadata below.
    has_new_clarification = any(et == "clarification" for et, _ in all_events)
    clar_qa_pairs = state.get("clarification_history", [])
    if clar_qa_pairs and not has_new_clarification:
        qa_event = {"type": "clarification_qa", "pairs": list(clar_qa_pairs)}
        all_events.append(("clarification_qa", qa_event))
        yield qa_event
        # Clear history now that it has been emitted
        state["clarification_history"] = []

    # ── Post-LLM output scan ──────────────────────────────
    if full_text:
        output_scan = scan_output(full_text)
        if not output_scan["is_safe"]:
            logger.warning(f"Post-LLM scan flagged: {output_scan['issues']}")

    # If no text_done was emitted (e.g. clarification-only responses), derive content
    # from events so the message is never saved as empty.
    if not full_text:
        for etype, ev in all_events:
            if etype == "clarification":
                clar = ev.get("clarification", {})
                full_text = clar.get("question", "")
                break
            if etype == "text_delta":
                full_text += ev.get("delta", "")

    # Save assistant message — persist all rich content so reload can reconstruct the full UI
    # all_events is a list of (event_type, event_dict) tuples.
    metadata = {}
    content_sql = None
    for etype, ev in all_events:
        if etype == "data":
            rows = ev.get("rows", [])
            metadata["data"] = {
                "rows": rows[:settings.METADATA_ROWS_CAP],  # Cap stored rows to prevent bloated JSONB
                "columns": ev.get("columns", []),
                "row_count": ev.get("row_count", len(rows)),
                "truncated": ev.get("truncated", False),
            }
            content_sql = ev.get("sql")
        if etype == "viz_config":
            cfg = ev.get("config", ev)  # config may be nested or flat
            metadata["viz_config"] = {
                "viz_type": cfg.get("viz_type", "table"),
                "available_views": cfg.get("available_views", ["table", "bar", "pie", "line"]),
                "primary_view": cfg.get("viz_type", "table"),
            }
        if etype == "clarification":
            clar = ev.get("clarification", ev)  # clarification may be nested
            metadata["clarification"] = {
                "question": clar.get("question", ""),
                "type": clar.get("type", ""),
                "options": clar.get("options", []),
            }
            # Store full clarification data in session so it can be restored on page reload
            state["clarification_display"] = {
                "question": clar.get("question", ""),
                "type": clar.get("type", ""),
                "options": clar.get("options", []),
                "mode": clar.get("mode", ""),
            }
        if etype == "clarification_qa":
            metadata["clarification_qa"] = ev.get("pairs", [])

    # Determine if this is a clarification-only response (no real content to persist).
    # Clarification popups are transient UI — only persist the final resolved response.
    is_clarification_only = "clarification" in metadata and "data" not in metadata and not content_sql

    if is_clarification_only:
        # Don't persist clarification-only assistant messages
        asst_msg_id = ""
    else:
        asst_msg_id = await session_mgr.save_message(
            session_id, "assistant", full_text,
            metadata=metadata, follow_ups=[],
            content_sql=content_sql, parent_message_id=user_msg_id,
            session_snapshot={
                "last_sql": state.get("last_sql"),
                "last_plan": state.get("last_plan"),
                "last_data": state.get("last_data"),
                "last_columns": state.get("last_columns"),
                "last_table": state.get("last_table"),
                "last_intent": state.get("last_intent"),
                "intent_chain": state.get("intent_chain"),
                "file_context": state.get("file_context"),
                "clarification_pending": state.get("clarification_pending"),
                "history": list(state.get("history", [])),
                "history_summary": state.get("history_summary"),
                "total_turns": state.get("total_turns"),
            }
        )

    if not is_clarification_only:
        # Append clarification Q&A pairs to history so they persist across sessions
        if metadata.get("clarification_qa"):
            qa_summary = " | ".join(
                f"Q: {p['question']} A: {p['answer']}" for p in metadata["clarification_qa"]
            )
            await session_mgr.append_history(state, "user", f"[Clarifications] {qa_summary}")
        await session_mgr.append_history(state, "assistant", full_text[:500])

    # Persist session title to DB if a pipeline generated one (first message)
    # Check multiple sources: session_state["title"], session_meta events, and _is_first_message flag
    session_title = state.get("title")
    for etype, ev in all_events:
        if etype == "session_meta":
            session_title = ev.get("session_title") or session_title
    logger.info(f"Title check: state_title='{state.get('title')}', resolved='{session_title}', is_first={is_first_message}, events={[(e, ev.get('session_title','')) for e,ev in all_events if e == 'session_meta']}")
    if session_title and session_title != settings.DEFAULT_SESSION_TITLE:
        try:
            from app.database import get_pool as _get_pool
            pool = _get_pool()
            schema = settings.APP_SCHEMA
            async with pool.acquire() as conn:
                await conn.execute(
                    f"UPDATE {schema}.chat_sessions SET title=$1 WHERE id=$2::uuid",
                    session_title, session_id,
                )
            logger.info(f"Session title persisted: '{session_title}' for {session_id}")
        except Exception as e:
            logger.warning(f"Failed to persist session title: {e}")

    # ── Implicit learning signal ────────────────────────────
    # Every time the pipeline returns real data (a `data` SSE event), record a
    # positive training signal: the user's query successfully mapped to a table.
    # Uses the last entry in sql_history (set by _update_session) to get the
    # original query text even for CLARIFICATION_REPLY re-runs.
    has_data_event = any(et == "data" for et, _ in all_events)
    if has_data_event and asst_msg_id and state.get("last_table"):
        from app.services.feedback_trainer import record_query_feedback
        from app.database import get_pool as _fb_pool
        sql_hist = state.get("sql_history", [])
        training_query = sql_hist[-1].get("message", message) if sql_hist else message
        try:
            await record_query_feedback(
                pool=_fb_pool(),
                schema=settings.APP_SCHEMA,
                session_id=session_id,
                message_id=asst_msg_id,
                query_text=training_query,
                resolved_table=state.get("last_table", ""),
                resolved_columns=state.get("last_columns", []),
                plan_json=state.get("last_plan", {}),
                rating=1,
            )
        except Exception as _fb_e:
            logger.debug(f"Implicit feedback record skipped: {_fb_e}")

    # Memory compression: compress old turns if history exceeds threshold
    from app.services.memory_compressor import maybe_compress
    await maybe_compress(state)

    try:
        await session_mgr.save(state)
    except Exception as e:
        logger.warning(f"Failed to save session state (client may have disconnected): {e}")

    log_pipeline_done(session_id=session_id, intent=intent)

    yield {"type": "typing_end"}
    yield {"type": "done", "message_id": asst_msg_id}
