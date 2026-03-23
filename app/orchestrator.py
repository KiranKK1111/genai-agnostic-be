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
from app.config import get_settings
from typing import AsyncIterator

logger = logging.getLogger(__name__)

# Global schema graph (loaded at startup)
_schema_graph: SchemaGraph | None = None

def set_schema_graph(graph: SchemaGraph):
    global _schema_graph
    _schema_graph = graph

def get_schema_graph() -> SchemaGraph:
    return _schema_graph


async def process_message(message: str, session_id: str = None, user_id: str = None,
                          user_name: str = "User", file_path: str = None,
                          file_name: str = None) -> AsyncIterator[dict]:
    """Main entry point. Processes a user message and yields SSE events."""
    settings = get_settings()
    session_mgr = SessionManager()
    state = await session_mgr.get_or_create(session_id, user_id)
    session_id = state["session_id"]

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

    # ── Step 3: Scope Validation ───────────────────────────
    scope = validate_scope(message)
    if not scope["in_scope"]:
        yield {"type": "step", "step_number": 1, "label": "Understanding your question..."}
        from app.services.llm_client import chat
        rejection = await chat([{"role": "user", "content": f"The user asked: \"{message}\"\nPolitely explain that you are designed only for database queries, file analysis, and conversation. You cannot: write code, generate images, browse the web. Keep it brief and friendly (2 sentences). Use an emoji."}])
        yield {"type": "text_done", "content": rejection}
        yield {"type": "follow_ups", "suggestions": [
            {"text": "Show me available tables", "type": "chip", "icon": "📊"},
            {"text": "What can you do?", "type": "question", "icon": "❓"},
        ]}
        yield {"type": "typing_end"}
        msg_id = await session_mgr.save_message(session_id, "user", message)
        await session_mgr.save_message(session_id, "assistant", rejection, parent_message_id=msg_id)
        yield {"type": "done", "message_id": msg_id}
        return

    # ── Step 4: Intent Classification ──────────────────────
    has_file = file_path is not None
    intent_result = await classify_intent(message, state, has_file=has_file)
    intent = intent_result["intent"]
    logger.info(f"Intent: {intent} (confidence={intent_result['confidence']}, method={intent_result['method']})")

    state["last_intent"] = intent
    state["intent_chain"] = (state.get("intent_chain", []) + [intent])[-10:]

    # Save user message
    user_msg_id = await session_mgr.save_message(session_id, "user", message)
    await session_mgr.append_history(state, "user", message)

    # ── Step 5: Pipeline Dispatch ──────────────────────────
    full_text = ""
    all_events = []

    if intent == "CHAT":
        async for event in execute_chat(message, state, user_name):
            all_events.append(event)
            yield event
            if event.get("type") == "text_done":
                full_text = event.get("content", "")

    elif intent in ("DB_QUERY", "DB_FOLLOW_UP"):
        is_follow = intent == "DB_FOLLOW_UP"
        async for event in execute_db_query(message, state, get_schema_graph(), is_follow_up=is_follow, user_name=user_name):
            all_events.append(event)
            yield event
            if event.get("type") == "text_done":
                full_text = event.get("content", "")
            if event.get("type") == "error":
                await log_query(user_id=user_id, session_id=session_id, user_prompt=message,
                               intent=intent, status="ERROR", error=event.get("message"))

    elif intent == "FILE_ANALYSIS" and file_path:
        async for event in execute_file_upload(file_path, file_name, message, state, user_name):
            all_events.append(event)
            yield event
            if event.get("type") == "text_done":
                full_text = event.get("content", "")

    elif intent == "FILE_FOLLOW_UP":
        async for event in execute_file_followup(message, state):
            all_events.append(event)
            yield event
            if event.get("type") == "text_done":
                full_text = event.get("content", "")

    elif intent == "VIZ_FOLLOW_UP":
        # Reuse cached data
        if state.get("last_data") and state.get("last_columns"):
            from app.pipelines.viz_engine import build_viz_clarification
            clar = build_viz_clarification(state["last_columns"])
            yield {"type": "clarification", "clarification": clar}
            state["clarification_pending"] = {"type": "viz_type"}
        else:
            yield {"type": "text_done", "content": "I don't have previous data to visualize. Please run a query first."}
            full_text = "I don't have previous data to visualize."

    elif intent == "HYBRID":
        async for event in execute_hybrid(message, state):
            all_events.append(event)
            yield event
            if event.get("type") == "text_done":
                full_text = event.get("content", "")

    elif intent == "CLARIFICATION_REPLY":
        # Re-run the pipeline with the user's clarification applied
        pending = state.get("clarification_pending") or {}
        response = state.get("clarification_response") or {}
        selected = response.get("selected_values")
        clar_type = pending.get("type", "")
        original_message = pending.get("message", message)

        # Frontend may send clarification as a regular chat message instead of /clarify.
        # Extract the selection from the message text if clarification_response wasn't set.
        if not selected and clar_type == "record_limit":
            msg_lower = message.strip().lower()
            if msg_lower in ("limited", "all"):
                selected = msg_lower
                logger.info(f"Record limit clarification: selected='{selected}' from chat message")
        if not selected and clar_type == "viz_type":
            msg_lower = message.strip().lower()
            if msg_lower in ("table", "bar", "pie", "line"):
                selected = msg_lower
                logger.info(f"Viz type clarification: selected='{selected}' from chat message")

        # Clear clarification state FIRST (prevents infinite loop)
        state["clarification_pending"] = None
        state["clarification_response"] = None

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
                clarified_message = f"{original_message} (filter on {resolved})"
            else:
                clarified_message = original_message

        elif clar_type == "viz_type":
            # User chose visualization type(s) — render data with summary
            if isinstance(selected, list):
                viz_types = selected
            elif isinstance(selected, str):
                viz_types = [s.strip() for s in selected.split(",")]
            else:
                viz_types = ["table"]
            primary_viz = viz_types[0] if viz_types else "table"

            if state.get("last_data") and state.get("last_columns"):
                from app.pipelines.viz_engine import build_viz_config
                from app.services.llm_client import chat_stream
                from app.services.response_beautifier import beautify, extract_follow_ups

                rows = state["last_data"]
                columns = state["last_columns"]
                row_count = state.get("last_row_count", len(rows))
                sql = state.get("last_sql", "")

                # Generate summary via LLM
                yield {"type": "step", "step_number": 1, "label": "Preparing response..."}
                response_prompt = f"The user asked: \"{original_message}\"\nSQL executed: {sql}\nReturned {row_count} rows with columns: {', '.join(columns)}.\nWrite a brief 2-3 sentence summary of the results. Use markdown and mention key numbers. End with FOLLOW_UPS: [\"suggestion\"]"
                full_text = ""
                async for token in chat_stream([{"role": "user", "content": response_prompt}]):
                    full_text += token
                    yield {"type": "text_delta", "delta": token}

                full_text = beautify(full_text)
                follow_ups = extract_follow_ups(full_text, intent="DB_QUERY")
                if "FOLLOW_UPS:" in full_text:
                    full_text = full_text.split("FOLLOW_UPS:")[0].strip()
                yield {"type": "text_done", "content": full_text}

                # Send viz config
                viz_cfg = build_viz_config(primary_viz, columns)
                viz_cfg["available_views"] = viz_types
                yield {"type": "viz_config", "config": viz_cfg}

                # Send data
                yield {"type": "data", "rows": rows, "columns": columns,
                       "row_count": row_count, "sql": sql, "truncated": False}
                yield {"type": "follow_ups", "suggestions": follow_ups}
            else:
                yield {"type": "text_done", "content": "I don't have previous data to visualize. Please run a query first."}
                full_text = "I don't have previous data to visualize."
            handled = True

        elif clar_type == "record_limit":
            stored_sql = pending.get("sql")
            stored_plan = pending.get("plan", {})
            if stored_sql and selected in ("limited", "all"):
                # Re-use the already-generated SQL; just add LIMIT for "limited"
                if selected == "limited":
                    limit_val = get_settings().RECORD_WARN_THRESHOLD
                    final_sql = stored_sql.rstrip().rstrip(";") + f" LIMIT {limit_val};"
                else:
                    final_sql = stored_sql
                state["last_sql"] = final_sql
                state["last_plan"] = stored_plan
                from app.pipelines.db_pipeline import execute_db_query_with_sql
                async for event in execute_db_query_with_sql(
                    final_sql, stored_plan, original_message, state, _schema_graph, user_name=user_name
                ):
                    all_events.append(event)
                    yield event
                    if event.get("type") == "text_done":
                        full_text = event.get("content", "")
                    if event.get("type") == "error":
                        await log_query(user_id=user_id, session_id=session_id,
                                       user_prompt=original_message, intent=intent,
                                       status="ERROR", error=event.get("message"))
                handled = True
            else:
                # Fallback: re-run the original query
                clarified_message = original_message

        if not handled:
            # If no clarification_response was set (user sent a plain message instead of
            # using /clarify), treat the current message as a regular DB query
            if not clarified_message:
                clarified_message = message

            # Re-run DB pipeline
            if _schema_graph:
                async for event in execute_db_query(clarified_message, state, _schema_graph,
                                                    is_follow_up=False, user_name=user_name):
                    all_events.append(event)
                    yield event
                    if event.get("type") == "text_done":
                        full_text = event.get("content", "")
                    if event.get("type") == "error":
                        await log_query(user_id=user_id, session_id=session_id,
                                       user_prompt=clarified_message, intent=intent,
                                       status="ERROR", error=event.get("message"))
            else:
                yield {"type": "text_done", "content": "I couldn't process that. Could you try rephrasing your question?"}
                full_text = "I couldn't process that."

    else:
        # Default to chat
        async for event in execute_chat(message, state, user_name):
            all_events.append(event)
            yield event
            if event.get("type") == "text_done":
                full_text = event.get("content", "")

    # ── Post-LLM output scan ──────────────────────────────
    if full_text:
        output_scan = scan_output(full_text)
        if not output_scan["is_safe"]:
            logger.warning(f"Post-LLM scan flagged: {output_scan['issues']}")

    # Save assistant message
    follow_ups = []
    metadata = {}
    content_sql = None
    for ev in all_events:
        if ev.get("type") == "follow_ups":
            follow_ups = ev.get("suggestions", [])
        if ev.get("type") == "data":
            metadata["data"] = {"row_count": ev.get("row_count"), "columns": ev.get("columns")}
            content_sql = ev.get("sql")

    asst_msg_id = await session_mgr.save_message(
        session_id, "assistant", full_text,
        metadata=metadata, follow_ups=follow_ups,
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
            "history_summary": state.get("history_summary"),
            "total_turns": state.get("total_turns"),
        }
    )

    await session_mgr.append_history(state, "assistant", full_text[:500])

    # Memory compression: compress old turns if history exceeds threshold
    from app.services.memory_compressor import maybe_compress
    await maybe_compress(state)

    await session_mgr.save(state)

    yield {"type": "typing_end"}
    yield {"type": "done", "message_id": asst_msg_id}
