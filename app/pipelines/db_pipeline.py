"""Database analysis pipeline — query planning, execution, visualization."""
import json
import logging
from app.services.query_planner import build_query_plan, generate_sql, amend_sql
from app.services.sql_executor import execute_sql
from app.services.explain_validator import validate_before_execute
from app.services.llm_client import chat_stream
from app.services.response_beautifier import beautify, extract_follow_ups
from app.services.title_generator import generate_title
from app.services.schema_inspector import SchemaGraph
from app.services.error_registry import get_error
from app.config import get_settings

logger = logging.getLogger(__name__)


async def execute_db_query(message: str, session_state: dict, schema_graph: SchemaGraph,
                           is_follow_up: bool = False, user_name: str = "User"):
    """Execute the DB analysis pipeline. Yields SSE events."""
    settings = get_settings()
    schema = settings.POSTGRES_SCHEMA

    yield {"type": "step", "step_number": 1, "label": "Understanding your question..."}

    if is_follow_up and session_state.get("last_sql"):
        # ── Follow-up: Amend existing SQL ──────────────────
        import re as _re
        # Strip LIMIT from previous SQL so the LLM works with the clean base query
        base_sql = _re.sub(r"\s*LIMIT\s+\d+", "", session_state["last_sql"], flags=_re.IGNORECASE)
        base_sql = base_sql.strip().rstrip(";") + ";"
        logger.info(f"Follow-up amend: original_sql={base_sql!r}, message={message!r}")
        yield {"type": "step", "step_number": 2, "label": "Analyzing follow-up context..."}
        sql = await amend_sql(base_sql, message, None, schema_graph)
        plan_dict = session_state.get("last_plan", {})
    else:
        # ── New query: Full 8-stage pipeline ───────────────
        yield {"type": "step", "step_number": 2, "label": "Analyzing database schema..."}
        yield {"type": "step", "step_number": 3, "label": "Identifying tables and columns..."}

        plan = await build_query_plan(message, schema_graph, session_state)
        plan_dict = plan.to_dict()

        if not plan.tables:
            domain_name = schema_graph.domain_name or "this"
            oos_prompt = f"""The user asked: "{message}"
This request is outside the scope of this tool.

Respond politely in 2-3 sentences. You MUST include this exact line in your response:
"This tool is only designed for extracting information related to the {domain_name} domain."

Then briefly suggest they try rephrasing their question using {domain_name}-related terms.
Do NOT reveal any internal details like table names, column names, or schema structure."""
            try:
                oos_response = ""
                async for token in chat_stream([{"role": "user", "content": oos_prompt}]):
                    oos_response += token
                    yield {"type": "text_delta", "delta": token}
                yield {"type": "text_done", "content": oos_response}
            except Exception:
                yield {"type": "text_done", "content": f"This tool is only designed for extracting information related to the {domain_name} domain. Please try rephrasing your question."}
            return

        # ── Ambiguity detection ──────────────────────────────
        # Check if the query maps ambiguously to multiple tables/columns/values
        from app.services.ambiguity_detector import detect_ambiguities
        filter_values = [f.get("value", "") for f in plan.filters if f.get("value")]
        ambiguity = await detect_ambiguities(message, plan.tables, filter_values, schema_graph)
        if ambiguity:
            yield {"type": "clarification", "clarification": ambiguity}
            session_state["clarification_pending"] = {
                "type": ambiguity["type"],
                "plan": plan_dict,
                "message": message,
            }
            return

        yield {"type": "step", "step_number": 4, "label": "Grounding filter values..."}
        yield {"type": "query_plan", "explanation": f"Tables: {', '.join(plan.tables)}. Filters: {json.dumps(plan.filters, default=str)}"}
        yield {"type": "step", "step_number": 5, "label": "Generating SQL..."}

        sql = await generate_sql(plan, schema, schema_graph)

    logger.info(f"Generated SQL: {sql}")

    # ── Stage 8a: EXPLAIN validation ───────────────────────
    explain_failed = False
    explain_error = None
    try:
        explain_result = await validate_before_execute(sql)
        if not explain_result["ok"]:
            explain_failed = True
            explain_error = explain_result.get("warning", "EXPLAIN validation failed")
            logger.warning(f"EXPLAIN validation failed: {explain_error}")
        elif explain_result.get("warning"):
            logger.info(f"EXPLAIN warning: {explain_result['warning']}")
    except Exception as e:
        logger.debug(f"EXPLAIN validation skipped: {e}")

    # ── Stage 8b: Execute (skip if EXPLAIN already caught the error) ───
    yield {"type": "step", "step_number": 6, "label": "Executing query..."}

    if explain_failed:
        result = {"error": explain_error, "code": "E002"}
    else:
        result = await execute_sql(sql)

    # Error retry: send error + actual schema to LLM for correction
    if "error" in result and result.get("code") == "E002":
        logger.info(f"SQL error, attempting LLM retry: {result['error']}")
        from app.services.llm_client import chat as llm_chat

        # Give LLM the actual table/column definitions so it can fix FK references
        schema_info = {}
        for tname in (plan_dict.get("tables") or []):
            if tname in schema_graph.tables:
                schema_info[tname] = list(schema_graph.tables[tname].columns.keys())

        retry_prompt = f"""The following SQL query failed. Fix it using the ACTUAL column names provided.

SQL: {sql}
Error: {result['error']}

ACTUAL table columns (use ONLY these column names):
{json.dumps(schema_info, indent=2, default=str)}

Rules:
- CRITICAL: Only generate SELECT queries
- CRITICAL: Always prefix table names with schema: {schema}.table_name (e.g. {schema}.customers, {schema}.accounts)
- CRITICAL: Do NOT include SQL comments (-- or /* */)
- Fix the column/table error using the actual column names above
- Keep the same intent and filters
- Only JOIN tables that have matching FK columns
- Return ONLY the corrected SQL, no explanation"""
        try:
            from app.services.query_planner import _extract_sql
            fixed_sql = await llm_chat([{"role": "user", "content": retry_prompt}], temperature=0.1)
            fixed_sql = _extract_sql(fixed_sql)
            logger.info(f"Retry SQL: {fixed_sql}")
            result = await execute_sql(fixed_sql)
            if "error" not in result:
                sql = fixed_sql  # Use the fixed SQL going forward
        except Exception as e:
            logger.error(f"SQL retry failed: {e}")

    if "error" in result:
        err = get_error(result.get("code", "E002"), result["error"])
        err["type"] = "error"
        yield err
        return

    row_count = result["row_count"]
    rows = result["rows"]
    columns = result["columns"]

    # Detect aggregation queries (COUNT, SUM, AVG, etc.) — single value or grouped
    import re as _re
    is_aggregation = bool(_re.search(
        r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", sql, _re.IGNORECASE
    )) and not _re.search(r"\bSELECT\s+\*\b", sql, _re.IGNORECASE)
    is_grouped_aggregation = is_aggregation and bool(_re.search(
        r"\bGROUP\s+BY\b", sql, _re.IGNORECASE
    ))

    # Skip record limit and viz clarifications for aggregation queries
    if not is_aggregation:
        # Record count check (applies to both new queries and follow-ups)
        if row_count > settings.RECORD_WARN_THRESHOLD:
            from app.builders.viz_config import record_limit_clarification
            clar = record_limit_clarification(row_count, settings.RECORD_WARN_THRESHOLD)
            yield {"type": "clarification", "clarification": clar}
            session_state["clarification_pending"] = {"type": "record_limit", "sql": sql, "plan": plan_dict}
            return

    # Update session
    await _update_session(session_state, sql, plan_dict, rows, columns, schema_graph)

    # Ask viz type for follow-ups (skip for aggregation queries)
    if is_follow_up and not is_aggregation:
        from app.builders.viz_config import viz_type_clarification
        viz_clar = viz_type_clarification(columns)
        yield {"type": "clarification", "clarification": viz_clar}
        session_state["clarification_pending"] = {"type": "viz_type", "message": message}
        return

    # Generate response text
    yield {"type": "step", "step_number": 7, "label": "Preparing response..."}

    # Detect yes/no questions for appropriate response style
    import re as _re
    is_yesno = bool(_re.match(r"^(is|are|does|do|can|has|have|was|were|will|should|did)\b", message.strip(), _re.IGNORECASE))

    # Build row data for context
    first_row_str = ""
    if rows:
        first_row_str = f"\nFirst row data: {json.dumps({k: str(v) for k, v in rows[0].items()}, default=str)}"

    if is_yesno and rows:
        # Yes/no question — answer directly based on the actual data
        first_row = rows[0] if rows else {}
        actual_values = {k: str(v) for k, v in first_row.items()}
        response_prompt = f"""The user asked: "{message}"
Query returned: {json.dumps(actual_values, default=str)}

Answer the yes/no question directly in 1-2 sentences based on the actual values returned.
- Compare the returned value against what the user asked about
- If it matches, answer "Yes" with the details
- If it doesn't match, answer "No" and tell them what the actual value is (e.g. "No, this loan is currently CLOSED")
Be conversational and concise. Do NOT mention SQL, tables, column names, or technical details.
End with FOLLOW_UPS: ["suggestion"]"""
    elif is_grouped_aggregation and rows:
        # Grouped aggregation — summarize all groups in the response
        all_rows_str = json.dumps(
            [{k: str(v) for k, v in row.items()} for row in rows],
            default=str
        )
        response_prompt = f"""The user asked: "{message}"
Query returned {row_count} groups:
{all_rows_str}

Summarize the breakdown in 2-4 sentences. Present each group and its count clearly using **bold** for numbers.
Do NOT reveal table names, column names, or SQL details.
End with FOLLOW_UPS: ["suggestion"]"""
    else:
        response_prompt = f"""The user asked: "{message}"
SQL executed: {sql}
Returned {row_count} rows with columns: {', '.join(columns)}.{first_row_str}
Write a brief 2-3 sentence summary of the results. Use markdown and mention key numbers.
Do NOT reveal table names, column names, or SQL details.
End with FOLLOW_UPS: ["suggestion"]"""

    full_text = ""
    async for token in chat_stream([{"role": "user", "content": response_prompt}]):
        # Check for cancellation between tokens
        if session_state.get("cancel_requested"):
            from app.services.cancel_handler import handle_cancel
            cancel_event = await handle_cancel(session_state["session_id"], full_text, "")
            yield cancel_event
            return
        full_text += token
        yield {"type": "text_delta", "delta": token}

    full_text = beautify(full_text)
    follow_ups = extract_follow_ups(full_text, intent="DB_QUERY")
    if "FOLLOW_UPS:" in full_text:
        full_text = full_text.split("FOLLOW_UPS:")[0].strip()

    yield {"type": "text_done", "content": full_text}

    # Aggregation queries: summary is enough, no data table or follow-ups
    if is_aggregation:
        return

    # Serialize rows for JSON (handle datetime etc)
    serializable_rows = []
    for row in rows:
        sr = {}
        for k, v in row.items():
            sr[k] = str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
        serializable_rows.append(sr)

    yield {"type": "data", "rows": serializable_rows, "columns": columns,
           "row_count": row_count, "sql": sql}
    yield {"type": "follow_ups", "suggestions": follow_ups}

    # Title on first message
    if session_state.get("total_turns", 0) == 0:
        title = await generate_title(message)
        yield {"type": "session_meta", "session_title": title}
        session_state["title"] = title


async def execute_db_query_with_sql(sql: str, plan_dict: dict, message: str,
                                    session_state: dict, schema_graph,
                                    user_name: str = "User"):
    """Execute a pre-built SQL query, save to session, then ask viz type clarification.
    Used after record_limit clarification — data is NOT shown yet, only the viz picker."""
    yield {"type": "step", "step_number": 1, "label": "Executing query..."}

    result = await execute_sql(sql, uncapped=True)

    if "error" in result:
        err = get_error(result.get("code", "E002"), result["error"])
        err["type"] = "error"
        yield err
        return

    row_count = result["row_count"]
    rows = result["rows"]
    columns = result["columns"]

    # Save data to session so viz_type handler can retrieve it later
    await _update_session(session_state, sql, plan_dict, rows, columns, schema_graph)
    session_state["last_row_count"] = row_count

    # Show only the viz type clarification — no data, no summary
    from app.builders.viz_config import viz_type_clarification
    viz_clar = viz_type_clarification(columns)
    yield {"type": "clarification", "clarification": viz_clar}
    session_state["clarification_pending"] = {"type": "viz_type", "message": message}


async def _update_session(state, sql, plan, rows, columns, schema_graph):
    state["last_sql"] = sql
    state["last_plan"] = plan
    # Serialize rows so they are JSON-safe (date/datetime → str)
    serialized = []
    for row in rows:
        sr = {}
        for k, v in row.items():
            sr[k] = str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
        serialized.append(sr)
    state["last_data"] = serialized
    state["last_columns"] = columns
    # Extract primary table
    for tname in schema_graph.tables:
        if tname in sql.lower():
            state["last_table"] = tname
            break
