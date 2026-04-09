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


async def _select_base_query(
    follow_up: str, sql_history: list[dict], session_state: dict, schema_graph: SchemaGraph
) -> tuple[str, dict]:
    """Given a follow-up message and query history, pick the best base query to amend.

    Example: If user asks about "customers in visakhapatnam" after queries:
      Q1: SELECT * FROM erp.erp_customers;                    ("get me all customers")
      Q2: SELECT * FROM erp.erp_customers WHERE state='AP';   ("what about those in AP")
      Q3: SELECT c.* FROM erp.erp_customers c JOIN erp.cart.. ("clients with 5+ cart items")

    The LLM should pick Q2 as the best base because visakhapatnam is a city in AP,
    so adding a city filter to Q2 makes more semantic sense than amending Q3.
    """
    from app.services.llm_client import chat_json

    # Build a numbered list of recent queries for the LLM
    history_lines = []
    for i, entry in enumerate(sql_history):
        msg = entry.get("message", "")
        sql = entry.get("sql", "")
        tables = ", ".join(entry.get("tables", []))
        history_lines.append(f"  Q{i+1}. \"{msg}\" → {sql}")

    history_text = "\n".join(history_lines)

    try:
        result = await chat_json([{"role": "user", "content":
            f'The user is asking a follow-up question about previous database queries.\n\n'
            f'Query history (oldest to newest):\n{history_text}\n\n'
            f'New follow-up: "{follow_up}"\n\n'
            f'Which previous query is the BEST BASE to amend for this follow-up?\n'
            f'Think about semantic relationships:\n'
            f'- If the follow-up adds a filter, which query has the right scope?\n'
            f'- If the follow-up narrows down results, which query is the right starting point?\n'
            f'- A city filter should build on a state/region query, not a JOIN query about a different topic\n'
            f'- "what about X" usually refers to the most recent query with a matching context\n'
            f'- If the follow-up is about filtering/refining and is vague (e.g. "filter by criteria"), '
            f'pick the simplest/broadest query that returns the main dataset (usually a SELECT * without JOINs)\n'
            f'- Prefer SELECT * queries over COUNT(*) queries when the user wants to filter records\n\n'
            f'Return JSON: {{"query_number": <1-based index>, "reason": "brief explanation"}}'}])
        chosen = result.get("query_number", len(sql_history))
        idx = max(0, min(chosen - 1, len(sql_history) - 1))
        logger.info(f"Base query selection: chose Q{idx+1} ({result.get('reason', '')})")
    except Exception as e:
        logger.warning(f"Base query selection failed: {e}, using last query")
        idx = len(sql_history) - 1

    entry = sql_history[idx]
    return entry["sql"], session_state.get("last_plan", {})


async def execute_db_query(message: str, session_state: dict, schema_graph: SchemaGraph,
                           is_follow_up: bool = False, user_name: str = "User"):
    """Execute the DB analysis pipeline. Yields SSE events."""
    settings = get_settings()
    schema = settings.POSTGRES_SCHEMA

    yield {"type": "step", "step_number": 1, "label": "Understanding your question..."}

    if is_follow_up and session_state.get("last_sql"):
        # ── Follow-up: Pick best base query from history, then amend ──
        import re as _re

        # Detect vague/open-ended filter requests — ask for specific criteria
        # instead of generating a bad SQL query from a vague question
        vague_filter = _re.search(
            r"\b(filter|narrow|refine|specific\s+criteria|certain\s+criteria|"
            r"would you like.*filter|filter.*by.*criteria|"
            r"based on.*criteria|by.*specific)\b",
            message, _re.IGNORECASE
        )
        # Only treat as vague if the message does NOT contain a concrete value/column
        # e.g. "filter by country India" is specific, "filter by specific criteria" is vague
        has_concrete_value = bool(_re.search(
            r"\b(=|is|equals|above|below|greater|less|more|than|in\s+\w{3,}|where)\b",
            message, _re.IGNORECASE
        ))
        if vague_filter and not has_concrete_value:
            # Use the SQL that generated the follow-up suggestions the user clicked.
            # This is deterministic — no LLM guessing needed.
            best_sql = session_state.get("last_follow_up_sql") or session_state.get("last_sql", "")
            columns = session_state.get("last_follow_up_columns") or session_state.get("last_columns", [])
            table_name = session_state.get("last_table", "")

            from app.builders.viz_config import filter_criteria_clarification
            clar = filter_criteria_clarification(columns, table_name)
            yield {"type": "clarification", "clarification": clar}
            session_state["clarification_pending"] = {
                "type": "filter_criteria",
                "sql": best_sql,                               # the correct SQL to amend
                "message": message,                            # original user message
            }
            return

        yield {"type": "step", "step_number": 2, "label": "Analyzing follow-up context..."}

        base_sql = session_state["last_sql"]
        plan_dict = session_state.get("last_plan", {})

        # If we have query history, ask LLM which query is the best base
        sql_history = session_state.get("sql_history", [])
        if len(sql_history) > 1:
            base_sql, plan_dict = await _select_base_query(message, sql_history, session_state, schema_graph)

        # Strip LIMIT from chosen base so the LLM works with the clean query
        base_sql = _re.sub(r"\s*LIMIT\s+\d+", "", base_sql, flags=_re.IGNORECASE)
        base_sql = base_sql.strip().rstrip(";") + ";"
        logger.info(f"Follow-up amend: base_sql={base_sql!r}, message={message!r}")
        sql = await amend_sql(base_sql, message, None, schema_graph)
        plan_dict = plan_dict
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
        # Skip values that were already resolved by a previous clarification
        from app.services.ambiguity_detector import detect_ambiguities
        filter_values = [f.get("value", "") for f in plan.filters if f.get("value")]
        resolved_ambiguities = session_state.get("resolved_ambiguities", [])
        ambiguity = await detect_ambiguities(message, plan.tables, filter_values, schema_graph, resolved_ambiguities)
        if ambiguity:
            yield {"type": "clarification", "clarification": ambiguity}
            session_state["clarification_pending"] = {
                "type": ambiguity["type"],
                "plan": plan_dict,
                "message": message,
                "token": ambiguity.get("token", ""),
            }
            return
        # Clear resolved ambiguities after successful query (no longer needed)
        session_state["resolved_ambiguities"] = []

        yield {"type": "step", "step_number": 4, "label": "Grounding filter values..."}
        yield {"type": "query_plan", "explanation": f"Tables: {', '.join(plan.tables)}. Filters: {json.dumps(plan.filters, default=str)}"}
        yield {"type": "step", "step_number": 5, "label": "Generating SQL..."}

        sql = await generate_sql(plan, schema, schema_graph)

    logger.info(f"Generated SQL: {sql}")

    # Detect aggregation queries (COUNT, SUM, AVG, etc.) — needed before EXPLAIN check
    import re as _re
    is_aggregation = bool(_re.search(
        r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", sql, _re.IGNORECASE
    )) and not _re.search(r"\bSELECT\s+\*\b", sql, _re.IGNORECASE)
    is_grouped_aggregation = is_aggregation and bool(_re.search(
        r"\bGROUP\s+BY\b", sql, _re.IGNORECASE
    ))

    # ── Stage 8a: EXPLAIN validation ───────────────────────
    explain_failed = False
    explain_error = None
    estimated_rows = None
    try:
        explain_result = await validate_before_execute(sql)
        explanation = explain_result.get("explanation", {})
        estimated_rows = explanation.get("estimated_rows")
        if not explain_result["ok"]:
            explain_failed = True
            explain_error = explain_result.get("warning", "EXPLAIN validation failed")
            logger.warning(f"EXPLAIN validation failed: {explain_error}")
        elif explain_result.get("warning"):
            logger.info(f"EXPLAIN warning: {explain_result['warning']}")
    except Exception as e:
        logger.debug(f"EXPLAIN validation skipped: {e}")

    # ── Stage 8a-ii: Pre-execution record limit check ─────
    # If EXPLAIN estimates more rows than the threshold, get the exact count.
    # Only ask record_limit clarification for >= 5000 rows.
    # For < 5000 rows, skip the limit question (will ask viz_type after execution).
    if not is_aggregation and estimated_rows and estimated_rows > settings.RECORD_WARN_THRESHOLD:
        # Get exact count instead of using EXPLAIN estimate
        import re as _re_count
        count_sql = _re_count.sub(r"SELECT\s+.*?\s+FROM", "SELECT COUNT(*) FROM", sql, count=1, flags=_re_count.IGNORECASE | _re_count.DOTALL)
        # Remove ORDER BY and LIMIT for count query
        count_sql = _re_count.sub(r"\s*ORDER BY[^;]*", "", count_sql, flags=_re_count.IGNORECASE)
        count_sql = _re_count.sub(r"\s*LIMIT\s+\d+", "", count_sql, flags=_re_count.IGNORECASE)
        try:
            count_result = await execute_sql(count_sql)
            if "error" not in count_result and count_result["rows"]:
                exact_count = list(count_result["rows"][0].values())[0]
                exact_count = int(exact_count) if exact_count else estimated_rows
            else:
                exact_count = estimated_rows
        except Exception:
            exact_count = estimated_rows

        # > 5000: ask record_limit (large dataset needs user confirmation)
        # <= 5000: skip limit question, let post-execution handle viz_type
        if exact_count > 5000:
            from app.builders.viz_config import record_limit_clarification
            clar = record_limit_clarification(exact_count, settings.RECORD_WARN_THRESHOLD)
            yield {"type": "clarification", "clarification": clar}
            session_state["clarification_pending"] = {
                "type": "record_limit", "sql": sql, "plan": plan_dict,
                "message": message, "token": "",
            }
            return

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

        # Give LLM the actual table/column definitions so it can fix FK references.
        # Include tables from the plan AND tables referenced in the failed SQL
        # (follow-ups may have added JOINs to tables not in the original plan).
        import re as _re2
        sql_tables = set(plan_dict.get("tables") or [])
        for tname in schema_graph.tables:
            if tname in sql.lower():
                sql_tables.add(tname)
        schema_info = {}
        for tname in sql_tables:
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
            from app.services.query_planner import _extract_sql, _ensure_schema_prefix
            fixed_sql = await llm_chat([{"role": "user", "content": retry_prompt}], temperature=0.1)
            fixed_sql = _extract_sql(fixed_sql)
            fixed_sql = _ensure_schema_prefix(fixed_sql, schema, set(schema_graph.tables.keys()))
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

    rows = result["rows"]
    columns = result["columns"]
    # Use actual row count for the summary (post-cap), but keep the DB total for the
    # record_limit clarification threshold check below.
    db_total_count = result["row_count"]
    row_count = len(rows)

    # Handle zero results — give a clear message instead of an empty response
    if row_count == 0:
        no_data_msg = f"No records were found matching your query. You might want to try different criteria or check if the data exists."
        yield {"type": "text_done", "content": no_data_msg}
        yield {"type": "follow_ups", "suggestions": [
            {"text": "Show me available data categories", "type": "chip", "icon": ""},
            {"text": "Try a broader search", "type": "chip", "icon": ""},
        ]}
        return

    # ── Clarification logic for non-aggregation queries ──────
    # <= 5000 rows: skip record_limit, ask viz_type directly (new OR follow-up)
    # > 5000 rows: ask record_limit first (user must confirm large loads)
    if not is_aggregation:
        if db_total_count > 5000:
            from app.builders.viz_config import record_limit_clarification
            clar = record_limit_clarification(db_total_count, settings.RECORD_WARN_THRESHOLD)
            yield {"type": "clarification", "clarification": clar}
            session_state["clarification_pending"] = {"type": "record_limit", "sql": sql, "plan": plan_dict, "message": message}
            return

        # <= 5000 rows: save session and ask viz_type
        await _update_session(session_state, sql, plan_dict, rows, columns, schema_graph, message)
        session_state["last_row_count"] = row_count
        from app.builders.viz_config import viz_type_clarification
        viz_clar = viz_type_clarification(columns)
        yield {"type": "clarification", "clarification": viz_clar}
        session_state["clarification_pending"] = {"type": "viz_type", "message": message}
        return

    # Aggregation queries — no clarification needed, just update session
    await _update_session(session_state, sql, plan_dict, rows, columns, schema_graph, message)
    session_state["last_row_count"] = row_count

    # Generate response text
    yield {"type": "step", "step_number": 7, "label": "Preparing response..."}

    # Detect yes/no questions for appropriate response style
    import re as _re
    is_yesno = bool(_re.match(r"^(is|are|does|do|can|has|have|was|were|will|should|did)\b", message.strip(), _re.IGNORECASE))

    # Build human-readable field labels from column names (strip underscores, title-case)
    def _friendly_cols(cols):
        return [c.replace('_', ' ').title() for c in cols]

    # Build a sanitized sample row using friendly labels (no raw column names)
    first_row_summary = ""
    if rows:
        friendly = _friendly_cols(columns)
        pairs = [f"{friendly[i]}: {str(v)}" for i, (k, v) in enumerate(rows[0].items()) if i < len(friendly)]
        first_row_summary = f"\nSample record: {', '.join(pairs)}"

    if is_yesno and rows:
        # Yes/no question — answer directly based on the actual data
        first_row = rows[0] if rows else {}
        friendly = _friendly_cols(list(first_row.keys()))
        sanitized = {friendly[i]: str(v) for i, (k, v) in enumerate(first_row.items()) if i < len(friendly)}
        response_prompt = f"""The user asked: "{message}"
Data found: {json.dumps(sanitized, default=str)}

Answer the yes/no question directly in 1-2 sentences based on the actual values returned.
- Compare the returned value against what the user asked about
- If it matches, answer "Yes" with the details
- If it doesn't match, answer "No" and tell them what the actual value is
Be conversational and concise.
IMPORTANT: Do NOT mention or reveal any database internals such as schema names, table names, column names, SQL queries, or technical implementation details.
End with FOLLOW_UPS: ["query1", "query2"] — suggest 2 actionable follow-up QUERIES the user could ask next about THIS data (e.g. "How many are in Bengaluru?", "Show customers with balance above 1 lakh"). Make them specific to the data returned, not generic."""
    elif is_grouped_aggregation and rows:
        # Grouped aggregation — summarize all groups using friendly labels
        friendly = _friendly_cols(columns)
        sanitized_rows = []
        for row in rows:
            sanitized_rows.append({friendly[i]: str(v) for i, (k, v) in enumerate(row.items()) if i < len(friendly)})
        all_rows_str = json.dumps(sanitized_rows, default=str)
        response_prompt = f"""The user asked: "{message}"
Found {row_count} groups:
{all_rows_str}

Summarize the breakdown in 2-4 sentences. Present each group and its count clearly using **bold** for numbers.
IMPORTANT: Do NOT mention or reveal any database internals such as schema names, table names, column names, SQL queries, or technical implementation details.
End with FOLLOW_UPS: ["query1", "query2"] — suggest 2 actionable follow-up QUERIES the user could ask next about THIS data (e.g. "How many are in Bengaluru?", "Show customers with balance above 1 lakh"). Make them specific to the data returned, not generic."""
    else:
        friendly_col_str = ', '.join(_friendly_cols(columns))
        response_prompt = f"""The user asked: "{message}"
Found {row_count} records with fields: {friendly_col_str}.{first_row_summary}
Write a brief 2-3 sentence summary of the results. Use markdown and mention key numbers.
IMPORTANT: Do NOT mention or reveal any database internals such as schema names, table names, column names, SQL queries, or technical implementation details. Use natural business language only.
End with FOLLOW_UPS: ["query1", "query2"] — suggest 2 actionable follow-up QUERIES the user could ask next about THIS data (e.g. "How many are in Bengaluru?", "Show customers with balance above 1 lakh"). Make them specific to the data returned, not generic."""

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
            sr[k] = v.isoformat() if hasattr(v, 'isoformat') else (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v)
        serializable_rows.append(sr)

    yield {"type": "data", "rows": serializable_rows, "columns": columns,
           "row_count": row_count, "sql": sql}
    yield {"type": "follow_ups", "suggestions": follow_ups}

    # Tag the SQL that generated these follow-ups so vague filter requests
    # can find the right base query even after other queries run in between
    session_state["last_follow_up_sql"] = sql
    session_state["last_follow_up_columns"] = columns

    # Title on first message
    if session_state.get("_is_first_message", False):
        title = await generate_title(message)
        yield {"type": "session_meta", "session_title": title}
        session_state["title"] = title


async def execute_db_query_with_sql(sql: str, plan_dict: dict, message: str,
                                    session_state: dict, schema_graph,
                                    user_name: str = "User"):
    """Execute a pre-built SQL query, save to session, then ask viz type clarification.
    Used after record_limit clarification — user has confirmed the row count."""
    yield {"type": "step", "step_number": 1, "label": "Executing query..."}

    result = await execute_sql(sql, uncapped=True)

    if "error" in result:
        err = get_error(result.get("code", "E002"), result["error"])
        err["type"] = "error"
        yield err
        return

    rows = result["rows"]
    columns = result["columns"]
    row_count = len(rows)

    # Cap rows for session storage (session can't hold millions of rows)
    # The full dataset goes to the frontend via the data event, but only a
    # subset is kept in session for follow-up context.
    session_rows = rows[:200]
    await _update_session(session_state, sql, plan_dict, session_rows, columns, schema_graph, message)
    session_state["last_row_count"] = row_count

    # Show only the viz type clarification — no data, no summary
    from app.builders.viz_config import viz_type_clarification
    viz_clar = viz_type_clarification(columns)
    yield {"type": "clarification", "clarification": viz_clar}
    session_state["clarification_pending"] = {"type": "viz_type", "message": message}


async def _update_session(state, sql, plan, rows, columns, schema_graph, message: str = ""):
    state["last_sql"] = sql
    state["last_plan"] = plan
    # Serialize rows so they are JSON-safe (date/datetime → str)
    # Cap at 200 rows for session storage — full data goes to frontend via SSE
    serialized = []
    for row in rows[:200]:
        sr = {}
        for k, v in row.items():
            sr[k] = v.isoformat() if hasattr(v, 'isoformat') else (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v)
        serialized.append(sr)
    state["last_data"] = serialized
    state["last_columns"] = columns
    # Extract primary table
    primary_table = None
    for tname in schema_graph.tables:
        if tname in sql.lower():
            state["last_table"] = tname
            if primary_table is None:
                primary_table = tname
            break

    # Append to sql_history so follow-ups can pick the best base query
    import re as _re_hist
    tables_in_sql = [t for t in schema_graph.tables if t in sql.lower()]
    has_join = bool(_re_hist.search(r"\bJOIN\b", sql, _re_hist.IGNORECASE))
    history = state.get("sql_history", [])
    history.append({
        "sql": sql,
        "message": message,
        "tables": tables_in_sql,
        "has_join": has_join,
        "columns": columns,
    })
    # Keep last 10 queries
    state["sql_history"] = history[-10:]
