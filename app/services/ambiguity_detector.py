"""Ambiguity detection — detect when user queries map to multiple tables, columns, or values.

Uses FAISS similarity gap analysis:
    If the top-2 results are within AMBIGUITY_GAP of each other,
    and the top result exceeds MIN_SIMILARITY, the reference is ambiguous.

Returns clarification payloads compatible with the SSE clarification protocol.

New in agentic architecture:
    detect_entity_ambiguity() — uses EntityResolver to search column-level entries
    across ALL tables *before* table resolution, so the pipeline asks the right
    clarifying question immediately rather than guessing the wrong table.
"""
import logging
from app.services.embedder import embed_single
from app.services.vector_search import search
from app.services.schema_inspector import SchemaGraph

logger = logging.getLogger(__name__)

from app.config import get_settings as _get_settings


async def detect_entity_ambiguity(
    user_message: str,
    schema_graph: SchemaGraph,
    resolved_ambiguities: list[dict] | None = None,
) -> dict | None:
    """
    NEW: Pre-table-resolution entity ambiguity check.

    Parses the user message with NLU, extracts entity phrases, then uses
    EntityResolver to search column-level schema_idx across ALL tables.

    If any entity is found in multiple tables → returns a clarification payload
    with user-friendly table labels (no raw names exposed).

    This check runs BEFORE the query planner's LLM extraction, so ambiguous
    queries like "show case status distribution" immediately ask which report
    to use rather than silently picking the wrong table.

    resolved_ambiguities: list of {token, resolved_table} dicts from prior turns.
    """
    from app.services.query_nlu import parse_query
    from app.services.entity_resolver import resolve_all_entities

    resolved_ambiguities = resolved_ambiguities or []
    # Build mapping: entity_phrase → already-resolved table
    resolved_map: dict[str, str] = {}
    for r in resolved_ambiguities:
        tok = r.get("token", "").lower().replace("_", " ")
        table = r.get("resolved_table") or r.get("value", "")
        if tok and table:
            resolved_map[tok] = table

    # Parse NLU entities from the message (async LLM-driven)
    nlu = await parse_query(user_message)
    if not nlu.entities:
        return None

    # Resolve all entities via vector search
    resolutions, first_clarification = await resolve_all_entities(
        nlu.entities, schema_graph, resolved_tables=resolved_map
    )

    if first_clarification:
        logger.info(
            f"detect_entity_ambiguity | ambiguity found: "
            f"entity='{first_clarification.get('entity_phrase')}' "
            f"options={[o['label'] for o in first_clarification.get('options', [])]}"
        )
    return first_clarification


async def detect_ambiguous_table(token: str, schema_graph: SchemaGraph) -> dict | None:
    """Check if a token maps ambiguously to multiple tables.

    Example: "account" could match "accounts" AND "account_logs"
    Returns clarification payload if ambiguous, None if clear.
    """
    query_text = f"table {token}"
    query_emb = await embed_single(query_text)
    results = await search("schema_idx", query_emb, k=3, query_text=query_text)

    # Filter to table-type entries only
    table_results = [r for r in results if r["payload"].get("type") == "table"]

    if len(table_results) < 2:
        return None

    top = table_results[0]["similarity"]
    second = table_results[1]["similarity"]

    # Ambiguous if top-2 are close AND both are above minimum
    _s = _get_settings()
    if top - second < _s.AMBIGUITY_GAP and second > _s.MIN_SIMILARITY:
        options = []
        for r in table_results[:3]:
            tname = r["payload"].get("table", "")
            if tname in schema_graph.tables:
                tmeta = schema_graph.tables[tname]
                # Use friendly labels — no raw table/column names
                friendly_name = tname.replace('_', ' ').title()
                desc = tmeta.description if hasattr(tmeta, 'description') and tmeta.description else f"{friendly_name} data"
                options.append({
                    "value": tname,
                    "label": friendly_name,
                    "description": desc,
                    "similarity": round(r["similarity"], 3),
                })
        if len(options) >= 2:
            return {
                "type": "ambiguous_table",
                "mode": "single_select",
                "support_for_custom_replies": False,
                "question": f"I found multiple data categories that could match '{token}'. Which one did you mean?",
                "options": options,
                "token": token,
            }
    return None


async def detect_ambiguous_column(token: str, tables: list[str],
                                   schema_graph: SchemaGraph) -> dict | None:
    """Check if a column reference is ambiguous across resolved tables.

    Example: "balance" exists in both "accounts" and "loans"
    Returns clarification payload if ambiguous, None if clear.
    """
    matches = []
    for tname in tables:
        if tname not in schema_graph.tables:
            continue
        for cname, cinfo in schema_graph.tables[tname].columns.items():
            # Exact or substring match
            if token.lower() in cname or cname in token.lower():
                friendly_col = cname.replace('_', ' ').title()
                friendly_table = tname.replace('_', ' ').title()
                matches.append({
                    "value": f"{tname}.{cname}",
                    "label": f"{friendly_col} in {friendly_table}",
                    "description": "",
                    "table": tname,
                    "column": cname,
                })

    if len(matches) >= 2:
        # Check if all matches are the same column name (common in FK joins — not ambiguous)
        unique_columns = set(m["column"] for m in matches)
        if len(unique_columns) == 1:
            return None  # Same column across tables (e.g., customer_id) — not ambiguous

        return {
            "type": "ambiguous_column",
            "mode": "single_select",
            "support_for_custom_replies": False,
            "question": f"'{token}' could refer to multiple fields. Which one did you mean?",
            "options": matches[:4],
            "token": token,
        }
    return None


async def detect_ambiguous_value(value: str, tables: list[str],
                                  schema_graph: SchemaGraph) -> dict | None:
    """Check if a filter value exists in multiple columns across resolved tables.

    Example: "ACTIVE" exists in accounts.status AND loans.loan_status
    Returns clarification payload if ambiguous, None if clear.
    """
    from app.database import get_pool
    from app.config import get_settings

    settings = get_settings()
    pool = get_pool()
    db_schema = settings.POSTGRES_SCHEMA

    matches = []
    try:
        async with pool.acquire() as conn:
            for tname in tables:
                if tname not in schema_graph.tables:
                    continue
                for cname, cinfo in schema_graph.tables[tname].columns.items():
                    if cinfo["data_type"] not in ("character varying", "text", "varchar"):
                        continue
                    try:
                        row = await conn.fetchrow(
                            f"SELECT {cname} FROM {db_schema}.{tname} WHERE {cname} ILIKE $1 LIMIT 1",
                            value
                        )
                        if row:
                            friendly_col = cname.replace('_', ' ').title()
                            friendly_table = tname.replace('_', ' ').title()
                            matches.append({
                                "value": f"{tname}.{cname}",
                                "label": f"'{value}' in {friendly_table} ({friendly_col})",
                                "description": "",
                                "table": tname,
                                "column": cname,
                                "actual_value": row[cname],
                            })
                    except Exception:
                        continue
    except Exception:
        return None

    if len(matches) >= 2:
        return {
            "type": "ambiguous_value",
            "mode": "single_select",
            "support_for_custom_replies": False,
            "question": f"'{value}' was found in multiple places. Which one did you mean?",
            "options": matches[:4],
            "token": value,
        }
    return None


async def detect_confusable_tables(plan_tables: list[str],
                                   schema_graph: SchemaGraph) -> dict | None:
    """Detect when a resolved table has a sibling with a similar name.

    Example: plan resolves to "questionnaire_report_daily" but
    "questionnaire_report" also exists — user may have meant either.

    Returns clarification payload if confusable sibling found, None otherwise.
    """
    all_tables = set(schema_graph.tables.keys())

    for resolved in plan_tables:
        # Find sibling tables: tables where one name is a prefix of the other
        siblings = []
        for other in all_tables:
            if other == resolved:
                continue
            # Check if one is a prefix/substring of the other
            if resolved.startswith(other) or other.startswith(resolved):
                siblings.append(other)

        if not siblings:
            continue

        # Build clarification options
        options = []
        # The "base" table (shorter name) — presented as non-daily/snapshot
        # The "extended" table (longer name) — presented as daily/refreshing
        candidates = [resolved] + siblings
        for tname in sorted(candidates, key=len):
            tmeta = schema_graph.tables.get(tname)
            if not tmeta:
                continue
            friendly = tname.replace("_", " ").title()
            desc = tmeta.description if tmeta and tmeta.description else f"{friendly} data"
            # Use the first sentence of description as label, or the friendly name
            label = desc.split(".")[0].strip() if desc and len(desc.split(".")[0]) < 60 else friendly
            options.append({
                "value": tname,
                "label": label,
                "description": desc,
            })

        if len(options) >= 2:
            return {
                "type": "ambiguous_table",
                "mode": "single_select",
                "support_for_custom_replies": False,
                "question": "Are you willing to view the daily refreshing data?",
                "options": options,
                "token": min(candidates, key=len).replace("_", " "),
            }

    return None


async def detect_ambiguities(user_message: str, plan_tables: list[str],
                              filter_values: list[str],
                              schema_graph: SchemaGraph,
                              resolved_ambiguities: list[dict] = None) -> dict | None:
    """Run all ambiguity checks. Returns first clarification found, or None.

    Check order (most impactful first):
        0. Entity-level cross-table ambiguity (NLU + vector search — NEW)
        1. Confusable sibling tables (e.g., report vs report_daily)
        2. Ambiguous tables (single token matches 2+ tables)
        3. Ambiguous filter values
        4. Ambiguous columns

    resolved_ambiguities: list of previously resolved ambiguities to skip
        (prevents re-asking the same question after user already answered)
    """
    import re
    resolved_ambiguities = resolved_ambiguities or []
    # Build a set of tokens that were already resolved by previous clarifications
    resolved_tokens = {r.get("token", "").lower() for r in resolved_ambiguities if r.get("token")}

    # ── Check 0: Entity-level cross-table column ambiguity (agentic NLU pass) ──
    # This runs BEFORE table resolution to catch "case status found in multiple
    # tables" patterns that the downstream LLM extraction would silently pick one.
    entity_resolved = any(r.get("type") == "ambiguous_entity" for r in resolved_ambiguities)
    if not entity_resolved:
        clar = await detect_entity_ambiguity(user_message, schema_graph, resolved_ambiguities)
        if clar and clar.get("token", "").lower() not in resolved_tokens:
            return clar

    # Check for confusable sibling tables first (e.g., report vs report_daily)
    # Skip if user already resolved this ambiguity
    confusable_resolved = any(r.get("type") == "ambiguous_table" for r in resolved_ambiguities)
    if not confusable_resolved:
        clar = await detect_confusable_tables(plan_tables, schema_graph)
        if clar and clar.get("token", "").lower() not in resolved_tokens:
            return clar

    # Extract tokens that might be table references
    tokens = user_message.lower().split()
    for token in tokens:
        if len(token) < 3 or token in resolved_tokens:
            continue
        # Only check tokens that partially match 2+ tables
        matching_tables = [t for t in schema_graph.tables if token in t or t in token]
        if len(matching_tables) >= 2:
            clar = await detect_ambiguous_table(token, schema_graph)
            if clar:
                return clar

    # Identify tables the user explicitly mentioned (not just joined in)
    msg_tokens = {w.lower() for w in re.findall(r"\w+", user_message)}
    user_mentioned_tables = set()
    for tname in plan_tables:
        tbase = tname.rstrip("s")
        if tname in msg_tokens or tbase in msg_tokens or tname + "s" in msg_tokens:
            user_mentioned_tables.add(tname)
            continue
        # Check synonyms
        for token in msg_tokens:
            resolved = schema_graph.resolve_table(token)
            if resolved == tname:
                user_mentioned_tables.add(tname)
                break

    # Check ambiguous values (e.g., "ACTIVE" in multiple columns)
    # Skip values already resolved by previous clarification answers
    if len(plan_tables) >= 2 and filter_values:
        for val in filter_values:
            if val.lower() in resolved_tokens:
                logger.info(f"Skipping ambiguity check for '{val}' — already resolved by user")
                continue
            clar = await detect_ambiguous_value(val, plan_tables, schema_graph)
            if clar and user_mentioned_tables:
                # If ambiguous matches span user-mentioned and non-mentioned tables,
                # keep only matches in user-mentioned tables — not truly ambiguous
                mentioned_matches = [o for o in clar["options"]
                                     if o.get("table") in user_mentioned_tables]
                if len(mentioned_matches) == 1:
                    # Only one match in user's tables — not ambiguous
                    logger.info(f"Ambiguity resolved: '{val}' scoped to {mentioned_matches[0]['table']} (user-mentioned)")
                    continue
            if clar:
                return clar

    return None
