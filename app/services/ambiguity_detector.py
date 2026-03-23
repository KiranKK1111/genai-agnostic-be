"""Ambiguity detection — detect when user queries map to multiple tables, columns, or values.

Uses FAISS similarity gap analysis:
    If the top-2 results are within AMBIGUITY_GAP of each other,
    and the top result exceeds MIN_SIMILARITY, the reference is ambiguous.

Returns clarification payloads compatible with the SSE clarification protocol.
"""
import logging
from app.services.embedder import embed_single
from app.services.vector_search import search
from app.services.schema_inspector import SchemaGraph

logger = logging.getLogger(__name__)

# If top-2 similarity scores are within this gap, it's ambiguous
AMBIGUITY_GAP = 0.12
# Minimum similarity to consider a result relevant at all
MIN_SIMILARITY = 0.50


async def detect_ambiguous_table(token: str, schema_graph: SchemaGraph) -> dict | None:
    """Check if a token maps ambiguously to multiple tables.

    Example: "account" could match "accounts" AND "account_logs"
    Returns clarification payload if ambiguous, None if clear.
    """
    query_emb = await embed_single(f"table {token}")
    results = await search("schema_idx", query_emb, k=3)

    # Filter to table-type entries only
    table_results = [r for r in results if r["payload"].get("type") == "table"]

    if len(table_results) < 2:
        return None

    top = table_results[0]["similarity"]
    second = table_results[1]["similarity"]

    # Ambiguous if top-2 are close AND both are above minimum
    if top - second < AMBIGUITY_GAP and second > MIN_SIMILARITY:
        options = []
        for r in table_results[:3]:
            tname = r["payload"].get("table", "")
            if tname in schema_graph.tables:
                tmeta = schema_graph.tables[tname]
                col_preview = ", ".join(list(tmeta.columns.keys())[:5])
                options.append({
                    "value": tname,
                    "label": tname,
                    "description": f"{len(tmeta.columns)} columns: {col_preview}...",
                    "similarity": round(r["similarity"], 3),
                })
        if len(options) >= 2:
            return {
                "type": "ambiguous_table",
                "mode": "single_select",
                "question": f"I found multiple tables that could match '{token}'. Which one did you mean?",
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
                matches.append({
                    "value": f"{tname}.{cname}",
                    "label": f"{tname}.{cname}",
                    "description": f"type: {cinfo['data_type']}",
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
            "question": f"'{token}' could refer to multiple columns. Which one did you mean?",
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
                            matches.append({
                                "value": f"{tname}.{cname}",
                                "label": f"'{value}' in {tname}.{cname}",
                                "description": f"Filter {tname}.{cname} = '{row[cname]}'",
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
            "question": f"'{value}' was found in multiple columns. Which one did you mean?",
            "options": matches[:4],
            "token": value,
        }
    return None


async def detect_ambiguities(user_message: str, plan_tables: list[str],
                              filter_values: list[str],
                              schema_graph: SchemaGraph) -> dict | None:
    """Run all ambiguity checks. Returns first clarification found, or None.

    Check order (most impactful first):
        1. Ambiguous tables
        2. Ambiguous filter values
        3. Ambiguous columns
    """
    import re

    # Extract tokens that might be table references
    tokens = user_message.lower().split()
    for token in tokens:
        if len(token) < 3:
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
    if len(plan_tables) >= 2 and filter_values:
        for val in filter_values:
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
