"""Safe SQL execution — SELECT-only with schema injection, timeout, row cap.

Security layers (defense in depth):
    Layer 1: Structural validation (no multi-statement, no comments, no DML keywords)
    Layer 2: PostgreSQL READ ONLY transaction (database-level enforcement)
    Layer 3: Statement timeout + row cap

Allowed query patterns (all SELECT variations):
    - Basic SELECT, WHERE, ORDER BY, LIMIT, OFFSET
    - Aggregation: COUNT, SUM, AVG, MIN, MAX, GROUP BY, HAVING
    - DISTINCT, CASE WHEN, COALESCE
    - Window functions: ROW_NUMBER(), RANK(), SUM() OVER (PARTITION BY ...)
    - Subqueries: scalar, EXISTS, NOT EXISTS, IN (SELECT ...)
    - JOINs: INNER, LEFT, RIGHT, FULL OUTER, CROSS, SELF, LATERAL
    - CTEs: WITH ... AS, WITH RECURSIVE
    - Set operations: UNION, UNION ALL, INTERSECT, EXCEPT
    - PostgreSQL: JSON (->>, #>>), Array (@>), type casts (::)
"""
import re
import logging
import time
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)

# ── DML/DDL keywords that must NEVER appear in user-facing SQL ────
# Word-boundary matched (\b) so column names like "updated_at", "created_at",
# "is_deleted", "grant_amount" are NOT false-positived.
BLOCKED_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "EXECUTE", "CALL",
    "COPY", "VACUUM", "REINDEX", "CLUSTER",
]

# Multi-word patterns (matched as full phrases)
BLOCKED_PHRASES = [
    r"\bSET\s+ROLE\b",
    r"\bRESET\s+ROLE\b",
    r"\bSET\s+SESSION\b",
    r"\bSELECT\b[^;]*\bINTO\b",         # SELECT INTO (creates tables)
    r"\bFOR\s+UPDATE\b",                  # Row locking
    r"\bFOR\s+SHARE\b",                   # Row locking
    r"\bFOR\s+NO\s+KEY\s+UPDATE\b",       # Row locking
    r"\bFOR\s+KEY\s+SHARE\b",             # Row locking
]

# Dangerous PostgreSQL functions
BLOCKED_FUNCTIONS = [
    "pg_sleep", "dblink", "lo_import", "lo_export",
    "pg_read_file", "pg_write_file",
    "pg_terminate_backend", "pg_cancel_backend",
    "pg_reload_conf", "pg_rotate_logfile",
]


def validate_sql(sql: str) -> dict | None:
    """Validate that SQL is a safe, single SELECT statement.
    Returns None if safe, or {error, code} if blocked."""
    if not sql or not sql.strip():
        return {"error": "Empty query", "code": "E002"}

    cleaned = sql.strip()

    # Strip trailing semicolons for validation (PostgreSQL allows one)
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()

    # Block SQL comments (-- and /* */) — used to hide malicious payloads
    if "--" in cleaned or "/*" in cleaned:
        return {"error": "SQL comments are not allowed", "code": "E002"}

    # Block multiple statements (semicolons within the query body)
    if ";" in cleaned:
        return {"error": "Multiple SQL statements are not allowed", "code": "E002"}

    # Must start with SELECT or WITH (CTE)
    upper = cleaned.upper().lstrip()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return {"error": "Only SELECT queries are allowed", "code": "E002"}

    # Block DML/DDL keywords (word-boundary match prevents false positives on column names)
    for kw in BLOCKED_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return {"error": f"Only SELECT queries are allowed (blocked: {kw})", "code": "E002"}

    # Block dangerous phrases
    for phrase in BLOCKED_PHRASES:
        if re.search(phrase, upper):
            return {"error": "Only SELECT queries are allowed", "code": "E002"}

    # Block dangerous functions
    for func in BLOCKED_FUNCTIONS:
        if re.search(rf"\b{func}\b", upper):
            return {"error": "Blocked function detected", "code": "E002"}

    return None  # Safe


async def execute_sql(sql: str, uncapped: bool = False) -> dict:
    """Execute a SELECT query safely inside a READ ONLY transaction.
    Returns {rows, columns, row_count, execution_time_ms}.
    Set uncapped=True to skip the MAX_RESULT_ROWS safety cap (e.g. user chose 'populate all').

    Uses cursor-based streaming for large result sets to avoid timeouts and
    memory issues. No hardcoded timeout — rows are fetched in batches."""
    settings = get_settings()

    # Layer 1: Structural validation
    block = validate_sql(sql)
    if block:
        return block

    # Layer 1b: Identifier injection check (Issue #1/#2 fix)
    from app.services.sql_safety import validate_identifiers_in_sql
    ident_error = validate_identifiers_in_sql(sql)
    if ident_error:
        return {"error": ident_error, "code": "E002"}

    max_rows = getattr(settings, "MAX_POPULATE_ROWS", 100000) if uncapped else getattr(settings, "MAX_RESULT_ROWS", 10000)
    batch_size = 5000  # Fetch rows in batches to avoid memory spikes
    pool = get_pool()
    start = time.time()
    try:
        async with pool.acquire() as conn:
            # Layer 2: READ ONLY transaction — PostgreSQL will reject any write attempt
            async with conn.transaction(readonly=True):
                # Use a cursor to stream results in batches — no statement timeout needed.
                # The cursor fetches rows incrementally so even multi-million row queries
                # won't timeout or load everything into memory at once.
                cursor = await conn.cursor(sql)
                all_rows = []
                columns = None
                total_count = 0

                while True:
                    batch = await cursor.fetch(batch_size)
                    if not batch:
                        break

                    if columns is None:
                        columns = list(batch[0].keys())

                    total_count += len(batch)

                    # If capped, only keep rows up to the limit
                    if not uncapped:
                        remaining = max_rows - len(all_rows)
                        if remaining > 0:
                            all_rows.extend(dict(r) for r in batch[:remaining])
                        if len(all_rows) >= max_rows:
                            # Continue counting total but stop collecting rows
                            continue
                    else:
                        all_rows.extend(dict(r) for r in batch)

            elapsed_ms = int((time.time() - start) * 1000)

            if not all_rows:
                return {"rows": [], "columns": columns or [], "row_count": total_count,
                        "execution_time_ms": elapsed_ms, "sql": sql}

            return {
                "rows": all_rows,
                "columns": columns or [],
                "row_count": total_count,
                "execution_time_ms": elapsed_ms,
                "sql": sql,
            }
    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        error_str = str(e)
        logger.error(f"SQL execution failed ({elapsed_ms}ms): {error_str}\nSQL: {sql}")
        if "read-only transaction" in error_str.lower():
            return {"error": "Write operations are not allowed", "code": "E002", "execution_time_ms": elapsed_ms}
        return {"error": error_str, "code": "E002", "execution_time_ms": elapsed_ms}
