"""SQL dialect helpers — PostgreSQL-specific query adjustments."""


def qualify_tables(sql: str, schema: str) -> str:
    """Ensure all table references use the schema prefix.
    Simple heuristic: if schema. is not before a known table name, add it."""
    # This is a lightweight helper; the main SQL generation in query_planner
    # already prefixes schema. This catches edge cases in follow-up amendments.
    if f"{schema}." not in sql and schema != "public":
        import re
        # Add schema before FROM/JOIN table references
        sql = re.sub(r"(FROM|JOIN)\s+(?!" + schema + r"\.)([a-zA-Z_][a-zA-Z0-9_]*)",
                     rf"\1 {schema}.\2", sql, flags=re.IGNORECASE)
    return sql


def add_limit(sql: str, limit: int) -> str:
    """Add LIMIT to a query if not already present."""
    if "LIMIT" not in sql.upper():
        sql = sql.rstrip(";").strip() + f" LIMIT {limit};"
    return sql


def remove_order_by(sql: str) -> str:
    """Remove ORDER BY clause (used for COUNT queries)."""
    import re
    return re.sub(r"\s+ORDER BY[^;]*", "", sql, flags=re.IGNORECASE)
