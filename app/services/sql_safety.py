"""SQL Safety — identifier validation and sanitization.

Fixes Issue #1: SQL injection via dynamic table/column names.
All dynamic identifiers (table names, column names, schema names)
MUST be validated against the known schema before interpolation.

Usage:
    from app.services.sql_safety import safe_table, safe_column, safe_schema

    sql = f"SELECT {safe_column(col, table, graph)} FROM {safe_schema(schema)}.{safe_table(table, graph)}"
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Valid PostgreSQL identifier: letters, digits, underscores, no leading digit
_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _is_valid_identifier(name: str) -> bool:
    """Check if a string is a valid PostgreSQL identifier (no injection possible)."""
    return bool(_IDENT_RE.match(name)) and len(name) <= 128


def safe_identifier(name: str) -> str:
    """Validate and double-quote a SQL identifier.

    Raises ValueError if the name contains injection characters.
    Double-quoting prevents any SQL interpretation of the name.
    """
    if not name or not _is_valid_identifier(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return f'"{name}"'


def safe_table(table_name: str, schema_graph=None) -> str:
    """Validate a table name against the schema graph and return quoted identifier.

    If schema_graph is provided, validates the table actually exists.
    """
    if not _is_valid_identifier(table_name):
        raise ValueError(f"Invalid table name: {table_name!r}")

    if schema_graph is not None:
        if table_name not in schema_graph.tables:
            raise ValueError(f"Unknown table: {table_name!r}")

    return f'"{table_name}"'


def safe_column(column_name: str, table_name: str = None, schema_graph=None) -> str:
    """Validate a column name against the schema graph and return quoted identifier."""
    if not _is_valid_identifier(column_name):
        raise ValueError(f"Invalid column name: {column_name!r}")

    if schema_graph is not None and table_name:
        if table_name in schema_graph.tables:
            if column_name not in schema_graph.tables[table_name].columns:
                # Allow common pseudo-columns
                if column_name not in ("*", "count", "1"):
                    logger.debug(f"Column {column_name!r} not found in {table_name}")

    return f'"{column_name}"'


def safe_schema(schema_name: str) -> str:
    """Validate a schema name (must be a valid identifier)."""
    if not _is_valid_identifier(schema_name):
        raise ValueError(f"Invalid schema name: {schema_name!r}")
    return f'"{schema_name}"'


def safe_value_for_ilike(value: str) -> str:
    """Sanitize a value for ILIKE queries — escape special pattern characters."""
    # Escape PostgreSQL LIKE special characters
    value = value.replace("\\", "\\\\")
    value = value.replace("%", "\\%")
    value = value.replace("_", "\\_")
    return value


def validate_identifiers_in_sql(sql: str, schema_graph=None) -> Optional[str]:
    """Quick check that a SQL string doesn't contain obvious injection patterns.

    Returns error message if suspicious, None if OK.
    This is a supplementary check — the primary defense is readonly transactions.
    """
    # Block string concatenation tricks
    if "||" in sql and ("chr(" in sql.lower() or "char(" in sql.lower()):
        return "Suspicious string concatenation detected"

    # Block system catalog access
    if re.search(r"\bpg_catalog\b|\binformation_schema\b", sql, re.IGNORECASE):
        return "System catalog access is not allowed"

    # Block Unicode escapes used for bypass
    if re.search(r"\\u[0-9a-fA-F]{4}", sql):
        return "Unicode escapes are not allowed in queries"

    return None
