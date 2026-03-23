"""Validate generated SQL using EXPLAIN before execution."""
import logging
from app.services.query_explainer import explain_sql

logger = logging.getLogger(__name__)


async def validate_before_execute(sql: str) -> dict:
    """Run EXPLAIN to validate SQL. Returns {ok, warning, explanation}."""
    result = await explain_sql(sql)

    if not result["valid"]:
        return {"ok": False, "warning": result.get("error", "EXPLAIN failed"),
                "explanation": result}

    warnings = []
    # Warn if estimated rows > 100k (may be slow)
    if result.get("estimated_rows") and result["estimated_rows"] > 100000:
        warnings.append(f"Estimated {result['estimated_rows']:,} rows — query may be slow")

    # Warn if cost is very high
    if result.get("estimated_cost") and result["estimated_cost"] > 50000:
        warnings.append(f"High estimated cost: {result['estimated_cost']:.0f}")

    return {
        "ok": True,
        "warning": "; ".join(warnings) if warnings else None,
        "explanation": result,
    }
