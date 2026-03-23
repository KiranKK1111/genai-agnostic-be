"""EXPLAIN validator — runs EXPLAIN on generated SQL before execution."""
import logging
from app.database import get_pool

logger = logging.getLogger(__name__)


async def explain_sql(sql: str) -> dict:
    """Run EXPLAIN on a SQL query. Returns {valid, plan_text, estimated_rows, estimated_cost}."""
    pool = get_pool()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"EXPLAIN {sql}")
            plan_lines = [row[0] for row in rows]
            plan_text = "\n".join(plan_lines)

            # Parse estimated rows from first line
            estimated_rows = None
            estimated_cost = None
            if plan_lines:
                first = plan_lines[0]
                if "rows=" in first:
                    try:
                        estimated_rows = int(first.split("rows=")[1].split()[0].rstrip(")"))
                    except (ValueError, IndexError):
                        pass
                if "cost=" in first:
                    try:
                        cost_part = first.split("cost=")[1].split()[0]
                        estimated_cost = float(cost_part.split("..")[1] if ".." in cost_part else cost_part)
                    except (ValueError, IndexError):
                        pass

            return {"valid": True, "plan_text": plan_text,
                    "estimated_rows": estimated_rows, "estimated_cost": estimated_cost}
    except Exception as e:
        logger.warning(f"EXPLAIN failed: {e}")
        return {"valid": False, "error": str(e), "plan_text": "", "estimated_rows": None, "estimated_cost": None}
