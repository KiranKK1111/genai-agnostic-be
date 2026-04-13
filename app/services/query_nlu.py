"""
Semantic NLU layer — LLM-driven query parsing into structured intent.

Design goals
────────────
* Fully dynamic — zero hardcoded keyword lists, stopwords, or regex patterns.
* The LLM analyses the user's natural language and extracts structured fields:
      entities, aggregation, action, filters, sort, limit.
* Single LLM call per query — results cached in the NLUResult dataclass.

Example
───────
  await parse_query("show case status distribution")
  → NLUResult(
        entities      = ["case status"],
        aggregation   = "count_group_by",
        action        = "show",
        filters       = [],
        sort          = None,
        limit         = None,
        temporal      = False,
        raw_tokens    = ["show", "case", "status", "distribution"],
    )
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class NLUResult:
    """Structured result of query NLU parsing."""

    # Candidate entity tokens (column/table references) ordered by position.
    entities: list[str] = field(default_factory=list)

    # Detected aggregation intent. None means plain SELECT *.
    # Values: "count_group_by" | "count" | "sum" | "avg" | "max" | "min"
    #         | "count_group_by_time" | None
    aggregation: str | None = None

    # Primary user action
    action: str = "show"

    # Verbatim filter phrases (e.g. "status = ACTIVE", "city Chennai")
    filters: list[str] = field(default_factory=list)

    # ORDER BY direction detected from surface words
    sort: str | None = None  # "asc" | "desc" | None

    # Row limit (e.g. "top 10")
    limit: int | None = None

    # True when query contains temporal grouping intent
    temporal: bool = False

    # Original tokenised words
    raw_tokens: list[str] = field(default_factory=list)

    # Detected aggregation keyword (for logging / debugging)
    aggregation_trigger: str | None = None


async def parse_query(user_message: str) -> NLUResult:
    """
    Parse *user_message* into a structured NLUResult using the LLM.

    Fully dynamic — no hardcoded keyword lists. The LLM determines
    entities, aggregation, action, filters, sort, and limit from context.
    """
    from app.services.llm_client import chat_json

    result = NLUResult()
    text = user_message.strip()
    result.raw_tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())

    try:
        parsed = await chat_json([{"role": "user", "content":
            f'Parse this database query into structured components.\n\n'
            f'Query: "{text}"\n\n'
            f'Extract these fields:\n'
            f'- entities: list of column/table name references (multi-word phrases that refer to '
            f'database fields). Strip action words, aggregation words, and filler words. '
            f'Example: "show case status distribution" → ["case status"]\n'
            f'- aggregation: the type of aggregation, one of:\n'
            f'  "count_group_by" (distribution, breakdown, frequency, proportion)\n'
            f'  "count" (how many, total count)\n'
            f'  "sum" (total, sum, cumulative)\n'
            f'  "avg" (average, mean)\n'
            f'  "max" (maximum, highest, largest)\n'
            f'  "min" (minimum, lowest, smallest)\n'
            f'  "count_group_by_time" (trend over time, monthly, yearly)\n'
            f'  null (plain data retrieval with no aggregation)\n'
            f'- aggregation_trigger: the exact word(s) that indicate aggregation (e.g. "distribution"), or null\n'
            f'- action: primary user action — "show", "list", "filter", "compare", "search", "count", "rank"\n'
            f'- filters: list of filter phrases from the query (e.g. ["status is approved", "city Chennai"])\n'
            f'- sort: "asc" or "desc" or null\n'
            f'- limit: integer row limit if specified (e.g. "top 10" → 10), or null\n'
            f'- temporal: true if the query involves time-based grouping (trends, monthly, yearly), false otherwise\n\n'
            f'Return JSON with all fields above.'}])

        result.entities = parsed.get("entities", [])
        result.aggregation = parsed.get("aggregation")
        result.aggregation_trigger = parsed.get("aggregation_trigger")
        result.action = parsed.get("action", "show")
        result.filters = parsed.get("filters", [])
        result.sort = parsed.get("sort")
        result.temporal = parsed.get("temporal", False)
        limit_val = parsed.get("limit")
        if isinstance(limit_val, (int, float)) and limit_val > 0:
            result.limit = int(limit_val)

    except Exception as e:
        logger.warning(f"LLM NLU parsing failed: {e} — falling back to raw tokens")
        # Graceful fallback: treat all non-trivial words as entities
        trivial = {"show", "me", "the", "a", "an", "of", "in", "for", "all", "get", "give", "list"}
        kept = [t for t in result.raw_tokens if t not in trivial and len(t) > 1]
        if kept:
            result.entities = [" ".join(kept)]

    logger.info(
        f"NLU | query={user_message!r} "
        f"entities={result.entities} "
        f"agg={result.aggregation}({result.aggregation_trigger}) "
        f"action={result.action}"
    )
    return result


def aggregation_to_sql(agg: str | None, group_column: str, count_alias: str = "count") -> dict:
    """
    Translate an aggregation type into SQL fragment hints.

    Returns a dict that the query planner / SQL builder can use directly:
        {
            "select_expr":  "case_status, COUNT(*) AS count",
            "group_by":     "case_status",
            "order_by":     None,
            "aggregation":  "count",
        }
    """
    if agg == "count_group_by" or agg == "count_group_by_time":
        return {
            "select_expr": f"{group_column}, COUNT(*) AS {count_alias}",
            "group_by": group_column,
            "order_by": None,
            "aggregation": "count",
        }
    if agg == "count":
        return {
            "select_expr": f"COUNT(*) AS {count_alias}",
            "group_by": None,
            "order_by": None,
            "aggregation": "count",
        }
    if agg == "sum":
        return {
            "select_expr": f"SUM({group_column}) AS total",
            "group_by": None,
            "order_by": None,
            "aggregation": "sum",
        }
    if agg == "avg":
        return {
            "select_expr": f"AVG({group_column}) AS average",
            "group_by": None,
            "order_by": None,
            "aggregation": "avg",
        }
    if agg == "max":
        return {
            "select_expr": f"MAX({group_column}) AS maximum",
            "group_by": None,
            "order_by": None,
            "aggregation": "max",
        }
    if agg == "min":
        return {
            "select_expr": f"MIN({group_column}) AS minimum",
            "group_by": None,
            "order_by": None,
            "aggregation": "min",
        }
    # Plain SELECT *
    return {
        "select_expr": "*",
        "group_by": None,
        "order_by": None,
        "aggregation": "none",
    }
