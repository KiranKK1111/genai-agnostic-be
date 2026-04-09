"""Dashboard API — serves KPIs, chart data, and table data for business reports.

Each domain (Climate Risk, Pureplay, RSRMA, CESRA, TESRA) gets:
  - /summary   → KPI cards (counts, % changes, status breakdowns)
  - /charts    → Pre-aggregated chart data (time series, distributions)
  - /table     → Paginated + filterable raw table data
"""
import logging
from datetime import date, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from app.database import get_pool
from app.config import get_settings
from app.api.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

SCHEMA = None


def _schema() -> str:
    global SCHEMA
    if SCHEMA is None:
        SCHEMA = get_settings().POSTGRES_SCHEMA
    return SCHEMA


# ── Helpers ────────────────────────────────────────────────


async def _query(sql: str, *args):
    pool = get_pool()
    async with pool.acquire() as conn:
        return [dict(r) for r in await conn.fetch(sql, *args)]


async def _query_one(sql: str, *args):
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, *args)
        return dict(row) if row else {}


def _parse_filters(table_alias: str, filters: Optional[str]) -> tuple[str, list]:
    """Parse 'col1:val1,col2:val2' filter string into WHERE clauses.

    Returns (where_clause, params_list). Uses parameterised queries.
    """
    if not filters:
        return "", []

    clauses = []
    params = []
    idx = 1
    for pair in filters.split(","):
        if ":" not in pair:
            continue
        col, val = pair.split(":", 1)
        col = col.strip()
        val = val.strip()
        if not col or not val or val.lower() == "all":
            continue
        # Sanitise column name (alphanumeric + underscore only)
        safe_col = "".join(c for c in col if c.isalnum() or c == "_")
        if not safe_col:
            continue
        clauses.append(f"{table_alias}.{safe_col} ILIKE ${idx}")
        params.append(f"%{val}%")
        idx += 1

    where = " AND ".join(clauses)
    return (f"AND {where}" if where else ""), params


# ══════════════════════════════════════════════════════════════
# METADATA — available domains and reports
# ══════════════════════════════════════════════════════════════


@router.get("/meta")
async def get_dashboard_meta(_user=Depends(get_current_user)):
    """Return the list of available dashboard domains and their sub-reports."""
    return {
        "domains": [
            {
                "id": "climate_risk",
                "label": "Climate Risk",
                "icon": "Flame",
                "reports": [
                    {"id": "cra_score", "label": "CRA Score Report", "table": "cra_score_report"},
                    {"id": "risk_trigger", "label": "Risk Trigger Report", "table": "risk_trigger_report"},
                    {"id": "questionnaire", "label": "Questionnaire Report", "table": "questionnaire_report"},
                    {"id": "case_workflow", "label": "Case Workflow Report", "table": "case_workflow_report"},
                    {"id": "advisor_memo", "label": "Advisor Memo Report", "table": "advisor_memo_report"},
                    {"id": "questionnaire_daily", "label": "Questionnaire Report (Daily)", "table": "questionnaire_report_daily"},
                ],
            },
            {
                "id": "pureplay",
                "label": "Pureplay",
                "icon": "Leaf",
                "reports": [
                    {"id": "pureplay", "label": "Pureplay Report", "table": "pureplay"},
                ],
            },
            {
                "id": "rsrma",
                "label": "RSRMA",
                "icon": "Shield",
                "reports": [
                    {"id": "rsrma", "label": "RSRMA Operational Status", "table": "rsrma"},
                ],
            },
            {
                "id": "cesra",
                "label": "CESRA",
                "icon": "BarChart3",
                "reports": [
                    {"id": "cesra_kpi", "label": "ESRM KPI Monitoring", "table": "esrm_kpi_monitoring"},
                    {"id": "cesra_cst", "label": "CST Monitoring", "table": "cst_monitoring"},
                ],
            },
            {
                "id": "tesra",
                "label": "TESRA",
                "icon": "Scale",
                "reports": [
                    {"id": "tesra", "label": "Equator Principles", "table": "tesra"},
                ],
            },
        ]
    }


# ══════════════════════════════════════════════════════════════
# GENERIC — works for ANY table
# ══════════════════════════════════════════════════════════════


@router.get("/report/{table_name}/summary")
async def get_report_summary(
    table_name: str,
    _user=Depends(get_current_user),
):
    """Return KPI summary cards for any business table.

    Auto-detects date columns, status columns, and generates:
      - Total record count
      - Status breakdown (counts per status)
      - Monthly volume (current vs previous month)
    """
    s = _schema()
    safe_table = "".join(c for c in table_name if c.isalnum() or c == "_")

    try:
        # Total count
        total = await _query_one(f"SELECT COUNT(*) as total FROM {s}.{safe_table}")

        # Find status-like columns (case_status, grading_status, etc.)
        cols = await _query(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = $1 AND table_name = $2 ORDER BY ordinal_position",
            s, safe_table
        )
        col_names = [c["column_name"] for c in cols]

        status_cols = [c for c in col_names if "status" in c or "rating" in c or "grading" in c or "brag" in c]
        date_cols = [c for c in col_names
                     if any(c == cn["column_name"] and cn["data_type"] in ("date", "timestamp without time zone", "timestamp with time zone")
                            for cn in cols)]

        # Status breakdown for first status column found
        status_breakdown = []
        if status_cols:
            sc = status_cols[0]
            status_breakdown = await _query(
                f"SELECT COALESCE({sc}::text, 'N/A') as status, COUNT(*) as count "
                f"FROM {s}.{safe_table} GROUP BY {sc} ORDER BY count DESC LIMIT 15"
            )

        # Monthly volume using first date column
        monthly_volume = []
        current_month_count = 0
        prev_month_count = 0
        if date_cols:
            dc = date_cols[0]
            monthly_volume = await _query(
                f"SELECT TO_CHAR(DATE_TRUNC('month', {dc}), 'YYYY-MM') as month, "
                f"COUNT(*) as count FROM {s}.{safe_table} "
                f"WHERE {dc} IS NOT NULL "
                f"GROUP BY DATE_TRUNC('month', {dc}) ORDER BY month"
            )
            # Current vs previous month
            now = date.today()
            first_of_month = now.replace(day=1)
            prev_month = (first_of_month - timedelta(days=1)).replace(day=1)
            counts = await _query(
                f"SELECT "
                f"COUNT(*) FILTER (WHERE {dc} >= $1) as current_month, "
                f"COUNT(*) FILTER (WHERE {dc} >= $2 AND {dc} < $1) as prev_month "
                f"FROM {s}.{safe_table}",
                first_of_month, prev_month
            )
            if counts:
                current_month_count = counts[0].get("current_month", 0)
                prev_month_count = counts[0].get("prev_month", 0)

        return {
            "table": safe_table,
            "total_records": total.get("total", 0),
            "status_column": status_cols[0] if status_cols else None,
            "status_breakdown": status_breakdown,
            "date_column": date_cols[0] if date_cols else None,
            "monthly_volume": monthly_volume,
            "current_month_count": current_month_count,
            "prev_month_count": prev_month_count,
            "columns": col_names,
        }
    except Exception as e:
        logger.error(f"Dashboard summary error for {safe_table}: {e}")
        raise HTTPException(status_code=400, detail=f"Could not generate summary for '{table_name}'")


@router.get("/report/{table_name}/charts")
async def get_report_charts(
    table_name: str,
    group_by: Optional[str] = Query(None, description="Column to group by"),
    date_col: Optional[str] = Query(None, description="Date column for time series"),
    granularity: str = Query("month", description="Time granularity: day, week, month, quarter, year"),
    filters: Optional[str] = Query(None, description="Filters as col1:val1,col2:val2"),
    _user=Depends(get_current_user),
):
    """Return chart-ready aggregated data for a business table."""
    s = _schema()
    safe_table = "".join(c for c in table_name if c.isalnum() or c == "_")
    filter_clause, filter_params = _parse_filters("t", filters)

    charts = {}

    try:
        # Time series chart (if date column provided or auto-detected)
        if date_col:
            safe_dc = "".join(c for c in date_col if c.isalnum() or c == "_")
            trunc = {"day": "day", "week": "week", "month": "month", "quarter": "quarter", "year": "year"}.get(granularity, "month")
            charts["time_series"] = await _query(
                f"SELECT TO_CHAR(DATE_TRUNC('{trunc}', t.{safe_dc}), 'YYYY-MM') as period, "
                f"COUNT(*) as count FROM {s}.{safe_table} t "
                f"WHERE t.{safe_dc} IS NOT NULL {filter_clause} "
                f"GROUP BY DATE_TRUNC('{trunc}', t.{safe_dc}) ORDER BY period",
                *filter_params
            )

        # Distribution chart (if group_by provided)
        if group_by:
            safe_gb = "".join(c for c in group_by if c.isalnum() or c == "_")
            charts["distribution"] = await _query(
                f"SELECT COALESCE(t.{safe_gb}::text, 'N/A') as category, COUNT(*) as count "
                f"FROM {s}.{safe_table} t WHERE 1=1 {filter_clause} "
                f"GROUP BY t.{safe_gb} ORDER BY count DESC LIMIT 20",
                *filter_params
            )

        # Auto-detect and return multiple chart datasets if no specific request
        if not date_col and not group_by:
            # Auto-detect columns
            cols = await _query(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = $1 AND table_name = $2 ORDER BY ordinal_position",
                s, safe_table
            )

            # Generate a time-series chart for EACH date column (up to 4)
            date_chart_count = 0
            for c in cols:
                if c["data_type"] in ("date", "timestamp without time zone", "timestamp with time zone") and date_chart_count < 4:
                    safe_dc = c["column_name"]
                    # Build readable title from column name: "approved_date" → "Volume by Approved Date"
                    col_label = safe_dc.replace("_", " ").title()
                    chart_key = f"volume_by_{safe_dc}"
                    result = await _query(
                        f"SELECT TO_CHAR(DATE_TRUNC('month', t.{safe_dc}), 'YYYY-MM') as period, "
                        f"COUNT(*) as count FROM {s}.{safe_table} t "
                        f"WHERE t.{safe_dc} IS NOT NULL {filter_clause} "
                        f"GROUP BY DATE_TRUNC('month', t.{safe_dc}) ORDER BY period",
                        *filter_params
                    )
                    if result:  # Only add if there's data
                        charts[chart_key] = result
                        date_chart_count += 1

            # If no date charts were found, fall back to status distribution charts
            if date_chart_count == 0:
                for c in cols:
                    if c["data_type"] in ("character varying", "text") and any(
                        kw in c["column_name"] for kw in ("status", "rating", "grading", "type", "version", "category")
                    ):
                        safe_sc = c["column_name"]
                        charts[f"distribution_{safe_sc}"] = await _query(
                            f"SELECT COALESCE(t.{safe_sc}::text, 'N/A') as category, COUNT(*) as count "
                            f"FROM {s}.{safe_table} t WHERE 1=1 {filter_clause} "
                            f"GROUP BY t.{safe_sc} ORDER BY count DESC LIMIT 15",
                            *filter_params
                        )

        return {"table": safe_table, "charts": charts}

    except Exception as e:
        logger.error(f"Dashboard chart error for {safe_table}: {e}")
        raise HTTPException(status_code=400, detail=f"Could not generate charts for '{table_name}'")


@router.get("/report/{table_name}/data")
async def get_report_data(
    table_name: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=10, le=500),
    sort_by: Optional[str] = Query(None),
    sort_dir: str = Query("asc", regex="^(asc|desc)$"),
    filters: Optional[str] = Query(None, description="Filters as col1:val1,col2:val2"),
    search: Optional[str] = Query(None, description="Global text search across all text columns"),
    _user=Depends(get_current_user),
):
    """Return paginated, filterable table data for a business report."""
    s = _schema()
    safe_table = "".join(c for c in table_name if c.isalnum() or c == "_")
    filter_clause, filter_params = _parse_filters("t", filters)

    try:
        # Get column metadata
        cols = await _query(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = $1 AND table_name = $2 ORDER BY ordinal_position",
            s, safe_table
        )
        if not cols:
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")

        col_names = [c["column_name"] for c in cols]
        text_cols = [c["column_name"] for c in cols if c["data_type"] in ("character varying", "text", "varchar")]

        # Build search clause
        search_clause = ""
        if search and text_cols:
            search_parts = " OR ".join(f"t.{c}::text ILIKE ${len(filter_params) + 1}" for c in text_cols[:8])
            search_clause = f"AND ({search_parts})"
            filter_params.append(f"%{search}%")

        # Total count (with filters)
        count_sql = f"SELECT COUNT(*) as total FROM {s}.{safe_table} t WHERE 1=1 {filter_clause} {search_clause}"
        total = await _query_one(count_sql, *filter_params)

        # Sort
        order = ""
        if sort_by and sort_by in col_names:
            order = f"ORDER BY t.{sort_by} {sort_dir.upper()} NULLS LAST"
        else:
            order = f"ORDER BY 1 ASC"

        # Paginated data
        offset = (page - 1) * page_size
        data_sql = (
            f"SELECT t.* FROM {s}.{safe_table} t "
            f"WHERE 1=1 {filter_clause} {search_clause} "
            f"{order} LIMIT {page_size} OFFSET {offset}"
        )
        rows = await _query(data_sql, *filter_params)

        # Serialise dates/timestamps to strings
        for row in rows:
            for k, v in row.items():
                if hasattr(v, "isoformat"):
                    row[k] = v.isoformat()

        # Get distinct values for filter dropdowns (low-cardinality columns only)
        filter_options = {}
        for c in cols:
            if c["data_type"] in ("character varying", "text") and any(
                kw in c["column_name"] for kw in ("status", "rating", "grading", "type", "version", "category", "country", "brag")
            ):
                distinct = await _query(
                    f"SELECT DISTINCT {c['column_name']}::text as val FROM {s}.{safe_table} "
                    f"WHERE {c['column_name']} IS NOT NULL ORDER BY val LIMIT 50"
                )
                filter_options[c["column_name"]] = [d["val"] for d in distinct]

        return {
            "table": safe_table,
            "columns": [{"name": c["column_name"], "type": c["data_type"]} for c in cols],
            "rows": rows,
            "total": total.get("total", 0),
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, -(-total.get("total", 0) // page_size)),
            "filter_options": filter_options,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard data error for {safe_table}: {e}")
        raise HTTPException(status_code=400, detail=f"Could not fetch data for '{table_name}'")


@router.get("/report/{table_name}/filters")
async def get_report_filters(
    table_name: str,
    _user=Depends(get_current_user),
):
    """Return available filter columns and their distinct values for a table."""
    s = _schema()
    safe_table = "".join(c for c in table_name if c.isalnum() or c == "_")

    try:
        cols = await _query(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = $1 AND table_name = $2 ORDER BY ordinal_position",
            s, safe_table
        )

        filters = []
        for c in cols:
            col_name = c["column_name"]
            data_type = c["data_type"]

            # Filterable text columns
            if data_type in ("character varying", "text"):
                distinct_count = await _query_one(
                    f"SELECT COUNT(DISTINCT {col_name}) as cnt FROM {s}.{safe_table}"
                )
                cnt = distinct_count.get("cnt", 0)
                if 0 < cnt <= 100:
                    values = await _query(
                        f"SELECT DISTINCT {col_name}::text as val FROM {s}.{safe_table} "
                        f"WHERE {col_name} IS NOT NULL ORDER BY val LIMIT 100"
                    )
                    filters.append({
                        "column": col_name,
                        "label": col_name.replace("_", " ").title(),
                        "type": "select",
                        "values": [v["val"] for v in values],
                    })

            # Date columns → date range filter
            elif data_type in ("date", "timestamp without time zone", "timestamp with time zone"):
                range_row = await _query_one(
                    f"SELECT MIN({col_name}) as min_date, MAX({col_name}) as max_date FROM {s}.{safe_table}"
                )
                min_d = range_row.get("min_date")
                max_d = range_row.get("max_date")
                filters.append({
                    "column": col_name,
                    "label": col_name.replace("_", " ").title(),
                    "type": "date_range",
                    "min": min_d.isoformat() if min_d else None,
                    "max": max_d.isoformat() if max_d else None,
                })

        return {"table": safe_table, "filters": filters}

    except Exception as e:
        logger.error(f"Dashboard filter error for {safe_table}: {e}")
        raise HTTPException(status_code=400, detail=f"Could not fetch filters for '{table_name}'")
