"""Visualization config builder — constructs ECharts-compatible configs and clarification payloads."""


# ── Chart config builders ─────────────────────────────

def bar_config(data: list[dict], x_col: str, y_cols: list[str],
               colors: list[str] = None) -> dict:
    colors = colors or ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
    return {
        "viz_type": "bar",
        "options": {
            "xAxis": {"type": "category", "data": [str(row.get(x_col, "")) for row in data]},
            "yAxis": {"type": "value"},
            "series": [
                {"name": col, "type": "bar", "data": [row.get(col, 0) for row in data]}
                for col in y_cols
            ],
            "legend": {"show": len(y_cols) > 1, "top": "top"},
            "color": colors[:len(y_cols)],
            "tooltip": {"trigger": "axis"},
        }
    }


def line_config(data: list[dict], x_col: str, y_cols: list[str],
                smooth: bool = True, colors: list[str] = None) -> dict:
    colors = colors or ["#2563EB", "#10B981", "#F59E0B", "#EF4444"]
    return {
        "viz_type": "line",
        "options": {
            "xAxis": {"type": "category", "data": [str(row.get(x_col, "")) for row in data]},
            "yAxis": {"type": "value"},
            "series": [
                {"name": col, "type": "line", "smooth": smooth,
                 "data": [row.get(col, 0) for row in data]}
                for col in y_cols
            ],
            "legend": {"show": len(y_cols) > 1, "top": "top"},
            "color": colors[:len(y_cols)],
            "tooltip": {"trigger": "axis"},
        }
    }


def pie_config(data: list[dict], name_col: str, value_col: str,
               colors: list[str] = None) -> dict:
    colors = colors or ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]
    return {
        "viz_type": "pie",
        "options": {
            "series": [{
                "type": "pie", "radius": ["40%", "70%"],
                "data": [{"name": str(row.get(name_col, "")), "value": row.get(value_col, 0)} for row in data],
            }],
            "legend": {"orient": "vertical", "left": "left"},
            "color": colors,
            "tooltip": {"trigger": "item"},
        }
    }


# ── Clarification builders ────────────────────────────

def viz_type_clarification(columns: list = None) -> dict:
    """Build viz type clarification payload (multi_select)."""
    return {
        "type": "viz_type", "mode": "multi_select",
        "support_for_custom_replies": False,
        "question": "How would you like to view the data?",
        "options": [
            {"value": "table", "label": "Table", "icon": "\U0001f5c2\ufe0f"},
            {"value": "bar", "label": "Bar graph", "icon": "\U0001f4ca"},
            {"value": "pie", "label": "Pie chart", "icon": "\U0001f967"},
            {"value": "donut", "label": "Donut chart", "icon": "\U0001f369"},
            {"value": "line", "label": "Line graph", "icon": "\U0001f4c8"},
        ]
    }


def axis_mode_clarification() -> dict:
    """Build axis mode clarification payload (single_select)."""
    return {
        "type": "axis_mode", "mode": "single_select",
        "support_for_custom_replies": False,
        "question": "Do you want to select X-axis and Y-axis on the fly or want to be specific?",
        "options": [
            {"value": "on_the_fly", "label": "On the fly (interactive dropdowns)"},
            {"value": "specific", "label": "Want to be specific (choose now)"},
        ]
    }


def axis_specific_clarification(columns: list[dict]) -> dict:
    """Build specific axis selection payload (radio + checkbox).
    columns: [{name, data_type}] where data_type is STRING/NUMBER/DATE."""
    string_cols = [c for c in columns if c.get("data_type") in ("STRING", "DATE")]
    number_cols = [c for c in columns if c.get("data_type") == "NUMBER"]
    all_cols = [{"value": c["name"], "label": c["name"], "data_type": c.get("data_type", "STRING")} for c in columns]

    return {
        "type": "axis_specific", "mode": "radio_checkbox",
        "question": "Select columns for your chart:",
        "x_axis_options": all_cols,
        "y_axis_options": [c for c in all_cols if c["data_type"] == "NUMBER"] or all_cols,
    }


def filter_criteria_clarification(columns: list[str], table_name: str = "") -> dict:
    """Build filter criteria clarification payload (text input).
    Shown when the user asks a vague filter question like 'filter by criteria'."""
    friendly_cols = [c.replace("_", " ").title() for c in columns]
    col_list = ", ".join(friendly_cols[:10])
    hint = f"Available fields: {col_list}" if friendly_cols else ""
    return {
        "type": "filter_criteria",
        "mode": "text_input",
        "support_for_custom_replies": True,
        "question": "What criteria would you like to filter by?",
        "placeholder": "e.g., country is India, balance above 50000, city is Mumbai",
        "hint": hint,
        "options": [],
    }


def record_limit_clarification(total_rows: int, limit: int) -> dict:
    """Build record limit clarification payload (single_select).
    When total_rows exceeds MAX_POPULATE_ROWS, the 'all' option is capped
    and the question warns the user about display limitations."""
    from app.config import get_settings
    max_populate = get_settings().MAX_POPULATE_ROWS

    note = None
    if total_rows > max_populate:
        question = (
            f"This query matches **{total_rows:,} records**. "
            f"How would you like to proceed?"
        )
        options = [
            {"value": "all", "label": f"Load maximum ({max_populate:,} records)"},
            {"value": "limited", "label": f"Load first {limit:,} records"},
        ]
        note = (
            f"**Note:** Displaying all {total_rows:,} records at once may not be visually "
            f"practical and could significantly slow down your browser. "
            f"The maximum you can load on-screen is {max_populate:,} records. "
            f"To access the full dataset, use the **Export** option after loading."
        )
    else:
        question = (
            f"This query returned **{total_rows:,} records**. "
            f"How would you like to proceed?"
        )
        options = [
            {"value": "all", "label": f"Load all {total_rows:,} records"},
            {"value": "limited", "label": f"Load first {limit:,} records"},
        ]

    result = {
        "type": "record_limit", "mode": "single_select",
        "support_for_custom_replies": True,
        "question": question,
        "options": options,
        "context": {"total_rows": total_rows, "limit": limit, "max_populate": max_populate},
    }
    if note:
        result["note"] = note
    return result
