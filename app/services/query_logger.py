"""Rich-formatted query pipeline logger.

Prints structured panels to the terminal for every stage of the
query pipeline: arrival → intent → plan → SQL → execution → retry.

All functions are synchronous (just terminal I/O) and safe to call
from async code. Thread-safe via Rich's internal locking.
"""
import time
import json
import logging
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.rule import Rule
from rich import box

console = Console(highlight=False)
logger = logging.getLogger(__name__)

# ── Colour palette ─────────────────────────────────────
_CLR_INTENT   = "bold cyan"
_CLR_PLAN     = "bold blue"
_CLR_SQL      = "bold green"
_CLR_EXEC     = "bold green"
_CLR_RETRY    = "bold yellow"
_CLR_ERROR    = "bold red"
_CLR_ARRIVAL  = "bold magenta"
_CLR_DIM      = "dim white"
_CLR_FILE     = "bold blue"

# ── Timing helpers ──────────────────────────────────────
_stage_start: dict[str, float] = {}


def _sid(session_id: Optional[str]) -> str:
    """Short 8-char session prefix for panel titles."""
    if not session_id:
        return ""
    return f" [{session_id[:8]}]"


# ═══════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════

def log_query_arrival(message: str, session_id: str = "", user_name: str = "User",
                      intent_hint: str = ""):
    """Log when a new query arrives at the orchestrator."""
    _stage_start[session_id] = time.monotonic()
    short = message if len(message) <= 120 else message[:117] + "..."
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style=_CLR_DIM, width=10)
    grid.add_column()
    grid.add_row("Session:", Text(session_id[:8] if session_id else "—", style="dim cyan"))
    grid.add_row("User:",    Text(user_name, style="white"))
    grid.add_row("Message:", Text(short, style="bold white"))
    if intent_hint:
        grid.add_row("Hint:",    Text(intent_hint, style=_CLR_DIM))
    console.print(Panel(grid, title="[bold magenta]▶  Query Received[/]",
                         border_style="magenta", box=box.ROUNDED, expand=False))


def log_intent(intent: str, confidence: float, method: str, session_id: str = ""):
    """Log the classified intent."""
    colour = {
        "DB_QUERY":           "cyan",
        "DB_FOLLOW_UP":       "cyan",
        "FILE_ANALYSIS":      "blue",
        "FILE_FOLLOW_UP":     "blue",
        "CHAT":               "white",
        "VIZ_FOLLOW_UP":      "yellow",
        "HYBRID":             "green",
        "DUAL_SEARCH":        "yellow",
        "CLARIFICATION_REPLY":"magenta",
    }.get(intent, "white")

    conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
    line = Text()
    line.append(f"  Intent  ", style="bold white on #1a1a2e")
    line.append(f"  {intent:<22}", style=f"bold {colour}")
    line.append(f"  {conf_bar}  ", style="dim green")
    line.append(f"{confidence:.0%}  ", style="green")
    line.append(f"via {method}", style=_CLR_DIM)
    console.print(line)


def log_scope_rejected(message: str, session_id: str = ""):
    """Log when scope validation rejects a query."""
    console.print(f"  [bold red]✗ Scope rejected[/]  [dim]{message[:80]}[/]")


def log_query_plan(plan_dict: dict, session_id: str = ""):
    """Log the structured query plan produced by build_query_plan()."""
    tables  = plan_dict.get("tables", [])
    columns = plan_dict.get("columns", [])
    filters = plan_dict.get("filters", [])
    joins   = plan_dict.get("join_hints", [])
    agg     = plan_dict.get("aggregation")

    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1),
                border_style="blue")
    tbl.add_column(style="bold dim", width=12)
    tbl.add_column(style="white")

    tbl.add_row("Tables",
                Text(", ".join(tables) if tables else "—", style="bold cyan"))
    tbl.add_row("Columns",
                Text(", ".join(columns) if columns else "*", style="cyan"))

    if filters:
        filter_lines = []
        for f in filters:
            col = f.get("column", "?")
            op  = f.get("operator", "=")
            val = f.get("value", "?")
            filter_lines.append(f"{col} {op} {val!r}")
        tbl.add_row("Filters", Text(" AND ".join(filter_lines), style="yellow"))
    else:
        tbl.add_row("Filters", Text("none", style=_CLR_DIM))

    if joins:
        tbl.add_row("Joins",
                    Text(", ".join(joins) if isinstance(joins, list) else str(joins),
                         style="green"))
    if agg:
        tbl.add_row("Aggregation", Text(str(agg), style="magenta"))

    console.print(Panel(tbl, title=f"[{_CLR_PLAN}]⚙  Query Plan{_sid(session_id)}[/]",
                         border_style="blue", box=box.ROUNDED, expand=False))


def log_sql_generated(sql: str, stage: str = "Generated", session_id: str = ""):
    """Log a generated (or amended) SQL statement with syntax highlighting."""
    syntax = Syntax(sql.strip(), "sql", theme="monokai", word_wrap=True,
                    background_color="default")
    label = "✦  SQL Generated" if stage == "Generated" else f"↺  SQL {stage}"
    console.print(Panel(syntax, title=f"[{_CLR_SQL}]{label}{_sid(session_id)}[/]",
                         border_style="green", box=box.ROUNDED))


def log_sql_retry(original_sql: str, error: str, fixed_sql: str = "",
                  session_id: str = ""):
    """Log the SQL error → retry → fixed SQL sequence."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="bold dim", width=10)
    grid.add_column(style="white")
    grid.add_row("Error:",    Text(error[:300], style="red"))
    grid.add_row("Original:", Text(original_sql[:200].strip(), style="dim"))
    console.print(Panel(grid, title=f"[{_CLR_RETRY}]⚠  SQL Error — Retrying{_sid(session_id)}[/]",
                         border_style="yellow", box=box.ROUNDED, expand=False))
    if fixed_sql:
        log_sql_generated(fixed_sql, stage="Retry Fixed", session_id=session_id)


def log_execution_result(row_count: int, columns: list[str], elapsed_ms: float,
                          sql: str = "", session_id: str = ""):
    """Log the execution result summary."""
    col_str = ", ".join(columns[:8])
    if len(columns) > 8:
        col_str += f" +{len(columns)-8} more"

    line = Text()
    line.append("  ✓ ", style="bold green")
    line.append(f"{row_count:,} rows", style="bold white")
    line.append("  ·  ", style=_CLR_DIM)
    line.append(f"{len(columns)} cols ", style="white")
    line.append(f"({col_str})", style=_CLR_DIM)
    line.append("  ·  ", style=_CLR_DIM)
    line.append(f"{elapsed_ms:.0f} ms", style="bold green" if elapsed_ms < 500 else "bold yellow")
    console.print(line)


def log_execution_error(error: str, sql: str = "", session_id: str = ""):
    """Log a SQL execution error."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="bold dim", width=8)
    grid.add_column(style="red")
    grid.add_row("Error:", Text(error[:400], style="red"))
    if sql:
        grid.add_row("SQL:", Text(sql[:200].strip(), style="dim"))
    console.print(Panel(grid, title=f"[{_CLR_ERROR}]✗  Execution Failed{_sid(session_id)}[/]",
                         border_style="red", box=box.ROUNDED, expand=False))


def log_explain_warning(warning: str, estimated_rows: Optional[int] = None,
                         session_id: str = ""):
    """Log an EXPLAIN plan warning (e.g. seq scan on large table)."""
    row_info = f"  (~{estimated_rows:,} rows)" if estimated_rows else ""
    console.print(f"  [{_CLR_RETRY}]⚠ EXPLAIN[/]  [dim]{warning[:200]}{row_info}[/]")


def log_ambiguity(ambiguity_type: str, token: str, options: list, session_id: str = ""):
    """Log when the query planner detects an ambiguity that needs clarification."""
    opts = ", ".join(str(o) for o in options[:5])
    if len(options) > 5:
        opts += f" +{len(options)-5} more"
    console.print(
        f"  [bold yellow]? Ambiguity[/]  [dim]{ambiguity_type}[/]  "
        f"[yellow]{token!r}[/]  →  [dim]{opts}[/]"
    )


def log_followup_base(chosen_sql: str, reason: str, query_num: int, session_id: str = ""):
    """Log which base query was selected for a follow-up amendment."""
    short = chosen_sql[:120].strip()
    console.print(
        f"  [bold cyan]↪ Follow-up[/]  [dim]base = Q{query_num}[/]  "
        f"[dim]{reason[:80]}[/]\n  [dim]{short}[/]"
    )


def log_file_pipeline(file_name: str, stage: str, session_id: str = ""):
    """Log file analysis pipeline stages."""
    icons = {
        "receiving":   "📥",
        "parsing":     "📄",
        "chunking":    "✂",
        "embedding":   "🔢",
        "indexing":    "📑",
        "analyzing":   "🔍",
        "followup":    "💬",
    }
    icon = icons.get(stage.lower(), "•")
    console.print(
        f"  [{_CLR_FILE}]{icon}  File[/]  [white]{file_name}[/]  "
        f"[dim]— {stage}[/]"
    )


def log_pipeline_done(session_id: str = "", intent: str = ""):
    """Log the end of a full pipeline run with total wall-clock time."""
    elapsed = time.monotonic() - _stage_start.pop(session_id, time.monotonic())
    console.print(
        Rule(
            f"[dim]done · {elapsed*1000:.0f} ms · {intent}[/]",
            style="dim",
            align="right",
        )
    )
