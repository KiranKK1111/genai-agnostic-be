"""
Agentic Entity Resolver — finds the right table + column for every NLU entity
by searching the schema vector index across ALL tables simultaneously.

Design
──────
Traditional approach: pick a table first, then look for columns inside it.
This misses cross-table ambiguity (same column name in multiple tables).

New approach:
  1. Embed the entity phrase ("case status")
  2. Search schema_idx for matching *column-level* entries across ALL tables
  3. Group hits by table
  4. If the same column-like entry appears in multiple tables → AMBIGUOUS
     → return a clarification payload with user-friendly table labels
  5. If unambiguous (or after user resolves) → return the resolved TableColumnMatch

This is the component that makes the pipeline truly agentic: it self-resolves
what it can, and surfaces only genuine ambiguity to the user.

Example
───────
  resolve_entity("case status", schema_graph)
  → EntityResolution(
        entity_phrase = "case status",
        matches = [
            TableColumnMatch(table="cra_score_report",     column="case_status", score=0.97),
            TableColumnMatch(table="case_workflow_report", column="case_status", score=0.95),
        ],
        is_ambiguous = True,
        clarification = {
            "type": "ambiguous_entity",
            "question": "I found 'Case Status' in multiple reports. Which one would you like?",
            "options": [
                {"value": "cra_score_report",     "label": "CRA Score Report",     ...},
                {"value": "case_workflow_report", "label": "Case Workflow Report", ...},
            ],
            "token": "case status",
        }
    )
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from app.services.vector_search import search
from app.services.schema_inspector import SchemaGraph
from app.config import get_settings

logger = logging.getLogger(__name__)


# ── Thresholds ────────────────────────────────────────────────────────────────
_MIN_COLUMN_SIMILARITY = 0.45   # Minimum score to consider a column match (lowered: underscore↔space variants score ~0.48)
_AMBIGUITY_GAP        = 0.08   # If top-2 are within this gap → ambiguous
_TOP_K_SEARCH         = 15     # How many schema_idx hits to retrieve per entity


@dataclass
class TableColumnMatch:
    """A single (table, column) candidate for an entity phrase."""
    table: str
    column: str
    score: float
    friendly_table: str = ""
    friendly_column: str = ""
    table_description: str = ""

    def __post_init__(self):
        if not self.friendly_table:
            self.friendly_table = self.table.replace("_", " ").title()
        if not self.friendly_column:
            self.friendly_column = self.column.replace("_", " ").title()


@dataclass
class EntityResolution:
    """Result of resolving one entity phrase."""
    entity_phrase: str
    matches: list[TableColumnMatch] = field(default_factory=list)
    is_ambiguous: bool = False
    # Set if ambiguity was detected — pass directly to the SSE clarification event
    clarification: dict | None = None
    # Set if unambiguous (or after user resolved) — the winner
    resolved_table: str | None = None
    resolved_column: str | None = None


async def resolve_entity(
    entity_phrase: str,
    schema_graph: SchemaGraph,
    resolved_table: str | None = None,
) -> EntityResolution:
    """
    Resolve *entity_phrase* to (table, column) using vector search.

    Parameters
    ──────────
    entity_phrase   Natural-language phrase extracted by NLU, e.g. "case status".
    schema_graph    Current schema graph (used for descriptions & friendly names).
    resolved_table  If the user already resolved the ambiguity for this entity,
                    pass the table name to skip re-asking.

    Returns
    ───────
    EntityResolution — see class docstring above.
    """
    from app.services.embedder import embed_single

    # Normalise: treat underscore and space equivalently
    normalised = entity_phrase.lower().replace("_", " ").strip()

    # Embed the entity phrase with "schema:" prefix for asymmetric retrieval
    query_text = f"column {normalised}"
    embedding = await embed_single(query_text)

    # Search schema_idx — column-level entries only
    raw_results = await search(
        "schema_idx",
        embedding,
        k=_TOP_K_SEARCH,
        query_text=query_text,
    )

    # Filter to column-level entries and apply minimum similarity threshold
    column_hits: list[dict] = []
    for r in raw_results:
        payload = r.get("payload", {})
        if payload.get("type") != "column":
            continue
        if r["similarity"] < _MIN_COLUMN_SIMILARITY:
            continue
        column_hits.append(r)

    if not column_hits:
        logger.info(f"EntityResolver | '{entity_phrase}': no column matches above threshold — trying direct schema scan")
        # ── Direct string-match fallback ──────────────────────────────────────
        # Vector similarity can miss exact column names when the embedding distance
        # between "case status" (spaces) and "case_status" (underscores) is below
        # the threshold. Walk the schema graph directly instead.
        normalised_ul = normalised.replace(" ", "_")          # "case_status"
        normalised_sp = normalised.replace("_", " ")          # "case status"

        direct_matches: list[TableColumnMatch] = []
        for tname, tmeta in schema_graph.tables.items():
            for cname in tmeta.columns:
                cname_sp = cname.lower().replace("_", " ")
                if cname.lower() == normalised_ul or cname_sp == normalised_sp:
                    desc = getattr(tmeta, "description", "") or ""
                    direct_matches.append(TableColumnMatch(
                        table=tname,
                        column=cname,
                        score=1.0,          # exact match → treat as perfect score
                        table_description=desc,
                    ))

        if not direct_matches:
            logger.info(f"EntityResolver | '{entity_phrase}': not found in schema by direct scan either")
            return EntityResolution(entity_phrase=normalised)

        logger.info(
            f"EntityResolver | '{entity_phrase}': direct scan found {len(direct_matches)} match(es) "
            f"in tables {[m.table for m in direct_matches]}"
        )

        if resolved_table:
            winner = next((m for m in direct_matches if m.table == resolved_table), direct_matches[0])
            return EntityResolution(
                entity_phrase=normalised,
                matches=direct_matches,
                is_ambiguous=False,
                resolved_table=winner.table,
                resolved_column=winner.column,
            )

        if len(direct_matches) == 1:
            return EntityResolution(
                entity_phrase=normalised,
                matches=direct_matches,
                is_ambiguous=False,
                resolved_table=direct_matches[0].table,
                resolved_column=direct_matches[0].column,
            )

        # Same column name in multiple tables → AMBIGUOUS
        options = _build_clarification_options(direct_matches, group_by_table=True,
                                               schema_graph=schema_graph)
        question = _build_clarification_question(entity_phrase, options)
        return EntityResolution(
            entity_phrase=normalised,
            matches=direct_matches,
            is_ambiguous=True,
            clarification={
                "type":                      "ambiguous_entity",
                "mode":                      "single_select",
                "support_for_custom_replies": False,
                "question":                  question,
                "options":                   options,
                "token":                     normalised,
                "entity_phrase":             entity_phrase,
            },
        )

    # Build TableColumnMatch objects
    matches: list[TableColumnMatch] = []
    for r in column_hits:
        payload = r["payload"]
        tname = payload.get("table", "")
        cname = payload.get("column", "")
        if not tname or not cname:
            continue

        # Fetch table description from schema_graph if available
        desc = ""
        if tname in schema_graph.tables:
            tmeta = schema_graph.tables[tname]
            desc = getattr(tmeta, "description", "") or ""

        matches.append(TableColumnMatch(
            table=tname,
            column=cname,
            score=r["similarity"],
            table_description=desc,
        ))

    if not matches:
        return EntityResolution(entity_phrase=normalised)

    # Sort by score descending
    matches.sort(key=lambda m: m.score, reverse=True)

    # ── Resolved by user already ──────────────────────────────────────────────
    if resolved_table:
        winner = next((m for m in matches if m.table == resolved_table), None)
        if not winner and matches:
            winner = matches[0]  # fallback to best score
        return EntityResolution(
            entity_phrase=normalised,
            matches=matches,
            is_ambiguous=False,
            resolved_table=winner.table if winner else None,
            resolved_column=winner.column if winner else None,
        )

    # ── Deduplicate by (table, column) ───────────────────────────────────────
    seen_tc: set[tuple[str, str]] = set()
    deduped: list[TableColumnMatch] = []
    for m in matches:
        key = (m.table, m.column)
        if key not in seen_tc:
            seen_tc.add(key)
            deduped.append(m)
    matches = deduped

    # ── Single match → unambiguous ───────────────────────────────────────────
    if len(matches) == 1:
        return EntityResolution(
            entity_phrase=normalised,
            matches=matches,
            is_ambiguous=False,
            resolved_table=matches[0].table,
            resolved_column=matches[0].column,
        )

    # ── Ambiguity check: same column name in multiple tables ─────────────────
    # Group by column name (the semantic entity, not the table)
    col_to_tables: dict[str, list[TableColumnMatch]] = {}
    for m in matches:
        col_to_tables.setdefault(m.column, []).append(m)

    # The most likely target column is the one with the highest top score
    best_column = matches[0].column
    tables_for_best = col_to_tables.get(best_column, [])

    # Is the same (exact) column name present in multiple tables?
    same_column_multi_table = len(tables_for_best) >= 2

    # Are two different columns competing (e.g. "case_status" vs "status")?
    top_score    = matches[0].score
    second_score = matches[1].score if len(matches) > 1 else 0.0
    score_gap    = top_score - second_score
    competing_columns = score_gap < _AMBIGUITY_GAP and matches[1].column != best_column

    is_ambiguous = same_column_multi_table or competing_columns

    if not is_ambiguous:
        # Clear winner
        return EntityResolution(
            entity_phrase=normalised,
            matches=matches,
            is_ambiguous=False,
            resolved_table=matches[0].table,
            resolved_column=matches[0].column,
        )

    # ── Build clarification payload ───────────────────────────────────────────
    # For same-column-multi-table: options are the tables
    # For competing-columns: options are (table, column) pairs
    options = _build_clarification_options(
        matches, same_column_multi_table, schema_graph
    )

    friendly_entity = normalised.replace("_", " ").title()
    question = _build_clarification_question(entity_phrase, options)

    clarification = {
        "type":                    "ambiguous_entity",
        "mode":                    "single_select",
        "support_for_custom_replies": False,
        "question":                question,
        "options":                 options,
        "token":                   normalised,
        "entity_phrase":           entity_phrase,
    }

    return EntityResolution(
        entity_phrase=normalised,
        matches=matches,
        is_ambiguous=True,
        clarification=clarification,
    )


def _build_clarification_options(
    matches: list[TableColumnMatch],
    group_by_table: bool,
    schema_graph: SchemaGraph,
) -> list[dict]:
    """Build the option list for the clarification popup.

    Never exposes raw table names — uses human-readable labels derived from
    descriptions or title-cased friendly names.
    """
    seen_tables: set[str] = set()
    options: list[dict] = []

    for m in matches:
        if group_by_table:
            # One option per distinct table
            if m.table in seen_tables:
                continue
            seen_tables.add(m.table)

        label = _table_friendly_label(m.table, m.table_description, schema_graph)
        raw_desc = m.table_description or f"Data from {label}"
        # Strip raw table name from description so the UI never shows snake_case names.
        # Replace "The <table_name> table" or bare <table_name> occurrences with the label.
        description = re.sub(
            r'\bThe\s+' + re.escape(m.table) + r'\s+table\b',
            f"This report",
            raw_desc,
            flags=re.IGNORECASE,
        )
        description = re.sub(re.escape(m.table), label, description, flags=re.IGNORECASE)

        options.append({
            "value":       m.table,
            "label":       label,
            "description": description,
            "column":      m.column,
            "similarity":  round(m.score, 3),
        })

        if len(options) >= 5:  # cap at 5 options to keep UI clean
            break

    return options


def _table_friendly_label(
    table_name: str,
    description: str,
    schema_graph: SchemaGraph,
) -> str:
    """Return a human-readable label for a table.

    Priority:
    1. LLM-generated description (first N words, title-cased)
    2. Title-case of the table name with underscores → spaces
    """
    if description:
        # Use first sentence of description, strip technical jargon
        first_sentence = description.split(".")[0].strip()
        if first_sentence and len(first_sentence) < 80:
            return first_sentence

    # Fallback: convert snake_case to Title Case words
    # e.g. "cra_score_report" → "CRA Score Report"
    parts = table_name.split("_")
    # Uppercase abbreviations (≤ 4 chars) and title-case the rest
    return " ".join(p.upper() if len(p) <= 3 else p.title() for p in parts)


def _build_clarification_question(entity_phrase: str, options: list[dict]) -> str:
    """Generate a natural clarification question for the user."""
    entity_title = entity_phrase.replace("_", " ").title()
    if len(options) == 2:
        a, b = options[0]["label"], options[1]["label"]
        return (
            f"I found '{entity_title}' in multiple reports. "
            f"Did you mean from **{a}** or **{b}**?"
        )
    labels = ", ".join(f"**{o['label']}**" for o in options[:3])
    return (
        f"I found '{entity_title}' in {len(options)} different reports "
        f"({labels}). Which one would you like to use?"
    )


async def resolve_all_entities(
    entity_phrases: list[str],
    schema_graph: SchemaGraph,
    resolved_tables: dict[str, str] | None = None,
) -> tuple[list[EntityResolution], dict | None]:
    """
    Resolve every entity phrase from NLU.

    Returns (resolutions, first_clarification_needed):
    - resolutions: one EntityResolution per phrase
    - first_clarification_needed: the clarification dict for the first
      ambiguous entity, or None if all entities resolved cleanly.

    The pipeline should yield the clarification immediately and pause —
    re-invoke with the user's answer filled into resolved_tables.
    """
    resolved_tables = resolved_tables or {}
    resolutions: list[EntityResolution] = []
    first_clarification: dict | None = None

    for phrase in entity_phrases:
        resolved_table = resolved_tables.get(phrase.lower().replace("_", " "))
        resolution = await resolve_entity(phrase, schema_graph, resolved_table)
        resolutions.append(resolution)

        if resolution.is_ambiguous and first_clarification is None:
            first_clarification = resolution.clarification

    return resolutions, first_clarification


def resolutions_to_table_column_map(
    resolutions: list[EntityResolution],
) -> dict[str, tuple[str, str]]:
    """
    Convert a list of resolved EntityResolutions to a map:
        entity_phrase → (table_name, column_name)

    Only includes unambiguously resolved entities.
    """
    result: dict[str, tuple[str, str]] = {}
    for r in resolutions:
        if not r.is_ambiguous and r.resolved_table and r.resolved_column:
            result[r.entity_phrase] = (r.resolved_table, r.resolved_column)
    return result
