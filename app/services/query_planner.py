"""8-stage query planner — translates natural language to structured QueryPlan."""
import logging
import json
import re
from thefuzz import fuzz
from app.services.llm_client import chat, chat_json
from app.services.schema_inspector import SchemaGraph, DOMAIN_SYNONYMS
from app.config import get_settings

logger = logging.getLogger(__name__)


def _dedup_in_clauses(sql: str) -> str:
    """Remove duplicate values in IN (...) clauses and drop redundant OR conditions."""
    # Deduplicate values inside IN clauses
    def _dedup_in(m):
        prefix = m.group(1)  # e.g. "c.city IN ("
        values_str = m.group(2)
        vals = re.findall(r"'([^']+)'", values_str)
        # Preserve order, remove duplicates
        seen = set()
        unique = []
        for v in vals:
            if v.lower() not in seen:
                seen.add(v.lower())
                unique.append(v)
        return f"{prefix}" + ", ".join(f"'{v}'" for v in unique) + ")"

    sql = re.sub(
        r"(\w+\.\w+\s+IN\s*\()([^)]+)\)",
        _dedup_in, sql, flags=re.IGNORECASE
    )

    # Remove "OR col = 'val'" if the same col has an IN clause containing that value
    for m in re.finditer(r"(\w+\.\w+)\s+IN\s*\(([^)]+)\)", sql, re.IGNORECASE):
        col = m.group(1)
        in_vals = {v.lower() for v in re.findall(r"'([^']+)'", m.group(2))}
        # Find and remove redundant OR conditions for same column
        for or_m in re.finditer(
            rf"\s+OR\s+{re.escape(col)}\s*=\s*'([^']+)'", sql, re.IGNORECASE
        ):
            if or_m.group(1).lower() in in_vals:
                sql = sql.replace(or_m.group(0), "")
                logger.info(f"_dedup_in_clauses: removed redundant OR {col} = '{or_m.group(1)}'")

    return sql


def _extract_sql(text: str) -> str:
    """Extract clean SQL from LLM response, stripping preamble and code fences."""
    text = text.strip()
    # If the response contains a code block, extract its content
    if "```" in text:
        # Find the first code block
        parts = text.split("```")
        # parts[1] is the content inside the first pair of backticks
        if len(parts) >= 2:
            block = parts[1]
            # Strip optional language tag (e.g. "sql\n")
            if block.startswith(("sql", "SQL")):
                block = block.split("\n", 1)[-1]
            text = block.strip()
    else:
        # No code block — strip any preamble text before SELECT/WITH
        match = re.search(r"(?i)((?:SELECT|WITH)\b.*)", text, flags=re.DOTALL)
        if match:
            text = match.group(1).strip()
    return text.rstrip(";") + ";"

# Load spaCy model for NER-based entity extraction (Stage 1)
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy NER model loaded (en_core_web_sm)")
except Exception as e:
    _nlp = None
    logger.warning(f"spaCy not available, using LLM-only extraction: {e}")


class QueryPlan:
    def __init__(self):
        self.tables: list[str] = []
        self.joins: list[dict] = []
        self.columns: list[str] = ["*"]
        self.filters: list[dict] = []
        self.aggregation: dict | None = None
        self.order_by: str | None = None
        self.limit: int | None = None
        self.group_by: str | None = None

    def to_dict(self) -> dict:
        return {
            "tables": self.tables, "joins": self.joins, "columns": self.columns,
            "filters": self.filters, "aggregation": self.aggregation,
            "order_by": self.order_by, "limit": self.limit, "group_by": self.group_by,
        }


async def _ground_value_full(value: str, plan_tables: list[str], schema_graph: SchemaGraph) -> tuple[str, str | None, str | None]:
    """Full 4-tier value grounding: value_index -> FAISS semantic -> live ILIKE -> fuzzy Levenshtein.
    Returns (grounded_value, table_name, column_name)."""
    val_lower = value.lower().strip()

    # ── Tier 1: Dynamic value index (built from actual DB values) ──
    idx_match = schema_graph.lookup_value(value, plan_tables)
    if idx_match:
        actual_val, tname, cname = idx_match
        logger.info(f"Value grounding (index): '{value}' -> '{actual_val}' in {tname}.{cname}")
        return actual_val, tname, cname

    # ── Tier 2: FAISS semantic search on values_idx ────────
    try:
        from app.services.vector_search import ground_value
        faiss_results = await ground_value(value, k=3)
        if faiss_results and faiss_results[0]["similarity"] > 0.80:
            best = faiss_results[0]
            logger.info(f"Value grounding (FAISS): '{value}' -> '{best['value']}' (similarity={best['similarity']:.2f})")
            return best["value"], best.get("table"), best.get("column")
    except Exception as e:
        logger.debug(f"FAISS value grounding skipped: {e}")

    # ── Tier 3: Live DB ILIKE scan ─────────────────────────
    # Use exact ILIKE for short values to avoid false positives (e.g. "in" matching "Singh")
    if len(val_lower) < 3:
        ilike_pattern = value
    else:
        ilike_pattern = f"%{value}%"
    try:
        from app.database import get_pool
        settings = get_settings()
        pool = get_pool()
        schema = settings.POSTGRES_SCHEMA

        async with pool.acquire() as conn:
            for tname in plan_tables:
                if tname not in schema_graph.tables:
                    continue
                for cname, cinfo in schema_graph.tables[tname].columns.items():
                    if cinfo["data_type"] not in ("character varying", "text", "varchar", "USER-DEFINED"):
                        continue
                    try:
                        row = await conn.fetchrow(
                            f"SELECT {cname} FROM {schema}.{tname} WHERE {cname} ILIKE $1 LIMIT 1",
                            ilike_pattern
                        )
                        if row:
                            exact_val = row[cname]
                            logger.info(f"Value grounding (ILIKE): '{value}' -> '{exact_val}' in {tname}.{cname}")
                            return exact_val, tname, cname
                    except Exception:
                        continue
    except Exception as e:
        logger.debug(f"ILIKE value grounding skipped: {e}")

    # ── Tier 4: Fuzzy Levenshtein on sample data ───────────
    best_match = None
    best_score = 0
    for tname in plan_tables:
        if tname not in schema_graph.tables:
            continue
        for sample_row in schema_graph.tables[tname].sample_rows:
            for cname, cval in sample_row.items():
                if not isinstance(cval, str):
                    continue
                score = fuzz.ratio(val_lower, cval.lower())
                if score > best_score and score >= 75:
                    best_score = score
                    best_match = (cval, tname, cname)

    if best_match:
        logger.info(f"Value grounding (fuzzy): '{value}' -> '{best_match[0]}' (score={best_score})")
        return best_match

    # ── Date normalization ─────────────────────────────────
    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    if val_lower in month_map:
        return str(month_map[val_lower]), None, None
    # "3rd month", "1st month" etc.
    ordinal_match = re.match(r"(\d+)(?:st|nd|rd|th)\s+month", val_lower)
    if ordinal_match:
        return ordinal_match.group(1), None, None

    # Fallback: return original value
    return value, None, None


async def _find_column_for_value(value: str, table_name: str, schema_graph: SchemaGraph) -> str | None:
    """Find which column a grounded value belongs to via live DB check."""
    try:
        from app.database import get_pool
        settings = get_settings()
        pool = get_pool()
        schema = settings.POSTGRES_SCHEMA

        if table_name not in schema_graph.tables:
            return None

        async with pool.acquire() as conn:
            for cname, cinfo in schema_graph.tables[table_name].columns.items():
                if cinfo["data_type"] not in ("character varying", "text", "varchar", "USER-DEFINED"):
                    continue
                try:
                    row = await conn.fetchrow(
                        f"SELECT 1 FROM {schema}.{table_name} WHERE {cname} = $1 LIMIT 1",
                        value
                    )
                    if row:
                        return cname
                except Exception:
                    continue
    except Exception:
        pass
    return None


def _spacy_extract(text: str, schema_graph: SchemaGraph) -> dict:
    """Extract entities using spaCy NER + custom pattern matching."""
    if _nlp is None:
        return {}

    doc = _nlp(text)
    table_hints = []
    filter_values = []

    # NER: extract named entities as potential filter values
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC", "ORG", "PERSON", "NORP", "PRODUCT"):
            filter_values.append(ent.text)
        elif ent.label_ in ("CARDINAL", "MONEY", "QUANTITY", "ORDINAL"):
            filter_values.append(ent.text)

    # Custom patterns: match tokens against known table/synonym names
    # First pass: check bigrams (multi-word synonyms take priority over individual tokens)
    bigram_token_indices = set()  # track tokens consumed by bigram matches
    for token in doc:
        if token.is_stop or token.is_punct or len(token.text) < 2:
            continue
        if token.i + 1 < len(doc):
            bigram = f"{token.text} {doc[token.i + 1].text}".lower()
            resolved_bi = schema_graph.resolve_table(bigram.replace(" ", "_"))
            if resolved_bi and resolved_bi not in table_hints:
                table_hints.append(resolved_bi)
                bigram_token_indices.add(token.i)
                bigram_token_indices.add(token.i + 1)

    # Second pass: individual tokens (skip tokens already consumed by bigrams)
    for token in doc:
        if token.is_stop or token.is_punct or len(token.text) < 2:
            continue
        if token.i in bigram_token_indices:
            continue
        token_lower = token.text.lower()
        resolved = schema_graph.resolve_table(token_lower)
        if resolved and resolved not in table_hints:
            table_hints.append(resolved)

    # Aggregation hints from dependency parse
    aggregation = "none"
    agg_map = {"how many": "count", "total": "sum",
               "average": "avg", "minimum": "min", "maximum": "max",
               "highest": "max", "lowest": "min"}
    # Single-word aggregation keywords (matched with word boundaries to avoid "account" matching "count")
    agg_words = {"count": "count", "avg": "avg", "sum": "sum"}
    text_lower = text.lower()
    for phrase, agg_type in agg_map.items():
        if phrase in text_lower:
            aggregation = agg_type
            break
    if aggregation == "none":
        for word, agg_type in agg_words.items():
            if re.search(rf"\b{word}\b", text_lower):
                aggregation = agg_type
                break

    # Sort/limit hints
    order_by = None
    limit = None
    sort_match = re.search(r"(?:sort|order)\s+(?:by)\s+(\w+)", text_lower)
    if sort_match:
        order_by = sort_match.group(1)
    limit_match = re.search(r"(?:top|first|limit)\s+(\d+)", text_lower)
    if limit_match:
        limit = int(limit_match.group(1))

    result = {}
    if table_hints:
        result["table_hints"] = table_hints
    if filter_values:
        result["filter_values"] = filter_values
    if aggregation != "none":
        result["aggregation"] = aggregation
    if order_by:
        result["order_by"] = order_by
    if limit:
        result["limit"] = limit

    return result


async def build_query_plan(user_message: str, schema_graph: SchemaGraph, session_state: dict = None) -> QueryPlan:
    """Build a QueryPlan from a natural language message using the 8-stage pipeline."""
    settings = get_settings()
    schema = settings.POSTGRES_SCHEMA
    plan = QueryPlan()

    # ── Stage 1: Entity extraction (spaCy NER + LLM) ──────
    # First pass: spaCy NER + custom patterns (zero LLM cost)
    spacy_entities = _spacy_extract(user_message, schema_graph)
    if spacy_entities:
        logger.info(f"Stage 1 spaCy entities: {spacy_entities}")

    # Second pass: LLM extraction (fills gaps spaCy missed)
    synonyms_display = {syn: tbl for syn, tbl in schema_graph.synonyms.items() if syn != tbl}
    extract_prompt = f"""Analyze this database query and extract entities.
Available tables: {', '.join(schema_graph.tables.keys())}
Known synonyms: {json.dumps(synonyms_display)}

User query: "{user_message}"

Rules:
- ONLY map user terms to tables if they exactly match a table name or a known synonym above
- Do NOT infer table mappings based on general knowledge (e.g. do NOT map "buyers" to "customers" unless "buyers" is a known synonym)
- If a term does not match any table or synonym, do NOT include it in table_hints
- Include unrecognized terms in filter_values instead

Return JSON only:
{{"table_hints": ["matched table names"],
  "filter_values": ["values to filter by"],
  "aggregation": "count|sum|avg|min|max|none",
  "order_by": "column name or null",
  "limit": null or number}}

Aggregation rules:
- Use "count" ONLY when the user explicitly asks "how many", "count", or "number of"
- Use "none" when the user says "show", "list", "display", "get", "find", "fetch", or "all" — these mean return rows, not a count"""

    llm_entities = await chat_json([{"role": "user", "content": extract_prompt}])

    # Merge: spaCy results take precedence, LLM fills gaps
    entities = {**llm_entities}
    if spacy_entities.get("table_hints"):
        existing = set(entities.get("table_hints", []))
        entities["table_hints"] = list(existing | set(spacy_entities["table_hints"]))
    if spacy_entities.get("filter_values"):
        existing = set(entities.get("filter_values", []))
        entities["filter_values"] = list(existing | set(spacy_entities["filter_values"]))
    if spacy_entities.get("aggregation") and entities.get("aggregation", "none") == "none":
        entities["aggregation"] = spacy_entities["aggregation"]

    # Deterministic override: "show all", "list", "display" etc. should never be count
    msg_lower = user_message.lower()
    if entities.get("aggregation") == "count" and not any(
        kw in msg_lower for kw in ["how many", "count", "number of", "total number"]
    ):
        if re.search(r"\b(show|list|display|get|find|fetch|retrieve|give)\b", msg_lower):
            entities["aggregation"] = "none"
    if spacy_entities.get("order_by") and not entities.get("order_by"):
        entities["order_by"] = spacy_entities["order_by"]
    if spacy_entities.get("limit") and not entities.get("limit"):
        entities["limit"] = spacy_entities["limit"]

    # Validate table_hints: only keep hints that trace back to a word or phrase in the user's message
    # via exact table name, known synonym, or plural/singular match
    validated_hints = []
    msg_tokens = {w.lower() for w in re.findall(r"\w+", user_message)}
    # Build bigrams from message for multi-word synonym matching (e.g. "account holders" -> "account_holders")
    msg_words = [w.lower() for w in re.findall(r"\w+", user_message)]
    msg_bigrams = {f"{msg_words[i]}_{msg_words[i+1]}" for i in range(len(msg_words) - 1)}

    for hint in entities.get("table_hints", []):
        hint_lower = hint.lower()
        # Direct: user said the table name or its singular/plural
        if hint_lower in msg_tokens or hint_lower.rstrip("s") in msg_tokens or hint_lower + "s" in msg_tokens:
            validated_hints.append(hint)
            continue
        # Reverse synonym: check if any user token is a known synonym for this table
        found = False
        for token in msg_tokens:
            resolved = schema_graph.resolve_table(token)
            if resolved == hint_lower:
                validated_hints.append(hint)
                found = True
                break
        if found:
            continue
        # Reverse synonym on bigrams: check multi-word phrases (e.g. "account_holders")
        for bigram in msg_bigrams:
            resolved = schema_graph.resolve_table(bigram)
            if resolved == hint_lower:
                validated_hints.append(hint)
                break
    if len(validated_hints) != len(entities.get("table_hints", [])):
        rejected = set(entities.get("table_hints", [])) - set(validated_hints)
        logger.info(f"Stage 1 rejected ungrounded table hints: {rejected}")
    entities["table_hints"] = validated_hints

    # Remove filter_values that are actually table synonyms (e.g. "account holders" resolved to "customers")
    # But retain non-table parts as filter values (e.g. "credit_card" → table "cards", keep "credit" as filter)
    if entities.get("filter_values"):
        resolved_tables = set(validated_hints)
        cleaned_filters = []
        for fv in entities["filter_values"]:
            fv_key = fv.lower().replace(" ", "_")
            resolved = schema_graph.resolve_table(fv_key)
            if resolved and resolved in resolved_tables:
                logger.info(f"Stage 1: removed '{fv}' from filter_values (resolved as table synonym)")
                # For compound synonyms, retain the non-table-name parts as filter values
                # e.g. "credit_card" or "credit card" → table "cards", keep "credit" as filter
                # Normalize: split on both spaces and underscores
                fv_parts = re.split(r"[\s_]+", fv.lower())
                table_base = resolved.rstrip("s")  # "cards" → "card"
                residual = [p for p in fv_parts if p != resolved and p != table_base
                            and p + "s" != resolved and p.rstrip("s") != table_base]
                if residual:
                    residual_val = " ".join(residual)
                    cleaned_filters.append(residual_val)
                    logger.info(f"Stage 1: retained '{residual_val}' as filter value from synonym '{fv}'")
                continue
            cleaned_filters.append(fv)
        entities["filter_values"] = cleaned_filters

    logger.info(f"Stage 1 merged entities: {entities}")

    # ── Stage 2a: Table resolution ─────────────────────────
    table_hints = entities.get("table_hints", [])
    for hint in table_hints:
        resolved = schema_graph.resolve_table(hint)
        if resolved and resolved not in plan.tables:
            plan.tables.append(resolved)

    # ── Stage 2a-ii: Primary table detection ──────────────
    # The table the user wants to SEE (subject of show/list/get) should be first.
    # Tables used only for filtering (in subordinate clauses) go after.
    if len(plan.tables) >= 2:
        # Find the first resolved table noun after the action verb
        # e.g. "show all the customers who are using credit cards" → primary = customers
        subject_match = re.search(
            r"\b(?:show|list|display|get|find|fetch|retrieve|give)\b"
            r"(?:\s+(?:all|me|the|my|every|each|a))*"  # skip filler words
            r"\s+(\w+)",              # the subject noun
            msg_lower
        )
        if subject_match:
            subject_word = subject_match.group(1)
            subject_table = schema_graph.resolve_table(subject_word)
            if subject_table and subject_table in plan.tables and plan.tables[0] != subject_table:
                plan.tables.remove(subject_table)
                plan.tables.insert(0, subject_table)
                logger.info(f"Stage 2a: reordered primary table to '{subject_table}' (subject of user query)")

    # ── Stage 2b: Value-implied table discovery ────────────
    # Only discover additional tables if we already have at least one table from Stage 2a.
    # Without a base table, filter values can't meaningfully imply related tables.
    if not plan.tables:
        logger.info("Stage 2b skipped: no base tables resolved from user message")

    for val in entities.get("filter_values", []) if plan.tables else []:
        val_lower = val.lower()
        # Check domain synonyms for table hints
        for syn_key, syn_table in DOMAIN_SYNONYMS.items():
            if val_lower in syn_key and syn_table in schema_graph.tables and syn_table not in plan.tables:
                plan.tables.append(syn_table)

        # Check value_index -> DOMAIN_SYNONYMS chain
        idx_match = schema_graph.lookup_value(val_lower)
        grounded = idx_match[0] if idx_match else val_lower
        for syn_key, syn_table in DOMAIN_SYNONYMS.items():
            if grounded.lower() in syn_key and syn_table in schema_graph.tables and syn_table not in plan.tables:
                plan.tables.append(syn_table)

        # Live ILIKE scan to discover tables implied by value
        # Skip if the value already exists in a column of an existing plan table
        val_found_in_plan = False
        try:
            from app.database import get_pool
            pool = get_pool()
            async with pool.acquire() as conn:
                for tname in plan.tables:
                    if tname not in schema_graph.tables:
                        continue
                    for cname, cinfo in schema_graph.tables[tname].columns.items():
                        if cinfo["data_type"] not in ("character varying", "text", "varchar", "USER-DEFINED"):
                            continue
                        try:
                            row = await conn.fetchrow(
                                f"SELECT 1 FROM {schema}.{tname} WHERE {cname} ILIKE $1 LIMIT 1",
                                f"%{val}%"
                            )
                            if row:
                                val_found_in_plan = True
                                break
                        except Exception:
                            continue
                    if val_found_in_plan:
                        break
        except Exception:
            pass

        if not val_found_in_plan:
            try:
                pool = get_pool()
                async with pool.acquire() as conn:
                    for tname, tmeta in schema_graph.tables.items():
                        if tname in plan.tables:
                            continue
                        for cname, cinfo in tmeta.columns.items():
                            if cinfo["data_type"] not in ("character varying", "text", "varchar", "USER-DEFINED"):
                                continue
                            try:
                                row = await conn.fetchrow(
                                    f"SELECT 1 FROM {schema}.{tname} WHERE {cname} ILIKE $1 LIMIT 1",
                                    f"%{val}%"
                                )
                                if row:
                                    # Validate via FK: does this table connect to existing plan tables?
                                    if not plan.tables or any(schema_graph.join_path(tname, pt) for pt in plan.tables):
                                        if tname not in plan.tables:
                                            plan.tables.append(tname)
                                            logger.info(f"Stage 2b: Value '{val}' implied table '{tname}' via {cname}")
                                    break
                            except Exception:
                                continue
            except Exception:
                pass

    # No LLM fallback for table resolution — if no table was resolved from
    # the user's message via exact match or known synonyms, the query is out of scope.
    # The pipeline will handle this with an appropriate response.

    # ── Stage 3: FK join discovery ─────────────────────────
    if len(plan.tables) >= 2:
        for i in range(len(plan.tables) - 1):
            jp = schema_graph.join_path(plan.tables[i], plan.tables[i + 1])
            if jp:
                plan.joins.append(jp)

    # ── Stage 4: Column grounding ──────────────────────────
    # (Columns resolved during value grounding and SQL generation)

    # ── Stage 5: Value grounding (full 3-tier) ─────────────
    for val in entities.get("filter_values", []):
        grounded_value, grounded_table, grounded_column = await _ground_value_full(val, plan.tables, schema_graph)

        # If grounding found the table and column, use them directly
        if grounded_table and grounded_column:
            plan.filters.append({
                "column": f"{grounded_table}.{grounded_column}",
                "operator": "=",
                "value": grounded_value,
            })
            continue

        # Otherwise find which column this value belongs to
        matched = False
        for tname in plan.tables:
            col = await _find_column_for_value(grounded_value, tname, schema_graph)
            if col:
                plan.filters.append({
                    "column": f"{tname}.{col}",
                    "operator": "=",
                    "value": grounded_value,
                })
                matched = True
                break

        # Last resort: match by data type
        if not matched:
            for tname in plan.tables:
                for cname, cinfo in schema_graph.tables[tname].columns.items():
                    if cinfo["data_type"] in ("character varying", "text", "varchar", "USER-DEFINED"):
                        plan.filters.append({
                            "column": f"{tname}.{cname}",
                            "operator": "=",
                            "value": grounded_value,
                        })
                        matched = True
                        break
                if matched:
                    break

    # ── Stage 6: Aggregation, ordering & limit ─────────────
    agg = entities.get("aggregation", "none")
    if agg and agg != "none":
        plan.aggregation = {"type": agg.upper()}

    plan.order_by = entities.get("order_by")
    plan.limit = entities.get("limit")

    # ── Stage 7: SQL generation from plan ──────────────────
    # (SQL is generated in db_pipeline using generate_sql)

    return plan


async def generate_sql(plan: QueryPlan, schema: str, schema_graph=None) -> str:
    """Generate SQL from a structured QueryPlan via LLM."""
    plan_json = json.dumps(plan.to_dict(), indent=2, default=str)

    primary_table = plan.tables[0] if plan.tables else ""
    has_joins = len(plan.tables) > 1

    # Determine SELECT clause — build it deterministically instead of letting LLM guess
    agg = plan.aggregation
    if agg and agg != "none":
        agg_type = agg if isinstance(agg, str) else agg.get("type", "")
        if agg_type.upper() == "COUNT":
            select_clause = f"SELECT COUNT(t1.*)"
        else:
            select_clause = None  # Let LLM decide for SUM/AVG/MIN/MAX
    elif has_joins:
        select_clause = f"SELECT t1.*"
    else:
        select_clause = "SELECT *"

    # Only include column names for joined/filter tables (not primary) to keep prompt small
    column_hint = ""
    if schema_graph and has_joins:
        for tname in plan.tables[1:]:
            if tname in schema_graph.tables:
                cols = list(schema_graph.tables[tname].columns.keys())
                column_hint += f"\n{tname} columns: {', '.join(cols)}"

    select_instruction = f"IMPORTANT: The SELECT clause MUST be exactly: {select_clause} (where t1 is the alias for '{primary_table}'). Do NOT list individual columns." if select_clause else ""

    prompt = f"""Generate a PostgreSQL SELECT from this plan. Return ONLY SQL, no explanation.
Schema: {schema}
Plan: {plan_json}
{select_instruction}
Rules: SELECT only. Prefix tables with {schema}. Use unique lowercase aliases. Use exact column/table names from plan. No SQL comments.{column_hint}"""

    sql = await chat([{"role": "user", "content": prompt}], temperature=0.1)
    return _extract_sql(sql)


async def amend_sql(original_sql: str, follow_up: str, plan: QueryPlan, schema_graph: SchemaGraph) -> str:
    """Amend an existing SQL query based on a follow-up message."""
    follow_lower = follow_up.lower()

    # ── Pass 1: Deterministic amendments ───────────────────
    amended = original_sql

    # GROUP BY / summarize by: "summarize by status", "summarize this data with respect to loan status"
    group_match = re.search(
        r"(?:summarize|group|break\s*down|split|categorize|distribute)\s+.*?"
        r"(?:by|with\s+respect\s+to|per|for\s+each)\s+(?:\w+\s+)*?(\w+)\s*$",
        follow_lower
    )
    if not group_match:
        group_match = re.search(r"(?:for|per)\s+each\s+(\w+)", follow_lower)
    if not group_match:
        group_match = re.search(r"count\s+(?:of\s+records?\s+)?(?:by|per|for\s+each)\s+(\w+)", follow_lower)
    if group_match:
        group_col = group_match.group(1)
        # Resolve column name against schema
        referenced = [t for t in schema_graph.tables if t in original_sql.lower()]
        for tname in referenced:
            resolved = schema_graph.resolve_column(tname, group_col)
            if resolved:
                group_col = resolved
                break
        # Find the table alias (e.g. "l" from "loans l")
        alias_match = re.search(r"FROM\s+\S+\s+(\w)\b", amended, re.IGNORECASE)
        col_ref = f"{alias_match.group(1)}.{group_col}" if alias_match else group_col
        # Remove existing GROUP BY if any
        amended = re.sub(r"\s*GROUP BY[^;]*", "", amended, flags=re.IGNORECASE)
        # Remove ORDER BY and LIMIT for group queries
        amended = re.sub(r"\s*ORDER BY[^;]*", "", amended, flags=re.IGNORECASE)
        amended = re.sub(r"\s*LIMIT\s+\d+", "", amended, flags=re.IGNORECASE)
        # Transform SELECT clause to include grouping column + COUNT
        amended = re.sub(
            r"SELECT\s+COUNT\s*\(\*\)",
            f"SELECT {col_ref}, COUNT(*)",
            amended, flags=re.IGNORECASE
        )
        amended = re.sub(
            r"SELECT\s+\*",
            f"SELECT {col_ref}, COUNT(*)",
            amended, flags=re.IGNORECASE
        )
        amended = amended.strip().rstrip(";")
        amended = f"{amended} GROUP BY {col_ref};"
        return amended

    # COUNT <-> SELECT * swap
    if any(kw in follow_lower for kw in ["list all", "show all", "get all", "all those", "all records"]):
        if "COUNT(*)" in amended.upper() or "COUNT (" in amended.upper():
            amended = amended.replace("SELECT COUNT(*)", "SELECT *").replace("SELECT count(*)", "SELECT *")
            return amended

    if any(kw in follow_lower for kw in ["how many", "count", "number of"]):
        # Strip GROUP BY and grouping columns — user wants a simple count
        if re.search(r"\bGROUP\s+BY\b", amended, re.IGNORECASE):
            amended = re.sub(r"\s*GROUP BY[^;]*", "", amended, flags=re.IGNORECASE)
            # Remove grouping column from SELECT, keep only COUNT(*)
            amended = re.sub(
                r"SELECT\s+\S+\s*,\s*COUNT\s*\(\*\)",
                "SELECT COUNT(*)",
                amended, flags=re.IGNORECASE
            )
        if "SELECT *" in amended.upper():
            amended = re.sub(r"SELECT\s+\*\s+FROM", "SELECT COUNT(*) FROM", amended, flags=re.IGNORECASE)
        # Remove ORDER BY and LIMIT for count queries
        amended = re.sub(r"ORDER BY.*?(;|$)", ";", amended, flags=re.IGNORECASE)
        amended = re.sub(r"LIMIT\s+\d+", "", amended, flags=re.IGNORECASE)
        amended = amended.strip().rstrip(";") + ";"
        # If the message only asks for count (no additional filters), return early
        # Otherwise fall through to LLM to handle filters (e.g. "how many in Gurgaon")
        count_only = re.sub(r"\b(how many|count|number of|of them|are there|do we have)\b", "", follow_lower, flags=re.IGNORECASE).strip()
        if len(count_only) < 3:
            return amended
        # Pass the COUNT version to the LLM for further amendment with filters
        original_sql = amended

    # Yes/no questions about a column value: "is this loan active?", "is it closed?"
    # Convert to SELECT of the relevant column instead of adding a filter
    yesno_match = re.match(r"^(is|are|does|do|was|were|has|have)\b", follow_lower)
    if yesno_match:
        # Extract the referenced tables from the original SQL
        referenced_tables = [t for t in schema_graph.tables if t in original_sql.lower()]
        # Find the value being asked about (e.g. "active", "closed", "approved")
        # by checking follow-up words against the value index
        asked_value = None
        asked_column = None
        for word in re.findall(r"[A-Za-z_]+", follow_up):
            if len(word) < 3:
                continue
            match = schema_graph.lookup_value(word, referenced_tables)
            if match:
                asked_value, _, asked_column = match
                break
        if asked_column:
            # Replace SELECT with the specific column and remove LIMIT
            # Keep existing WHERE clause from the previous query
            where_match = re.search(r"(WHERE\s+.+?)(?:\s+ORDER BY|\s+LIMIT|\s*;)", amended, re.IGNORECASE | re.DOTALL)
            where_clause = where_match.group(1) if where_match else ""
            # Find which table has this column
            table_ref = referenced_tables[0] if referenced_tables else ""
            schema_name = schema_graph.tables[table_ref].schema if table_ref in schema_graph.tables else ""
            amended = f"SELECT {asked_column} FROM {schema_name}.{table_ref} {where_clause};"
            logger.info(f"Yes/no amendment: SELECT {asked_column} for value '{asked_value}'")
            return amended

    # ORDER BY injection: "sort by name", "order by balance desc"
    sort_match = re.search(r"(?:sort|order)\s+(?:by|them\s+by)\s+(\w+)\s*(asc|desc)?", follow_lower)
    if sort_match:
        sort_col = sort_match.group(1)
        sort_dir = (sort_match.group(2) or "ASC").upper()
        # Resolve column name
        if schema_graph:
            for tname in (plan.tables if plan else schema_graph.tables.keys()):
                if tname in schema_graph.tables:
                    resolved = schema_graph.resolve_column(tname, sort_col)
                    if resolved:
                        sort_col = resolved
                        break
        # Remove existing ORDER BY if any, then add new one
        amended = re.sub(r"\s*ORDER BY[^;]*", "", amended, flags=re.IGNORECASE)
        amended = amended.strip().rstrip(";")
        amended = f"{amended} ORDER BY {sort_col} {sort_dir};"
        return amended

    # LIMIT modification: "top 10", "first 5", "limit 20"
    limit_match = re.search(r"(?:top|first|limit|show)\s+(\d+)", follow_lower)
    if limit_match:
        limit_val = int(limit_match.group(1))
        # Remove existing LIMIT if any, then add new one
        amended = re.sub(r"\s*LIMIT\s+\d+", "", amended, flags=re.IGNORECASE)
        amended = amended.strip().rstrip(";")
        amended = f"{amended} LIMIT {limit_val};"
        return amended

    # ── Pass 2: LLM amendment ──────────────────────────────
    # Only include tables already referenced in the original SQL
    referenced_tables = {}
    for tname, tmeta in schema_graph.tables.items():
        if tname in original_sql.lower():
            referenced_tables[tname] = list(tmeta.columns.keys())

    # Get actual distinct values from the DB for referenced tables
    # so the LLM can map user terms to real values (e.g. "Visakhapatnam" -> "Vizag")
    column_values = schema_graph.get_column_values(list(referenced_tables.keys()))

    prompt = f"""Amend this SQL query based on the follow-up question.

Original SQL: {original_sql}
Follow-up: "{follow_up}"
Tables and columns: {json.dumps(referenced_tables, default=str)}

ACTUAL values in the database (use ONLY these values in WHERE clauses):
{json.dumps(column_values, indent=2, default=str)}

Rules:
- CRITICAL: Only generate SELECT queries
- CRITICAL: Do NOT include SQL comments (-- or /* */)
- CRITICAL: Only use values from the ACTUAL values list above. Map user terms to the closest matching actual value (e.g. if user says "Visakhapatnam" but actual values have "Vizag", use "Vizag")
- Preserve the schema prefix on all table names exactly as in the original SQL
- Start from the original SQL and modify what's necessary to address the follow-up
- If the user asks to "summarize by", "group by", "break down by", or "for each", restructure the SELECT to include the grouping column and add a GROUP BY clause
- Do NOT add JOINs unless the follow-up explicitly asks for data from another table
- Return ONLY the amended SQL, no explanation"""

    amended = await chat([{"role": "user", "content": prompt}], temperature=0.1)
    amended = _extract_sql(amended)

    # ── Pass 3: Ground new string literals against DB ──────
    amended = await ground_sql_literals(amended, schema_graph)

    # ── Pass 4: Deduplicate IN clause values ──────────────
    amended = _dedup_in_clauses(amended)

    return amended


async def ground_sql_literals(sql: str, schema_graph: SchemaGraph) -> str:
    """Post-process SQL to canonicalize string literals against actual DB values.

    Strategy:
    1. Parse column-value pairs from SQL (e.g. c.city = 'Visakhapatnam')
    2. Check if the value exists in that column's known values (from value_index)
    3. If not, find the best match from that column's actual values via LLM
    4. For unmatched literals, fall back to value_index lookup
    """
    literals = re.findall(r"'([^']+)'", sql)
    if not literals:
        return sql

    # ── Step 1: Extract column-value context from SQL ──
    # Match patterns like: column = 'value', column IN ('val1', 'val2')
    col_value_pairs = {}  # literal -> (table_alias_or_name, column)
    # Pattern: alias.column = 'value' or column = 'value'
    for m in re.finditer(r"(\w+)\.(\w+)\s*=\s*'([^']+)'", sql):
        col_value_pairs[m.group(3)] = (m.group(1), m.group(2))
    # Also match IN clauses: alias.column IN ('val1', 'val2', ...)
    for m in re.finditer(r"(\w+)\.(\w+)\s+IN\s*\(([^)]+)\)", sql, re.IGNORECASE):
        alias, col = m.group(1), m.group(2)
        for val in re.findall(r"'([^']+)'", m.group(3)):
            col_value_pairs[val] = (alias, col)

    # ── Step 2: Resolve table aliases to actual table names ──
    alias_map = {}
    for m in re.finditer(r"(\w+)\.(\w+)\s+(\w+)\b", sql):
        # schema.table alias pattern
        alias_map[m.group(3)] = m.group(2)
    for m in re.finditer(r"\b(\w+)\s+(\w{1,3})\b", sql):
        if m.group(1).lower() in [t for t in schema_graph.tables]:
            alias_map[m.group(2)] = m.group(1)

    for literal in literals:
        if literal.upper() in ("TRUE", "FALSE", "NULL", "0", "1"):
            continue

        # ── Check if value exists in value_index (exact match) ──
        exact = schema_graph.lookup_value(literal)
        if exact and exact[0] == literal:
            continue  # Value exists in DB as-is

        # ── Get the column context for this literal ──
        if literal in col_value_pairs:
            alias, col = col_value_pairs[literal]
            tname = alias_map.get(alias, alias)

            # Get known values for this specific column
            column_values = schema_graph.get_column_values([tname])
            col_key = f"{tname}.{col}"
            known_values = column_values.get(col_key, [])

            if known_values:
                # Check if literal already matches a known value (case-insensitive)
                if any(v.lower() == literal.lower() for v in known_values):
                    # Fix casing if needed
                    for v in known_values:
                        if v.lower() == literal.lower() and v != literal:
                            sql = sql.replace(f"'{literal}'", f"'{v}'")
                            logger.info(f"ground_sql_literals (case fix): '{literal}' -> '{v}'")
                    continue

                # Value not in DB — ask LLM to pick the best match
                try:
                    best = await _llm_resolve_value(literal, known_values)
                    if best and best != literal:
                        sql = sql.replace(f"'{literal}'", f"'{best}'")
                        logger.info(f"ground_sql_literals (LLM): '{literal}' -> '{best}' in {col_key}")
                except Exception as e:
                    logger.debug(f"ground_sql_literals LLM resolve failed: {e}")
                continue

        # ── Fallback: value_index fuzzy lookup (no column context) ──
        match = schema_graph.lookup_value(literal)
        if match:
            canonical, _, _ = match
            if canonical != literal:
                sql = sql.replace(f"'{literal}'", f"'{canonical}'")
                logger.info(f"ground_sql_literals (index): '{literal}' -> '{canonical}'")

    return sql


async def _llm_resolve_value(user_term: str, known_values: list[str]) -> str | None:
    """Ask LLM to map a user term to the closest matching actual DB value."""
    prompt = f"""The user used the term "{user_term}" but the database only contains these values:
{json.dumps(known_values)}

Which value from the list above is the user most likely referring to?
If none match, respond with exactly: NONE
Otherwise respond with ONLY the exact matching value from the list, nothing else."""

    result = await chat([{"role": "user", "content": prompt}], temperature=0.0)
    result = result.strip().strip("'\"")
    if result == "NONE" or result not in known_values:
        return None
    return result
