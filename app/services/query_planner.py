"""8-stage query planner — translates natural language to structured QueryPlan."""
import logging
import json
import re
from thefuzz import fuzz
from app.services.llm_client import chat, chat_json, is_llm_error
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


def _ensure_schema_prefix(sql: str, schema: str, known_tables: set[str]) -> str:
    """Deterministically prefix bare table names with the schema.

    LLMs (especially smaller ones like llama3.1) often ignore the instruction
    to prefix tables with the schema. This fixes it post-hoc by scanning the
    SQL for any known table name that isn't already prefixed with 'schema.'.

    Example:
        "FROM erp_customers AS t1" → "FROM erp.erp_customers AS t1"
    """
    if not known_tables or not schema:
        return sql

    for table in sorted(known_tables, key=len, reverse=True):
        # Match bare table name NOT already preceded by 'schema.'
        # Handles: FROM table, JOIN table, FROM table AS, etc.
        pattern = re.compile(
            r'(?<!' + re.escape(schema) + r'\.)' +     # negative lookbehind: not already prefixed
            r'\b(' + re.escape(table) + r')\b' +        # the table name
            r'(?!\s*\.)',                                # not followed by '.' (not a schema ref itself)
            re.IGNORECASE
        )
        sql = pattern.sub(f'{schema}.{table}', sql)

    return sql

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
    settings = get_settings()
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
        if faiss_results and faiss_results[0]["similarity"] > settings.FAISS_VALUE_SIMILARITY:
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
                if score > best_score and score >= settings.FUZZY_MATCH_THRESHOLD:
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

    # Normalize numeric shorthands in filter values (e.g. "50k" → 50000, "2M" → 2000000)
    normalized_filters = []
    for fv in filter_values:
        norm_match = re.match(r'^(\d+(?:\.\d+)?)\s*([kKmMbBcCrR]|cr|lakh|lakhs|crore|crores)$', fv.strip())
        if norm_match:
            num = float(norm_match.group(1))
            suffix = norm_match.group(2).lower()
            multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000,
                           "c": 10_000_000, "cr": 10_000_000, "crore": 10_000_000, "crores": 10_000_000,
                           "r": 100_000, "lakh": 100_000, "lakhs": 100_000}
            multiplier = multipliers.get(suffix, 1)
            expanded = int(num * multiplier)
            normalized_filters.append(str(expanded))
            logger.info(f"Stage 1: normalized '{fv}' -> {expanded}")
        else:
            normalized_filters.append(fv)
    filter_values = normalized_filters

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

    # ── Stage 0: Semantic NLU pass (fast, no LLM) ─────────
    # Run before spaCy and LLM so aggregation intent (e.g. "distribution" →
    # count_group_by) is established authoritatively from surface semantics,
    # not re-derived (and potentially overridden) by the LLM extraction step.
    from app.services.query_nlu import parse_query as _parse_nlu
    nlu_result = await _parse_nlu(user_message)
    logger.info(
        f"Stage 0 NLU | entities={nlu_result.entities} "
        f"agg={nlu_result.aggregation}({nlu_result.aggregation_trigger}) "
        f"action={nlu_result.action} limit={nlu_result.limit}"
    )

    # ── Stage 1: Entity extraction (spaCy NER + LLM) ──────
    # First pass: spaCy NER + custom patterns (zero LLM cost)
    spacy_entities = _spacy_extract(user_message, schema_graph)
    if spacy_entities:
        logger.info(f"Stage 1 spaCy entities: {spacy_entities}")

    # Second pass: LLM extraction (fills gaps spaCy missed)
    synonyms_display = {syn: tbl for syn, tbl in schema_graph.synonyms.items() if syn != tbl}

    # Build domain-aware table descriptions for the LLM
    table_descriptions = {}
    for tname, tmeta in schema_graph.tables.items():
        desc_parts = []
        if tmeta.description:
            desc_parts.append(tmeta.description)
        cols = list(tmeta.columns.keys())[:10]
        desc_parts.append(f"columns: {', '.join(cols)}")
        table_descriptions[tname] = ". ".join(desc_parts)

    # Load semantic expansions if available (generated during seeding)
    semantic_hints = ""
    try:
        from app.database import get_pool
        pool = get_pool()
        async with pool.acquire() as conn:
            cached = await conn.fetchrow(
                f"SELECT description FROM {settings.APP_SCHEMA}.schema_index "
                f"WHERE table_name = '_semantic_expansions' AND column_name IS NULL"
            )
            if cached and cached["description"]:
                expansions = json.loads(cached["description"])
                # Format: "balance → money, funds, wealth"
                expansion_lines = []
                for key, terms in expansions.items():
                    if isinstance(terms, list) and terms:
                        flat = [str(t) for t in terms if not isinstance(t, (list, dict))]
                        if flat:
                            expansion_lines.append(f"  {key} → {', '.join(flat[:5])}")
                if expansion_lines:
                    semantic_hints = "\nSemantic mappings (use these to resolve conceptual queries):\n" + "\n".join(expansion_lines[:30])
    except Exception:
        pass

    tables_info = "\n".join(f"  {t}: {d}" for t, d in table_descriptions.items())

    # Surface NLU context to the LLM so it can use detected aggregation + entities
    nlu_context = ""
    if nlu_result.aggregation:
        nlu_context += (
            f"\nNLU pre-analysis detected:\n"
            f"  - Aggregation intent: {nlu_result.aggregation} "
            f"(triggered by word '{nlu_result.aggregation_trigger}')\n"
            f"  - Entity candidates: {nlu_result.entities}\n"
            f"  - Action: {nlu_result.action}\n"
            f"Use these hints to set 'aggregation' and identify the right table/column.\n"
            f"For 'count_group_by' aggregation use 'count' in the JSON response.\n"
        )

    extract_prompt = f"""Analyze this database query and extract entities.

Available tables and their descriptions:
{tables_info}

Known synonyms: {json.dumps(synonyms_display)}
{semantic_hints}
{nlu_context}
User query: "{user_message}"

Rules:
- Map user terms to tables using table names, known synonyms, descriptions, OR semantic mappings above
- Use the table descriptions and semantic mappings to resolve conceptual queries
  (e.g. if user says "money" and semantic mappings show "balance → money, funds", map to the table containing balance)
- If a term matches a semantic mapping, include the corresponding table in table_hints
- Include filter values (specific names, places, dates, numbers) in filter_values
- If the user asks about a concept that maps to a specific column via semantic mappings, include the column in order_by if relevant

CRITICAL — table_hints must include ALL tables needed to answer the query:
Step 1: Identify the primary entity (what the user wants to see)
Step 2: Think about what OTHER tables are needed to satisfy the query.
  - If the query references a relationship between entities, include all related tables
  - "account holders" = customers who have accounts → needs customers + accounts
  - "credit card transactions" = transactions done via credit cards → needs transactions + cards
  - "customers with loans" = needs customers + loans
Step 3: Include all identified tables in table_hints
Step 4: If a term acts as BOTH a table reference AND a qualifier/filter, include it in BOTH
  table_hints and filter_values.
  - "credit card transactions" → table_hints: ["transactions", "cards"], filter_values: ["credit"]
    (because "credit" qualifies the card_type)
  - "personal loans" → table_hints: ["loans"], filter_values: ["personal"]
  - "savings accounts" → table_hints: ["accounts"], filter_values: ["savings"]

Return JSON only:
{{"table_hints": ["matched table names"],
  "primary_entity": "the table whose DATA the user wants to see (not tables used only for filtering/joining)",
  "filter_values": ["values to filter by"],
  "aggregation": "count|sum|avg|min|max|none",
  "order_by": "column name or null",
  "limit": null or number}}

primary_entity rules:
- This is the table whose rows/details the user wants to RETRIEVE
- Example: "get me all account holders" → user wants CUSTOMER details, so primary_entity = "customers" (not "accounts")
- Example: "show me all loans in AP" → user wants LOAN details, so primary_entity = "loans"
- Example: "customers who have credit cards" → user wants CUSTOMER details, so primary_entity = "customers"

Aggregation rules:
- Use "count" ONLY when the user is explicitly asking for a NUMBER/COUNT (e.g. "how many", "count of", "number of", "total count")
- Use "none" for ALL other cases — especially when the user wants to SEE/RETRIEVE data
- "get me all X", "show all X", "list all X", "fetch all X", "display X" → aggregation MUST be "none"
- When in doubt, use "none" — it is better to return data rows than an unwanted count"""

    llm_entities = await chat_json([{"role": "user", "content": extract_prompt}])

    if is_llm_error(llm_entities):
        logger.warning(f"Stage 1 LLM entity extraction failed: {llm_entities.get('detail', llm_entities.get('error'))}. Falling back to spaCy-only entities.")
        llm_entities = {}   # treat as if LLM returned nothing — spaCy results will drive the plan

    # Merge: spaCy results take precedence, LLM fills gaps
    entities = {**llm_entities}
    if spacy_entities.get("table_hints"):
        existing = set(entities.get("table_hints", []))
        entities["table_hints"] = list(existing | set(spacy_entities["table_hints"]))
    if spacy_entities.get("filter_values"):
        existing = set(entities.get("filter_values", []))
        entities["filter_values"] = list(existing | set(spacy_entities["filter_values"]))
    # Deduplicate filter values (case-insensitive) — keep the first occurrence's casing
    if entities.get("filter_values"):
        seen = set()
        deduped = []
        for fv in entities["filter_values"]:
            key = fv.strip().lower()
            if key not in seen:
                seen.add(key)
                deduped.append(fv)
        entities["filter_values"] = deduped
    # Normalize any numeric shorthands in merged filter values
    if entities.get("filter_values"):
        _norm = []
        for fv in entities["filter_values"]:
            nm = re.match(r'^(\d+(?:\.\d+)?)\s*([kKmMbBcCrR]|cr|lakh|lakhs|crore|crores)$', fv.strip())
            if nm:
                num = float(nm.group(1))
                suffix = nm.group(2).lower()
                mults = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000,
                         "c": 10_000_000, "cr": 10_000_000, "crore": 10_000_000, "crores": 10_000_000,
                         "r": 100_000, "lakh": 100_000, "lakhs": 100_000}
                _norm.append(str(int(num * mults.get(suffix, 1))))
            else:
                _norm.append(fv)
        entities["filter_values"] = _norm
    if spacy_entities.get("aggregation") and entities.get("aggregation", "none") == "none":
        entities["aggregation"] = spacy_entities["aggregation"]

    # Safety override: if the LLM returned "count" but the user clearly wants to
    # retrieve/list data (not count it), correct it. The user must explicitly ask
    # for a count — ambiguous cases default to listing rows.
    #
    # EXCEPTION: if NLU detected a semantic aggregation word (e.g. "distribution",
    # "breakdown", "trend"), trust it — these words unambiguously mean group+count.
    msg_lower = user_message.lower()
    explicitly_counting = any(kw in msg_lower for kw in ["how many", "count of", "number of", "total number", "total count"])
    nlu_agg_authoritative = nlu_result.aggregation in (
        "count_group_by", "count_group_by_time", "count", "sum", "avg", "max", "min"
    )
    if entities.get("aggregation") == "count" and not explicitly_counting and not nlu_agg_authoritative:
        logger.info(f"Stage 1: overriding aggregation 'count' → 'none' (no explicit count keywords in message)")
        entities["aggregation"] = "none"

    # Apply NLU-detected aggregation when LLM returned 'none' but NLU found a
    # semantic aggregation trigger (e.g. "distribution" → count_group_by).
    if nlu_result.aggregation and entities.get("aggregation", "none") == "none":
        # Map NLU agg type back to the planner's vocabulary
        nlu_to_plan = {
            "count_group_by":      "count",
            "count_group_by_time": "count",
            "count":               "count",
            "sum":                 "sum",
            "avg":                 "avg",
            "max":                 "max",
            "min":                 "min",
        }
        mapped = nlu_to_plan.get(nlu_result.aggregation)
        if mapped:
            logger.info(
                f"Stage 1: NLU semantic aggregation '{nlu_result.aggregation}' "
                f"(trigger='{nlu_result.aggregation_trigger}') → applying '{mapped}'"
            )
            entities["aggregation"] = mapped

    # Store count_group_by hint in session state so SQL builder knows to add GROUP BY
    if nlu_result.aggregation in ("count_group_by", "count_group_by_time"):
        if session_state is not None:
            session_state["_nlu_group_by"] = True
            session_state["_nlu_aggregation"] = nlu_result.aggregation
    if spacy_entities.get("order_by") and not entities.get("order_by"):
        entities["order_by"] = spacy_entities["order_by"]
    if spacy_entities.get("limit") and not entities.get("limit"):
        entities["limit"] = spacy_entities["limit"]

    # Validate table_hints: keep hints that trace back to user's message
    # via exact name, synonym, plural/singular, OR semantic expansion
    validated_hints = []
    msg_tokens = {w.lower() for w in re.findall(r"\w+", user_message)}
    msg_words = [w.lower() for w in re.findall(r"\w+", user_message)]
    msg_bigrams = {f"{msg_words[i]}_{msg_words[i+1]}" for i in range(len(msg_words) - 1)}

    # Load semantic expansions for validation
    semantic_expansion_map: dict[str, set[str]] = {}
    try:
        from app.database import get_pool as _get_pool
        _pool = _get_pool()
        import asyncio
        async with _pool.acquire() as _conn:
            _cached = await _conn.fetchrow(
                f"SELECT description FROM {settings.APP_SCHEMA}.schema_index "
                f"WHERE table_name = '_semantic_expansions' AND column_name IS NULL"
            )
            if _cached and _cached["description"]:
                _expansions = json.loads(_cached["description"])
                for key, terms in _expansions.items():
                    table_name = key.split(".")[0]  # "accounts.balance" → "accounts"
                    if isinstance(terms, list):
                        for term in terms:
                            if isinstance(term, (list, dict)):
                                continue
                            for word in str(term).lower().split():
                                semantic_expansion_map.setdefault(word, set()).add(table_name)
    except Exception:
        pass

    for hint in entities.get("table_hints", []):
        hint_lower = hint.lower()
        # Validate: the table must exist in the schema (reject hallucinated tables)
        if hint_lower in schema_graph.tables:
            validated_hints.append(hint)
            continue
        # Try resolving as a synonym
        resolved = schema_graph.resolve_table(hint_lower)
        if resolved and resolved in schema_graph.tables:
            validated_hints.append(resolved)
            continue
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
    # If the user resolved an entity ambiguity in a prior turn, honour that choice
    # by inserting the resolved table at the front before processing LLM hints.
    forced_table = session_state.get("resolved_table") if session_state else None
    if forced_table and forced_table in schema_graph.tables:
        plan.tables.append(forced_table)
        logger.info(f"Stage 2a: using session-resolved table '{forced_table}' (ambiguous_entity resolved)")

    table_hints = entities.get("table_hints", [])
    for hint in table_hints:
        resolved = schema_graph.resolve_table(hint)
        if resolved and resolved not in plan.tables:
            plan.tables.append(resolved)

    # ── Stage 2a-ii: Primary table detection ──────────────
    # If a table was force-resolved from ambiguity, it always wins first position.
    # Only use the LLM's primary_entity to reorder when there is no forced table.
    if forced_table and forced_table in plan.tables:
        if plan.tables[0] != forced_table:
            plan.tables.remove(forced_table)
            plan.tables.insert(0, forced_table)
        # forced_table is now at position 0; do NOT let LLM primary_entity override it
    elif len(plan.tables) >= 2:
        primary_hint = entities.get("primary_entity", "")
        if primary_hint:
            primary_table = schema_graph.resolve_table(primary_hint)
            if primary_table and primary_table in plan.tables and plan.tables[0] != primary_table:
                plan.tables.remove(primary_table)
                plan.tables.insert(0, primary_table)
                logger.info(f"Stage 2a: reordered primary table to '{primary_table}' (LLM primary_entity)")

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
    # Check for pre-resolved ambiguities from session state
    resolved_ambiguities = session_state.get("resolved_ambiguities", []) if session_state else []
    resolved_map = {}  # token -> (table, column)
    for ra in resolved_ambiguities:
        token = ra.get("token", "").lower()
        resolved_ref = ra.get("resolved", "")
        if token and "." in resolved_ref:
            rt, rc = resolved_ref.split(".", 1)
            resolved_map[token] = (rt, rc)

    for val in entities.get("filter_values", []):
        # If this value was already resolved by a clarification, use the resolved column directly
        if val.lower() in resolved_map:
            res_table, res_column = resolved_map[val.lower()]
            # Ground the actual value in the resolved column
            grounded_value = val
            # Look up the actual DB value
            actual = schema_graph.lookup_value(val, [res_table])
            if actual:
                grounded_value = actual[0]
            else:
                # Try case-insensitive match from value index
                from app.database import get_pool
                try:
                    pool = get_pool()
                    async with pool.acquire() as conn:
                        row = await conn.fetchrow(
                            f"SELECT {res_column} FROM {settings.POSTGRES_SCHEMA}.{res_table} WHERE {res_column} ILIKE $1 LIMIT 1",
                            f"%{val}%"
                        )
                        if row:
                            grounded_value = row[res_column]
                except Exception:
                    pass
            plan.filters.append({
                "column": f"{res_table}.{res_column}",
                "operator": "=",
                "value": grounded_value,
            })
            logger.info(f"Stage 5: used resolved ambiguity for '{val}' → {res_table}.{res_column} = '{grounded_value}'")
            continue

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

    # ── Stage 5b: Remove unnecessary JOINed tables ─────────
    # After value grounding, check if any non-primary table is actually needed.
    # A JOINed table is unnecessary if no filter references it.
    # EXCEPTION: the user-resolved (forced) table is always kept regardless of filters.
    if len(plan.tables) >= 2:
        filter_tables = set()
        for f in plan.filters:
            col = f.get("column", "")
            if "." in col:
                filter_tables.add(col.split(".")[0])
        primary = plan.tables[0]
        needed_tables = [primary]
        for t in plan.tables[1:]:
            if t in filter_tables or t == forced_table:
                needed_tables.append(t)
            else:
                logger.info(f"Stage 5b: removed unnecessary table '{t}' (no filters reference it)")
        if len(needed_tables) != len(plan.tables):
            plan.tables = needed_tables
            # Also clean up joins that reference removed tables
            plan.joins = [j for j in plan.joins if all(
                t in needed_tables for t in [j.get("left_table", ""), j.get("right_table", "")]
            )] if plan.joins else []

    # ── Stage 6: Aggregation, ordering & limit ─────────────
    agg = entities.get("aggregation", "none")
    if agg and agg != "none":
        plan.aggregation = {"type": agg.upper()}

    plan.order_by = entities.get("order_by")
    plan.limit = entities.get("limit")

    # ── Stage 6b: GROUP BY column for count_group_by aggregation ──────────
    # When NLU detects "distribution"/"breakdown"/etc., resolve which column
    # to group by so generate_sql can produce deterministic GROUP BY SQL.
    if nlu_result.aggregation in ("count_group_by", "count_group_by_time") and plan.tables:
        primary_table = plan.tables[0]
        group_col = None

        # Priority 1: session-state resolved entity → use its column
        if session_state:
            for ra in session_state.get("resolved_ambiguities", []):
                if ra.get("type") == "ambiguous_entity":
                    res_table = ra.get("resolved_table") or ra.get("value", "")
                    if res_table == primary_table:
                        token = ra.get("token", "").replace(" ", "_")
                        col = schema_graph.resolve_column(primary_table, token)
                        if col:
                            group_col = col
                            break

        # Priority 2: match NLU entity phrases against primary table columns
        if not group_col:
            for entity_phrase in nlu_result.entities:
                col = schema_graph.resolve_column(
                    primary_table, entity_phrase.replace(" ", "_")
                )
                if col:
                    group_col = col
                    break

        if group_col:
            plan.group_by = group_col
            logger.info(
                f"Stage 6b: GROUP BY='{group_col}' on table='{primary_table}' "
                f"(NLU agg={nlu_result.aggregation})"
            )
            if session_state is not None:
                session_state["_nlu_group_by"] = group_col
                session_state["_nlu_aggregation"] = nlu_result.aggregation
        else:
            logger.info(
                f"Stage 6b: could not resolve GROUP BY column for NLU agg "
                f"'{nlu_result.aggregation}' on table='{primary_table}' "
                f"(entities={nlu_result.entities}) — falling back to LLM SQL"
            )

    # ── Stage 7: SQL generation from plan ──────────────────
    # (SQL is generated in db_pipeline using generate_sql)

    return plan


async def generate_sql(plan: QueryPlan, schema: str, schema_graph=None) -> str:
    """Generate SQL from a structured QueryPlan via LLM.

    Fast path: when plan.group_by is set with COUNT aggregation on a single table,
    the SQL is built deterministically (no LLM) for reliability.
    """
    primary_table = plan.tables[0] if plan.tables else ""
    has_joins = len(plan.tables) > 1
    group_by_col = plan.group_by  # set by Stage 6b for count_group_by queries

    agg = plan.aggregation
    agg_type = ""
    if agg and agg != "none":
        agg_type = agg if isinstance(agg, str) else agg.get("type", "")

    # ── Fast path: deterministic GROUP BY COUNT (no LLM needed) ──────────────
    if agg_type.upper() == "COUNT" and group_by_col and primary_table and not has_joins:
        alias = "t1"
        where_parts = []
        for f in plan.filters:
            col = f.get("column", "")
            op = f.get("operator", "=")
            val = f.get("value", "")
            if "." in col:
                col = col.split(".", 1)[1]
            if isinstance(val, str):
                where_parts.append(f"{alias}.{col} {op} '{val}'")
            else:
                where_parts.append(f"{alias}.{col} {op} {val}")
        where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        sql = (
            f"SELECT {alias}.{group_by_col}, COUNT(*) AS count"
            f" FROM {schema}.{primary_table} {alias}"
            f"{where_clause}"
            f" GROUP BY {alias}.{group_by_col}"
            f" ORDER BY count DESC;"
        )
        logger.info(f"generate_sql: deterministic count_group_by → {sql}")
        return sql

    # ── LLM path for complex queries ─────────────────────────────────────────
    plan_json = json.dumps(plan.to_dict(), indent=2, default=str)

    # Determine SELECT clause deterministically
    if agg_type.upper() == "COUNT" and group_by_col:
        # JOIN case with group_by — give LLM a precise instruction
        alias = "t1"
        select_clause = f"SELECT {alias}.{group_by_col}, COUNT(*) AS count"
        group_by_instruction = f"IMPORTANT: Add 'GROUP BY {alias}.{group_by_col} ORDER BY count DESC' at the end."
    elif agg_type.upper() == "COUNT":
        select_clause = f"SELECT COUNT(t1.*)"
        group_by_instruction = ""
    else:
        select_clause = None  # Let LLM decide for SUM/AVG/MIN/MAX
        group_by_instruction = ""

    if not agg_type:
        if has_joins:
            select_clause = f"SELECT t1.*"
        else:
            select_clause = "SELECT *"
        group_by_instruction = ""

    # Include column names + table descriptions for domain-aware SQL generation
    column_hint = ""
    if schema_graph:
        for tname in plan.tables:
            if tname in schema_graph.tables:
                tmeta = schema_graph.tables[tname]
                cols = list(tmeta.columns.keys())
                desc = f" — {tmeta.description}" if tmeta.description else ""
                column_hint += f"\n{tname}{desc}\n  columns: {', '.join(cols)}"

    select_instruction = (
        f"IMPORTANT: The SELECT clause MUST be exactly: {select_clause} "
        f"(where t1 is the alias for '{primary_table}'). Do NOT list individual columns."
        if select_clause else ""
    )

    prompt = f"""Generate a PostgreSQL SELECT from this plan. Return ONLY SQL, no explanation.
Schema: {schema}
Plan: {plan_json}
{select_instruction}
{group_by_instruction}

Table context:{column_hint}

Rules: SELECT only. Prefix tables with {schema}. Use unique lowercase aliases. Use exact column/table names from plan. No SQL comments."""

    sql = await chat([{"role": "user", "content": prompt}], temperature=0.1)
    sql = _extract_sql(sql)
    # Deterministically ensure schema prefix (LLM often forgets it)
    if schema_graph:
        sql = _ensure_schema_prefix(sql, schema, set(schema_graph.tables.keys()))
    return sql


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
    # Include tables already referenced in the SQL
    referenced_tables = {}
    for tname, tmeta in schema_graph.tables.items():
        if tname in original_sql.lower():
            referenced_tables[tname] = list(tmeta.columns.keys())

    # Also discover related tables that might be needed for the follow-up.
    # The follow-up may reference entities in other tables (e.g. "personal loans"
    # requires JOINing the loans table even though the original query is on customers).
    related_tables = {}
    for tname, tmeta in schema_graph.tables.items():
        if tname in referenced_tables:
            continue
        # Check if any word in the follow-up maps to this table or its columns
        for word in re.findall(r"[A-Za-z_]+", follow_up.lower()):
            if len(word) < 3:
                continue
            resolved_table = schema_graph.resolve_table(word)
            if resolved_table == tname:
                related_tables[tname] = list(tmeta.columns.keys())
                break
            for cname in tmeta.columns:
                if word in cname or cname in word:
                    related_tables[tname] = list(tmeta.columns.keys())
                    break
            if tname in related_tables:
                break

    all_tables = {**referenced_tables, **related_tables}

    # Get actual distinct values from the DB for all relevant tables
    # so the LLM can map user terms to real values (e.g. "Visakhapatnam" -> "Vizag")
    column_values = schema_graph.get_column_values(list(all_tables.keys()))

    # Extract schema prefix from original SQL for consistency
    schema_prefix = ""
    schema_match = re.search(r"FROM\s+(\w+)\.", original_sql, re.IGNORECASE)
    if schema_match:
        schema_prefix = schema_match.group(1)

    related_section = ""
    if related_tables:
        related_section = f"\nRelated tables (JOIN if needed): {json.dumps(related_tables, default=str)}"

    prompt = f"""Amend this SQL query based on the follow-up question.

Original SQL: {original_sql}
Follow-up: "{follow_up}"
Tables in current query: {json.dumps(referenced_tables, default=str)}{related_section}

ACTUAL values in the database (use ONLY these values in WHERE clauses):
{json.dumps(column_values, indent=2, default=str)}

Rules:
- CRITICAL: Only generate SELECT queries
- CRITICAL: Do NOT include SQL comments (-- or /* */)
- CRITICAL: ONLY use column names that exist in the tables listed above — NEVER invent or guess column names
- CRITICAL: Every non-aggregated column in SELECT must appear in the GROUP BY clause (GROUP BY consistency)
- CRITICAL: Only use values from the ACTUAL values list above. Map user terms to the closest matching actual value (e.g. if user says "Visakhapatnam" but actual values have "Vizag", use "Vizag")
- CRITICAL: PRESERVE all existing WHERE conditions from the original SQL unless the user explicitly asks to remove or change them. The follow-up ADDS to or refines the existing query
- If the user says "get all", "list all", "show all" etc., change SELECT to return rows (SELECT *) not COUNT(*). Keep the existing filters and JOINs unless the user says to remove them
- If the original has COUNT(*) and the user asks to "list" or "show" or "get" records, change to SELECT *
- Preserve the schema prefix ({schema_prefix}) on all table names exactly as in the original SQL
- Start from the original SQL and modify only what is necessary to address the follow-up
- If the follow-up references data from a related table, add a JOIN using the appropriate foreign key
- Return ONLY the amended SQL, no explanation"""

    try:
        amended = await chat([{"role": "user", "content": prompt}], temperature=0.1)
    except RuntimeError as e:
        logger.warning(f"amend_sql LLM call failed ({e}); falling back to original SQL")
        return original_sql
    amended = _extract_sql(amended)

    # Ensure schema prefix survived the LLM amendment
    amended = _ensure_schema_prefix(amended, schema_prefix, set(schema_graph.tables.keys()))

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
