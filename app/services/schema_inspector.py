"""Schema introspection — reads tables, columns, FKs from information_schema."""
import json
import logging
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)

# Populated dynamically at startup by LLM from actual schema
DOMAIN_SYNONYMS: dict[str, str] = {}

# Max distinct values per column to build the value_index (low-cardinality columns only)
_VALUE_INDEX_MAX_CARDINALITY = 200


class TableMeta:
    def __init__(self, name: str, schema: str):
        self.name = name
        self.schema = schema
        self.columns: dict[str, dict] = {}
        self.fk_out: dict[str, dict] = {}  # col -> {table, column}
        self.fk_in: list[dict] = []
        self.sample_rows: list[dict] = []
        self.description: str = ""

    @property
    def full_name(self):
        return f"{self.schema}.{self.name}"


class SchemaGraph:
    def __init__(self):
        self.tables: dict[str, TableMeta] = {}
        self.synonyms: dict[str, str] = {}
        # Dynamic reverse index: lowercase_value -> list of (actual_value, table, column)
        # Built from distinct values in low-cardinality text columns
        self.value_index: dict[str, list[tuple[str, str, str]]] = {}
        # Domain name inferred from table descriptions (e.g. "Banking", "Healthcare")
        self.domain_name: str = ""

    def resolve_table(self, token: str) -> str | None:
        token_lower = token.lower().strip()
        # Exact match
        if token_lower in self.tables:
            return token_lower
        # Dynamic synonyms (LLM-generated + auto plural/singular)
        if token_lower in self.synonyms:
            return self.synonyms[token_lower]
        # Plural/singular fallback
        if token_lower.endswith("s") and token_lower[:-1] in self.tables:
            return token_lower[:-1]
        if token_lower.endswith("s") and token_lower[:-1] in self.synonyms:
            return self.synonyms[token_lower[:-1]]
        if token_lower + "s" in self.tables:
            return token_lower + "s"
        return None

    def resolve_column(self, table_name: str, token: str) -> str | None:
        if table_name not in self.tables:
            return None
        token_lower = token.lower().strip().replace(" ", "_")
        cols = self.tables[table_name].columns
        # Exact
        if token_lower in cols:
            return token_lower
        # Partial
        for cname in cols:
            if token_lower in cname or cname in token_lower:
                return cname
        return None

    def lookup_value(self, value: str, plan_tables: list[str] | None = None) -> tuple[str, str, str] | None:
        """Look up a value in the dynamic index. Returns (actual_value, table, column) or None.
        Tries exact match first, then fuzzy match (Levenshtein + substring).
        If plan_tables is given, prefer matches in those tables."""
        val_lower = value.lower().strip()
        if not val_lower:
            return None

        # ── Exact match ──
        matches = self.value_index.get(val_lower)
        if matches:
            return self._pick_best_match(matches, plan_tables)

        # ── Fuzzy match: Levenshtein on similar-length strings ──
        from thefuzz import fuzz
        best_match = None
        best_score = 0
        for key, entries in self.value_index.items():
            # Skip very short keys (< 3 chars) for fuzzy — too ambiguous
            if len(key) < 3 or len(val_lower) < 3:
                continue
            # Only compare strings of similar length to avoid false positives
            # (e.g. "visakhapatnam" vs "ap" should never match)
            len_ratio = min(len(val_lower), len(key)) / max(len(val_lower), len(key))
            if len_ratio < 0.4:
                continue
            score = fuzz.ratio(val_lower, key)
            if score > best_score and score >= 75:
                best_score = score
                best_match = entries
        if best_match:
            return self._pick_best_match(best_match, plan_tables)

        return None

    def _pick_best_match(self, matches: list[tuple[str, str, str]],
                         plan_tables: list[str] | None) -> tuple[str, str, str]:
        """From a list of (value, table, column) matches, prefer one in plan_tables."""
        if not plan_tables:
            return matches[0]
        for actual_val, tname, cname in matches:
            if tname in plan_tables:
                return (actual_val, tname, cname)
        return matches[0]

    def get_column_values(self, table_names: list[str]) -> dict[str, list[str]]:
        """Get distinct values per column for the given tables from the value index.
        Returns {table.column: [val1, val2, ...]}."""
        result: dict[str, list[str]] = {}
        for _key, entries in self.value_index.items():
            for actual_val, tname, cname in entries:
                if tname in table_names:
                    col_key = f"{tname}.{cname}"
                    if col_key not in result:
                        result[col_key] = []
                    if actual_val not in result[col_key]:
                        result[col_key].append(actual_val)
        return result

    def join_path(self, table_a: str, table_b: str) -> dict | None:
        if table_a not in self.tables or table_b not in self.tables:
            return None
        ta = self.tables[table_a]
        # Direct FK from A to B
        for col, fk in ta.fk_out.items():
            if fk["table"] == table_b:
                return {"type": "INNER JOIN", "left": table_a, "right": table_b,
                        "on": f"{table_a}.{col} = {table_b}.{fk['column']}"}
        # Direct FK from B to A
        tb = self.tables[table_b]
        for col, fk in tb.fk_out.items():
            if fk["table"] == table_a:
                return {"type": "INNER JOIN", "left": table_a, "right": table_b,
                        "on": f"{table_b}.{col} = {table_a}.{fk['column']}"}
        # Shared column name heuristic
        for cola in ta.columns:
            for colb in tb.columns:
                if cola == colb and ("_id" in cola or cola == "id"):
                    return {"type": "INNER JOIN", "left": table_a, "right": table_b,
                            "on": f"{table_a}.{cola} = {table_b}.{colb}"}
        return None


async def inspect_schema() -> SchemaGraph:
    settings = get_settings()
    schema = settings.POSTGRES_SCHEMA
    pool = get_pool()
    graph = SchemaGraph()

    async with pool.acquire() as conn:
        # Load tables and columns
        rows = await conn.fetch("""
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = $1
            ORDER BY table_name, ordinal_position
        """, schema)

        for row in rows:
            tname = row["table_name"]
            if tname not in graph.tables:
                graph.tables[tname] = TableMeta(tname, schema)
            graph.tables[tname].columns[row["column_name"]] = {
                "data_type": row["data_type"],
                "nullable": row["is_nullable"] == "YES",
            }

        # Load FK relationships
        fk_rows = await conn.fetch("""
            SELECT
                tc.table_name AS from_table,
                kcu.column_name AS from_column,
                ccu.table_name AS to_table,
                ccu.column_name AS to_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = $1
        """, schema)

        for fk in fk_rows:
            ft = fk["from_table"]
            if ft in graph.tables:
                graph.tables[ft].fk_out[fk["from_column"]] = {
                    "table": fk["to_table"], "column": fk["to_column"]
                }

        # Load sample rows (5 per table)
        for tname, tmeta in graph.tables.items():
            try:
                samples = await conn.fetch(f"SELECT * FROM {schema}.{tname} LIMIT 5")
                tmeta.sample_rows = [dict(r) for r in samples]
            except Exception:
                pass

        # Build dynamic value index from all low-cardinality columns (text, enum, etc.)
        # Skip numeric/date types that don't benefit from value indexing
        _SKIP_TYPES = {"integer", "bigint", "smallint", "numeric", "double precision", "real",
                       "date", "timestamp without time zone", "timestamp with time zone",
                       "boolean", "bytea", "uuid", "json", "jsonb"}
        indexed_values = 0
        for tname, tmeta in graph.tables.items():
            for cname, cinfo in tmeta.columns.items():
                if cinfo["data_type"] in _SKIP_TYPES:
                    continue
                try:
                    count_row = await conn.fetchrow(
                        f"SELECT COUNT(DISTINCT {cname}) as cnt FROM {schema}.{tname}"
                    )
                    distinct_count = count_row["cnt"] if count_row else 0
                    if distinct_count == 0 or distinct_count > _VALUE_INDEX_MAX_CARDINALITY:
                        continue
                    rows = await conn.fetch(
                        f"SELECT DISTINCT {cname} FROM {schema}.{tname} "
                        f"WHERE {cname} IS NOT NULL LIMIT {_VALUE_INDEX_MAX_CARDINALITY}"
                    )
                    for row in rows:
                        val = row[cname]
                        if val and isinstance(val, str) and val.strip():
                            key = val.lower().strip()
                            if key not in graph.value_index:
                                graph.value_index[key] = []
                            graph.value_index[key].append((val, tname, cname))
                            indexed_values += 1
                except Exception:
                    continue
        logger.info(f"  Value index built: {indexed_values} values, {len(graph.value_index)} unique keys")

    # Generate LLM descriptions for tables (cached in schema_index table)
    app_schema = settings.APP_SCHEMA
    try:
        from app.services.llm_client import chat_json
        generated = 0
        async with pool.acquire() as conn:
            for tname, tmeta in graph.tables.items():
                # Check cache first
                cached = await conn.fetchrow(
                    f"SELECT description FROM {app_schema}.schema_index WHERE table_name = $1 AND column_name IS NULL AND description IS NOT NULL",
                    tname
                )
                if cached and cached["description"]:
                    tmeta.description = cached["description"]
                    continue

                # Generate via LLM
                col_info = ", ".join(f"{c} ({v['data_type']})" for c, v in list(tmeta.columns.items())[:15])
                desc_result = await chat_json([{"role": "user", "content":
                    f"Describe this database table in one sentence.\nTable: {tname}\nColumns: {col_info}\nReturn: {{\"description\": \"...\"}}"
                }])
                tmeta.description = desc_result.get("description", tname)
                generated += 1

                # Persist to cache (upsert via delete + insert for NULL column_name)
                await conn.execute(
                    f"DELETE FROM {app_schema}.schema_index WHERE table_name = $1 AND column_name IS NULL",
                    tname
                )
                await conn.execute(
                    f"INSERT INTO {app_schema}.schema_index (table_name, description) VALUES ($1, $2)",
                    tname, tmeta.description
                )

        cached_count = len(graph.tables) - generated
        if generated > 0:
            logger.info(f"  Generated LLM descriptions for {generated} tables ({cached_count} cached)")
        else:
            logger.info(f"  Loaded {cached_count} table descriptions from cache")
    except Exception as e:
        logger.warning(f"LLM description generation skipped: {e}")

    # Build synonym map: auto singular/plural + LLM-generated domain synonyms
    graph.synonyms = {}
    for tname in graph.tables:
        if tname.endswith("s"):
            graph.synonyms[tname[:-1]] = tname
        else:
            graph.synonyms[tname + "s"] = tname

    # Generate domain-specific synonyms via LLM (cached in schema_index)
    try:
        from app.services.llm_client import chat_json
        table_list = list(graph.tables.keys())
        schema_name = settings.POSTGRES_SCHEMA

        async with pool.acquire() as conn:
            # Check cache
            cached = await conn.fetchrow(
                f"SELECT description FROM {app_schema}.schema_index "
                f"WHERE table_name = '_synonyms' AND column_name IS NULL"
            )
            if cached and cached["description"]:
                synonyms = json.loads(cached["description"])
            else:
                # Ask LLM for domain synonyms based on actual tables
                col_summaries = {}
                for t in table_list:
                    col_summaries[t] = list(graph.tables[t].columns.keys())[:10]

                syn_result = await chat_json([{"role": "user", "content":
                    f"""Given this {schema_name} database schema, generate synonyms that users might use to refer to each table.
Only include synonyms that are relevant to this specific domain/schema.

Tables and their columns:
{json.dumps(col_summaries, indent=2)}

Return JSON: {{"synonyms": {{"synonym_word": "actual_table_name", ...}}}}
Example: {{"synonyms": {{"client": "customers", "acct": "accounts", "txn": "transactions"}}}}

Rules:
- Only include domain-appropriate synonyms (e.g. don't map "buyer" to "customers" in banking)
- Include common abbreviations
- Include singular/plural variants
- Include multi-word domain phrases using underscores (e.g. "account_holder": "customers", "credit_card": "cards")
- Each synonym must map to exactly one table from the list above"""}])
                synonyms = syn_result.get("synonyms", {})

                # Cache it
                await conn.execute(
                    f"DELETE FROM {app_schema}.schema_index WHERE table_name = '_synonyms' AND column_name IS NULL"
                )
                await conn.execute(
                    f"INSERT INTO {app_schema}.schema_index (table_name, description) VALUES ('_synonyms', $1)",
                    json.dumps(synonyms)
                )

            graph.synonyms.update(synonyms)
            # Also update the module-level DOMAIN_SYNONYMS for use by intent classifier
            global DOMAIN_SYNONYMS
            DOMAIN_SYNONYMS = dict(graph.synonyms)
            logger.info(f"  Domain synonyms: {len(synonyms)} entries (from {'cache' if cached and cached['description'] else 'LLM'})")
    except Exception as e:
        logger.warning(f"Domain synonym generation skipped: {e}")

    # Infer domain name from table descriptions (cached)
    try:
        from app.services.llm_client import chat_json
        async with pool.acquire() as conn:
            cached = await conn.fetchrow(
                f"SELECT description FROM {app_schema}.schema_index "
                f"WHERE table_name = '_domain' AND column_name IS NULL"
            )
            if cached and cached["description"]:
                graph.domain_name = cached["description"]
            else:
                descriptions = [t.description for t in graph.tables.values() if t.description]
                # Derive a hint from schema name (e.g. "eds_banking" -> "banking")
                schema_hint = schema.replace("_", " ").strip()
                domain_result = await chat_json([{"role": "user", "content":
                    f"""What is the specific business domain for a database called "{schema_hint}" with this data?
{'; '.join(descriptions)}
Return JSON with a precise single-word domain name: {{"domain": "..."}}
Be specific: use "Banking" not "Financial", "Healthcare" not "Medical", "Retail" not "Commerce"."""}])
                graph.domain_name = domain_result.get("domain", schema.split("_")[-1].title())
                await conn.execute(
                    f"DELETE FROM {app_schema}.schema_index WHERE table_name = '_domain' AND column_name IS NULL"
                )
                await conn.execute(
                    f"INSERT INTO {app_schema}.schema_index (table_name, description) VALUES ('_domain', $1)",
                    graph.domain_name
                )
        logger.info(f"  Domain: {graph.domain_name}")
    except Exception as e:
        graph.domain_name = schema.split("_")[-1].title() if "_" in schema else schema.title()
        logger.warning(f"Domain inference skipped: {e}, using '{graph.domain_name}'")

    logger.info(f"Schema inspected: {len(graph.tables)} tables, {sum(len(t.columns) for t in graph.tables.values())} columns")
    return graph
