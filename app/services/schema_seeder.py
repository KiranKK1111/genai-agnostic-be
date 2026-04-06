"""Schema seeder — embed table/column names and distinct values into FAISS + BM25 + PostgreSQL.

Architecture:
    1. Collect all corpus text ENRICHED with LLM descriptions + domain synonyms
    2. Generate semantic expansions via LLM (related terms, synonyms, concepts)
    3. Fit the SentenceEncoder on the full enriched corpus
    4. Generate embeddings via the fitted encoder
    5. Batch insert into PostgreSQL (single transaction)
    6. Build FAISS in-memory indexes (dense/semantic search)
    7. Build BM25 in-memory indexes (sparse/keyword search)

The enrichment step is critical for semantic understanding:
    Without: "column balance in table accounts, type numeric"
    With:    "column balance money funds wealth amount in table accounts bank financial, type numeric"
    Now "how much money" matches "balance" because they share expanded vocabulary.
"""
import logging
import json
from app.services.embedder import embed_texts
from app.services.sentence_encoder import fit_encoder
from app.services.faiss_manager import batch_add_to_index, clear_index
from app.services.bm25_manager import build_bm25_index, clear_bm25_index
from app.services.schema_inspector import SchemaGraph, DOMAIN_SYNONYMS
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)

def _max_distinct():
    return get_settings().SCHEMA_MAX_DISTINCT_VALUES


async def _generate_semantic_expansions(graph: SchemaGraph) -> dict[str, list[str]]:
    """Use LLM to generate semantic expansions for each table and column.

    Returns: {"table_name.column_name": ["synonym1", "synonym2", ...]}

    This is the key to semantic understanding — the LLM knows that:
      "balance" → money, funds, wealth, amount, financial
      "email"   → contact, address, communication, mail
      "city"    → location, place, area, town, region
    """
    from app.services.llm_client import chat_json

    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    # Check cache first
    try:
        async with pool.acquire() as conn:
            cached = await conn.fetchrow(
                f"SELECT description FROM {schema}.schema_index "
                f"WHERE table_name = '_semantic_expansions' AND column_name IS NULL"
            )
            if cached and cached["description"]:
                expansions = json.loads(cached["description"])
                logger.info(f"  Loaded cached semantic expansions ({len(expansions)} entries)")
                return expansions
    except Exception:
        pass

    # Generate via LLM
    expansions = {}
    try:
        # Build a compact schema summary for the LLM
        schema_summary = {}
        for tname, tmeta in graph.tables.items():
            cols = {c: v["data_type"] for c, v in list(tmeta.columns.items())[:15]}
            schema_summary[tname] = cols

        # Use domain name for context-specific synonyms
        domain = graph.domain_name or settings.POSTGRES_SCHEMA
        # Include table descriptions for richer context
        table_descs = {}
        for tname, tmeta in graph.tables.items():
            if tmeta.description:
                table_descs[tname] = tmeta.description

        result = await chat_json([{"role": "user", "content":
            f"""You are a {domain} domain expert. Generate semantic synonyms and related terms
that {domain} business users would use when searching this database.

IMPORTANT: All synonyms must be specific to the {domain} domain.
- In Banking: "balance" → money, funds, savings (financial terms)
- In Healthcare: "balance" → outstanding bill, copay (medical billing terms)
- In Retail: "balance" → store credit, gift card amount
Choose terms that match the {domain} context ONLY.

Domain: {domain}

Schema:
{json.dumps(schema_summary, indent=2)}

Table descriptions:
{json.dumps(table_descs, indent=2) if table_descs else "Not available"}

For EACH table and column, list {domain}-specific words that a business user would naturally say.
Include:
- Domain jargon and abbreviations common in {domain}
- Business concepts that map to these tables/columns in {domain}
- Natural language phrases {domain} users would type
- Do NOT include generic IT/database terms

Return JSON:
{{
  "table_name": ["domain_synonym1", "business_term", "user_phrase"],
  "table_name.column_name": ["domain_synonym1", "business_term", "concept"]
}}

Use the first table from the schema as a guide — generate {domain}-appropriate synonyms for ALL tables and their key columns.

Rules:
- All terms MUST be relevant to the {domain} domain specifically
- Include {domain}-specific abbreviations and jargon
- Include conceptual phrases a {domain} user would naturally type
- 3-8 terms per entry
- Only include genuinely relevant terms for {domain}"""}])

        expansions = result if isinstance(result, dict) else {}

        # Remove any error keys
        expansions.pop("error", None)
        expansions.pop("raw", None)

        # Cache for future use
        if expansions:
            try:
                async with pool.acquire() as conn:
                    await conn.execute(
                        f"DELETE FROM {schema}.schema_index WHERE table_name = '_semantic_expansions'",
                    )
                    await conn.execute(
                        f"INSERT INTO {schema}.schema_index (table_name, description) VALUES ($1, $2)",
                        "_semantic_expansions", json.dumps(expansions)
                    )
            except Exception:
                pass

        logger.info(f"  Generated semantic expansions for {len(expansions)} entries via LLM")

    except Exception as e:
        logger.warning(f"  Semantic expansion generation failed: {e}")

    return expansions


def _collect_schema_texts(
    graph: SchemaGraph,
    expansions: dict[str, list[str]],
) -> tuple[list[str], list[dict]]:
    """Collect schema texts ENRICHED with LLM descriptions + semantic expansions.

    Before enrichment:
        "column balance in table accounts, type numeric"
    After enrichment:
        "column balance money funds wealth amount in table accounts bank financial, type numeric.
         The accounts table stores customer bank account information"
    """
    texts = []
    payloads = []

    # Build reverse synonym map: actual_table → [synonyms]
    reverse_synonyms: dict[str, list[str]] = {}
    for syn, actual in graph.synonyms.items():
        reverse_synonyms.setdefault(actual, []).append(syn)

    for tname, tmeta in graph.tables.items():
        col_names = ", ".join(tmeta.columns.keys())
        # Readable name: "erp_customers" → "erp customers" for query matching
        tname_readable = tname.replace("_", " ")

        # Enrich table text with:
        # 1. Readable name (so "erp customers" matches "erp_customers")
        # 2. LLM description (e.g., "stores customer profile information")
        # 3. Domain synonyms (e.g., "client, patron, acct_holder")
        # 4. Semantic expansions (e.g., "bank account, financial, deposit")
        parts = [f"table {tname} {tname_readable}: {col_names}"]
        if tmeta.description:
            parts.append(tmeta.description)
        table_syns = reverse_synonyms.get(tname, [])
        if table_syns:
            parts.append(f"also known as: {', '.join(table_syns)}")
        table_expansions = expansions.get(tname, [])
        if table_expansions:
            # Flatten in case LLM returned nested lists
            flat = [str(t) for t in table_expansions if not isinstance(t, (list, dict))]
            if flat:
                parts.append(f"related: {', '.join(flat)}")

        enriched_text = ". ".join(parts)
        texts.append(enriched_text)
        payloads.append({
            "type": "table", "table": tname,
            "description": col_names,
            "columns": list(tmeta.columns.keys()),
        })

        # Enrich each column with readable names + semantic expansions
        for cname, cinfo in tmeta.columns.items():
            cname_readable = cname.replace("_", " ")
            parts = [f"column {cname} {cname_readable} in table {tname} {tname_readable}, type {cinfo['data_type']}"]
            col_key = f"{tname}.{cname}"
            col_expansions = expansions.get(col_key, [])
            if col_expansions:
                flat = [str(t) for t in col_expansions if not isinstance(t, (list, dict))]
                if flat:
                    parts.append(f"related: {', '.join(flat)}")

            enriched_text = ". ".join(parts)
            texts.append(enriched_text)
            payloads.append({
                "type": "column", "table": tname, "column": cname,
                "description": f"{cname} ({cinfo['data_type']})",
                "data_type": cinfo["data_type"],
            })

    return texts, payloads


async def _collect_value_texts(graph: SchemaGraph) -> tuple[list[str], list[dict]]:
    """Collect distinct value texts for embedding."""
    settings = get_settings()
    pool = get_pool()
    db_schema = settings.POSTGRES_SCHEMA

    texts = []
    payloads = []
    logger.info(f"  Scanning {db_schema} for indexable values...")

    async with pool.acquire() as conn:
        for tname, tmeta in graph.tables.items():
            for cname, cinfo in tmeta.columns.items():
                if cinfo["data_type"] not in ("character varying", "text", "varchar", "USER-DEFINED"):
                    continue

                try:
                    count_row = await conn.fetchrow(
                        f"SELECT COUNT(DISTINCT {cname}) as cnt FROM {db_schema}.{tname}"
                    )
                    distinct_count = count_row["cnt"] if count_row else 0

                    if distinct_count == 0 or distinct_count > _max_distinct():
                        continue

                    rows = await conn.fetch(
                        f"SELECT DISTINCT {cname} FROM {db_schema}.{tname} WHERE {cname} IS NOT NULL ORDER BY {cname} LIMIT {_max_distinct()}"
                    )

                    for row in rows:
                        val = row[cname]
                        if val and isinstance(val, str) and len(val.strip()) > 0:
                            texts.append(val)
                            payloads.append({
                                "value": val,
                                "table": tname,
                                "column": cname,
                            })
                except Exception as e:
                    logger.warning(f"Skipping {tname}.{cname} for values_idx: {e}")
                    continue

    logger.info(f"  Found {len(texts)} indexable values across {len(set(p['table'] for p in payloads)) if payloads else 0} tables")
    return texts, payloads


async def seed_schema_index(schema_texts: list[str], schema_payloads: list[dict]):
    """Embed schema entries into schema_idx (FAISS + BM25 + PostgreSQL)."""
    await clear_index("schema_idx")
    clear_bm25_index("schema_idx")

    if not schema_texts:
        logger.warning("No schema entries to seed")
        return

    logger.info(f"Seeding schema_idx with {len(schema_texts)} entries...")

    embeddings = await embed_texts(schema_texts, mode="schema")

    metadata_ids = await batch_add_to_index(
        "schema_idx", embeddings, schema_payloads, contents=schema_texts
    )

    build_bm25_index("schema_idx", schema_texts, metadata_ids)
    logger.info(f"  schema_idx seeded: {len(schema_texts)} entries (FAISS + BM25)")


async def seed_values_index(value_texts: list[str], value_payloads: list[dict]):
    """Embed distinct values into values_idx (FAISS + BM25 + PostgreSQL)."""
    await clear_index("values_idx")
    clear_bm25_index("values_idx")

    if not value_texts:
        logger.info("  values_idx: no indexable values found")
        return

    logger.info(f"Seeding values_idx with {len(value_texts)} values...")

    embeddings = await embed_texts(value_texts, mode="value")

    metadata_ids = await batch_add_to_index(
        "values_idx", embeddings, value_payloads, contents=value_texts
    )

    build_bm25_index("values_idx", value_texts, metadata_ids)
    logger.info(
        f"  values_idx seeded: {len(value_texts)} values "
        f"across {len(set(p['table'] for p in value_payloads))} tables (FAISS + BM25)"
    )


async def seed_all(graph: SchemaGraph):
    """Seed both schema_idx and values_idx.

    Flow:
      1. Generate semantic expansions via LLM (cached after first run)
      2. Collect enriched corpus texts (schema + descriptions + expansions + values)
      3. Fit SentenceEncoder on the full enriched corpus (Stage 3: pre-training)
      4. Train neural refiner with contrastive learning (Stages 9-12)
      5. Embed and index everything (using SVD + neural refiner)
    """
    # Step 1: Generate semantic expansions via LLM
    logger.info("Generating semantic expansions for schema...")
    expansions = await _generate_semantic_expansions(graph)

    # Step 2: Collect ALL enriched text
    schema_texts, schema_payloads = _collect_schema_texts(graph, expansions)
    value_texts, value_payloads = await _collect_value_texts(graph)

    # Step 3: Fit the SentenceEncoder on the full enriched corpus (Stage 3)
    from app.services.sentence_encoder import get_encoder
    full_corpus = schema_texts + value_texts
    if full_corpus:
        logger.info(f"Fitting SentenceEncoder on {len(full_corpus)} enriched corpus texts...")
        fit_encoder(full_corpus)

    # Step 4: Train neural refiner (Stages 9, 10, 11, 12)
    encoder = get_encoder()
    if encoder.is_fitted:
        try:
            from app.services.neural_trainer import train_neural_refiner, set_refiner
            logger.info("Training neural refiner (Stages 9-12)...")
            refiner = await train_neural_refiner(encoder, graph)
            set_refiner(refiner)
            logger.info("  ✓ Neural refiner trained")
        except Exception as e:
            logger.warning(f"  Neural training skipped: {e}")

    # Step 5: Embed and index (now uses SVD + neural refiner)
    await seed_schema_index(schema_texts, schema_payloads)
    await seed_values_index(value_texts, value_payloads)
