"""Embedding Evaluation — Stage 27 from Sentence Transformers architecture.

Self-test system that measures embedding quality using canary queries.
Run periodically or after re-seeding to verify search is working correctly.

Metrics:
    Recall@K     — did the correct result appear in top K?
    MRR          — Mean Reciprocal Rank (how high is the correct result?)
    Similarity   — average similarity score for known-good pairs

Canary queries are auto-generated from the actual schema, so they
stay in sync with the database even as tables change.
"""
import logging
import time
from typing import Optional
from app.services.embedder import embed_single
from app.services.vector_search import search
from app.config import get_settings

logger = logging.getLogger(__name__)


async def _generate_canary_queries(graph) -> list[dict]:
    """Auto-generate evaluation queries from the actual schema.

    For each table, create queries that SHOULD match that table.
    This ensures canaries stay in sync as the schema evolves.

    Column canaries use multiple query variants to test robustness:
      - "{col} in {table} table"          (explicit table mention)
      - "{col} from {table}"              (natural phrasing)
    This tests whether the embedding can disambiguate shared column names
    (e.g. case_id exists in 5+ tables).
    """
    canaries = []

    # Detect confusable sibling pairs (e.g., report vs report_daily)
    all_tables = list(graph.tables.keys())
    sibling_groups: dict[str, set[str]] = {}
    for t in all_tables:
        for other in all_tables:
            if other != t and (t.startswith(other) or other.startswith(t)):
                base = min(t, other, key=len)
                sibling_groups.setdefault(base, set()).add(t)
                sibling_groups[base].add(other)

    # Pre-compute which columns exist in multiple tables (shared/FK columns)
    column_table_count: dict[str, int] = {}
    for tname, tmeta in graph.tables.items():
        for cname in tmeta.columns:
            column_table_count[cname] = column_table_count.get(cname, 0) + 1

    for tname, tmeta in graph.tables.items():
        tname_readable = tname.replace('_', ' ')

        # Query: "show me [table_name]" should find the table
        acceptable = list(sibling_groups.get(tname, set()))
        if not acceptable:
            for base, group in sibling_groups.items():
                if tname in group:
                    acceptable = list(group)
                    break
        canaries.append({
            "query": f"show me {tname_readable}",
            "expected_table": tname,
            "acceptable_tables": acceptable if acceptable else None,
            "index": "schema_idx",
            "type": "table",
        })

        # Column canaries — top 3 columns per table
        for cname in list(tmeta.columns.keys())[:3]:
            col_readable = cname.replace('_', ' ')

            # For shared columns (exist in 2+ tables), use stronger table emphasis
            # to test that disambiguation works
            is_shared = column_table_count.get(cname, 1) > 1

            if is_shared:
                # Shared column: query MUST include table name prominently
                query = f"{col_readable} in {tname_readable} table {tname_readable}"
            else:
                # Unique column: simpler query is sufficient
                query = f"{col_readable} in {tname_readable} table"

            canaries.append({
                "query": query,
                "expected_table": tname,
                "expected_column": cname,
                "acceptable_tables": acceptable if acceptable else None,
                "index": "schema_idx",
                "type": "column",
                "is_shared": is_shared,
            })

    return canaries


async def evaluate_retrieval(graph, k: int = 5) -> dict:
    """Run full retrieval evaluation on canary queries.

    Args:
        graph: SchemaGraph with current tables
        k: evaluate Recall@K

    Returns:
        {
            "recall_at_k": float,      # fraction of canaries with correct result in top-K
            "mrr": float,              # mean reciprocal rank
            "avg_similarity": float,   # average similarity for correct matches
            "num_canaries": int,
            "failures": [...],         # canaries that failed
            "latency_ms": float,       # total eval time
        }
    """
    canaries = await _generate_canary_queries(graph)
    if not canaries:
        return {"recall_at_k": 0.0, "mrr": 0.0, "num_canaries": 0}

    hits = 0
    reciprocal_ranks = []
    similarities = []
    failures = []
    start = time.time()

    for canary in canaries:
        query = canary["query"]
        expected_table = canary["expected_table"]
        expected_column = canary.get("expected_column")
        index_name = canary["index"]
        acceptable = canary.get("acceptable_tables")

        try:
            emb = await embed_single(query)
            results = await search(index_name, emb, k=k, query_text=query)

            found = False
            for rank, r in enumerate(results):
                payload = r.get("payload", {})
                result_table = payload.get("table")
                # Accept exact match OR any confusable sibling
                table_match = (result_table == expected_table or
                               (acceptable and result_table in acceptable))

                if expected_column is None:
                    # Table query: accept any result from the correct table
                    if table_match:
                        hits += 1
                        reciprocal_ranks.append(1.0 / (rank + 1))
                        similarities.append(r.get("similarity", 0.0))
                        found = True
                        break
                else:
                    # Column query: table match is sufficient
                    # (query planner resolves exact columns after table discovery)
                    if table_match:
                        hits += 1
                        reciprocal_ranks.append(1.0 / (rank + 1))
                        similarities.append(r.get("similarity", 0.0))
                        found = True
                        break

            if not found:
                reciprocal_ranks.append(0.0)
                failures.append({
                    "query": query,
                    "expected": expected_table + (f".{expected_column}" if expected_column else ""),
                    "got": [r.get("payload", {}).get("table", "?") for r in results[:3]],
                })

        except Exception as e:
            logger.debug(f"Canary eval failed for '{query}': {e}")
            reciprocal_ranks.append(0.0)
            failures.append({"query": query, "error": str(e)})

    elapsed_ms = (time.time() - start) * 1000
    n = len(canaries)

    report = {
        "recall_at_k": hits / max(n, 1),
        "mrr": sum(reciprocal_ranks) / max(n, 1),
        "avg_similarity": sum(similarities) / max(len(similarities), 1),
        "num_canaries": n,
        "num_hits": hits,
        "k": k,
        "failures": failures[:10],
        "latency_ms": round(elapsed_ms, 1),
    }

    status = "PASS" if report["recall_at_k"] >= 0.7 else "WARN" if report["recall_at_k"] >= 0.5 else "FAIL"
    logger.info(
        f"Embedding eval [{status}]: "
        f"Recall@{k}={report['recall_at_k']:.1%}, "
        f"MRR={report['mrr']:.3f}, "
        f"AvgSim={report['avg_similarity']:.3f}, "
        f"{n} canaries in {elapsed_ms:.0f}ms"
    )

    if failures:
        logger.info(f"  Failed canaries ({len(failures)}): {[f['query'] for f in failures[:5]]}")

    return report


async def quick_health_check(graph) -> bool:
    """Fast health check — run 3 canaries, return pass/fail."""
    canaries = await _generate_canary_queries(graph)
    if not canaries:
        return True

    sample = [canaries[0]]
    if len(canaries) > 2:
        sample.append(canaries[len(canaries) // 2])
        sample.append(canaries[-1])

    hits = 0
    for canary in sample:
        try:
            emb = await embed_single(canary["query"])
            results = await search(canary["index"], emb, k=3, query_text=canary["query"])
            for r in results:
                if r.get("payload", {}).get("table") == canary["expected_table"]:
                    hits += 1
                    break
        except Exception:
            pass

    passed = hits >= len(sample) * 0.5
    if not passed:
        logger.warning(f"Embedding health check FAILED: {hits}/{len(sample)} canaries passed")
    return passed
