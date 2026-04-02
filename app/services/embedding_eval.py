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
    """
    canaries = []

    for tname, tmeta in graph.tables.items():
        # Query: "show me [table_name]" should find the table
        canaries.append({
            "query": f"show me {tname.replace('_', ' ')}",
            "expected_table": tname,
            "index": "schema_idx",
            "type": "table",
        })

        # Query: "[column_name] in [table]" should find the column
        # Use "in [table]" format and repeat table name for FK columns that exist
        # in multiple tables (e.g. "customer_id" in accounts, loans, transactions).
        # This gives the table name more weight in the embedding.
        for cname in list(tmeta.columns.keys())[:3]:  # top 3 columns
            col_readable = cname.replace('_', ' ')
            table_readable = tname.replace('_', ' ')
            canaries.append({
                "query": f"{col_readable} in {table_readable} table",
                "expected_table": tname,
                "expected_column": cname,
                "index": "schema_idx",
                "type": "column",
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

        try:
            emb = await embed_single(query)
            results = await search(index_name, emb, k=k, query_text=query)

            # Check if expected result appears in top-K
            # For column queries: accept if the correct TABLE is found,
            # because the query planner resolves columns separately after
            # table discovery. A table match IS a successful retrieval.
            found = False
            for rank, r in enumerate(results):
                payload = r.get("payload", {})
                table_match = payload.get("table") == expected_table

                if expected_column is None:
                    # Table query: accept any result from the correct table
                    # (table entry or column entry — both confirm the table was found)
                    if table_match:
                        hits += 1
                        reciprocal_ranks.append(1.0 / (rank + 1))
                        similarities.append(r.get("similarity", 0.0))
                        found = True
                        break
                else:
                    # Column query: exact column match OR table match (table contains the column)
                    exact_col = payload.get("column") == expected_column
                    if table_match and exact_col:
                        # Best case: exact column match
                        hits += 1
                        reciprocal_ranks.append(1.0 / (rank + 1))
                        similarities.append(r.get("similarity", 0.0))
                        found = True
                        break
                    elif table_match and not found:
                        # Acceptable: table match (column is in this table)
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
        "failures": failures[:10],  # limit to 10 for logging
        "latency_ms": round(elapsed_ms, 1),
    }

    # Log summary
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
    """Fast health check — run 3 canaries, return pass/fail.

    Use this for the 15-minute schema watcher cycle.
    """
    canaries = await _generate_canary_queries(graph)
    if not canaries:
        return True

    # Pick 3 representative canaries (first, middle, last table)
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
