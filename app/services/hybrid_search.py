"""Hybrid search — Reciprocal Rank Fusion of PostgreSQL dense + sparse results.

Dense:  dot product on normalized real[] vectors (cosine similarity)
Sparse: tsvector/tsquery full-text search with ts_rank
Both run as SQL queries — no in-memory indexes.
"""
import logging
from app.services.faiss_manager import dense_search, sparse_search
from app.config import get_settings

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    dense_results: list[tuple[int, float]],
    sparse_results: list[tuple[int, float]],
    k: int = 60,
    alpha: float = 0.5,
) -> list[tuple[int, float]]:
    """Combine dense and sparse results via RRF.

    Args:
        dense_results:  [(metadata_id, similarity_score)] from dot product
        sparse_results: [(metadata_id, ts_rank_score)] from tsvector
        k:     RRF constant (higher = less weight to top ranks). Default 60.
        alpha: weight for dense results (1-alpha for sparse). Default 0.5 (equal).

    Returns:
        [(metadata_id, fused_score)] sorted by fused score descending.
    """
    scores: dict[int, float] = {}

    for rank, (doc_id, _) in enumerate(dense_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + alpha / (k + rank + 1)

    for rank, (doc_id, _) in enumerate(sparse_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1 - alpha) / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


async def hybrid_search(
    index_name: str,
    query_text: str,
    query_embedding: list[float],
    k: int = 5,
    dense_weight: float = None,
    retrieval_k: int = None,
) -> list[tuple[int, float]]:
    """Run PostgreSQL dense + sparse search, fuse with RRF.

    Both searches hit PostgreSQL directly — no in-memory indexes.
    """
    settings = get_settings()
    if dense_weight is None:
        dense_weight = settings.HYBRID_DENSE_WEIGHT
    if retrieval_k is None:
        retrieval_k = settings.HYBRID_RETRIEVAL_K

    # Dense search (cosine similarity via dot product on real[])
    dense_results = await dense_search(index_name, query_embedding, k=retrieval_k)

    # Sparse search (keyword matching via tsvector/tsquery)
    sparse_results = await sparse_search(index_name, query_text, k=retrieval_k)

    # Fallbacks
    if not sparse_results:
        return dense_results[:k]
    if not dense_results:
        return sparse_results[:k]

    # Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion(
        dense_results, sparse_results, k=60, alpha=dense_weight
    )

    logger.debug(
        f"Hybrid search '{index_name}': "
        f"dense={len(dense_results)}, sparse={len(sparse_results)}, "
        f"fused={len(fused)} -> top-{k}"
    )

    return fused[:k]
