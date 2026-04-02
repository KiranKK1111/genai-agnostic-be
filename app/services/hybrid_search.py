"""Hybrid search — Reciprocal Rank Fusion of FAISS dense + BM25 sparse results.

Implements Stage 19 from Sentence Transformers architecture.
Dense vectors miss exact keyword matches; BM25 misses semantic similarity.
Combining both via RRF dominates either alone.

BEIR NDCG@10 benchmark (reference):
    BM25 only:          43.0
    Dense only:         49.2
    Dense + BM25 RRF:   57.8  <- this is what we implement
"""
import logging
from typing import Optional
from app.services.faiss_manager import search_index
from app.services.bm25_manager import search_bm25

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    dense_results: list[tuple[int, float]],
    sparse_results: list[tuple[int, float]],
    k: int = 60,
    alpha: float = 0.5,
) -> list[tuple[int, float]]:
    """Combine dense (FAISS) and sparse (BM25) results via RRF.

    Args:
        dense_results:  [(metadata_id, similarity_score)] from FAISS
        sparse_results: [(metadata_id, bm25_score)] from BM25
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
    """Run FAISS dense + BM25 sparse search, fuse with RRF.

    Args:
        index_name:      which index to search (schema_idx, values_idx, chunks_idx)
        query_text:      raw query text for BM25
        query_embedding: query vector for FAISS
        k:               final number of results to return
        dense_weight:    alpha for RRF (0.6 = slightly favor dense/semantic)
        retrieval_k:     how many candidates to pull from each source before fusion

    Returns:
        [(metadata_id, fused_score)] top-k results.
    """
    from app.config import get_settings
    settings = get_settings()
    if dense_weight is None:
        dense_weight = settings.HYBRID_DENSE_WEIGHT
    if retrieval_k is None:
        retrieval_k = settings.HYBRID_RETRIEVAL_K

    # FAISS dense search (semantic similarity)
    from app.services.faiss_manager import get_faiss_index
    idx = await get_faiss_index(index_name)
    dense_results = idx.search(query_embedding, k=retrieval_k)

    # BM25 sparse search (keyword matching)
    sparse_results = search_bm25(index_name, query_text, k=retrieval_k)

    # If BM25 index is empty, fall back to dense-only
    if not sparse_results:
        return dense_results[:k]

    # If FAISS index is empty, fall back to sparse-only
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
