"""LLM-based reranker — uses existing Ollama as a cross-encoder substitute.

Implements Stage 14 from Sentence Transformers architecture.
A cross-encoder reads both query and document TOGETHER through the transformer,
enabling token-level interactions that bi-encoders miss.

Since we don't want additional model downloads, we use the existing Ollama LLM
to score relevance of (query, candidate) pairs — achieving ~70-80% of the
quality gain of a dedicated cross-encoder model.

Production pattern (Retrieve-then-Rerank):
    1. FAISS/BM25 retrieves top-K candidates (fast, ~5ms)
    2. LLM reranker scores top-K pairs (accurate, ~200ms for K=5)
    3. Return reranked top-N results
"""
import logging
import json
from app.services.llm_client import chat_json
from app.config import get_settings

logger = logging.getLogger(__name__)

# System prompt for the reranker
_RERANK_SYSTEM = """You are a relevance scoring engine. Given a query and a list of candidate texts,
score each candidate's relevance to the query on a scale of 0.0 to 1.0.

Rules:
- 1.0 = perfect match, directly answers or matches the query
- 0.7+ = clearly relevant
- 0.4-0.7 = somewhat related
- 0.0-0.4 = not relevant
- Be precise and consistent

Return JSON: {"scores": [0.85, 0.32, ...]} in the same order as candidates."""


async def rerank(
    query: str,
    candidates: list[dict],
    text_key: str = "content",
    top_n: int = 5,
) -> list[dict]:
    """Rerank candidates using the existing LLM as a cross-encoder substitute.

    Args:
        query:      the user's search query
        candidates: list of dicts, each containing text to score
        text_key:   which key in each candidate dict holds the text
        top_n:      how many results to return after reranking

    Returns:
        Reranked candidates (top_n), each with an added 'rerank_score' field.
    """
    if not candidates:
        return []

    if len(candidates) <= 1:
        for c in candidates:
            c["rerank_score"] = c.get("similarity", 1.0)
        return candidates

    # Build the candidate list for the LLM
    candidate_texts = []
    for i, c in enumerate(candidates):
        text = c.get(text_key, "") or c.get("payload", {}).get("description", "") or str(c)
        candidate_texts.append(f"[{i+1}] {text[:500]}")  # truncate long texts

    prompt = (
        f"Query: {query}\n\n"
        f"Candidates:\n" + "\n".join(candidate_texts) + "\n\n"
        f"Score each candidate's relevance to the query (0.0 to 1.0)."
    )

    try:
        result = await chat_json(
            messages=[{"role": "user", "content": prompt}],
            system=_RERANK_SYSTEM,
        )

        scores = result.get("scores", [])

        # Validate scores
        if len(scores) != len(candidates):
            logger.warning(
                f"Reranker returned {len(scores)} scores for {len(candidates)} candidates, "
                "falling back to original order"
            )
            for c in candidates:
                c["rerank_score"] = c.get("similarity", 0.5)
            return candidates[:top_n]

        # Attach scores and sort
        for c, score in zip(candidates, scores):
            try:
                c["rerank_score"] = float(score)
            except (ValueError, TypeError):
                c["rerank_score"] = c.get("similarity", 0.5)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_n]

    except Exception as e:
        logger.warning(f"Reranker failed ({e}), returning original order")
        for c in candidates:
            c["rerank_score"] = c.get("similarity", 0.5)
        return candidates[:top_n]
