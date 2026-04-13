"""Embedding client — local sentence encoder, zero external models.

Uses the custom SentenceEncoder (sentence_encoder.py) which mimics
the Sentence Transformers pipeline entirely in Python + numpy:
    Stage 2:  Tokenization (subword-like)
    Stage 4:  TF-IDF Embedding
    Stage 5:  SVD Projection
    Stage 7:  Dense Projection (learned rotation)
    Stage 8:  L2 Normalization
    Stage 15: Asymmetric encoding (query vs document)
    Stage 16: Instruction prefixes (task-specific)

No fastembed, no sentence-transformers, no Ollama embeddings API.
"""
import asyncio
import logging
from app.config import get_settings

logger = logging.getLogger(__name__)


async def embed_texts(texts: list[str], mode: str = "query") -> list[list[float]]:
    """Embed a batch of texts using the local SentenceEncoder.

    Runs the synchronous numpy/SVD encoder in a thread executor so it doesn't
    block the asyncio event loop while embedding many chunks (e.g. file uploads).

    Args:
        texts: input strings
        mode: encoding mode (Stage 15 asymmetric + Stage 16 prefix)
              "query"   — user search queries (boosts rare terms)
              "schema"  — table/column descriptions
              "value"   — database values
              "passage" — RAG document chunks
    """
    if not texts:
        return []

    from app.services.sentence_encoder import encode_texts
    return await asyncio.to_thread(encode_texts, texts, mode=mode)


async def embed_single(text: str, mode: str = "query") -> list[float]:
    """Embed a single text. Returns one embedding vector."""
    results = await embed_texts([text], mode=mode)
    return results[0] if results else [0.0] * get_settings().EMBEDDING_DIMENSIONS
