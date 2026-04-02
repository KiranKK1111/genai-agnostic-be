"""BM25 sparse search — pure math, no model download needed.

Implements Okapi BM25 (Stage 19 from Sentence Transformers architecture).
Runs alongside FAISS dense search to catch exact keyword matches that
dense embeddings miss (e.g., model names, codes, IDs, domain terms).

Architecture:
    BM25 indexes mirror FAISS indexes (schema_idx, values_idx, chunks_idx).
    Each index stores tokenized documents and their TF/IDF statistics.
    At query time, BM25 scores lexical overlap; FAISS scores semantic similarity.
    Reciprocal Rank Fusion (hybrid_search.py) combines both result sets.
"""
import math
import re
import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing."""
    return re.findall(r"[a-z0-9_]+", text.lower())


class BM25Index:
    """In-memory BM25 index for a single document collection.

    Parameters match Okapi BM25 defaults:
        k1=1.5  term frequency saturation
        b=0.75  document length normalization
    """

    def __init__(self, k1: float = None, b: float = None):
        from app.config import get_settings
        s = get_settings()
        self.k1 = k1 if k1 is not None else s.BM25_K1
        self.b = b if b is not None else s.BM25_B
        self.docs: list[list[str]] = []       # tokenized documents
        self.doc_ids: list[int] = []           # parallel metadata IDs
        self.doc_contents: list[str] = []      # original text for reference
        self.df: Counter = Counter()           # document frequency per term
        self.doc_len: list[int] = []           # token count per doc
        self.avg_dl: float = 0.0               # average document length
        self.n_docs: int = 0
        self._initialized = False

    def build(self, contents: list[str], metadata_ids: list[int]):
        """Build BM25 index from raw text documents."""
        self.docs = []
        self.doc_ids = list(metadata_ids)
        self.doc_contents = list(contents)
        self.df = Counter()
        self.doc_len = []

        for text in contents:
            tokens = _tokenize(text)
            self.docs.append(tokens)
            self.doc_len.append(len(tokens))
            # Count each unique term once per document
            for term in set(tokens):
                self.df[term] += 1

        self.n_docs = len(self.docs)
        self.avg_dl = sum(self.doc_len) / max(self.n_docs, 1)
        self._initialized = True
        logger.info(f"BM25 index built: {self.n_docs} docs, {len(self.df)} unique terms")

    def add(self, content: str, metadata_id: int):
        """Add a single document to the index."""
        if not self._initialized:
            self.build([content], [metadata_id])
            return

        tokens = _tokenize(content)
        self.docs.append(tokens)
        self.doc_ids.append(metadata_id)
        self.doc_contents.append(content)
        self.doc_len.append(len(tokens))
        for term in set(tokens):
            self.df[term] += 1
        self.n_docs += 1
        self.avg_dl = sum(self.doc_len) / self.n_docs

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Search for top-k documents. Returns [(metadata_id, bm25_score)]."""
        if not self._initialized or self.n_docs == 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for i in range(self.n_docs):
            score = self._score_doc(query_tokens, i)
            if score > 0:
                scores.append((self.doc_ids[i], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def _score_doc(self, query_tokens: list[str], doc_idx: int) -> float:
        """Compute BM25 score for a single document."""
        doc_tokens = self.docs[doc_idx]
        dl = self.doc_len[doc_idx]
        tf_doc = Counter(doc_tokens)
        score = 0.0

        for term in query_tokens:
            if term not in tf_doc:
                continue
            tf = tf_doc[term]
            df = self.df.get(term, 0)

            # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

            # TF normalization with length penalty
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1e-9))
            )

            score += idf * tf_norm

        return score

    def clear(self):
        """Reset the index."""
        self.docs = []
        self.doc_ids = []
        self.doc_contents = []
        self.df = Counter()
        self.doc_len = []
        self.avg_dl = 0.0
        self.n_docs = 0
        self._initialized = False

    @property
    def count(self) -> int:
        return self.n_docs


# ── Global BM25 index registry ─────────────────────────
_bm25_indexes: dict[str, BM25Index] = {}


def get_bm25_index(index_name: str) -> BM25Index:
    """Get or create a BM25 index by name."""
    if index_name not in _bm25_indexes:
        _bm25_indexes[index_name] = BM25Index()
    return _bm25_indexes[index_name]


def build_bm25_index(index_name: str, contents: list[str], metadata_ids: list[int]):
    """Build a BM25 index from content strings and their metadata IDs."""
    idx = get_bm25_index(index_name)
    idx.build(contents, metadata_ids)


def search_bm25(index_name: str, query: str, k: int = 5) -> list[tuple[int, float]]:
    """Search a BM25 index. Returns [(metadata_id, bm25_score)]."""
    idx = get_bm25_index(index_name)
    return idx.search(query, k=k)


def clear_bm25_index(index_name: str):
    """Clear a BM25 index."""
    if index_name in _bm25_indexes:
        _bm25_indexes[index_name].clear()


def add_to_bm25_index(index_name: str, content: str, metadata_id: int):
    """Add a single document to a BM25 index."""
    idx = get_bm25_index(index_name)
    idx.add(content, metadata_id)
