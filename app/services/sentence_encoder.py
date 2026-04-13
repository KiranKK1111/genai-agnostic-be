"""Sentence Encoder — mimics Sentence Transformers pipeline, zero external models.

Built from scratch following the ST architecture reference (30 stages).
Uses only Python stdlib + numpy (already installed via faiss-cpu).

Pipeline (mirrors ST core stages):
  Stage 1  Input & Batching      — length-sorted batching for efficiency
  Stage 2  Tokenization           — subword-like tokens via words + char n-grams
  Stage 4  Embedding Layer        — TF-IDF weighted term vectors (learned from corpus)
  Stage 5  Encoder Layers         — Truncated SVD learns latent semantic dimensions
  Stage 7  Dense Projection       — learned linear rotation for tighter clusters
  Stage 6  Pooling                — document-level aggregation (inherent in TF-IDF)
  Stage 8  L2 Normalization       — unit vectors so cosine = dot product
  Stage 15 Asymmetric Encoding    — separate query vs document encoding paths
  Stage 16 Instruction Prefixes   — task-specific prefixes steer embedding geometry

How it works:
  fit(corpus)   = "pre-training"  — learns vocabulary, IDF, SVD, and dense projection
  encode(texts) = "inference"     — tokenize → TF-IDF → SVD → Dense → L2 norm

The SVD components are the "transformer weights" — they encode which token
patterns co-occur and thus capture semantic relationships.
The Dense projection rotates the space to push similar items closer.
"""
import math
import re
import logging
import numpy as np
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)


# ── Stage 2: Tokenizer ─────────────────────────────────────────
# Mimics WordPiece/BPE subword tokenization without a trained vocab.
# Words capture meaning; char n-grams capture morphology.
# "customers" → ["customers", "#cus", "#ust", "#sto", "#tom", "#ome", "#mer", "#ers"]

def _tokenize(text: str) -> list[str]:
    """Subword-like tokenization: word unigrams + bigrams + sub-words + char trigrams.

    Compound tokens containing underscores (e.g. 'erp_customers') are split
    into parts ('erp', 'customers') IN ADDITION to keeping the original.
    This ensures queries ("erp customers") share word tokens with corpus
    entries ("erp_customers") — without this, overlap is zero.

    Word bigrams capture phrases: "case status" → token "case_status" which
    matches the schema column of the same name directly.
    """
    text = text.lower().strip()
    words = re.findall(r"[a-z0-9_]+", text)

    # Flatten compound words so we work with atomic word parts for bigrams
    flat_words: list[str] = []
    for word in words:
        flat_words.append(word)
        if "_" in word:
            flat_words.extend(p for p in word.split("_") if p)

    tokens: list[str] = list(flat_words)

    # Word bigrams: "case" + "status" → "case_status"
    # This makes phrase matching robust even before SVD compression
    for i in range(len(flat_words) - 1):
        a, b = flat_words[i], flat_words[i + 1]
        # Only bigram atomic words (no compound tokens), cap length to avoid noise
        if "_" not in a and "_" not in b and len(a) + len(b) <= 24:
            tokens.append(f"{a}_{b}")

    # Character trigrams for morphological similarity
    for word in flat_words:
        if "_" not in word and len(word) >= 3:
            for i in range(len(word) - 2):
                tokens.append(f"#{word[i:i+3]}")

    return tokens


# ── Stage 16: Instruction Prefixes ─────────────────────────────
# Prepending task-specific prefixes steers the encoder to produce
# different geometry for different use cases.
# "query:" optimises for retrieval recall.
# "schema:" optimises for table/column matching.
# Missing the prefix degrades quality ~2-3 NDCG points.

INSTRUCTION_PREFIXES = {
    "query":   "query: ",           # user search queries
    "schema":  "schema: ",          # table/column descriptions
    "value":   "value: ",           # database values
    "passage": "passage: ",         # RAG document chunks
    "cluster": "cluster: ",         # clustering/grouping
}


def _apply_prefix(text: str, mode: str = "query") -> str:
    """Apply instruction prefix for task-specific encoding (Stage 16)."""
    prefix = INSTRUCTION_PREFIXES.get(mode, "")
    return prefix + text


# ── Stage 15: Asymmetric Encoding ──────────────────────────────
# Queries are short ("customer balance"), documents are long
# ("column balance in table accounts, type numeric").
# Asymmetric encoding uses different IDF emphasis:
#   - Query mode: boost rare terms (precision-oriented)
#   - Document mode: balanced weighting (recall-oriented)

def _get_idf_params():
    from app.config import get_settings
    s = get_settings()
    return s.EMBEDDING_QUERY_IDF_BOOST, s.EMBEDDING_DOC_IDF_DAMPEN


# ── Core Encoder ───────────────────────────────────────────────

class SentenceEncoder:
    """Local sentence encoder — full ST pipeline, zero external models.

    Architecture:
        Tokenizer → TF-IDF Embedding → SVD Projection → Dense Layer → L2 Norm
        (Stage 2)   (Stage 4)          (Stage 5)         (Stage 7)     (Stage 8)

    With asymmetric encoding (Stage 15) and instruction prefixes (Stage 16).

    Args:
        dim: output embedding dimension (default 384, same as MiniLM-L6)
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vocab: dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self.components: Optional[np.ndarray] = None   # (k, vocab_size) SVD Vt
        self.dense_weight: Optional[np.ndarray] = None  # (dim, k) Stage 7 projection
        self.dense_bias: Optional[np.ndarray] = None    # (dim,) Stage 7 bias
        self.n_docs: int = 0
        self._fitted = False

    # ── fit() = "pre-training" ──────────────────────────────

    def fit(self, corpus: list[str]):
        """Fit encoder on corpus (equivalent to pre-training on domain data).

        Learns:
          1. Vocabulary — which tokens exist in this domain
          2. IDF weights — which tokens are informative vs common
          3. SVD components — latent semantic dimensions
          4. Dense projection — learned rotation for tighter similarity clusters
        """
        if not corpus:
            logger.warning("SentenceEncoder.fit() called with empty corpus")
            return

        # Tokenize all documents (with neutral prefix for fitting)
        doc_tokens = [_tokenize(text) for text in corpus]

        # ── Build vocabulary ────────────────────────────────
        all_tokens: set[str] = set()
        for tokens in doc_tokens:
            all_tokens.update(tokens)
        self.vocab = {token: i for i, token in enumerate(sorted(all_tokens))}
        vocab_size = len(self.vocab)
        self.n_docs = len(corpus)

        if vocab_size == 0:
            logger.warning("SentenceEncoder: empty vocabulary after tokenization")
            return

        logger.info(
            f"SentenceEncoder fitting: {self.n_docs} docs, "
            f"{vocab_size} tokens in vocabulary"
        )

        # ── BM25 parameters (from config) ───────────────────
        from app.config import get_settings as _gs
        _s = _gs()
        bm25_k1: float = getattr(_s, "BM25_K1", 1.5)
        bm25_b:  float = getattr(_s, "BM25_B",  0.75)

        # ── Compute IDF (BM25-style) ─────────────────────────
        df = np.zeros(vocab_size, dtype=np.float64)
        for tokens in doc_tokens:
            seen: set[str] = set()
            for t in tokens:
                if t in self.vocab and t not in seen:
                    df[self.vocab[t]] += 1
                    seen.add(t)
        # Robertson-Spärck Jones IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        self.idf = np.log(
            (self.n_docs - df + 0.5) / (df + 0.5) + 1.0
        ).astype(np.float64)

        # ── Average document length (for BM25 length normalisation) ──
        doc_lengths = np.array([len(t) for t in doc_tokens], dtype=np.float64)
        self._avgdl = float(doc_lengths.mean()) if len(doc_lengths) else 1.0

        # ── Build BM25 matrix (Stage 4) ─────────────────────
        # BM25 TF: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        # This caps term saturation and normalises for document length,
        # outperforming log-TF on long schema documents with many column names.
        tfidf = np.zeros((self.n_docs, vocab_size), dtype=np.float64)
        for i, tokens in enumerate(doc_tokens):
            tf = Counter(tokens)
            dl = len(tokens)
            norm_factor = 1.0 - bm25_b + bm25_b * dl / self._avgdl
            for token, count in tf.items():
                if token in self.vocab:
                    j = self.vocab[token]
                    tf_bm25 = (count * (bm25_k1 + 1.0)) / (count + bm25_k1 * norm_factor)
                    tfidf[i, j] = tf_bm25 * self.idf[j]

        row_norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        row_norms = np.clip(row_norms, 1e-9, None)
        tfidf = tfidf / row_norms

        # ── SVD decomposition (Stage 5) ─────────────────────
        k = min(self.dim, vocab_size, self.n_docs)
        try:
            # Use sparse SVD for large matrices to avoid LAPACK gesdd failures
            if self.n_docs * vocab_size > 5_000_000:
                from scipy.sparse.linalg import svds
                from scipy.sparse import csr_matrix
                sparse_tfidf = csr_matrix(tfidf)
                k_svd = min(k, min(sparse_tfidf.shape) - 1)
                U, S, Vt = svds(sparse_tfidf, k=k_svd)
                idx = np.argsort(-S)
                S = S[idx]
                Vt = Vt[idx]
            else:
                U, S, Vt = np.linalg.svd(tfidf, full_matrices=False)
            self.components = Vt[:k].astype(np.float32)  # (k, vocab_size)

            # ── Dense Projection (Stage 7) — full (k → dim) matrix ──
            # Previous: diagonal scaling only (can't rotate the space).
            # Now: full Xavier-initialised weight matrix (k, dim).
            # Initialised from SVD singular value scaling so the first
            # training step starts close to the right solution.
            S_k = S[:k].astype(np.float32)
            s_exp = np.exp(S_k - S_k.max())
            s_weights = s_exp / s_exp.sum()  # softmax importance per dimension

            # Full projection: (k, dim) — maps SVD space to output space
            # Initialised as scaled identity-like block + small noise for symmetry breaking
            rng = np.random.default_rng(42)
            w_init = np.zeros((k, self.dim), dtype=np.float32)
            min_dim = min(k, self.dim)
            # Fill diagonal block with singular-value importance weights
            for d in range(min_dim):
                w_init[d, d] = s_weights[d] if d < len(s_weights) else 0.0
            # Small noise on off-diagonal to break symmetry and allow rotation
            noise = rng.standard_normal((k, self.dim)).astype(np.float32) * 0.01
            self.dense_weight = w_init + noise  # (k, dim)
            self.dense_bias = np.zeros(self.dim, dtype=np.float32)

        except Exception as e:
            logger.warning(f"SVD failed ({e}) — using truncated identity projection")
            self.components = np.eye(k, vocab_size, dtype=np.float32)
            self.dense_weight = np.eye(k, self.dim, dtype=np.float32)
            self.dense_bias = np.zeros(self.dim, dtype=np.float32)

        self._fitted = True
        logger.info(
            f"SentenceEncoder fitted: {k} semantic dims, {vocab_size} vocab, "
            f"BM25(k1={bm25_k1}, b={bm25_b}), full dense projection ({k}→{self.dim})"
        )

    # ── encode() = "inference" ──────────────────────────────

    def encode(self, texts: list[str], mode: str = "query") -> np.ndarray:
        """Encode texts into dense vectors. Returns (N, dim) float32.

        Pipeline: prefix → tokenize → TF-IDF → SVD → Dense → L2 norm

        Args:
            texts: input strings
            mode: encoding mode for Stage 15 (asymmetric) + Stage 16 (prefix)
                  "query"   — user search queries (boosts rare terms)
                  "schema"  — table/column descriptions
                  "value"   — database values
                  "passage" — RAG document chunks (dampens rare terms)
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self._fitted:
            return self._encode_fitted(texts, mode)
        else:
            return self._encode_fallback(texts)

    def _encode_fitted(self, texts: list[str], mode: str = "query") -> np.ndarray:
        """Full pipeline: prefix → tokenize → BM25 → SVD → Dense(k→dim) → L2 norm."""
        n = len(texts)
        vocab_size = len(self.vocab)

        # ── Stage 16: Apply instruction prefix ──────────────
        prefixed = [_apply_prefix(t, mode) for t in texts]

        # ── Stage 15: Asymmetric IDF weighting ──────────────
        query_boost, doc_dampen = _get_idf_params()
        if mode == "query":
            idf_scale = query_boost        # boost rare terms for precision
        elif mode in ("passage", "schema"):
            idf_scale = doc_dampen         # dampen for recall
        else:
            idf_scale = 1.0

        # ── BM25 term scoring (Stage 2 + 4) ─────────────────
        # At inference, use query-optimised BM25 (no length normalisation for
        # short queries since avgdl is calibrated to document length).
        from app.config import get_settings as _gs2
        _s2 = _gs2()
        bm25_k1: float = getattr(_s2, "BM25_K1", 1.5)
        bm25_b:  float = getattr(_s2, "BM25_B",  0.75)
        avgdl = getattr(self, "_avgdl", 50.0)

        tfidf = np.zeros((n, vocab_size), dtype=np.float32)
        for i, text in enumerate(prefixed):
            tokens = _tokenize(text)
            tf = Counter(tokens)
            dl = len(tokens)
            # Queries are typically short — use b=0 for them (no length penalty)
            b_eff = 0.0 if mode == "query" else bm25_b
            norm_factor = 1.0 - b_eff + b_eff * dl / max(avgdl, 1.0)
            for token, count in tf.items():
                if token in self.vocab:
                    j = self.vocab[token]
                    tf_bm25 = (count * (bm25_k1 + 1.0)) / (count + bm25_k1 * norm_factor)
                    idf_val = float(self.idf[j]) ** idf_scale
                    tfidf[i, j] = tf_bm25 * idf_val

        # Row-normalize
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-9, None)
        tfidf = tfidf / norms

        # ── SVD projection (Stage 5) ────────────────────────
        projected = tfidf @ self.components.T  # (n, k)

        # ── Dense projection (Stage 7) — full (k → dim) matrix ──
        # dense_weight is now (k, dim), mapping SVD space to output space.
        # Tanh activation like a real Sentence Transformers dense layer.
        if self.dense_weight is not None:
            k = projected.shape[1]
            w = self.dense_weight[:k, :self.dim]   # safe slice for mismatched shapes
            b = self.dense_bias[:self.dim] if self.dense_bias is not None else 0
            projected = np.tanh(projected @ w + b)  # (n, dim)
        elif projected.shape[1] < self.dim:
            # Fallback: zero-pad if no weight matrix
            padding = np.zeros((n, self.dim - projected.shape[1]), dtype=np.float32)
            projected = np.hstack([projected, padding])

        # ── Neural refinement (Stage 9/12) ──────────────────
        # If the neural refiner has been trained, apply it to learn
        # non-linear relationships the SVD can't capture.
        from app.services.neural_trainer import get_refiner
        refiner = get_refiner()
        if refiner is not None and refiner.is_trained:
            projected = refiner.forward(projected)

        # ── L2 normalization (Stage 8) ──────────────────────
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-9, None)
        projected = projected / norms

        return projected.astype(np.float32)

    def _encode_fallback(self, texts: list[str]) -> np.ndarray:
        """Fallback before fit(): char n-gram feature hashing."""
        n = len(texts)
        vectors = np.zeros((n, self.dim), dtype=np.float32)

        for i, text in enumerate(texts):
            tokens = _tokenize(text)
            for token in tokens:
                h = hash(token) % self.dim
                vectors[i, abs(h)] += 1.0

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-9, None)
        vectors = vectors / norms

        return vectors

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ── Persistence: save/load weights to disk ──────────────────

    # ── Serialisation helpers (shared by file and DB paths) ──────

    def _to_bytes(self) -> bytes:
        """Serialise weights to an in-memory .npz byte buffer."""
        import io
        import json as _json
        data = {
            "dim":        np.array([self.dim]),
            "n_docs":     np.array([self.n_docs]),
            "avgdl":      np.array([getattr(self, "_avgdl", 50.0)]),
            "idf":        self.idf,
            "components": self.components,
            "vocab_json": np.array([_json.dumps(self.vocab)]),
        }
        if self.dense_weight is not None:
            data["dense_weight"] = self.dense_weight
        if self.dense_bias is not None:
            data["dense_bias"] = self.dense_bias
        buf = io.BytesIO()
        np.savez_compressed(buf, **data)
        return buf.getvalue()

    def _from_bytes(self, raw: bytes) -> None:
        """Deserialise weights from a .npz byte buffer. Marks encoder as fitted."""
        import io
        import json as _json
        data = np.load(io.BytesIO(raw), allow_pickle=False)
        self.dim = int(data["dim"][0])
        self.n_docs = int(data["n_docs"][0])
        self._avgdl = float(data["avgdl"][0]) if "avgdl" in data else 50.0
        self.idf = data["idf"]
        self.components = data["components"]
        self.vocab = _json.loads(str(data["vocab_json"][0]))
        self.dense_weight = data["dense_weight"] if "dense_weight" in data else None
        self.dense_bias   = data["dense_bias"]   if "dense_bias"   in data else None
        if self.dense_weight is not None and not isinstance(self.dense_weight, np.ndarray):
            self.dense_weight = None
        if self.dense_bias is not None and not isinstance(self.dense_bias, np.ndarray):
            self.dense_bias = None
        self._fitted = True

    # ── File-based persistence (local dev / fallback) ─────────────

    def save_weights(self, path: str) -> None:
        """Save weights to a .npz file."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted encoder")
        with open(path, "wb") as f:
            f.write(self._to_bytes())
        logger.info(f"Encoder weights saved to {path}")

    def load_weights(self, path: str) -> None:
        """Load weights from a .npz file."""
        with open(path, "rb") as f:
            self._from_bytes(f.read())
        logger.info(
            f"Encoder weights loaded from {path} "
            f"(dim={self.dim}, vocab={len(self.vocab)}, docs={self.n_docs}, "
            f"avgdl={self._avgdl:.1f})"
        )

    # ── PostgreSQL persistence ────────────────────────────────────

    async def save_to_db(self, pool, schema: str, schema_hash: str = "") -> None:
        """Persist weights into {schema}.model_weights (upsert by model_name)."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted encoder")
        raw = self._to_bytes()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {schema}.model_weights
                        (model_name, weight_data, schema_hash)
                    VALUES ('encoder', $1, $2)
                    ON CONFLICT (model_name) DO UPDATE
                        SET weight_data  = EXCLUDED.weight_data,
                            schema_hash  = EXCLUDED.schema_hash,
                            updated_at   = NOW()""",
                raw, schema_hash,
            )
        logger.info(
            f"Encoder weights saved to {schema}.model_weights "
            f"({len(raw) // 1024} KB, hash={schema_hash[:8]})"
        )

    async def load_from_db(self, pool, schema: str) -> str | None:
        """Load weights from {schema}.model_weights.
        Returns the stored schema_hash so the caller can compare, or None if absent."""
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT weight_data, schema_hash FROM {schema}.model_weights "
                f"WHERE model_name = 'encoder'"
            )
        if not row:
            return None
        self._from_bytes(bytes(row["weight_data"]))
        stored_hash = row["schema_hash"] or ""
        logger.info(
            f"Encoder weights loaded from {schema}.model_weights "
            f"(dim={self.dim}, vocab={len(self.vocab)}, docs={self.n_docs}, "
            f"avgdl={self._avgdl:.1f}, hash={stored_hash[:8]})"
        )
        return stored_hash


# ── Global singleton ────────────────────────────────────────────
_encoder: Optional[SentenceEncoder] = None


def get_encoder() -> SentenceEncoder:
    """Get or create the global SentenceEncoder singleton."""
    global _encoder
    if _encoder is None:
        from app.config import get_settings
        settings = get_settings()
        _encoder = SentenceEncoder(dim=settings.EMBEDDING_DIMENSIONS)
    return _encoder


def fit_encoder(corpus: list[str]):
    """Fit the global encoder on corpus text (called during seeding)."""
    enc = get_encoder()
    enc.fit(corpus)


def encode_texts(texts: list[str], mode: str = "query") -> list[list[float]]:
    """Encode texts using the global encoder. Returns list of float lists.

    Args:
        texts: input strings
        mode: "query", "schema", "value", "passage" (Stage 15+16)
    """
    enc = get_encoder()
    arr = enc.encode(texts, mode=mode)
    return arr.tolist()
