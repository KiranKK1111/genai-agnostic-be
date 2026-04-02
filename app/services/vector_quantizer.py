"""Vector Quantisation — Stage 23 from Sentence Transformers architecture.

Compresses stored embedding vectors (not model weights) for storage reduction.
Two strategies:
    Scalar:  float32 → uint8  (4x reduction, <1% accuracy loss)
    Binary:  float32 → 1 bit  (32x reduction, ~2-3% accuracy loss with rescore)

For 10M vectors at dim=384:
    float32:   14.4 GB
    int8:       3.6 GB   (scalar)
    binary:     450 MB   (binary)

Binary quantisation with rescoring:
    1. Fast Hamming distance on binary vectors → top-1000 candidates
    2. Rescore those 1000 using original float32 vectors
    3. Recovers ~95% of full-precision accuracy
"""
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


# ── Scalar Quantisation: float32 → uint8 (4x compression) ─────

class ScalarQuantizer:
    """Map each float dimension into [0, 255] range.

    Learns min/max per dimension from a calibration set,
    then quantises any vector into uint8.
    """

    def __init__(self):
        self.mins: Optional[np.ndarray] = None     # (dim,)
        self.scales: Optional[np.ndarray] = None   # (dim,)
        self._fitted = False

    def fit(self, vectors: np.ndarray):
        """Learn quantisation parameters from a set of vectors."""
        self.mins = vectors.min(axis=0).astype(np.float32)
        maxes = vectors.max(axis=0).astype(np.float32)
        self.scales = ((maxes - self.mins) / 255.0).astype(np.float32)
        self.scales = np.clip(self.scales, 1e-9, None)
        self._fitted = True
        logger.info(f"ScalarQuantizer fitted on {len(vectors)} vectors")

    def quantize(self, vectors: np.ndarray) -> np.ndarray:
        """Quantize float32 vectors to uint8. Returns (N, dim) uint8."""
        if not self._fitted:
            self.fit(vectors)
        q = ((vectors - self.mins) / self.scales).clip(0, 255).astype(np.uint8)
        return q

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        """Reconstruct float32 vectors from uint8. Returns (N, dim) float32."""
        return quantized.astype(np.float32) * self.scales + self.mins

    def similarity(self, query: np.ndarray, quantized_db: np.ndarray) -> np.ndarray:
        """Compute approximate cosine similarity using dequantized vectors."""
        db_approx = self.dequantize(quantized_db)
        # L2 normalize both
        q_norm = query / np.clip(np.linalg.norm(query, axis=-1, keepdims=True), 1e-9, None)
        d_norm = db_approx / np.clip(np.linalg.norm(db_approx, axis=1, keepdims=True), 1e-9, None)
        return (q_norm @ d_norm.T).flatten()

    @property
    def compression_ratio(self) -> float:
        return 4.0  # float32 (4 bytes) → uint8 (1 byte)


# ── Binary Quantisation: float32 → 1 bit (32x compression) ────

class BinaryQuantizer:
    """Binarize vectors: 1 if dimension > 0, else 0. Pack 8 dims per byte.

    After L2 normalisation, the sign of each dimension captures the
    most important bit of information. Hamming distance on binary
    vectors approximates cosine similarity.
    """

    def quantize(self, vectors: np.ndarray) -> np.ndarray:
        """Quantize float32 vectors to packed binary. Returns (N, dim//8) uint8."""
        bits = (vectors > 0).astype(np.uint8)
        return np.packbits(bits, axis=1)

    def hamming_similarity(self, query_bin: np.ndarray, db_bin: np.ndarray) -> np.ndarray:
        """Compute Hamming similarity (1 - normalised Hamming distance).

        Returns similarity scores in [0, 1] range.
        """
        xor = np.bitwise_xor(query_bin, db_bin)
        # Count differing bits
        bit_diffs = np.unpackbits(xor, axis=1).sum(axis=1)
        total_bits = query_bin.shape[1] * 8
        return 1.0 - (bit_diffs / total_bits)

    def search(self, query: np.ndarray, db_vectors: np.ndarray,
               db_bin: np.ndarray, k: int = 5,
               rescore_k: int = 100) -> list[tuple[int, float]]:
        """Binary search with rescoring (two-pass strategy).

        Pass 1: Fast Hamming distance on binary → top rescore_k candidates
        Pass 2: Exact cosine similarity on float32 for those candidates

        Returns [(index, similarity_score)] sorted descending.
        """
        # Quantize query
        q_bin = self.quantize(query.reshape(1, -1))

        # Pass 1: Hamming distance (fast, approximate)
        ham_scores = self.hamming_similarity(q_bin, db_bin)
        candidate_idxs = np.argsort(ham_scores)[::-1][:rescore_k]

        # Pass 2: Exact cosine on candidates (accurate)
        q_norm = query / np.clip(np.linalg.norm(query), 1e-9, None)
        candidates = db_vectors[candidate_idxs]
        c_norms = np.linalg.norm(candidates, axis=1, keepdims=True)
        c_norms = np.clip(c_norms, 1e-9, None)
        candidates = candidates / c_norms

        exact_scores = (candidates @ q_norm).flatten()
        top_k_local = np.argsort(exact_scores)[::-1][:k]

        results = [
            (int(candidate_idxs[i]), float(exact_scores[i]))
            for i in top_k_local
        ]
        return results

    @property
    def compression_ratio(self) -> float:
        return 32.0  # float32 (32 bits) → 1 bit


# ── Global instances ───────────────────────────────────────────
_scalar_quantizer: Optional[ScalarQuantizer] = None
_binary_quantizer: Optional[BinaryQuantizer] = None


def get_scalar_quantizer() -> ScalarQuantizer:
    global _scalar_quantizer
    if _scalar_quantizer is None:
        _scalar_quantizer = ScalarQuantizer()
    return _scalar_quantizer


def get_binary_quantizer() -> BinaryQuantizer:
    global _binary_quantizer
    if _binary_quantizer is None:
        _binary_quantizer = BinaryQuantizer()
    return _binary_quantizer
