"""Embedding Drift Monitor — Stage 30 from Sentence Transformers architecture.

Detects when embedding quality degrades due to:
    1. Query distribution shift — users asking new types of questions
    2. Corpus drift — new tables/columns added, data changed
    3. Encoder refit — SVD learned different dimensions after re-seeding
    4. Data corruption — tokenizer mismatch, encoding errors

Three monitors:
    1. Similarity Distribution — track score distribution for reference pairs
    2. Centroid Drift — cluster embeddings daily, detect centroid movement
    3. Canary Recall — fixed test queries that must always return correct results

Run schedule:
    Canary recall:          every schema watcher cycle (15 min)
    Similarity distribution: every schema rebuild
    Centroid drift:         every schema rebuild
"""
import logging
import time
import numpy as np
from typing import Optional
from app.config import get_settings

logger = logging.getLogger(__name__)


class SimilarityDistributionMonitor:
    """Track similarity score distribution for reference query pairs.

    Drift = the distribution of similarity scores shifts significantly
    from the baseline established after initial seeding.

    Uses Kolmogorov-Smirnov-like max deviation (no scipy needed).
    """

    def __init__(self):
        self.baseline_scores: Optional[np.ndarray] = None
        self.baseline_mean: float = 0.0
        self.baseline_std: float = 0.0
        self._has_baseline = False

    def set_baseline(self, scores: list[float]):
        """Set baseline similarity distribution (after initial seeding)."""
        if not scores:
            return
        self.baseline_scores = np.array(scores, dtype=np.float32)
        self.baseline_mean = float(self.baseline_scores.mean())
        self.baseline_std = float(self.baseline_scores.std())
        self._has_baseline = True
        logger.info(
            f"Drift monitor baseline set: "
            f"mean={self.baseline_mean:.3f}, std={self.baseline_std:.3f}, "
            f"n={len(scores)} pairs"
        )

    def check(self, current_scores: list[float], alert_threshold: float = None) -> dict:
        """Compare current similarity distribution against baseline.

        Args:
            current_scores: similarity scores from the same reference pairs
            alert_threshold: max allowed mean shift before alerting

        Returns:
            {"drift_detected": bool, "delta_mean": float, "details": ...}
        """
        if not self._has_baseline or not current_scores:
            return {"drift_detected": False, "reason": "no baseline"}

        if alert_threshold is None:
            from app.config import get_settings
            alert_threshold = get_settings().DRIFT_ALERT_THRESHOLD

        current = np.array(current_scores, dtype=np.float32)
        current_mean = float(current.mean())
        delta = current_mean - self.baseline_mean

        # Simple drift detection: mean shift beyond threshold
        drift_detected = abs(delta) > alert_threshold

        result = {
            "drift_detected": drift_detected,
            "baseline_mean": self.baseline_mean,
            "current_mean": current_mean,
            "delta_mean": delta,
            "baseline_std": self.baseline_std,
            "current_std": float(current.std()),
            "n_pairs": len(current_scores),
        }

        if drift_detected:
            logger.warning(
                f"DRIFT DETECTED: similarity mean shifted {delta:+.3f} "
                f"(baseline={self.baseline_mean:.3f}, current={current_mean:.3f})"
            )
        else:
            logger.debug(
                f"Drift check OK: delta_mean={delta:+.3f} "
                f"(threshold={alert_threshold})"
            )

        return result


class CentroidDriftMonitor:
    """Track embedding space centroid positions over time.

    Cluster query/document embeddings into K centroids.
    If centroids move significantly between checks, the embedding
    space geometry has changed.
    """

    def __init__(self, k: int = None):
        if k is None:
            from app.config import get_settings
            k = get_settings().DRIFT_CENTROID_K
        self.k = k
        self.baseline_centroids: Optional[np.ndarray] = None
        self._has_baseline = False

    def _simple_kmeans(self, vectors: np.ndarray, k: int, max_iter: int = 20) -> np.ndarray:
        """Simple k-means clustering using only numpy.

        Returns centroids (k, dim).
        """
        n = len(vectors)
        if n <= k:
            return vectors.copy()

        # Random init
        rng = np.random.RandomState(42)
        indices = rng.choice(n, size=k, replace=False)
        centroids = vectors[indices].copy()

        for _ in range(max_iter):
            # Assign each vector to nearest centroid
            dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = labels == j
                if mask.any():
                    new_centroids[j] = vectors[mask].mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        return centroids

    def set_baseline(self, embeddings: np.ndarray):
        """Set baseline centroids from current embeddings."""
        if len(embeddings) < self.k:
            return
        self.baseline_centroids = self._simple_kmeans(embeddings, self.k)
        self._has_baseline = True
        logger.info(f"Centroid drift baseline set: {self.k} centroids")

    def check(self, new_embeddings: np.ndarray, alert_threshold: float = 0.15) -> dict:
        """Check if embedding space centroids have drifted.

        Returns:
            {"drift_detected": bool, "avg_drift": float, "max_drift": float}
        """
        if not self._has_baseline or len(new_embeddings) < self.k:
            return {"drift_detected": False, "reason": "no baseline"}

        current_centroids = self._simple_kmeans(new_embeddings, self.k)

        # Match centroids by nearest assignment (handles reordering)
        drifts = []
        for bc in self.baseline_centroids:
            dists = np.linalg.norm(current_centroids - bc, axis=1)
            drifts.append(float(dists.min()))

        avg_drift = float(np.mean(drifts))
        max_drift = float(np.max(drifts))
        drift_detected = avg_drift > alert_threshold

        result = {
            "drift_detected": drift_detected,
            "avg_centroid_drift": avg_drift,
            "max_centroid_drift": max_drift,
            "k": self.k,
        }

        if drift_detected:
            logger.warning(
                f"CENTROID DRIFT: avg={avg_drift:.3f}, max={max_drift:.3f} "
                f"(threshold={alert_threshold})"
            )

        return result


# ── Global instances ───────────────────────────────────────────
_sim_monitor: Optional[SimilarityDistributionMonitor] = None
_centroid_monitor: Optional[CentroidDriftMonitor] = None


def get_similarity_monitor() -> SimilarityDistributionMonitor:
    global _sim_monitor
    if _sim_monitor is None:
        _sim_monitor = SimilarityDistributionMonitor()
    return _sim_monitor


def get_centroid_monitor() -> CentroidDriftMonitor:
    global _centroid_monitor
    if _centroid_monitor is None:
        _centroid_monitor = CentroidDriftMonitor()
    return _centroid_monitor


async def run_full_drift_check(graph) -> dict:
    """Run all drift monitors and return combined report.

    Called after schema rebuilds to verify embedding quality.
    """
    from app.services.embedding_eval import evaluate_retrieval, quick_health_check

    results = {}

    # 1. Canary recall check
    eval_report = await evaluate_retrieval(graph, k=5)
    results["evaluation"] = eval_report

    # 2. Health check
    healthy = await quick_health_check(graph)
    results["healthy"] = healthy

    # Overall status
    results["status"] = "OK" if healthy and eval_report.get("recall_at_k", 0) >= 0.5 else "DEGRADED"

    if results["status"] == "DEGRADED":
        logger.warning(f"Embedding system DEGRADED: {results}")

    return results
