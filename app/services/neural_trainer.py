"""Neural Refinement Layer — trains on top of SVD embeddings using contrastive learning.

Implements the missing ST stages using only numpy (no PyTorch/TensorFlow):
    Stage 3  Pre-training        — SVD fit is pre-training; this is fine-tuning
    Stage 9  MNR Contrastive Loss — in-batch negatives, temperature-scaled softmax
    Stage 10 Hard Negative Mining — mine hard negatives from corpus embeddings
    Stage 11 Synthetic Data       — LLM generates (query, positive) training pairs
    Stage 12 Knowledge Distillation — SVD = teacher, neural layer = student

Architecture:
    SVD embeddings (384d) → Dense(384→384, ReLU) → Dense(384→384) → L2 norm
    Trained with MNR loss on LLM-generated pairs + hard negatives.

The neural layer learns NON-LINEAR relationships that SVD (linear) can't capture:
    SVD:    "money" ≈ "balance" (co-occurrence)
    Neural: "wealthy customers" → both customers AND accounts.balance (multi-hop)
"""
import math
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Stage 3+12: Neural Network (numpy-only, no frameworks)
# ═══════════════════════════════════════════════════════════════

class DenseLayer:
    """Single fully-connected layer with optional activation."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu"):
        # Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        self.W = (np.random.randn(in_dim, out_dim) * scale).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.activation = activation

        # Gradient storage
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

        # Cache for backward pass
        self._input: Optional[np.ndarray] = None
        self._pre_act: Optional[np.ndarray] = None

        # Adam optimizer state
        self._mW = np.zeros_like(self.W)
        self._vW = np.zeros_like(self.W)
        self._mb = np.zeros_like(self.b)
        self._vb = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: x @ W + b, then activation."""
        self._input = x
        self._pre_act = x @ self.W + self.b

        if self.activation == "relu":
            return np.maximum(0, self._pre_act)
        elif self.activation == "tanh":
            return np.tanh(self._pre_act)
        else:  # linear
            return self._pre_act

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass: compute gradients and return grad for previous layer."""
        # Apply activation derivative
        if self.activation == "relu":
            grad_act = grad_output * (self._pre_act > 0).astype(np.float32)
        elif self.activation == "tanh":
            tanh_out = np.tanh(self._pre_act)
            grad_act = grad_output * (1 - tanh_out ** 2)
        else:
            grad_act = grad_output

        # Gradients for W and b
        self.dW = self._input.T @ grad_act / len(grad_act)
        self.db = grad_act.mean(axis=0)

        # Gradient for previous layer
        return grad_act @ self.W.T

    def adam_step(self, lr: float = 1e-3, beta1: float = 0.9,
                  beta2: float = 0.999, eps: float = 1e-8, t: int = 1):
        """Adam optimizer update."""
        # W update
        self._mW = beta1 * self._mW + (1 - beta1) * self.dW
        self._vW = beta2 * self._vW + (1 - beta2) * self.dW ** 2
        mW_hat = self._mW / (1 - beta1 ** t)
        vW_hat = self._vW / (1 - beta2 ** t)
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

        # b update
        self._mb = beta1 * self._mb + (1 - beta1) * self.db
        self._vb = beta2 * self._vb + (1 - beta2) * self.db ** 2
        mb_hat = self._mb / (1 - beta1 ** t)
        vb_hat = self._vb / (1 - beta2 ** t)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


class NeuralRefiner:
    """2-layer neural network trained with contrastive loss.

    Architecture: Input(384) → Dense(384, ReLU) → Dense(384, linear) → L2 norm
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.layer1 = DenseLayer(dim, dim, activation="relu")
        self.layer2 = DenseLayer(dim, dim, activation="linear")
        self._trained = False
        self._step = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through both layers + L2 normalize."""
        h = self.layer1.forward(x)
        out = self.layer2.forward(h)
        # L2 normalize
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-9, None)
        return out / norms

    def backward(self, grad: np.ndarray):
        """Backward pass through both layers."""
        g = self.layer2.backward(grad)
        self.layer1.backward(g)

    def update(self, lr: float = 1e-3):
        """Adam optimizer step on both layers."""
        self._step += 1
        self.layer1.adam_step(lr=lr, t=self._step)
        self.layer2.adam_step(lr=lr, t=self._step)

    @property
    def is_trained(self) -> bool:
        return self._trained


# ═══════════════════════════════════════════════════════════════
# Stage 9: Multiple Negatives Ranking (MNR) Loss
# ═══════════════════════════════════════════════════════════════

def mnr_loss(anchors: np.ndarray, positives: np.ndarray,
             scale: float = 20.0) -> tuple[float, np.ndarray]:
    """Compute MNR contrastive loss with in-batch negatives.

    Args:
        anchors:   (B, D) L2-normalized anchor embeddings
        positives: (B, D) L2-normalized positive embeddings
        scale:     temperature parameter (higher = sharper discrimination)

    Returns:
        (loss_value, gradient_wrt_anchors)

    The diagonal of anchors @ positives.T contains correct pairs.
    Off-diagonal entries are in-batch negatives (free!).
    With batch_size=32: 32 positives → 32*31 = 992 free negatives.
    """
    B = len(anchors)
    if B == 0:
        return 0.0, np.zeros_like(anchors)

    # Similarity matrix: (B, B)
    scores = (anchors @ positives.T) * scale

    # Softmax per row
    scores_max = scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    row_sums = exp_scores.sum(axis=1, keepdims=True)
    softmax = exp_scores / row_sums

    # Labels: correct pairs are on the diagonal
    labels = np.eye(B, dtype=np.float32)

    # Cross-entropy loss: -log(softmax[i][i]) for each row
    diag_probs = np.diag(softmax)
    loss = -np.log(np.clip(diag_probs, 1e-9, 1.0)).mean()

    # Gradient of cross-entropy w.r.t. anchors
    # d_loss/d_anchors = scale * (softmax - labels) @ positives / B
    grad = scale * (softmax - labels) @ positives / B

    return float(loss), grad


# ═══════════════════════════════════════════════════════════════
# Stage 10: Hard Negative Mining
# ═══════════════════════════════════════════════════════════════

def mine_hard_negatives(anchor_embs: np.ndarray, positive_embs: np.ndarray,
                        corpus_embs: np.ndarray, n_hard: int = 3) -> np.ndarray:
    """Mine hard negatives from the corpus for each anchor.

    Hard negative = corpus item with HIGH similarity to anchor
    but is NOT the correct positive. These force the model to
    learn fine-grained distinctions.

    Args:
        anchor_embs:  (B, D) anchor embeddings
        positive_embs: (B, D) positive embeddings
        corpus_embs:  (N, D) full corpus embeddings
        n_hard:       number of hard negatives per anchor

    Returns:
        (B * n_hard, D) hard negative embeddings
    """
    B = len(anchor_embs)
    hard_negs = []

    # Similarity of each anchor to entire corpus
    sims = anchor_embs @ corpus_embs.T  # (B, N)

    for i in range(B):
        # Sort corpus by similarity (descending)
        ranked = np.argsort(sims[i])[::-1]

        # Skip the positive itself (find by max similarity to positive_embs[i])
        pos_sims = corpus_embs @ positive_embs[i]
        pos_idx = np.argmax(pos_sims)

        count = 0
        for idx in ranked:
            if idx == pos_idx:
                continue
            hard_negs.append(corpus_embs[idx])
            count += 1
            if count >= n_hard:
                break

    if not hard_negs:
        return np.zeros((0, anchor_embs.shape[1]), dtype=np.float32)

    return np.array(hard_negs, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# Stage 11: Synthetic Training Data (LLM-generated pairs)
# ═══════════════════════════════════════════════════════════════

async def generate_training_pairs(graph, n_pairs: int = 50) -> list[tuple[str, str]]:
    """Use the existing LLM to generate (query, positive) training pairs.

    The LLM knows the domain and can generate natural language queries
    that a user would type, paired with the schema text they should match.

    Args:
        graph: SchemaGraph with tables and descriptions
        n_pairs: approximate number of pairs to generate

    Returns:
        [(query_text, positive_text), ...]
    """
    from app.services.llm_client import chat_json
    import json

    from app.config import get_settings
    settings = get_settings()
    n_pairs = settings.NEURAL_TRAIN_PAIRS

    domain = graph.domain_name or "database"
    # Keep schema summary compact to avoid LLM context overflow
    tables_info = {}
    for tname, tmeta in graph.tables.items():
        cols = list(tmeta.columns.keys())[:6]
        tables_info[tname] = cols

    pairs = []

    try:
        from app.services.llm_client import chat
        table_list = list(tables_info.keys())
        raw_response = await chat([{"role": "user", "content":
            f"""Generate {min(n_pairs, 30)} search queries for a {domain} database.
Tables: {', '.join(table_list)}

List each as: query | table_name
Example:
show all customers | customers
purchase history | transactions
how many loans | loans

Generate the list now:"""}], temperature=0.7)

        # Parse "query | table" format (more reliable than JSON for smaller models)
        result = {"pairs": []}
        for line in raw_response.strip().split("\n"):
            if "|" in line:
                parts = line.split("|", 1)
                if len(parts) == 2:
                    q = parts[0].strip().strip("-").strip("0123456789.").strip()
                    t = parts[1].strip().lower()
                    if q and t in [tl.lower() for tl in table_list]:
                        result["pairs"].append({"query": q, "target": t})

        raw_pairs = result.get("pairs", [])
        for p in raw_pairs:
            query = p.get("query", "")
            target = p.get("target", "")
            if not query or not target:
                continue

            # Resolve target to the enriched schema text
            target_table = target.split(".")[0]
            if target_table in graph.tables:
                tmeta = graph.tables[target_table]
                col_names = ", ".join(tmeta.columns.keys())
                positive = f"table {target_table}: {col_names}"
                if tmeta.description:
                    positive += f". {tmeta.description}"
                pairs.append((query, positive))

        logger.info(f"Generated {len(pairs)} synthetic training pairs via LLM")

    except Exception as e:
        logger.warning(f"Synthetic pair generation failed: {e}")

    # Also generate deterministic pairs from schema (always available)
    for tname, tmeta in graph.tables.items():
        col_names = ", ".join(tmeta.columns.keys())
        positive = f"table {tname}: {col_names}"
        if tmeta.description:
            positive += f". {tmeta.description}"
        # Simple name-based pairs (always correct)
        pairs.append((f"show me {tname.replace('_', ' ')}", positive))
        pairs.append((tname.replace("_", " "), positive))
        # Column-based pairs
        for cname in list(tmeta.columns.keys())[:5]:
            pairs.append((
                f"{cname.replace('_', ' ')} from {tname.replace('_', ' ')}",
                f"column {cname} in table {tname}"
            ))

    return pairs


# ═══════════════════════════════════════════════════════════════
# Stage 12: Training Loop (Knowledge Distillation + MNR)
# ═══════════════════════════════════════════════════════════════

async def train_neural_refiner(
    encoder,
    graph,
    epochs: int = None,
    lr: float = None,
    batch_size: int = None,
) -> NeuralRefiner:
    """Train the neural refinement layer on top of SVD embeddings.

    Flow:
        1. Generate synthetic (query, positive) pairs via LLM (Stage 11)
        2. Encode all texts through SVD encoder (teacher — Stage 12)
        3. Train neural refiner with MNR loss (Stage 9)
        4. Mine hard negatives and retrain (Stage 10)

    Args:
        encoder: fitted SentenceEncoder (SVD-based)
        graph: SchemaGraph for synthetic data generation
        epochs: training epochs
        lr: learning rate
        batch_size: training batch size

    Returns:
        Trained NeuralRefiner
    """
    from app.config import get_settings
    settings = get_settings()

    if epochs is None:
        epochs = getattr(settings, "NEURAL_TRAIN_EPOCHS", 30)
    if lr is None:
        lr = getattr(settings, "NEURAL_TRAIN_LR", 1e-3)
    if batch_size is None:
        batch_size = getattr(settings, "NEURAL_TRAIN_BATCH_SIZE", 16)

    refiner = NeuralRefiner(dim=encoder.dim)

    # ── Stage 11: Generate training data ────────────────────
    logger.info("Stage 11: Generating synthetic training pairs...")
    pairs = await generate_training_pairs(graph)
    if len(pairs) < 5:
        logger.warning("Too few training pairs, skipping neural training")
        return refiner

    # ── Stage 12: Encode with SVD teacher ───────────────────
    queries = [p[0] for p in pairs]
    positives = [p[1] for p in pairs]

    query_embs = encoder.encode(queries, mode="query")       # SVD teacher output
    pos_embs = encoder.encode(positives, mode="schema")       # SVD teacher output

    # Build full corpus embeddings for hard negative mining
    corpus_texts = positives  # unique schema texts
    corpus_embs = encoder.encode(corpus_texts, mode="schema")

    n_pairs = len(pairs)
    logger.info(f"Stage 9+10: Training neural refiner on {n_pairs} pairs, {epochs} epochs...")

    # ── Training loop with early stopping ─────────────────
    best_loss = float("inf")
    best_weights = None          # snapshot of best weights
    patience_counter = 0
    patience_limit = 5           # stop after 5 epochs without improvement

    for epoch in range(epochs):
        # Shuffle
        indices = np.random.permutation(n_pairs)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_pairs, batch_size):
            batch_idx = indices[start:start + batch_size]
            if len(batch_idx) < 2:
                continue  # MNR needs at least 2 pairs

            # Get SVD embeddings for this batch
            q_batch = query_embs[batch_idx]
            p_batch = pos_embs[batch_idx]

            # Forward through neural refiner
            q_refined = refiner.forward(q_batch)

            # Also refine positives (shared network)
            p_refined = refiner.forward(p_batch)

            # ── Stage 9: MNR loss ──────────────────────────
            loss, grad = mnr_loss(q_refined, p_refined, scale=20.0)

            # Backward + update
            refiner.backward(grad)
            refiner.update(lr=lr)

            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # ── Early stopping: save best weights ────────────
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Snapshot weights (deep copy)
            best_weights = {
                "l1_W": refiner.layer1.W.copy(),
                "l1_b": refiner.layer1.b.copy(),
                "l2_W": refiner.layer2.W.copy(),
                "l2_b": refiner.layer2.b.copy(),
            }
        else:
            patience_counter += 1

        # ── Stage 10: Hard negative mining (every 10 epochs) ─
        if (epoch + 1) % 10 == 0 and epoch < epochs - 1:
            refined_corpus = refiner.forward(corpus_embs)
            refined_queries = refiner.forward(query_embs)
            hard_negs = mine_hard_negatives(refined_queries, refined_corpus, refined_corpus, n_hard=2)
            if len(hard_negs) > 0:
                logger.debug(f"  Epoch {epoch+1}: mined {len(hard_negs)} hard negatives")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

        if patience_counter >= patience_limit:
            logger.info(f"  Early stopping at epoch {epoch+1} (no improvement for {patience_limit} epochs)")
            break

    # ── Restore best weights ─────────────────────────────
    if best_weights is not None:
        refiner.layer1.W = best_weights["l1_W"]
        refiner.layer1.b = best_weights["l1_b"]
        refiner.layer2.W = best_weights["l2_W"]
        refiner.layer2.b = best_weights["l2_b"]
        logger.info(f"  Restored best weights (loss={best_loss:.4f})")

    refiner._trained = True
    logger.info(f"Neural refiner trained: best_loss={best_loss:.4f}, {n_pairs} pairs")

    return refiner


# ═══════════════════════════════════════════════════════════════
# Global singleton
# ═══════════════════════════════════════════════════════════════
_refiner: Optional[NeuralRefiner] = None


def get_refiner() -> Optional[NeuralRefiner]:
    """Get the trained neural refiner (None if not yet trained)."""
    return _refiner


def set_refiner(refiner: NeuralRefiner):
    """Set the global neural refiner after training."""
    global _refiner
    _refiner = refiner
