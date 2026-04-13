"""Feedback-driven online learning — fine-tunes NeuralRefiner from user interactions.

Learning loop (triggered by schema_watcher every SCHEMA_WATCH_INTERVAL_MIN minutes):

    ┌─────────────────────────────────────────────────────────────┐
    │  Collect signals                                            │
    │   • Implicit +1  — every successful SQL query execution     │
    │   • Explicit +1  — user clicks thumbs-up                   │
    │   • Explicit -1  — user clicks thumbs-down (not used yet)  │
    │     (stored in {schema}.feedback with trained_on=false)    │
    ├─────────────────────────────────────────────────────────────┤
    │  Build training pairs (only NEW rows, trained_on=false)     │
    │   query_text  → resolved_table's schema description         │
    │   e.g. "case status distribution" → "table cra_score_report:│
    │          case_id, case_status, score ..."                   │
    ├─────────────────────────────────────────────────────────────┤
    │  Fine-tune NeuralRefiner (MNR contrastive loss, few epochs) │
    │   • Low LR (1e-4) prevents catastrophic forgetting          │
    │   • Small batch (≤16) with in-batch negatives               │
    ├─────────────────────────────────────────────────────────────┤
    │  Persist updated weights → mstr_app.model_weights          │
    │  Mark used rows → trained_on=true (incremental learning)    │
    └─────────────────────────────────────────────────────────────┘

Over time the refiner learns YOUR domain's natural-language → schema mapping.
After ~50 successful queries the embedding quality measurably improves.
"""
import json
import logging
import numpy as np
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)

# Defaults (overridden by Settings if present)
_MIN_PAIRS_DEFAULT = 10
_EPOCHS_DEFAULT = 10
_LR_DEFAULT = 1e-4
_DAYS_BACK_DEFAULT = 30
_MAX_PAIRS = 500   # cap to avoid OOM on large deployments


# ── Schema text builder ───────────────────────────────────────

def _make_positive_text(table: str, columns: list, graph) -> str:
    """Build a positive schema text for a (table, columns) pair.

    Mirrors the format used during initial seeding so the encoder sees
    the same text distribution at fine-tune time as at pre-train time.
    """
    if graph and table in graph.tables:
        tmeta = graph.tables[table]
        col_names = ", ".join(list(tmeta.columns.keys())[:12])
        text = f"table {table}: {col_names}"
        if tmeta.description:
            text += f". {tmeta.description}"
        return text
    if columns:
        return f"table {table}: {', '.join(str(c) for c in columns[:12])}"
    return f"table {table}"


# ── Signal recording (called from orchestrator + message_actions) ─

async def record_query_feedback(
    pool,
    schema: str,
    session_id: str,
    message_id: str,
    query_text: str,
    resolved_table: str,
    resolved_columns: list,
    plan_json: dict,
    rating: int,
) -> None:
    """Persist one implicit training signal into {schema}.feedback.

    Called automatically after every successful DB query execution (rating=1).
    Failures are NOT recorded as negatives because SQL errors don't reliably
    indicate wrong entity resolution — they might be temporary issues.

    Safe to fail silently — this is a best-effort background operation.
    """
    if not query_text or not resolved_table:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {schema}.feedback
                        (user_id, session_id, message_id, rating,
                         query_text, resolved_table, resolved_columns, plan_json,
                         intent_at_time, trained_on)
                    VALUES ('system', $1::uuid, $2::uuid, $3,
                            $4, $5, $6, $7::jsonb, 'DB_QUERY', false)""",
                session_id,
                message_id,
                rating,
                query_text.strip(),
                resolved_table.strip(),
                list(resolved_columns or []),
                json.dumps(plan_json or {}),
            )
    except Exception as e:
        logger.debug(f"record_query_feedback skipped: {e}")


# ── Training data loader ──────────────────────────────────────

async def load_feedback_pairs(
    pool,
    schema: str,
    graph,
    days_back: int = _DAYS_BACK_DEFAULT,
) -> list[tuple[str, str]]:
    """Load untrained positive feedback and convert to (query, positive_text) pairs.

    Only returns rows with trained_on=false so each signal is used exactly once.
    Returns [(query_text, positive_schema_text), ...].
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT query_text, resolved_table, resolved_columns
                    FROM {schema}.feedback
                    WHERE rating = 1
                      AND trained_on = false
                      AND query_text IS NOT NULL
                      AND resolved_table IS NOT NULL
                      AND created_at > now() - interval '{days_back} days'
                    ORDER BY created_at DESC
                    LIMIT {_MAX_PAIRS}""",
            )
    except Exception as e:
        logger.debug(f"load_feedback_pairs: {e}")
        return []

    pairs: list[tuple[str, str]] = []
    for row in rows:
        q = (row["query_text"] or "").strip()
        t = (row["resolved_table"] or "").strip()
        cols = list(row["resolved_columns"] or [])
        if q and t:
            pos_text = _make_positive_text(t, cols, graph)
            pairs.append((q, pos_text))
    return pairs


async def _count_pending_pairs(pool, schema: str, days_back: int) -> int:
    """Fast count of untrained positive rows — used to decide whether to train."""
    try:
        async with pool.acquire() as conn:
            return await conn.fetchval(
                f"""SELECT COUNT(*) FROM {schema}.feedback
                    WHERE rating = 1
                      AND trained_on = false
                      AND query_text IS NOT NULL
                      AND resolved_table IS NOT NULL
                      AND created_at > now() - interval '{days_back} days'"""
            ) or 0
    except Exception:
        return 0


async def _mark_feedback_as_trained(pool, schema: str, days_back: int) -> int:
    """Flip trained_on=true for all recently processed positive rows.

    Returns the number of rows marked.
    """
    try:
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"""UPDATE {schema}.feedback
                    SET trained_on = true
                    WHERE rating = 1
                      AND trained_on = false
                      AND query_text IS NOT NULL
                      AND resolved_table IS NOT NULL
                      AND created_at > now() - interval '{days_back} days'"""
            )
        count_str = (result or "").split()[-1]
        return int(count_str) if count_str.isdigit() else 0
    except Exception as e:
        logger.debug(f"_mark_feedback_as_trained: {e}")
        return 0


# ── Main fine-tuning entry point ──────────────────────────────

async def fine_tune_from_feedback(
    pool,
    schema: str,
    graph,
    encoder,
    min_pairs: int = _MIN_PAIRS_DEFAULT,
) -> tuple[bool, int]:
    """Incrementally fine-tune the NeuralRefiner on accumulated user feedback.

    Args:
        pool:      asyncpg connection pool
        schema:    app schema name (e.g. 'mstr_app')
        graph:     SchemaGraph (to build positive text for each resolved table)
        encoder:   fitted SentenceEncoder (teacher — encodes both query + schema)
        min_pairs: minimum new pairs required to trigger a training run

    Returns:
        (trained: bool, n_pairs: int)
    """
    settings = get_settings()
    days_back = getattr(settings, "FEEDBACK_DAYS_BACK", _DAYS_BACK_DEFAULT)
    epochs    = getattr(settings, "FEEDBACK_FINE_TUNE_EPOCHS", _EPOCHS_DEFAULT)
    lr        = getattr(settings, "FEEDBACK_FINE_TUNE_LR", _LR_DEFAULT)
    min_pairs = getattr(settings, "FEEDBACK_MIN_PAIRS", min_pairs)
    batch_size = getattr(settings, "NEURAL_TRAIN_BATCH_SIZE", 16)

    # Fast count check before loading all rows
    pending = await _count_pending_pairs(pool, schema, days_back)
    if pending < min_pairs:
        logger.debug(
            f"Feedback trainer: {pending} new pairs available (min={min_pairs}) — skipping"
        )
        return False, pending

    # Load full pairs for encoding
    pairs = await load_feedback_pairs(pool, schema, graph, days_back=days_back)
    n = len(pairs)
    if n < min_pairs:
        return False, n

    logger.info(f"Feedback trainer: fine-tuning on {n} pairs ({epochs} epochs, lr={lr})")

    queries   = [p[0] for p in pairs]
    positives = [p[1] for p in pairs]

    # Encode with the SVD teacher (encoder already fitted to this schema)
    try:
        q_embs = encoder.encode(queries,   mode="query")
        p_embs = encoder.encode(positives, mode="schema")
    except Exception as e:
        logger.warning(f"Feedback trainer: encoding failed: {e}")
        return False, 0

    # Get or create the neural refiner
    from app.services.neural_trainer import get_refiner, NeuralRefiner, mnr_loss, set_refiner
    refiner = get_refiner()
    if refiner is None:
        refiner = NeuralRefiner(dim=encoder.dim)

    best_loss = float("inf")
    best_state: dict | None = None

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            idx = perm[start: start + batch_size]
            if len(idx) < 2:
                continue  # MNR requires ≥2 pairs

            q_ref = refiner.forward(q_embs[idx])
            p_ref = refiner.forward(p_embs[idx])

            loss, grad = mnr_loss(q_ref, p_ref, scale=20.0)
            refiner.backward(grad)
            refiner.update(lr=lr)

            epoch_loss += loss
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {
                "l1_W": refiner.layer1.W.copy(),
                "l1_b": refiner.layer1.b.copy(),
                "l2_W": refiner.layer2.W.copy(),
                "l2_b": refiner.layer2.b.copy(),
            }

    # Restore best weights
    if best_state is not None:
        refiner.layer1.W = best_state["l1_W"]
        refiner.layer1.b = best_state["l1_b"]
        refiner.layer2.W = best_state["l2_W"]
        refiner.layer2.b = best_state["l2_b"]

    refiner._trained = True
    set_refiner(refiner)

    # Persist updated weights to DB
    try:
        await refiner.save_to_db(pool, schema)
        logger.info(
            f"Feedback trainer: refiner updated — best_loss={best_loss:.4f}, {n} pairs"
        )
    except Exception as e:
        logger.warning(f"Feedback trainer: save failed: {e}")

    # Mark all consumed rows so they aren't reused
    marked = await _mark_feedback_as_trained(pool, schema, days_back=days_back)
    logger.info(f"Feedback trainer: {marked} feedback rows marked as trained")

    return True, n
