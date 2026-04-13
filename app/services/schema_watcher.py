"""Schema watcher — periodic re-inspection + housekeeping cleanup."""
import asyncio
import logging
from app.services.schema_inspector import inspect_schema
from app.services.schema_seeder import seed_all
from app.services.rag_indexer import get_indexer
from app.services.faiss_manager import invalidate_retrieval_cache, cleanup_stale_cache, build_all_ivf_clusters
from app.services.feedback_trainer import fine_tune_from_feedback
from app.orchestrator import set_schema_graph
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)

_task: asyncio.Task | None = None


async def _cleanup_expired(settings):
    """Purge expired sessions, orphaned file chunks, and stale data."""
    pool = get_pool()
    schema = settings.APP_SCHEMA
    ttl_hours = settings.SESSION_TTL_HOURS
    try:
        async with pool.acquire() as conn:
            # Delete expired KV store entries
            await conn.execute(
                f"DELETE FROM {schema}.kv_store WHERE expires_at IS NOT NULL AND expires_at < now()"
            )

            # Find expired sessions
            expired = await conn.fetch(
                f"SELECT id FROM {schema}.chat_sessions WHERE updated_at < now() - interval '{ttl_hours} hours'"
            )
            if expired:
                expired_ids = [r["id"] for r in expired]
                # Delete orphaned file chunks for expired sessions
                await conn.execute(
                    f"DELETE FROM {schema}.file_chunks WHERE session_id = ANY($1::uuid[])",
                    expired_ids
                )
                # Delete orphaned chunk embeddings
                for sid in expired_ids:
                    await conn.execute(
                        f"""DELETE FROM {schema}.embedding_metadata
                            WHERE index_name = 'chunks_idx'
                              AND payload->>'session_id' = $1""",
                        str(sid)
                    )
                logger.info(f"Cleanup: purged {len(expired_ids)} expired sessions and their file chunks")

            # Prune old query logs (keep last 7 days)
            await conn.execute(
                f"DELETE FROM {schema}.query_logs WHERE created_at < now() - interval '7 days'"
            )

            # Prune old retrieval traces (keep last 7 days)
            await conn.execute(
                f"DELETE FROM {schema}.retrieval_trace WHERE created_at < now() - interval '7 days'"
            )
    except Exception as e:
        logger.debug(f"Cleanup skipped: {e}")


async def _watch_loop():
    settings = get_settings()
    interval = settings.SCHEMA_WATCH_INTERVAL_MIN * 60
    while True:
        await asyncio.sleep(interval)
        try:
            # ── Housekeeping cleanup ──────────────────────────
            await _cleanup_expired(settings)
            await cleanup_stale_cache()

            # ── Schema re-inspection ──────────────────────────
            logger.info("Schema watcher: re-inspecting schema...")
            graph = await inspect_schema()
            set_schema_graph(graph)

            # Invalidate retrieval cache BEFORE rebuilding indexes
            await invalidate_retrieval_cache()

            # Use RAGIndexer for incremental re-indexing (skips unchanged content)
            indexer = get_indexer()
            stats_list = await indexer.reindex_all(graph, force=False)
            for s in stats_list:
                logger.info(f"Schema watcher re-index: {s}")

            # Rebuild IVF cluster indexes to pick up newly added/changed vectors
            await build_all_ivf_clusters()

            # ── Incremental online learning from feedback ─────────────────
            # Fine-tune the NeuralRefiner on any feedback that has accumulated
            # since the last training run (trained_on=false rows only).
            # Runs silently if there are fewer than FEEDBACK_MIN_PAIRS new pairs.
            try:
                from app.services.sentence_encoder import get_encoder as _get_enc
                _pool = get_pool()
                trained, n_fb = await fine_tune_from_feedback(
                    pool=_pool,
                    schema=settings.APP_SCHEMA,
                    graph=graph,
                    encoder=_get_enc(),
                )
                if trained:
                    logger.info(
                        f"Schema watcher: neural refiner fine-tuned on {n_fb} feedback pairs"
                    )
            except Exception as e:
                logger.debug(f"Feedback fine-tuning skipped: {e}")

            # Run drift check (Stage 30) after re-seeding
            from app.services.drift_monitor import run_full_drift_check
            try:
                drift_report = await run_full_drift_check(graph)
                if drift_report.get("status") == "DEGRADED":
                    logger.warning(f"Schema watcher: embedding quality degraded after refresh")
            except Exception as e:
                logger.debug(f"Drift check skipped: {e}")

            logger.info(f"Schema watcher: refreshed ({len(graph.tables)} tables)")
        except Exception as e:
            logger.error(f"Schema watcher error: {e}")


def start_watcher():
    global _task
    _task = asyncio.create_task(_watch_loop())
    logger.info("Schema watcher started")


def stop_watcher():
    global _task
    if _task:
        _task.cancel()
        _task = None
