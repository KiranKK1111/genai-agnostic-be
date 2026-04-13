"""FastAPI application factory — startup, CORS, routers."""
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.database import create_pool, close_pool, get_pool
from app.startup_validator import validate_startup, is_llm_available
from app.services.schema_inspector import inspect_schema
from app.orchestrator import set_schema_graph
from app.api import chat, auth, admin, dashboard

from rich.logging import RichHandler
from rich.console import Console

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        console=Console(stderr=False),
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        show_path=False,
        markup=True,
    )],
)
# Quieten noisy third-party loggers
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("asyncpg").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def _ensure_model_weights_table(pool, schema: str) -> None:
    """Create mstr_app.model_weights if it doesn't exist.

    Stores serialised numpy weights for the SVD encoder and neural refiner
    so they survive container restarts without a persistent filesystem.
    """
    try:
        async with pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema}.model_weights (
                    model_name   VARCHAR(64)  PRIMARY KEY,
                    weight_data  BYTEA        NOT NULL,
                    schema_hash  VARCHAR(32)  NOT NULL DEFAULT '',
                    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
                )
            """)
        logger.info(f"model_weights table ready ({schema}.model_weights)")
    except Exception as e:
        logger.warning(f"_ensure_model_weights_table: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    settings = get_settings()

    # ── Startup ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("GenAI Dashboard Backend — Starting up")
    logger.info("=" * 60)

    # Validate prerequisites
    await validate_startup()

    # Create database pool
    pool = await create_pool()

    # Run database migrations
    logger.info("Running database migrations...")
    db_dir = os.path.join(os.path.dirname(__file__), "..", "db")
    import time as _mig_time

    # Step 1: Drop legacy HNSW objects (safe, idempotent — runs silently after first cleanup)
    drop_sql_path = os.path.join(db_dir, "drop_hnsw_legacy.sql")
    if os.path.exists(drop_sql_path):
        with open(drop_sql_path, encoding="utf-8") as f:
            drop_sql = f.read().replace("{app_schema}", settings.APP_SCHEMA)
        async with pool.acquire() as conn:
            await conn.execute(drop_sql)

    # Step 2: Create/verify FAISS + PostgreSQL tables
    init_sql_path = os.path.join(db_dir, "init.sql")
    if os.path.exists(init_sql_path):
        with open(init_sql_path, encoding="utf-8") as f:
            sql = f.read().replace("{app_schema}", settings.APP_SCHEMA)
        async with pool.acquire() as conn:
            await conn.execute(sql)
        logger.info("  ✓ Database tables created/verified")

    # Step 3: Run incremental migrations (safe, idempotent)
    for migration_file in ["fix_feedback_table.sql", "migrate_pg_search.sql",
                           "add_feedback_training_columns.sql",
                           "add_ivf_clusters.sql",
                           "add_uploaded_files.sql",
                           "add_user_name_column.sql"]:
        mig_path = os.path.join(db_dir, migration_file)
        if os.path.exists(mig_path):
            with open(mig_path, encoding="utf-8") as f:
                mig_sql = f.read().replace("{app_schema}", settings.APP_SCHEMA)
            t0 = _mig_time.time()
            async with pool.acquire() as conn:
                await conn.execute(mig_sql)
            elapsed = (_mig_time.time() - t0) * 1000
            if elapsed > 500:
                logger.warning(f"  ⚠ Migration {migration_file} took {elapsed:.0f}ms")
            else:
                logger.info(f"  ✓ {migration_file} ({elapsed:.0f}ms)")

    # Schema introspection
    logger.info("Running schema introspection...")
    graph = await inspect_schema()
    set_schema_graph(graph)
    logger.info(f"  ✓ Schema loaded: {len(graph.tables)} tables")

    # Seed vector indexes: check if schema changed since last run.
    # If unchanged and encoder weights exist on disk, skip re-seeding (fast startup).
    # If changed, re-seed and save encoder weights for next restart.
    import hashlib as _hl
    from app.services.schema_seeder import seed_all
    from app.services.sentence_encoder import get_encoder

    # Compute schema fingerprint (tables + columns)
    schema_parts = sorted(
        f"{t}:{','.join(sorted(meta.columns.keys()))}"
        for t, meta in graph.tables.items()
    )
    schema_hash = _hl.sha256("|".join(schema_parts).encode()).hexdigest()[:16]

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    # ── Ensure weight-storage + checksum tables exist ──────────────────────
    from app.services.rag_indexer import ensure_checksums_table
    await ensure_checksums_table()
    await _ensure_model_weights_table(pool, settings.APP_SCHEMA)

    # ── Load encoder weights from DB; skip re-seeding if schema unchanged ──
    needs_reseed = True
    enc = get_encoder()
    try:
        stored_hash = await enc.load_from_db(pool, settings.APP_SCHEMA)
        if stored_hash == schema_hash:
            logger.info(f"  ✓ Encoder loaded from DB (schema hash: {schema_hash})")
            needs_reseed = False
        else:
            logger.info(
                f"  Schema changed ({stored_hash[:8] if stored_hash else 'none'} → "
                f"{schema_hash[:8]}) — will re-seed"
            )
    except Exception as e:
        logger.warning(f"  ⚠ Encoder DB load failed: {e} — will re-seed")

    # ── Always try to load the neural refiner from DB ──────────────────────
    try:
        from app.services.neural_trainer import NeuralRefiner, set_refiner
        refiner = NeuralRefiner(dim=settings.EMBEDDING_DIMENSIONS)
        loaded = await refiner.load_from_db(pool, settings.APP_SCHEMA)
        if loaded:
            set_refiner(refiner)
            logger.info("  ✓ Neural refiner loaded from DB")
    except Exception as e:
        logger.warning(f"  ⚠ Neural refiner DB load failed: {e}")

    if needs_reseed:
        logger.info("Seeding PostgreSQL vector indexes...")
        try:
            await seed_all(graph)
            # Persist encoder + neural refiner to DB
            enc = get_encoder()
            if enc.is_fitted:
                await enc.save_to_db(pool, settings.APP_SCHEMA, schema_hash=schema_hash)
            from app.services.neural_trainer import get_refiner
            trained_refiner = get_refiner()
            if trained_refiner is not None and trained_refiner.is_trained:
                await trained_refiner.save_to_db(pool, settings.APP_SCHEMA)
                logger.info("  ✓ Neural refiner saved to DB")
            logger.info(f"  ✓ Indexes seeded + weights saved to DB (schema hash: {schema_hash})")
        except Exception as e:
            logger.error(f"  ✗ Seeding failed: {e}")
    else:
        logger.info("  ✓ Skipped re-seeding (schema unchanged, weights in DB)")

    # Build PostgreSQL IVF cluster indexes in background (non-blocking)
    async def _build_ivf_background():
        try:
            from app.services.faiss_manager import build_all_ivf_clusters
            logger.info("Building IVF cluster indexes in PostgreSQL (background)...")
            ivf_counts = await build_all_ivf_clusters()
            logger.info(f"  ✓ IVF clusters ready: {ivf_counts}")
        except Exception as e:
            logger.warning(f"  ⚠ IVF cluster build failed: {e}")

    # Run background tasks (non-blocking — server starts immediately)
    import asyncio as _aio

    _aio.create_task(_build_ivf_background())

    # Run embedding evaluation in background (non-blocking)
    async def _run_eval_background():
        try:
            from app.services.embedding_eval import evaluate_retrieval
            eval_report = await evaluate_retrieval(graph, k=5)
            recall = eval_report.get("recall_at_k", 0)
            if recall < 0.5:
                logger.warning(f"  ⚠ Embedding quality low: Recall@5={recall:.1%}")
            else:
                logger.info(f"  ✓ Embedding eval: Recall@5={recall:.1%}, MRR={eval_report.get('mrr', 0):.3f}")
        except Exception as e:
            logger.debug(f"Embedding eval skipped: {e}")

    _aio.create_task(_run_eval_background())

    # Create upload directory
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    # Start schema watcher background task
    from app.services.schema_watcher import start_watcher
    start_watcher()

    logger.info("=" * 60)
    logger.info(f"Server ready at http://localhost:8000")
    logger.info(f"Database: {settings.POSTGRES_SCHEMA} ({len(graph.tables)} tables)")
    if is_llm_available():
        logger.info(f"LLM: {settings.AI_FACTORY_MODEL} via Ollama")
    else:
        logger.warning(f"LLM: UNAVAILABLE — start Ollama and pull {settings.AI_FACTORY_MODEL}")
    logger.info(f"Embeddings: local SentenceEncoder ({settings.EMBEDDING_DIMENSIONS}d, no external model)")
    logger.info(f"Auth: {'enabled' if settings.AUTH_ENABLED else 'disabled (local mode)'}")
    logger.info("=" * 60)

    yield

    # ── Shutdown ───────────────────────────────────────────
    from app.services.schema_watcher import stop_watcher
    stop_watcher()
    await close_pool()
    logger.info("Server shut down cleanly")


# Create app
app = FastAPI(
    title="GenAI Dashboard Backend",
    description="Text-to-SQL Chat Agent with RAG File Analysis",
    version="2.0.0",
    lifespan=lifespan,
)

# Request logging middleware (Issue #11)
import time as _time

@app.middleware("http")
async def log_requests(request, call_next):
    start = _time.time()
    response = await call_next(request)
    elapsed = (_time.time() - start) * 1000
    if not request.url.path.startswith("/docs") and not request.url.path.startswith("/openapi"):
        logger.info(f"{request.method} {request.url.path} {response.status_code} {elapsed:.0f}ms")
    return response

# CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(chat.router)
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(dashboard.router)


@app.get("/")
async def root():
    return {"message": "GenAI Dashboard Backend", "version": "2.0.0", "docs": "/docs"}
