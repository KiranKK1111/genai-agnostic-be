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
from app.api import chat, auth, admin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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
    for migration_file in ["fix_feedback_table.sql"]:
        mig_path = os.path.join(db_dir, migration_file)
        if os.path.exists(mig_path):
            with open(mig_path, encoding="utf-8") as f:
                mig_sql = f.read().replace("{app_schema}", settings.APP_SCHEMA)
            async with pool.acquire() as conn:
                await conn.execute(mig_sql)

    # Schema introspection
    logger.info("Running schema introspection...")
    graph = await inspect_schema()
    set_schema_graph(graph)
    logger.info(f"  ✓ Schema loaded: {len(graph.tables)} tables")

    # Seed FAISS + BM25 indexes on every startup.
    # The TF-IDF+SVD encoder is recomputed from scratch each time (no persisted weights),
    # so stored vectors from a previous run would be in a different embedding space.
    # Re-seeding ensures vectors and encoder are always in sync.
    logger.info("Seeding FAISS + BM25 vector indexes...")
    from app.services.faiss_manager import load_all_indexes, get_faiss_index, clear_index
    from app.services.schema_seeder import seed_all
    try:
        await seed_all(graph)
        logger.info("  ✓ FAISS indexes seeded (schema_idx + values_idx)")
    except Exception as e:
        logger.error(f"  ✗ Seeding failed: {e}")
        # Fallback: try to load stale vectors (better than nothing)
        try:
            await load_all_indexes()
            logger.warning("  ⚠ Loaded stale vectors as fallback")
        except Exception:
            pass

    # Run embedding evaluation (Stage 27) after seeding/loading
    from app.services.embedding_eval import evaluate_retrieval
    try:
        eval_report = await evaluate_retrieval(graph, k=5)
        recall = eval_report.get("recall_at_k", 0)
        if recall < 0.5:
            logger.warning(f"  ⚠ Embedding quality low: Recall@5={recall:.1%}")
        else:
            logger.info(f"  ✓ Embedding eval: Recall@5={recall:.1%}, MRR={eval_report.get('mrr', 0):.3f}")
    except Exception as e:
        logger.debug(f"Embedding eval skipped: {e}")

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


@app.get("/")
async def root():
    return {"message": "GenAI Dashboard Backend", "version": "2.0.0", "docs": "/docs"}
