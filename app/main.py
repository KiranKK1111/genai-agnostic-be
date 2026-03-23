"""FastAPI application factory — startup, CORS, routers."""
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.database import create_pool, close_pool, get_pool
from app.startup_validator import validate_startup
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

    # Load existing FAISS indexes from PostgreSQL (persists across restarts)
    logger.info("Loading FAISS indexes from PostgreSQL...")
    from app.services.faiss_manager import load_all_indexes, get_faiss_index
    await load_all_indexes()

    # Only seed if schema_idx is empty (first run or after manual reset)
    schema_idx = await get_faiss_index("schema_idx")
    if schema_idx.count == 0:
        logger.info("Seeding FAISS vector indexes (first run)...")
        from app.services.schema_seeder import seed_all
        await seed_all(graph)
        logger.info("  ✓ FAISS indexes seeded (schema_idx + values_idx)")
    else:
        logger.info(f"  ✓ FAISS indexes already populated (schema_idx={schema_idx.count}, skipping seed)")

    # Create upload directory
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    # Create default user if AUTH_ENABLED and no users exist
    if settings.AUTH_ENABLED:
        async with pool.acquire() as conn:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {settings.APP_SCHEMA}.users")
            if count == 0:
                import bcrypt as _bcrypt
                hashed = _bcrypt.hashpw("admin".encode(), _bcrypt.gensalt()).decode()
                await conn.execute(
                    f"""INSERT INTO {settings.APP_SCHEMA}.users (username, email, hashed_password, role)
                        VALUES ('admin', 'admin@local', $1, 'admin')""",
                    hashed
                )
                logger.info("  ✓ Default admin user created (username: admin, password: admin)")

    # Start schema watcher background task
    from app.services.schema_watcher import start_watcher
    start_watcher()

    logger.info("=" * 60)
    logger.info(f"Server ready at http://localhost:8000")
    logger.info(f"Database: {settings.POSTGRES_SCHEMA} ({len(graph.tables)} tables)")
    logger.info(f"LLM: {settings.AI_FACTORY_MODEL} via Ollama")
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
