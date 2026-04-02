"""Admin endpoints — health, reseed, cleanup, audit."""
from fastapi import APIRouter, Depends
from app.database import health_check, get_pool
from app.config import get_settings
from app.auth import get_current_user, require_admin, User

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/health")
async def health():
    """Health check — no auth required (used by load balancers)."""
    from app.startup_validator import is_llm_available
    db_ok = await health_check()
    llm_ok = is_llm_available()
    # Server is healthy if DB is up; LLM being down = degraded, not unhealthy
    status = "healthy" if db_ok else "unhealthy"
    if db_ok and not llm_ok:
        status = "degraded"
    return {"status": status, "database": db_ok, "llm": llm_ok}


@router.post("/reseed-schema")
async def reseed_schema(user: User = Depends(require_admin)):
    """Re-run schema introspection and rebuild indexes. Admin only."""
    from app.services.schema_inspector import inspect_schema
    from app.services.schema_seeder import seed_all
    from app.orchestrator import set_schema_graph
    graph = await inspect_schema()
    set_schema_graph(graph)
    await seed_all(graph)
    return {"status": "reseeded", "tables": len(graph.tables)}


@router.get("/embedding-eval")
async def embedding_eval(user: User = Depends(require_admin)):
    """Run embedding quality evaluation (Stage 27). Admin only."""
    from app.services.embedding_eval import evaluate_retrieval
    from app.orchestrator import get_schema_graph as _schema_graph_fn
    _schema_graph = _schema_graph_fn()
    if _schema_graph is None:
        return {"error": "Schema not loaded yet"}
    report = await evaluate_retrieval(_schema_graph, k=5)
    return report


@router.get("/drift-check")
async def drift_check(user: User = Depends(require_admin)):
    """Run embedding drift monitoring (Stage 30). Admin only."""
    from app.services.drift_monitor import run_full_drift_check
    from app.orchestrator import get_schema_graph as _schema_graph_fn
    _schema_graph = _schema_graph_fn()
    if _schema_graph is None:
        return {"error": "Schema not loaded yet"}
    report = await run_full_drift_check(_schema_graph)
    return report


@router.post("/cleanup")
async def cleanup(user: User = Depends(require_admin)):
    """Purge expired KV entries. Admin only."""
    from app.services.kv_store import KVStore
    kv = KVStore()
    await kv.cleanup_expired()
    return {"status": "cleaned"}


@router.get("/audit-log")
async def get_audit_log(limit: int = None, user: User = Depends(require_admin)):
    """View audit log. Admin only."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT * FROM {schema}.audit_log ORDER BY created_at DESC LIMIT $1",
            limit or settings.AUDIT_LOG_LIMIT
        )
        return [dict(r) for r in rows]


@router.get("/feedback/summary")
async def feedback_summary(user: User = Depends(require_admin)):
    """Aggregate user feedback scores. Admin only."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        total = await conn.fetchrow(
            f"""SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE reaction='like') as likes,
                    COUNT(*) FILTER (WHERE reaction='dislike') as dislikes
                FROM {schema}.chat_messages WHERE reaction IS NOT NULL"""
        )
        recent = await conn.fetch(
            f"""SELECT reaction, reaction_comment, created_at
                FROM {schema}.chat_messages WHERE reaction IS NOT NULL
                ORDER BY created_at DESC LIMIT 20"""
        )
        return {
            "total_reactions": total["total"] if total else 0,
            "likes": total["likes"] if total else 0,
            "dislikes": total["dislikes"] if total else 0,
            "satisfaction_rate": round(total["likes"] / max(total["total"], 1) * 100, 1) if total else 0,
            "recent": [dict(r) for r in recent],
        }
