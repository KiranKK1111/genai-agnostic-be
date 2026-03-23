"""Admin endpoints — health, reseed, cleanup, audit."""
from fastapi import APIRouter, Depends
from app.database import health_check, get_pool
from app.config import get_settings
from app.auth import get_current_user, require_admin, User

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/health")
async def health():
    """Health check — no auth required (used by load balancers)."""
    db_ok = await health_check()
    return {"status": "healthy" if db_ok else "unhealthy", "database": db_ok}


@router.post("/reseed-schema")
async def reseed_schema(user: User = Depends(require_admin)):
    """Re-run schema introspection and rebuild indexes. Admin only."""
    from app.services.schema_inspector import inspect_schema
    from app.orchestrator import set_schema_graph
    graph = await inspect_schema()
    set_schema_graph(graph)
    return {"status": "reseeded", "tables": len(graph.tables)}


@router.post("/cleanup")
async def cleanup(user: User = Depends(require_admin)):
    """Purge expired KV entries. Admin only."""
    from app.services.kv_store import KVStore
    kv = KVStore()
    await kv.cleanup_expired()
    return {"status": "cleaned"}


@router.get("/audit-log")
async def get_audit_log(limit: int = 50, user: User = Depends(require_admin)):
    """View audit log. Admin only."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT * FROM {schema}.audit_log ORDER BY created_at DESC LIMIT $1", limit
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
