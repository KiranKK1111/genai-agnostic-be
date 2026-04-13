"""Chat API endpoints — SSE streaming, file upload, clarify, sessions, message actions."""
import json
import os
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from app.orchestrator import process_message
from app.services.session import SessionManager
from app.database import get_pool
from app.config import get_settings
from app.builders.response import sse_event
from app.auth import get_current_user, User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ClarifyRequest(BaseModel):
    session_id: str
    clarification_type: str
    selected_values: str | dict | list


class ReactRequest(BaseModel):
    reaction: Optional[str] = None  # "like", "dislike", or null to clear
    comment: Optional[str] = None


class EditRequest(BaseModel):
    new_content: str


# ── Main Chat SSE Endpoint ─────────────────────────────────
@router.post("")
async def chat_stream(req: ChatRequest, user: User = Depends(get_current_user)):
    """Main SSE streaming endpoint with markdown-aware buffering."""
    import asyncio
    from app.services.stream_buffer import StreamBuffer

    async def event_generator():
        buffer = StreamBuffer()

        async def produce():
            try:
                async for event in process_message(
                    message=req.message,
                    session_id=req.session_id,
                    user_id=user.id,
                    user_name=(user.name or user.username),
                ):
                    event_type = event.pop("type", "data")
                    await buffer.put(event_type, event)
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                logger.info("Client disconnected during stream")
            except Exception as e:
                logger.error(f"Stream error: {e}", exc_info=True)
                await buffer.put("error", {"code": "E099", "message": str(e), "recoverable": False})
                await buffer.put("done", {"message_id": ""})
            finally:
                await buffer.close()

        # Run the producer concurrently so consumed events can stream to the
        # client in real time — otherwise step/progress events arrive in a
        # single burst after the whole pipeline finishes.
        producer_task = asyncio.create_task(produce())
        try:
            async for sse_str in buffer.stream():
                yield sse_str
        finally:
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except (asyncio.CancelledError, Exception):
                    pass

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── File Upload ────────────────────────────────────────────
@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    message: str = Form(""),
    session_id: str = Form(None),
    user: User = Depends(get_current_user),
):
    """Upload a file and persist it in PostgreSQL, then stream analysis."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    ext = os.path.splitext(file.filename)[1].lower().lstrip(".")
    if ext not in settings.allowed_file_types_list:
        raise HTTPException(400, f"Unsupported file type: .{ext}. Allowed: {settings.ALLOWED_FILE_TYPES}")

    content = await file.read()
    mime_type = file.content_type or f"application/{ext}"

    # Persist file bytes in PostgreSQL — session_id may be NULL for brand-new sessions
    async with pool.acquire() as conn:
        file_id = await conn.fetchval(
            f"""INSERT INTO {schema}.uploaded_files
                    (session_id, user_id, file_name, file_size, mime_type, file_data)
                VALUES ($1::uuid, $2, $3, $4, $5, $6)
                RETURNING id""",
            session_id,  # None is fine — file_pipeline will update it after session is known
            user.id,
            file.filename,
            len(content),
            mime_type,
            content,
        )
    file_id_str = str(file_id)

    async def event_generator():
        try:
            async for event in process_message(
                message=message or f"Analyze this file: {file.filename}",
                session_id=session_id,
                user_id=user.id,
                file_id=file_id_str,
                file_name=file.filename,
                file_size=len(content),
                file_mime=mime_type,
                user_name=(user.name or user.username),
            ):
                event_type = event.pop("type", "data")
                yield sse_event(event_type, event)
        except Exception as e:
            logger.error(f"Upload stream error: {e}", exc_info=True)
            yield sse_event("error", {"code": "E099", "message": str(e), "recoverable": False})
            yield sse_event("done", {"message_id": ""})

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache"})


# ── File Download ──────────────────────────────────────────
@router.get("/files/{file_id}")
async def download_file(file_id: str, user: User = Depends(get_current_user)):
    """Download a previously uploaded file by its UUID."""
    from fastapi.responses import Response as FastAPIResponse
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""SELECT file_name, mime_type, file_data, user_id
                FROM {schema}.uploaded_files WHERE id=$1::uuid""",
            file_id,
        )
    if not row:
        raise HTTPException(404, "File not found")
    if row["user_id"] and row["user_id"] != user.id and user.role != "admin":
        raise HTTPException(403, "Access denied")

    return FastAPIResponse(
        content=bytes(row["file_data"]),
        media_type=row["mime_type"] or "application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{row["file_name"]}"'},
    )


# ── Session File List ──────────────────────────────────────
@router.get("/session/{session_id}/files")
async def list_session_files(session_id: str, user: User = Depends(get_current_user)):
    """List all files uploaded in a session (metadata only, no bytes)."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    async with pool.acquire() as conn:
        # Verify session ownership
        owner = await conn.fetchval(
            f"SELECT user_id FROM {schema}.chat_sessions WHERE id=$1::uuid", session_id
        )
        if owner is not None and owner != user.id and user.role != "admin":
            raise HTTPException(403, "Access denied")

        rows = await conn.fetch(
            f"""SELECT id, file_name, file_size, mime_type, uploaded_at
                FROM {schema}.uploaded_files
                WHERE session_id=$1::uuid
                ORDER BY uploaded_at ASC""",
            session_id,
        )
    return [dict(r) for r in rows]


# ── Clarification Response ─────────────────────────────────
@router.post("/clarify")
async def clarify(req: ClarifyRequest, user: User = Depends(get_current_user)):
    """Submit user's clarification response. Stores selection for pipeline re-run."""
    session_mgr = SessionManager()
    state = await session_mgr.get_or_create(req.session_id)
    if state.get("user_id") and state["user_id"] != user.id:
        raise HTTPException(403, "Access denied.")

    # Handle dismiss — user closed the clarification popup
    if req.clarification_type == "__dismissed":
        state["clarification_pending"] = None
        state["clarification_response"] = None
        state["clarification_display"] = None
        state["clarification_history"] = []
        await session_mgr.save(state)
        return {"status": "dismissed", "session_id": req.session_id}

    # Store the user's selection so CLARIFICATION_REPLY handler can use it
    state["clarification_response"] = {
        "type": req.clarification_type,
        "selected_values": req.selected_values,
    }

    # Accumulate Q&A pair in clarification_history so all pairs can be
    # displayed together after the user finishes answering all questions.
    pending = state.get("clarification_pending") or {}
    display = state.get("clarification_display") or {}
    question_text = display.get("question") or pending.get("question", "")
    # Derive a human-readable answer label
    selected = req.selected_values
    if selected == "__default":
        answer_text = "Skipped (using default)"
    elif isinstance(selected, list):
        answer_text = ", ".join(str(v) for v in selected)
    elif isinstance(selected, dict):
        answer_text = str(selected)
    else:
        answer_text = str(selected)
    # Try to resolve option labels for display
    options = display.get("options") or pending.get("options") or []
    if options and isinstance(selected, str):
        for opt in options:
            if opt.get("value") == selected:
                answer_text = opt.get("label", selected)
                break

    history = state.get("clarification_history", [])
    history.append({"question": question_text, "answer": answer_text, "type": req.clarification_type})
    state["clarification_history"] = history

    # Keep clarification_pending so intent classifier routes to CLARIFICATION_REPLY
    # (it gets cleared after the pipeline re-runs in orchestrator)

    await session_mgr.save(state)
    return {"status": "ok", "session_id": req.session_id}


# ── Cancel Generation ──────────────────────────────────────
@router.post("/cancel/{session_id}")
async def cancel_generation(session_id: str, user: User = Depends(get_current_user)):
    session_mgr = SessionManager()
    state = await session_mgr.get_or_create(session_id)
    if state.get("user_id") and state["user_id"] != user.id:
        raise HTTPException(403, "Access denied.")
    state["cancel_requested"] = True
    await session_mgr.save(state)
    return {"status": "cancelled"}


# ── Session CRUD ───────────────────────────────────────────
@router.get("/sessions")
async def list_sessions(user: User = Depends(get_current_user)):
    """List sessions for the authenticated user only."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""SELECT id, title, created_at, updated_at,
                       (SELECT COUNT(*) FROM {schema}.chat_messages WHERE session_id=s.id AND is_active=true) as message_count
                FROM {schema}.chat_sessions s
                WHERE s.user_id = $1
                ORDER BY updated_at DESC LIMIT $2""",
            user.id, settings.CHAT_HISTORY_LIMIT
        )
        return [dict(r) for r in rows]


@router.get("/session/{session_id}")
async def get_session(session_id: str, user: User = Depends(get_current_user)):
    """Get session messages — only if owned by authenticated user."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    # Verify ownership
    async with pool.acquire() as conn:
        owner = await conn.fetchval(
            f"SELECT user_id FROM {schema}.chat_sessions WHERE id=$1::uuid", session_id
        )
        if owner is not None and owner != user.id:
            raise HTTPException(403, "Access denied")
    session_mgr = SessionManager()
    messages = await session_mgr.get_session_messages(session_id)
    # Include pending clarification so the frontend can restore the popup on reload
    state = await session_mgr.kv.get(f"session:{session_id}")
    pending_clar = None
    if state and state.get("clarification_pending") and state.get("clarification_display"):
        pending_clar = state["clarification_display"]
    return {"session_id": session_id, "messages": messages, "pending_clarification": pending_clar}


@router.delete("/session/{session_id}")
async def delete_session(session_id: str, user: User = Depends(get_current_user)):
    """Delete session — only if owned by authenticated user."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        # Only delete if owned by user (or user is admin)
        result = await conn.execute(
            f"DELETE FROM {schema}.chat_sessions WHERE id=$1::uuid AND (user_id=$2 OR $3)",
            session_id, user.id, user.role == "admin"
        )
    # Clean up all session data: chunks, embeddings, uploaded files
    async with pool.acquire() as conn:
        await conn.execute(
            f"""DELETE FROM {schema}.embedding_metadata
                WHERE index_name = 'chunks_idx'
                  AND payload->>'session_id' = $1""",
            session_id
        )
        await conn.execute(
            f"DELETE FROM {schema}.file_chunks WHERE session_id = $1::uuid",
            session_id
        )
        await conn.execute(
            f"DELETE FROM {schema}.uploaded_files WHERE session_id = $1::uuid",
            session_id
        )
    from app.services.kv_store import KVStore
    kv = KVStore()
    await kv.delete(f"session:{session_id}")
    return {"status": "deleted"}


@router.put("/session/{session_id}/rename")
async def rename_session(session_id: str, title: str = Query(...), user: User = Depends(get_current_user)):
    """Rename session — only if owned by authenticated user."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE {schema}.chat_sessions SET title=$1 WHERE id=$2::uuid AND (user_id=$3 OR $4)",
            title, session_id, user.id, user.role == "admin"
        )
    return {"status": "renamed", "title": title}


# ── Message ownership helper ───────────────────────────────
async def _assert_message_owner(message_id: str, user: User, allow_admin: bool = True):
    """Raise 404 if message not found; 403 if the session owner != user."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        owner = await conn.fetchval(
            f"""SELECT cs.user_id FROM {schema}.chat_messages cm
                JOIN {schema}.chat_sessions cs ON cs.id = cm.session_id
                WHERE cm.id=$1::uuid""",
            message_id
        )
    if owner is None:
        raise HTTPException(404, "Message not found.")
    if owner != user.id and not (allow_admin and user.role == "admin"):
        raise HTTPException(403, "Access denied.")


# ── Message Actions ────────────────────────────────────────
@router.post("/message/{message_id}/react")
async def react_to_message_endpoint(message_id: str, req: ReactRequest, user: User = Depends(get_current_user)):
    await _assert_message_owner(message_id, user)
    from app.services.message_actions import react_to_message
    try:
        result = await react_to_message(message_id, req.reaction, req.comment, user_id=user.id)
        return result
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/message/{message_id}/retry")
async def retry_message(message_id: str, user: User = Depends(get_current_user)):
    """Regenerate an assistant response."""
    await _assert_message_owner(message_id, user)
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    async with pool.acquire() as conn:
        msg = await conn.fetchrow(f"SELECT * FROM {schema}.chat_messages WHERE id=$1::uuid", message_id)
        if not msg:
            raise HTTPException(404, "Message not found")
        if msg["role"] != "assistant":
            raise HTTPException(400, "Can only retry assistant messages")

        # Find parent user message
        parent_msg = await conn.fetchrow(
            f"SELECT * FROM {schema}.chat_messages WHERE id=$1::uuid", str(msg["parent_message_id"])
        )
        if not parent_msg:
            raise HTTPException(400, "Parent message not found")

        # Deactivate old response
        await conn.execute(f"UPDATE {schema}.chat_messages SET is_active=false WHERE id=$1::uuid", message_id)

    # Restore session context from parent message snapshot (Issue #14)
    if parent_msg.get("session_snapshot"):
        try:
            import json as _json
            snapshot = _json.loads(parent_msg["session_snapshot"]) if isinstance(parent_msg["session_snapshot"], str) else parent_msg["session_snapshot"]
            from app.services.session import SessionManager
            sm = SessionManager()
            state = await sm.get_or_create(str(msg["session_id"]), user.id)
            for key in (
                "last_sql", "last_plan", "last_data", "last_columns", "last_table",
                "last_intent", "intent_chain", "file_context",
                "history", "history_summary", "total_turns",
                "clarification_pending",
            ):
                if key in snapshot:
                    state[key] = snapshot[key]
            # Always reset transient flags — never restore stale in-flight state
            state["viz_suggestion_pending"] = False
            state["cancel_requested"] = False
            await sm.save(state)
        except Exception:
            pass  # Non-critical — session will still work without snapshot

    # Re-process
    async def event_generator():
        async for event in process_message(
            message=parent_msg["content"],
            session_id=str(msg["session_id"]),
            user_id=user.id,
            user_name=(user.name or user.username),
        ):
            event_type = event.pop("type", "data")
            yield sse_event(event_type, event)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Search ─────────────────────────────────────────────────
@router.get("/search")
async def search_messages(q: str = Query(..., min_length=1), limit: int = None, user: User = Depends(get_current_user)):
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""SELECT m.id, m.session_id, m.content, m.role, m.created_at,
                       s.title as session_title
                FROM {schema}.chat_messages m
                JOIN {schema}.chat_sessions s ON m.session_id = s.id
                WHERE m.content ILIKE $1 AND m.is_active=true
                  AND s.user_id = $2
                ORDER BY m.created_at DESC LIMIT $3""",
            f"%{q}%", user.id, limit or settings.DEFAULT_SEARCH_LIMIT
        )
        return [dict(r) for r in rows]

# ── Autocomplete / Suggest ─────────────────────────────────
@router.get("/suggest")
async def suggest(q: str = Query(..., min_length=1), user: User = Depends(get_current_user)):
    from app.services.autocomplete import get_suggestions
    return await get_suggestions(q, user_id=user.id)


# ── Export ─────────────────────────────────────────────────
class ExportRequest(BaseModel):
    session_id: str
    format: str = "csv"  # csv, xlsx, json

@router.post("/export")
async def export_data(req: ExportRequest, user: User = Depends(get_current_user)):
    """Export the last query result from a session."""
    from app.services.data_exporter import export_data as do_export
    from fastapi.responses import FileResponse
    _ALLOWED_FORMATS = {"csv", "json", "xlsx", "pdf"}
    if req.format not in _ALLOWED_FORMATS:
        raise HTTPException(400, f"Unsupported format '{req.format}'. Allowed: {', '.join(sorted(_ALLOWED_FORMATS))}")
    session_mgr = SessionManager()
    state = await session_mgr.get_or_create(req.session_id)
    if state.get("user_id") and state["user_id"] != user.id:
        raise HTTPException(403, "Access denied.")
    data = state.get("last_data")
    columns = state.get("last_columns")
    if not data or not columns:
        raise HTTPException(400, "No data to export. Run a query first.")
    path = await do_export(data, columns, format=req.format)
    return FileResponse(path, filename=os.path.basename(path),
                        media_type="application/octet-stream")


# ── Message Edit ───────────────────────────────────────────
@router.put("/message/{message_id}/edit")
async def edit_message(message_id: str, req: EditRequest, user: User = Depends(get_current_user)):
    from app.services.message_actions import edit_user_message
    # Verify message belongs to this user
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        owner = await conn.fetchval(
            f"""SELECT cs.user_id FROM {schema}.chat_messages cm
                JOIN {schema}.chat_sessions cs ON cs.id = cm.session_id
                WHERE cm.id=$1::uuid""",
            message_id
        )
    if owner is not None and owner != user.id and user.role != "admin":
        raise HTTPException(403, "Access denied.")
    try:
        result = await edit_user_message(message_id, req.new_content)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


# ── Message Resend ─────────────────────────────────────────
@router.post("/message/{message_id}/resend")
async def resend_message(message_id: str, user: User = Depends(get_current_user)):
    """Re-send the same user prompt to get a new response."""
    await _assert_message_owner(message_id, user)
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        msg = await conn.fetchrow(
            f"SELECT * FROM {schema}.chat_messages WHERE id=$1::uuid", message_id
        )
        if not msg:
            raise HTTPException(404, "Message not found")
        if msg["role"] != "user":
            raise HTTPException(400, "Can only resend user messages")

    async def event_generator():
        async for event in process_message(
            message=msg["content"],
            session_id=str(msg["session_id"]),
            user_id=user.id,
            user_name=(user.name or user.username),
        ):
            event_type = event.pop("type", "data")
            yield sse_event(event_type, event)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Message Content (for copy) ─────────────────────────────
@router.get("/message/{message_id}/content")
async def get_message_content(message_id: str, format: str = Query("raw"), user: User = Depends(get_current_user)):
    await _assert_message_owner(message_id, user)
    from app.services.message_actions import get_message_content as get_content
    try:
        content = await get_content(message_id, format=format)
        return {"content": content, "format": format}
    except ValueError as e:
        raise HTTPException(404, str(e))


# ── Message Versions ───────────────────────────────────────
@router.get("/message/{message_id}/versions")
async def get_versions(message_id: str, user: User = Depends(get_current_user)):
    await _assert_message_owner(message_id, user)
    from app.services.version_manager import get_message_versions
    versions = await get_message_versions(message_id)
    return {"versions": versions, "total": len(versions)}


# ── Saved Queries CRUD ─────────────────────────────────────
class SaveQueryRequest(BaseModel):
    name: str
    prompt: str
    sql: Optional[str] = None
    viz_config: Optional[dict] = None
    tags: Optional[list[str]] = None

@router.post("/queries/save")
async def save_query(req: SaveQueryRequest, user: User = Depends(get_current_user)):
    from app.services.saved_queries import save_query as do_save
    query_id = await do_save(
        user_id=user.id, name=req.name, prompt=req.prompt,
        sql=req.sql, viz_config=req.viz_config, tags=req.tags
    )
    return {"id": query_id, "status": "saved"}

@router.get("/queries")
async def list_queries(user: User = Depends(get_current_user)):
    from app.services.saved_queries import list_saved_queries
    return await list_saved_queries(user_id=user.id)

@router.post("/queries/{query_id}/run")
async def run_saved_query(query_id: str, user: User = Depends(get_current_user)):
    """Re-execute a saved query."""
    from app.services.saved_queries import get_saved_query, increment_run_count
    query = await get_saved_query(query_id, user_id=user.id)
    if not query:
        raise HTTPException(404, "Saved query not found")

    await increment_run_count(query_id)

    # Re-process the original prompt through the full pipeline
    async def event_generator():
        async for event in process_message(
            message=query["original_prompt"],
            user_id=user.id,
            user_name=(user.name or user.username),
        ):
            event_type = event.pop("type", "data")
            yield sse_event(event_type, event)

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache"})


@router.delete("/queries/{query_id}")
async def delete_query(query_id: str, user: User = Depends(get_current_user)):
    from app.services.saved_queries import delete_saved_query
    await delete_saved_query(query_id, user_id=user.id)
    return {"status": "deleted"}
