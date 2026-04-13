"""File analysis pipeline — load from DB, chunk, embed, RAG retrieve, synthesize."""
import logging
from app.services.embedder import embed_single
from app.services.vector_search import search
from app.services.llm_client import chat_stream
from app.services.response_beautifier import beautify
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)


async def execute_file_upload(file_id: str, file_name: str, message: str,
                               session_state: dict, user_name: str = "User"):
    """Process an uploaded file (already stored in DB) + optional query.

    Loads raw bytes from {schema}.uploaded_files, indexes them into chunks_idx,
    then generates an initial response. Yields SSE events.
    """
    import time as _t
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    _pipeline_start = _t.time()
    logger.info(f"[file_pipeline] START file_id={file_id[:8]}… file_name={file_name}")
    yield {"type": "step", "step_number": 1, "label": "Receiving file"}

    # ── Load bytes from PostgreSQL ────────────────────────────
    _t0 = _t.time()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT file_data, file_name FROM {schema}.uploaded_files WHERE id=$1::uuid",
            file_id,
        )
    if not row:
        logger.error(f"[file_pipeline] file not found in DB: {file_id}")
        yield {"type": "error", "code": "E099",
               "message": f"Uploaded file not found (id={file_id})", "recoverable": False}
        return

    file_bytes: bytes = bytes(row["file_data"])
    file_name = file_name or row["file_name"]
    logger.info(
        f"[file_pipeline] loaded bytes from DB: {len(file_bytes)} bytes "
        f"in {(_t.time() - _t0) * 1000:.0f}ms"
    )

    # ── Update session_id on the uploaded_files row ───────────
    # The session may have been created after the file was stored (new session).
    async with pool.acquire() as conn:
        await conn.execute(
            f"""UPDATE {schema}.uploaded_files
                SET session_id = $1::uuid
                WHERE id = $2::uuid AND session_id IS NULL""",
            session_state["session_id"], file_id,
        )

    yield {"type": "step", "step_number": 2, "label": "Extracting content"}
    yield {"type": "step", "step_number": 3, "label": "Splitting into chunks"}
    yield {"type": "step", "step_number": 4, "label": "Creating embeddings"}

    # ── RAGIndexer: parse + chunk + embed + store ─────────────
    logger.info(f"[file_pipeline] starting index_file for '{file_name}'")
    _t0 = _t.time()
    from app.services.rag_indexer import get_indexer
    indexer = get_indexer()
    _stats, chunk_records = await indexer.index_file(
        file_bytes, file_name, session_state["session_id"]
    )
    logger.info(
        f"[file_pipeline] index_file done: {len(chunk_records)} chunks "
        f"in {(_t.time() - _t0) * 1000:.0f}ms"
    )

    # ── Persist chunk records in file_chunks table (single batch insert) ──
    if chunk_records:
        _t0 = _t.time()
        async with pool.acquire() as conn:
            await conn.executemany(
                f"""INSERT INTO {schema}.file_chunks
                    (session_id, file_name, chunk_index, chunk_text, node_id)
                    VALUES ($1::uuid, $2, $3, $4, $5)""",
                [
                    (
                        session_state["session_id"],
                        file_name,
                        rec["chunk_index"],
                        rec["chunk_text"],
                        str(rec["node_id"]),
                    )
                    for rec in chunk_records
                ],
            )
        logger.info(
            f"[file_pipeline] persisted {len(chunk_records)} file_chunks rows "
            f"in {(_t.time() - _t0) * 1000:.0f}ms"
        )

    # ── Update session state ──────────────────────────────────
    file_ctx = {
        "file_id":     file_id,
        "file_name":   file_name,
        "chunk_count": _stats.added,
        "session_id":  session_state["session_id"],
    }
    session_state["file_context"] = file_ctx

    file_history = session_state.get("file_history", [])
    file_history.append({
        **file_ctx,
        "upload_message": message or f"Uploaded {file_name}",
    })
    session_state["file_history"] = file_history

    yield {"type": "step", "step_number": 5, "label": "Analyzing content"}

    # ── Generate initial response ─────────────────────────────
    query   = message or f"Summarize the uploaded file {file_name}"
    context = "\n\n".join(r["chunk_text"][:500] for r in chunk_records[:5])

    full_text = ""
    async for token in chat_stream([{"role": "user", "content": (
        f"File: {file_name}\nContent:\n{context}\n\n"
        f"User message: {query}\n\n"
        "Respond naturally to the user's message first, then provide a helpful analysis of the file. "
        "Use markdown formatting. Use **bold** for key terms and numbers. "
        "Do NOT use pipe-symbol tables (| col | col |) — they render as broken symbols. "
        "Do NOT start with any heading label like 'Summary:' or 'Analysis:'. "
        "MANDATORY: End with a follow-up question suggesting ACTIONABLE next steps the user can perform in this tool. The tool supports: asking follow-up questions about the file, summarizing specific sections, comparing file data with the database, and exporting. Reference specific content from the file. Do NOT ask hypothetical/analytical questions — only suggest things the tool can actually do. Never skip this."
    )}]):
        full_text += token
        yield {"type": "text_delta", "delta": token}

    full_text = beautify(full_text)
    if "FOLLOW_UPS:" in full_text:
        full_text = full_text.split("FOLLOW_UPS:")[0].strip()

    yield {"type": "text_done", "content": full_text}
    # Emit the file_id so the UI can offer a download button
    yield {"type": "file_stored", "file_id": file_id, "file_name": file_name}

    # NOTE: session title is set by the orchestrator from the user's first
    # prompt — we no longer override it with an LLM summary here.


async def _select_base_file(message: str, file_history: list[dict]) -> dict:
    """Given a follow-up message and file history, pick which file the user is asking about."""
    if len(file_history) == 1:
        return file_history[0]
    if not message:
        return file_history[-1]

    from app.services.llm_client import chat_json

    history_lines = []
    for i, entry in enumerate(file_history):
        fname  = entry.get("file_name", "unknown")
        msg    = entry.get("upload_message", "")
        chunks = entry.get("chunk_count", 0)
        history_lines.append(f"  F{i+1}. \"{fname}\" ({chunks} chunks) — uploaded with: \"{msg}\"")

    try:
        result = await chat_json([{"role": "user", "content":
            f'The user has uploaded multiple files in this session:\n'
            + "\n".join(history_lines)
            + f'\n\nThe user now asks: "{message}"\n\n'
            f'Which file is the user most likely asking about?\n'
            f'If unclear, pick the most recently uploaded file.\n\n'
            f'Return JSON: {{"file_number": <1-based index>, "reason": "brief explanation"}}'}])
        chosen = result.get("file_number", len(file_history))
        idx    = max(0, min(chosen - 1, len(file_history) - 1))
        logger.info(
            f"File selection: chose F{idx+1} ({file_history[idx].get('file_name')}) "
            f"— {result.get('reason', '')}"
        )
    except Exception as e:
        logger.warning(f"File selection failed: {e}, using last file")
        idx = len(file_history) - 1

    return file_history[idx]


async def execute_file_followup(message: str, session_state: dict):
    """Answer a follow-up question about an uploaded file using RAG."""
    settings = get_settings()
    pool     = get_pool()
    schema   = settings.APP_SCHEMA

    yield {"type": "step", "step_number": 1, "label": "Searching file content"}

    file_history = session_state.get("file_history", [])
    if len(file_history) > 1:
        file_ctx = await _select_base_file(message, file_history)
    else:
        file_ctx = session_state.get("file_context", {})

    target_file = file_ctx.get("file_name", "")
    session_state["file_context"] = file_ctx

    query_emb = await embed_single(message)
    results   = await search(
        "chunks_idx", query_emb, k=5,
        filter_key="session_id",
        filter_value=session_state["session_id"],
        query_text=message,
    )

    if target_file and len(file_history) > 1:
        file_results = [r for r in results if r.get("payload", {}).get("file_name") == target_file]
        if file_results:
            results = file_results

    chunks = []
    for r in results:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT chunk_text FROM {schema}.file_chunks WHERE node_id=$1",
                r["node_id"],
            )
            if row:
                chunks.append(row["chunk_text"])

    if not chunks:
        yield {"type": "text_done",
               "content": "I couldn't find relevant content in the uploaded file for your question. Could you rephrase?"}
        return

    yield {"type": "step", "step_number": 2, "label": "Generating response"}

    context   = "\n\n".join(chunks)
    full_text = ""
    async for token in chat_stream([{"role": "user", "content":
        f"File: {target_file or 'unknown'}\nRelevant content:\n{context}\n\n"
        f"Question: {message}\n\n"
        "Answer based on the file content. Use markdown with **bold** for key terms. "
        "Do NOT use pipe-symbol tables (| col | col |) — they render as broken symbols. "
        "Do NOT start with any heading label like 'Summary:' or 'Answer:'. "
        "MANDATORY: End with a follow-up question suggesting ACTIONABLE next steps the user can perform in this tool. The tool supports: asking follow-up questions about the file, summarizing specific sections, comparing file data with the database, and exporting. Reference specific content from the file. Do NOT ask hypothetical/analytical questions — only suggest things the tool can actually do. Never skip this."}]):
        full_text += token
        yield {"type": "text_delta", "delta": token}

    full_text = beautify(full_text)
    if "FOLLOW_UPS:" in full_text:
        full_text = full_text.split("FOLLOW_UPS:")[0].strip()

    yield {"type": "text_done", "content": full_text}
