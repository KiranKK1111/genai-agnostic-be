"""File analysis pipeline — upload, chunk, embed, RAG retrieve, synthesize."""
import os
import logging
import shutil
import uuid
from app.services.file_parser import parse_file, chunk_text
from app.services.embedder import embed_texts, embed_single
from app.services.sentence_encoder import fit_encoder, get_encoder
from app.services.vector_search import insert_node, search
from app.services.llm_client import chat_stream
from app.services.response_beautifier import beautify, extract_follow_ups
from app.services.title_generator import generate_title
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)


async def execute_file_upload(file_path: str, file_name: str, message: str,
                               session_state: dict, user_name: str = "User"):
    """Process a file upload + optional query. Yields SSE events."""
    settings = get_settings()

    yield {"type": "step", "step_number": 1, "label": "Receiving file..."}

    # Save to session directory
    session_dir = os.path.join(settings.UPLOAD_DIR, session_state["session_id"])
    os.makedirs(session_dir, exist_ok=True)
    dest = os.path.join(session_dir, file_name)
    shutil.copy2(file_path, dest)

    yield {"type": "step", "step_number": 2, "label": "Extracting content..."}
    content = await parse_file(dest)

    yield {"type": "step", "step_number": 3, "label": "Splitting into chunks..."}
    chunks = chunk_text(content, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    yield {"type": "step", "step_number": 4, "label": "Creating embeddings..."}
    # Expand encoder vocabulary with file content so SVD covers file terms
    encoder = get_encoder()
    if encoder.is_fitted:
        # Refit on existing corpus + new chunks for better coverage
        from app.database import get_pool
        pool = get_pool()
        schema = settings.APP_SCHEMA
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"SELECT content FROM {schema}.embedding_metadata WHERE content IS NOT NULL AND content != ''"
                )
            existing = [r["content"] for r in rows]
            fit_encoder(existing + [c[:500] for c in chunks])
        except Exception as e:
            logger.warning(f"Encoder refit on file chunks failed: {e}")
    # Use "passage" mode for document chunks (Stage 15+16)
    embeddings = await embed_texts([c[:500] for c in chunks], mode="passage")

    # Store chunks in DB and FAISS
    pool = get_pool()
    schema = settings.APP_SCHEMA
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        node_id = await insert_node("chunks_idx", emb, {
            "session_id": session_state["session_id"],
            "file_name": file_name, "chunk_index": i,
            "chunk_text": chunk[:500],
        })
        async with pool.acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {schema}.file_chunks (session_id, file_name, chunk_index, chunk_text, node_id)
                    VALUES ($1::uuid, $2, $3, $4, $5)""",
                session_state["session_id"], file_name, i, chunk, str(node_id)
            )

    # Update session — track active file AND append to file history
    file_ctx = {
        "file_name": file_name, "chunk_count": len(chunks),
        "session_id": session_state["session_id"],
    }
    session_state["file_context"] = file_ctx

    file_history = session_state.get("file_history", [])
    file_history.append({
        **file_ctx,
        "upload_message": message or f"Uploaded {file_name}",
    })
    session_state["file_history"] = file_history

    yield {"type": "step", "step_number": 5, "label": "Analyzing content..."}

    # Generate response
    query = message or f"Summarize the uploaded file {file_name}"
    context = "\n\n".join(chunks[:5])  # Use first 5 chunks for initial response

    full_text = ""
    async for token in chat_stream([{"role": "user", "content": (
        f"File: {file_name}\nContent:\n{context}\n\n"
        f"User message: {query}\n\n"
        "Respond naturally to the user's message first, then provide a helpful analysis of the file. "
        "Use markdown formatting."
    )}]):
        full_text += token
        yield {"type": "text_delta", "delta": token}

    full_text = beautify(full_text)
    follow_ups = extract_follow_ups(full_text, intent="FILE_ANALYSIS")
    if "FOLLOW_UPS:" in full_text:
        full_text = full_text.split("FOLLOW_UPS:")[0].strip()

    yield {"type": "text_done", "content": full_text}
    yield {"type": "follow_ups", "suggestions": follow_ups}

    if session_state.get("_is_first_message", False):
        title = await generate_title(f"File: {file_name}")
        yield {"type": "session_meta", "session_title": title}
        session_state["title"] = title


async def _select_base_file(message: str, file_history: list[dict]) -> dict:
    """Given a follow-up message and file history, pick which file the user is asking about.

    If there's only one file, return it directly. Otherwise, ask the LLM to decide
    based on file names and the user's question context.
    """
    if len(file_history) == 1:
        return file_history[0]

    # Last file is the default
    if not message:
        return file_history[-1]

    from app.services.llm_client import chat_json

    history_lines = []
    for i, entry in enumerate(file_history):
        fname = entry.get("file_name", "unknown")
        msg = entry.get("upload_message", "")
        chunks = entry.get("chunk_count", 0)
        history_lines.append(f"  F{i+1}. \"{fname}\" ({chunks} chunks) — uploaded with: \"{msg}\"")

    history_text = "\n".join(history_lines)

    try:
        result = await chat_json([{"role": "user", "content":
            f'The user has uploaded multiple files in this session:\n{history_text}\n\n'
            f'The user now asks: "{message}"\n\n'
            f'Which file is the user most likely asking about?\n'
            f'Consider: file names, upload context, and the question content.\n'
            f'If unclear, pick the most recently uploaded file.\n\n'
            f'Return JSON: {{"file_number": <1-based index>, "reason": "brief explanation"}}'}])
        chosen = result.get("file_number", len(file_history))
        idx = max(0, min(chosen - 1, len(file_history) - 1))
        logger.info(f"File selection: chose F{idx+1} ({file_history[idx].get('file_name')}) — {result.get('reason', '')}")
    except Exception as e:
        logger.warning(f"File selection failed: {e}, using last file")
        idx = len(file_history) - 1

    return file_history[idx]


async def execute_file_followup(message: str, session_state: dict):
    """Answer a follow-up question about an uploaded file using RAG.

    When multiple files exist in the session, uses LLM to pick the right file.
    Searches chunks filtered by file_name (not just session_id) so results
    come from the correct file.
    """
    settings = get_settings()

    yield {"type": "step", "step_number": 1, "label": "Searching file content..."}

    # Pick the right file from history (or use current file_context)
    file_history = session_state.get("file_history", [])
    if len(file_history) > 1:
        file_ctx = await _select_base_file(message, file_history)
    else:
        file_ctx = session_state.get("file_context", {})

    target_file = file_ctx.get("file_name", "")

    # Update active file_context to the selected file
    session_state["file_context"] = file_ctx

    # Embed the question
    query_emb = await embed_single(message)

    # Search chunks (hybrid: dense FAISS + sparse BM25)
    # Filter by session_id first, then narrow to target file
    results = await search("chunks_idx", query_emb, k=5,
                           filter_key="session_id", filter_value=session_state["session_id"],
                           query_text=message)

    # Filter results to target file if multiple files exist
    if target_file and len(file_history) > 1:
        file_results = [r for r in results if r.get("payload", {}).get("file_name") == target_file]
        # Fall back to all results if file-specific filtering yields nothing
        if file_results:
            results = file_results

    # Retrieve full chunk text
    pool = get_pool()
    schema = settings.APP_SCHEMA
    chunks = []
    for r in results:
        payload = r.get("payload", {})
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT chunk_text FROM {schema}.file_chunks WHERE node_id=$1", r["node_id"]
            )
            if row:
                chunks.append(row["chunk_text"])

    if not chunks:
        yield {"type": "text_done", "content": "I couldn't find relevant content in the uploaded file for your question. Could you rephrase?"}
        return

    yield {"type": "step", "step_number": 2, "label": "Generating response..."}

    context = "\n\n".join(chunks)
    full_text = ""
    async for token in chat_stream([{"role": "user", "content": f"File: {target_file or 'unknown'}\nRelevant content:\n{context}\n\nQuestion: {message}\n\nAnswer based on the file content. Use markdown."}]):
        full_text += token
        yield {"type": "text_delta", "delta": token}

    full_text = beautify(full_text)
    follow_ups = extract_follow_ups(full_text, intent="FILE_ANALYSIS")
    if "FOLLOW_UPS:" in full_text:
        full_text = full_text.split("FOLLOW_UPS:")[0].strip()

    yield {"type": "text_done", "content": full_text}
    yield {"type": "follow_ups", "suggestions": follow_ups}
