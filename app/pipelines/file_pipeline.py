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

    # Update session
    session_state["file_context"] = {
        "file_name": file_name, "chunk_count": len(chunks),
        "session_id": session_state["session_id"]
    }

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

    if session_state.get("total_turns", 0) == 0:
        title = await generate_title(f"File: {file_name}")
        yield {"type": "session_meta", "session_title": title}
        session_state["title"] = title


async def execute_file_followup(message: str, session_state: dict):
    """Answer a follow-up question about an uploaded file using RAG."""
    settings = get_settings()
    file_ctx = session_state.get("file_context", {})

    yield {"type": "step", "step_number": 1, "label": "Searching file content..."}

    # Embed the question
    query_emb = await embed_single(message)

    # Search chunks (hybrid: dense FAISS + sparse BM25)
    results = await search("chunks_idx", query_emb, k=5,
                           filter_key="session_id", filter_value=session_state["session_id"],
                           query_text=message)

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
    async for token in chat_stream([{"role": "user", "content": f"File: {file_ctx.get('file_name', 'unknown')}\nRelevant content:\n{context}\n\nQuestion: {message}\n\nAnswer based on the file content. Use markdown."}]):
        full_text += token
        yield {"type": "text_delta", "delta": token}

    full_text = beautify(full_text)
    follow_ups = extract_follow_ups(full_text, intent="FILE_ANALYSIS")
    if "FOLLOW_UPS:" in full_text:
        full_text = full_text.split("FOLLOW_UPS:")[0].strip()

    yield {"type": "text_done", "content": full_text}
    yield {"type": "follow_ups", "suggestions": follow_ups}
