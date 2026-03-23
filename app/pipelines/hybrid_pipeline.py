"""Hybrid pipeline — compare file content with database results."""
import logging
from app.services.embedder import embed_single
from app.services.vector_search import search
from app.services.llm_client import chat_stream
from app.services.response_beautifier import beautify, extract_follow_ups
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)


async def execute_hybrid(message: str, session_state: dict):
    """Compare file content with database results. Yields SSE events."""
    settings = get_settings()
    schema = settings.APP_SCHEMA

    yield {"type": "step", "step_number": 1, "label": "Gathering file context..."}

    # Get relevant file chunks
    query_emb = await embed_single(message)
    file_results = await search("chunks_idx", query_emb, k=3,
                                 filter_key="session_id", filter_value=session_state["session_id"])

    pool = get_pool()
    file_chunks = []
    for r in file_results:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT chunk_text FROM {schema}.file_chunks WHERE node_id=$1", r["node_id"]
            )
            if row:
                file_chunks.append(row["chunk_text"])

    yield {"type": "step", "step_number": 2, "label": "Comparing with database results..."}

    # Get cached DB data
    db_data = session_state.get("last_data", [])
    db_columns = session_state.get("last_columns", [])
    db_summary = f"Database results: {len(db_data)} rows, columns: {', '.join(db_columns[:10])}"
    if db_data:
        db_summary += f"\nSample: {str(db_data[:3])[:500]}"

    file_summary = f"File content:\n{'\n'.join(file_chunks[:3])}"

    yield {"type": "step", "step_number": 3, "label": "Generating comparison..."}

    prompt = f"""The user wants to compare file data with database results.

{db_summary}

{file_summary}

User question: "{message}"

Provide a clear comparison. Highlight matches, discrepancies, and insights. Use markdown formatting."""

    full_text = ""
    async for token in chat_stream([{"role": "user", "content": prompt}]):
        full_text += token
        yield {"type": "text_delta", "delta": token}

    full_text = beautify(full_text)
    follow_ups = extract_follow_ups(full_text, intent="HYBRID")
    if "FOLLOW_UPS:" in full_text:
        full_text = full_text.split("FOLLOW_UPS:")[0].strip()

    yield {"type": "text_done", "content": full_text}
    yield {"type": "follow_ups", "suggestions": follow_ups}
