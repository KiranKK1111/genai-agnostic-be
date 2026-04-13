"""Hybrid pipeline — compare file content with database results."""
import logging
from app.services.embedder import embed_single
from app.services.vector_search import search
from app.services.llm_client import chat_stream
from app.services.response_beautifier import beautify
from app.database import get_pool
from app.config import get_settings

logger = logging.getLogger(__name__)


async def execute_hybrid(message: str, session_state: dict):
    """Compare file content with database results. Yields SSE events."""
    settings = get_settings()
    schema = settings.APP_SCHEMA

    yield {"type": "step", "step_number": 1, "label": "Gathering file context"}

    # Get relevant file chunks
    query_emb = await embed_single(message)
    file_results = await search("chunks_idx", query_emb, k=3,
                                 filter_key="session_id", filter_value=session_state["session_id"],
                                 query_text=message)

    pool = get_pool()
    file_chunks = []
    for r in file_results:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT chunk_text FROM {schema}.file_chunks WHERE node_id=$1", r["node_id"]
            )
            if row:
                file_chunks.append(row["chunk_text"])

    yield {"type": "step", "step_number": 2, "label": "Retrieving database results"}
    yield {"type": "step", "step_number": 3, "label": "Comparing datasets"}

    # Get cached DB data
    db_data = session_state.get("last_data", [])
    db_columns = session_state.get("last_columns", [])
    friendly_cols = ', '.join(c.replace('_', ' ').title() for c in db_columns[:10])
    db_summary = f"Database results: {len(db_data)} records with fields: {friendly_cols}"
    if db_data:
        # Sanitize sample rows with friendly labels
        sample_rows = []
        for row in db_data[:3]:
            sanitized = {k.replace('_', ' ').title(): str(v) for k, v in row.items()}
            sample_rows.append(sanitized)
        db_summary += f"\nSample: {str(sample_rows)[:500]}"

    file_summary = f"File content:\n{chr(10).join(file_chunks[:3])}"

    yield {"type": "step", "step_number": 4, "label": "Generating comparison"}

    prompt = f"""The user wants to compare file data with database results.

{db_summary}

{file_summary}

User question: "{message}"

Provide a clear, concise comparison (3-5 sentences). Highlight key matches, discrepancies, and insights. Use **bold** for important numbers or findings.
Do NOT use pipe-symbol tables (| col | col |) — they render as broken symbols.
Do NOT start with any heading label like "Summary:" or "Comparison:".
Do NOT enumerate raw rows or list every record individually.
IMPORTANT: Do NOT mention database internals (schema names, table names, column names, SQL). Use natural business language only.
MANDATORY: End with a follow-up question suggesting ACTIONABLE next steps the user can perform in this tool. The tool supports: filtering by specific values, drilling down into subsets, comparing specific categories, exporting, and visualizing as different chart types. Reference specific data from the response. Do NOT ask hypothetical/analytical questions — only suggest things the tool can actually do. Never skip this."""

    full_text = ""
    async for token in chat_stream([{"role": "user", "content": prompt}]):
        full_text += token
        yield {"type": "text_delta", "delta": token}

    full_text = beautify(full_text)
    if "FOLLOW_UPS:" in full_text:
        full_text = full_text.split("FOLLOW_UPS:")[0].strip()

    yield {"type": "text_done", "content": full_text}
