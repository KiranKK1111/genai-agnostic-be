"""Chat pipeline — greetings, capability questions, general conversation."""
import logging
from app.services.llm_client import chat_stream
from app.services.response_beautifier import beautify
from app.services.text_utils import smart_capitalize_token
from app.config import get_settings


# No hardcoded greeting regex — intent classifier determines this dynamically.

logger = logging.getLogger(__name__)

_SYSTEM_BASE = """You are a friendly AI assistant for the {domain_name} database dashboard.
The user's name is {user_name}.

YOUR CAPABILITIES:
- Answer general questions and have conversations
- Query the {domain_name} database using natural language
- Analyze uploaded files (PDF, XLSX, CSV, DOCX, TXT, JSON)
- Compare file data with database results
- Generate visualizations (Table, Bar, Pie, Line charts)

DATABASE CONTEXT (this is the ONLY source of examples you may reference):
{schema_context}

FORMATTING RULES:
- Use markdown: **bold** for key terms, `code` for technical values
- Use emojis sparingly for warmth (max 1-2 per response):
  Greetings: \U0001f44b  Success: \u2705  Data: \U0001f4ca  Warning: \u26a0\ufe0f
- Show SQL in fenced code blocks: ```sql ... ```
- Format large numbers with commas: 1,245 not 1245
- Keep responses concise: 2-4 sentences for chat, 3-6 for data
- NEVER use pipe-symbol markdown tables (| col | col |) — they render as broken terminal symbols
- NEVER start a response with a heading label like "Summary:" or "Answer:"
- CAPITALIZATION: Any acronym or abbreviation (e.g. {domain_name}, SQL, CSV, PDF, KPI, CRM, ERP, API, URL, ID) MUST always be written in UPPERCASE. Never render {domain_name} as "Mstr" or "mstr" — always "{domain_name}".

EXAMPLE RULE (STRICT):
- NEVER invent example topics like "sales numbers", "customer information", "employee records", "product data", etc. unless those entities actually exist in the DATABASE CONTEXT above.
- If you want to show an example, it MUST reference the actual tables/columns listed in DATABASE CONTEXT. If none are relevant, give NO example at all.
- Do NOT say things like "for example, are you looking for X or Y?". Only suggest concrete things the user can do against the real schema.

ENDING RULE:
- MANDATORY: End every response with a follow-up question suggesting ACTIONABLE next steps the user can perform in this tool. The tool supports: querying the database with natural language, uploading files for analysis, comparing file data with database data, and generating visualizations. Reference the real tables from DATABASE CONTEXT when suggesting query ideas. Do NOT ask hypothetical questions — only suggest things the tool can actually do. Never skip this.

SCOPE BOUNDARY:
- If asked to do something outside your capabilities (write code, generate images, browse the web), politely explain your scope."""


def _build_schema_context(schema_graph) -> str:
    """Summarize the active schema so the LLM grounds its examples in real tables."""
    if schema_graph is None or not getattr(schema_graph, "tables", None):
        return "(No schema metadata available.)"
    tables = list(schema_graph.tables.items())
    lines: list[str] = []
    if getattr(schema_graph, "domain_name", ""):
        lines.append(f"Domain: {schema_graph.domain_name}")
    lines.append(f"Tables ({len(tables)}):")
    for tname, tmeta in tables[:12]:
        cols = list(tmeta.columns.keys())[:6]
        friendly = ", ".join(cols)
        desc = f" — {tmeta.description}" if getattr(tmeta, "description", "") else ""
        lines.append(f"  - {tname} ({friendly}){desc}")
    if len(tables) > 12:
        lines.append(f"  ...and {len(tables) - 12} more tables")
    return "\n".join(lines)


async def execute_chat(message: str, session_state: dict, user_name: str = "User"):
    """Execute the chat pipeline. Yields SSE events."""
    settings = get_settings()
    raw_schema = settings.POSTGRES_SCHEMA

    # Pull the domain name from the live schema graph when available; fall back
    # to a smart capitalization of the schema name (so "mstr" becomes "MSTR"
    # instead of the default Title-case "Mstr").
    try:
        from app.orchestrator import get_schema_graph as _get_graph
        graph = _get_graph()
    except Exception:
        graph = None

    if graph is not None and getattr(graph, "domain_name", ""):
        domain_name = " ".join(smart_capitalize_token(t) for t in graph.domain_name.split())
    else:
        domain_name = " ".join(
            smart_capitalize_token(t) for t in raw_schema.replace("_", " ").split()
        )

    display_name = user_name if not user_name.isdigit() else "User"

    # ── Progressive steps so the user sees meaningful activity ───────────
    yield {"type": "step", "step_number": 1, "label": "Understanding your message"}

    # The unified LLM intent classifier already determined this is CHAT.
    schema_context = _build_schema_context(graph)
    system = (
        _SYSTEM_BASE
        .replace("{user_name}", display_name)
        .replace("{domain_name}", domain_name)
        .replace("{schema_context}", schema_context)
    )

    yield {"type": "step", "step_number": 2, "label": "Recalling conversation context"}

    # Build messages: [compressed summary] + [last 5 turns] + [current message]
    messages = []
    summary = session_state.get("history_summary", "")
    if summary:
        messages.append({"role": "system", "content": f"Previous conversation summary: {summary}"})
    for turn in session_state.get("history", [])[-5:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

    yield {"type": "step", "step_number": 3, "label": "Thinking of a response"}
    yield {"type": "step", "step_number": 4, "label": "Composing reply"}

    # Stream response (with cancellation check)
    full_text = ""
    async for token in chat_stream(messages, system=system):
        if session_state.get("cancel_requested"):
            from app.services.cancel_handler import handle_cancel
            cancel_event = await handle_cancel(session_state["session_id"], full_text, "")
            yield cancel_event
            return
        full_text += token
        # Don't stream FOLLOW_UPS marker to frontend — it's metadata, not content
        if "FOLLOW_UPS:" not in full_text:
            yield {"type": "text_delta", "delta": token}

    # Beautify
    full_text = beautify(full_text)

    # Clean follow-ups marker from displayed text
    if "FOLLOW_UPS:" in full_text:
        full_text = full_text.split("FOLLOW_UPS:")[0].strip()

    yield {"type": "text_done", "content": full_text}

    # NOTE: session title is set by the orchestrator from the user's first
    # prompt (see orchestrator.py:160-178). We intentionally do NOT override
    # it here with an LLM-generated summary — that produced misleading titles
    # like "New Message from Unknown User" for short greetings.
