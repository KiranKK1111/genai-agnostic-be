"""Chat pipeline — greetings, capability questions, general conversation."""
import re
import logging
from app.services.llm_client import chat_stream, chat
from app.services.response_beautifier import beautify, extract_follow_ups
from app.services.title_generator import generate_title
from app.config import get_settings

_GREETING_RE = re.compile(
    r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy|greetings|what'?s\s+up|sup|thanks|thank\s*you|bye|ok)\s*[!?.]*$",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)

_SYSTEM_BASE = """You are a friendly AI assistant for the {domain_name} database dashboard.
The user's name is {user_name}.

YOUR CAPABILITIES:
- Answer general questions and have conversations
- Query the {domain_name} database using natural language
- Analyze uploaded files (PDF, XLSX, CSV, DOCX, TXT, JSON)
- Compare file data with database results
- Generate visualizations (Table, Bar, Pie, Line charts)

FORMATTING RULES:
- Use markdown: **bold** for key terms, `code` for technical values
- Use emojis sparingly for warmth (max 1-2 per response):
  Greetings: \U0001f44b  Success: \u2705  Data: \U0001f4ca  Warning: \u26a0\ufe0f
- Show SQL in fenced code blocks: ```sql ... ```
- Format large numbers with commas: 1,245 not 1245
- Keep responses concise: 2-4 sentences for chat, 3-6 for data

SCOPE BOUNDARY:
- If asked to do something outside your capabilities (write code, generate images, browse the web), politely explain your scope."""

_FOLLOW_UP_INSTRUCTION = """

FOLLOW-UP SUGGESTIONS:
- End every response with 2-3 follow-up suggestions.
- Format as: FOLLOW_UPS: ["suggestion 1", "suggestion 2"]"""


async def execute_chat(message: str, session_state: dict, user_name: str = "User"):
    """Execute the chat pipeline. Yields SSE events."""
    settings = get_settings()
    raw_schema = settings.POSTGRES_SCHEMA
    domain_name = raw_schema.replace("_", " ").title()
    display_name = user_name if not user_name.isdigit() else "User"

    is_greeting = bool(_GREETING_RE.match(message.strip()))

    # For greetings: don't ask LLM to generate follow-ups at all
    # This prevents FOLLOW_UPS: text from appearing in the response
    if is_greeting:
        system = _SYSTEM_BASE.replace("{user_name}", display_name).replace("{domain_name}", domain_name)
    else:
        system = (_SYSTEM_BASE + _FOLLOW_UP_INSTRUCTION).replace("{user_name}", display_name).replace("{domain_name}", domain_name)

    # Build messages: [compressed summary] + [last 5 turns] + [current message]
    messages = []
    summary = session_state.get("history_summary", "")
    if summary:
        messages.append({"role": "system", "content": f"Previous conversation summary: {summary}"})
    for turn in session_state.get("history", [])[-5:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

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

    # Extract and emit follow-up suggestions (skip for greetings)
    if not is_greeting:
        follow_ups = extract_follow_ups(full_text, intent="CHAT")
        yield {"type": "follow_ups", "suggestions": follow_ups}

    # Generate title for first message
    if session_state.get("_is_first_message", False):
        # For greetings, use the message itself as the title instead of asking LLM
        # (LLM tends to produce misleading titles like "Starting a New Conversation")
        if is_greeting:
            title = message.strip().capitalize()
        else:
            title = await generate_title(message)
        yield {"type": "session_meta", "session_title": title}
        session_state["title"] = title
        logger.info(f"Generated session title: '{title}'")
