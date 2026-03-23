"""Auto-generate session titles from first message."""
from app.services.llm_client import chat


async def generate_title(message: str) -> str:
    """Generate a 4-8 word title from a user message."""
    try:
        result = await chat(
            [{"role": "user", "content": f"Summarize this in exactly 4-6 words as a chat title. No quotes, no punctuation at end.\n\n{message}"}],
            temperature=0.3
        )
        title = result.strip().strip('"').strip("'")
        # Ensure reasonable length
        words = title.split()
        if len(words) > 8:
            title = " ".join(words[:8])
        return title or "New Chat"
    except Exception:
        return "New Chat"
