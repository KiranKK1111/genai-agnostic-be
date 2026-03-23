"""Token budget estimator for LLM context management."""

# Rough token estimation: 1 token ≈ 4 chars for English text
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string."""
    if not text:
        return 0
    return len(text) // CHARS_PER_TOKEN


def truncate_to_budget(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def build_context_budget(system: str, history: list[dict], user_msg: str,
                         max_context: int = 6000) -> list[dict]:
    """Build conversation messages that fit within the token budget."""
    system_tokens = estimate_tokens(system)
    user_tokens = estimate_tokens(user_msg)
    remaining = max_context - system_tokens - user_tokens - 200  # 200 token buffer

    if remaining <= 0:
        return [{"role": "user", "content": truncate_to_budget(user_msg, max_context - system_tokens - 100)}]

    # Add history from most recent, working backwards
    messages = []
    for turn in reversed(history):
        turn_tokens = estimate_tokens(turn.get("content", ""))
        if remaining - turn_tokens < 0:
            break
        messages.insert(0, turn)
        remaining -= turn_tokens

    messages.append({"role": "user", "content": user_msg})
    return messages
