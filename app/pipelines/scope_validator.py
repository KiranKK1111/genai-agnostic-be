"""Scope validation — determines if prompt is in-scope or out-of-scope."""
import re

OUT_OF_SCOPE_PATTERNS = re.compile(
    r"\b(write\s+(me\s+)?code|generate\s+(an?\s+)?image|browse\s+(the\s+)?web|search\s+(the\s+)?internet|"
    r"play\s+(a\s+)?(game|music|video)|translate\s+to|write\s+(a\s+)?(poem|song|essay|story)|"
    r"hack|exploit|password|illegal)\b",
    re.IGNORECASE
)

def validate_scope(message: str) -> dict:
    """Check if message is within application scope. Returns {in_scope, reason}."""
    if OUT_OF_SCOPE_PATTERNS.search(message):
        return {"in_scope": False, "reason": "out_of_scope_pattern"}
    return {"in_scope": True, "reason": ""}
