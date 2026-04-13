"""Shared text helpers."""
import re


# Common English abbreviations that should always be uppercase when they appear
# as standalone tokens in titles / domain names.
_KNOWN_ACRONYMS = {
    "mstr", "crm", "erp", "sql", "sdm", "sdb", "kpi", "api", "url", "id",
    "hr", "it", "qa", "ai", "ml", "ui", "ux", "pdf", "csv", "xlsx", "docx",
    "json", "xml", "db", "rdbms", "oltp", "olap", "etl", "elt", "sso",
}

_VOWEL_RE = re.compile(r"[aeiou]", re.IGNORECASE)


def smart_capitalize_token(token: str) -> str:
    """Capitalize a single word, uppercasing it if it looks like an acronym.

    Rules:
      - Already has 2+ uppercase letters (e.g. "MSTR", "SQL") → leave as-is.
      - Known abbreviation → upper()
      - Short (≤ 4 chars) and mostly consonants → upper() — catches unknown acronyms.
      - Otherwise → Title case.

    Examples:
        "mstr"    → "MSTR"
        "MSTR"    → "MSTR"
        "banking" → "Banking"
        "erp"     → "ERP"
        "Corporate" → "Corporate"
    """
    if not token:
        return token
    stripped = token.strip()
    if not stripped:
        return token

    # Already has multiple uppercase letters — assume it's intentional (MSTR, SQL, etc.)
    upper_count = sum(1 for c in stripped if c.isupper())
    if upper_count >= 2:
        return stripped

    lower = stripped.lower()
    if lower in _KNOWN_ACRONYMS:
        return stripped.upper()

    # Short, mostly-consonant tokens are very likely acronyms
    if len(lower) <= 4:
        vowels = len(_VOWEL_RE.findall(lower))
        consonants = len(lower) - vowels
        if consonants >= 3 and vowels <= 1:
            return stripped.upper()

    return stripped[:1].upper() + stripped[1:].lower()
