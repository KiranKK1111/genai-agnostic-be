"""Response beautification — post-process LLM output."""
import re
import logging

logger = logging.getLogger(__name__)


def _validate_markdown(text: str) -> str:
    """Ensure all opened markdown tags are properly closed."""
    # Fix unclosed bold markers
    bold_count = len(re.findall(r"\*\*", text))
    if bold_count % 2 != 0:
        text += "**"

    # Fix unclosed inline code markers
    code_count = text.count("`") - 3 * text.count("```")
    if code_count % 2 != 0:
        text += "`"

    # Fix unclosed code fences
    fence_count = len(re.findall(r"```", text))
    if fence_count % 2 != 0:
        text += "\n```"

    # Fix unclosed italic markers (single *)
    # Count standalone * (not part of ** or ***)
    stripped = re.sub(r"\*\*\*|\*\*", "", text)
    italic_count = stripped.count("*")
    if italic_count % 2 != 0:
        text += "*"

    return text


def beautify(text: str) -> str:
    """Post-process LLM response text."""
    if not text:
        return text

    # Step 1: Markdown validation — ensure all opened tags are closed
    text = _validate_markdown(text)

    # Step 2: SQL syntax hints — add ```sql language hints to code fences
    text = re.sub(r"```\s*\n(SELECT|WITH|INSERT)", r"```sql\n\1", text, flags=re.IGNORECASE)

    # Step 3: Number formatting — large numbers with comma separators
    def add_commas(match):
        num = match.group(0)
        if len(num) >= 4 and "." not in num:
            return f"{int(num):,}"
        return num
    text = re.sub(r"\b\d{4,}\b", add_commas, text)

    return text


def extract_follow_ups(text: str, intent: str = None) -> list[dict]:
    """Extract or generate follow-up suggestions."""
    follow_ups = []
    # Try to extract from LLM output
    if "FOLLOW_UPS:" in text:
        import json
        try:
            json_part = text.split("FOLLOW_UPS:")[1].strip()
            suggestions = json.loads(json_part)
            for s in suggestions[:3]:
                follow_ups.append({"text": s, "type": "chip", "icon": ""})
        except Exception:
            pass

    # Generate defaults based on intent if none extracted
    if not follow_ups:
        if intent == "DB_QUERY":
            follow_ups = [
                {"text": "Show as bar chart", "type": "action", "icon": "\U0001f4ca"},
                {"text": "Export to CSV", "type": "action", "icon": "\U0001f4e5"},
            ]
        elif intent == "FILE_ANALYSIS":
            follow_ups = [
                {"text": "Summarize the key points", "type": "question", "icon": "\u2753"},
                {"text": "Compare with database", "type": "action", "icon": "\U0001f504"},
            ]
        elif intent == "CHAT":
            follow_ups = [
                {"text": "Show me all tables", "type": "chip", "icon": "\U0001f4ca"},
                {"text": "Upload a file", "type": "action", "icon": "\U0001f4c1"},
            ]

    return follow_ups
