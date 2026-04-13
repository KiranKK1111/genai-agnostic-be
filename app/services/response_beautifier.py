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

    # Step 4: Convert inline "•" bullet sequences to proper newline-separated markdown
    # Handles LLM output like "• Item A • Item B • Item C" all on a single line
    if "•" in text:
        # "• " at the very start of a line → "- "
        text = re.sub(r"^\s*•\s+", "- ", text, flags=re.MULTILINE)
        # " • " mid-line (between two items) → newline + "- "
        text = re.sub(r"\s*•\s+", "\n- ", text)

    # Step 4b: Strip markdown pipe-syntax table lines (they render as raw | symbols)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Drop table rows (| col | col |) and separator rows (| --- | --- |)
        if stripped.startswith("|") and stripped.endswith("|"):
            continue
        cleaned.append(line)
    # Collapse consecutive blank lines left by removed table rows
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))

    # Step 5: Strip stale heading labels the LLM tends to add
    text = re.sub(r"^\*{0,2}Summary\*{0,2}[\s:]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\*{0,2}Follow-?ups?\*{0,2}[\s:]*\n?", "", text, flags=re.IGNORECASE | re.MULTILINE)

    return text.strip()


def _viz_chip_label(viz_type: str) -> str:
    """Convert a viz type key into a human-readable chip label dynamically."""
    vt = viz_type.lower().replace("_", " ")
    # If the type name doesn't already include "chart"/"graph"/"view", append "chart"
    if vt not in ("table",) and not any(w in vt for w in ("chart", "graph", "view", "map")):
        vt += " chart"
    return vt


def _suggestion_mentions_viz(text: str, viz_type: str) -> bool:
    """Return True if a suggestion string is promoting the given viz type.
    Matches on the viz type key itself (e.g. 'bar') appearing in the text.
    """
    return viz_type.lower() in text.lower()


def extract_follow_ups(
    text: str,
    intent: str = None,
    active_viz: str = None,
    available_viz_types: list[str] | None = None,
) -> list[dict]:
    """Extract or generate follow-up suggestions.

    active_viz           : chart type already rendered — suggestions for this type are filtered.
    available_viz_types  : full list of viz types the backend supports for this dataset.
                           When provided, follow-ups are built dynamically from this list
                           (excluding active_viz) so there are no hardcoded type names here.
    """
    follow_ups = []

    # ── Extract from LLM output ──────────────────────────────
    if "FOLLOW_UPS:" in text:
        import json
        try:
            json_part = text.split("FOLLOW_UPS:")[1].strip()
            suggestions = json.loads(json_part)
            for s in suggestions[:3]:
                # Drop suggestions that promote the chart already on screen
                if active_viz and _suggestion_mentions_viz(s, active_viz):
                    continue
                follow_ups.append({"text": s, "type": "chip", "icon": ""})
        except Exception:
            pass

    # ── Dynamic fallback when LLM produced no FOLLOW_UPS ─────
    # Build viz chips dynamically from available types (if data intent)
    if not follow_ups and intent == "DB_QUERY" and available_viz_types:
        for vtype in available_viz_types:
            if vtype == active_viz:
                continue
            follow_ups.append({
                "text": f"Show as {_viz_chip_label(vtype)}",
                "type": "action",
                "icon": "",
            })

    return follow_ups
