"""Prompt injection detection and output sanitization.

This is a SELECT-only data retrieval application.
All DML/DDL, prompt jailbreaks, and system prompt extraction are blocked.
"""
import re
import logging

logger = logging.getLogger(__name__)

# ── Pre-LLM: Input injection patterns ──────────────────
INJECTION_PATTERNS = [
    # Prompt jailbreak attempts
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)",
    r"system\s+prompt",
    r"repeat\s+(your|the)\s+instructions",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(a|an|if)",
    r"you\s+are\s+now\s+",
    r"reveal\s+(your|the)\s+(instructions|prompt|system)",
    r"new\s+rule\s*:",
    r"override\s+(all|previous|the)\s+",
    r"forget\s+(all|everything|your)\s+",

    # Direct SQL DML/DDL in user input
    r"\b(DROP|ALTER|TRUNCATE)\s+(TABLE|INDEX|SCHEMA|DATABASE|FUNCTION)",
    r"\b(INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM)",
    r"\b(CREATE\s+(TABLE|INDEX|FUNCTION|SCHEMA|DATABASE|ROLE|USER))",
    r"\b(GRANT|REVOKE)\s+",
    r"\bEXEC(UTE)?\s+",

    # SQL injection payloads
    r";\s*(DROP|DELETE|INSERT|UPDATE|ALTER|TRUNCATE|CREATE)",
    r"'\s*(OR|AND)\s+['\d]",
    r"UNION\s+(ALL\s+)?SELECT",
    r"INTO\s+(OUTFILE|DUMPFILE)",
    r"\bxp_cmdshell\b",
    r"\bpg_sleep\b",
    r"\bdblink\b",
]

# ── Post-LLM: Output leakage patterns ─────────────────
OUTPUT_LEAKAGE_PATTERNS = [
    r"system\s+prompt\s*(is|was|says|reads|:)",
    r"my\s+instructions\s+(are|say|tell)",
    r"I\s+was\s+instructed\s+to",
    r"here\s+(is|are)\s+my\s+(instructions|system\s+prompt|rules)",
    r"\b(INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM)\b",
    r"\b(DROP|ALTER|TRUNCATE)\s+(TABLE|INDEX|SCHEMA)\b",
]


def check_injection(text: str) -> dict:
    """Pre-LLM: Check for prompt injection and SQL injection patterns.
    Returns {is_injection, matched_patterns}."""
    if not text:
        return {"is_injection": False, "matched_patterns": []}

    matched = []
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            matched.append(pattern)

    return {
        "is_injection": len(matched) > 0,
        "matched_patterns": matched,
    }


def scan_output(text: str) -> dict:
    """Post-LLM: Scan LLM response for prompt leakage and DML statements.
    Returns {is_safe, issues, sanitized_text}."""
    if not text:
        return {"is_safe": True, "issues": [], "sanitized_text": text}

    issues = []

    # Check for prompt/instruction leakage
    for pattern in OUTPUT_LEAKAGE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            issues.append(f"Output leakage detected: '{match.group()}'")

    # Check for SQL DML in non-code-block text
    text_no_code = re.sub(r"```[\s\S]*?```", "", text)
    dml_pattern = r"\b(INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|DROP\s+TABLE|ALTER\s+TABLE|TRUNCATE)\b"
    dml_match = re.search(dml_pattern, text_no_code, re.IGNORECASE)
    if dml_match:
        issues.append(f"DML statement in response: '{dml_match.group()}'")

    if issues:
        logger.warning(f"Post-LLM scan flagged: {issues}")

    return {
        "is_safe": len(issues) == 0,
        "issues": issues,
        "sanitized_text": text,
    }
