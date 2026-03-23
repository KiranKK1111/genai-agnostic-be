"""Centralized error code registry (E001–E099)."""

ERROR_REGISTRY = {
    # SQL errors
    "E001": {"category": "SQL", "message": "Query execution timed out", "recoverable": True,
             "suggestion": "Try adding filters to narrow results or ask for a subset."},
    "E002": {"category": "SQL", "message": "SQL execution failed", "recoverable": True,
             "suggestion": "Rephrase your question — I'll generate a new query."},
    "E003": {"category": "SQL", "message": "No tables found matching your query", "recoverable": True,
             "suggestion": "Try mentioning a specific table name or ask 'show me all tables'."},
    "E004": {"category": "SQL", "message": "Column not found", "recoverable": True,
             "suggestion": "Check column names — ask 'show columns in [table]' to see available fields."},
    "E005": {"category": "SQL", "message": "Value not found in database", "recoverable": True,
             "suggestion": "The value may be spelled differently in the database. Try a broader search."},
    "E006": {"category": "SQL", "message": "JOIN path not found between tables", "recoverable": True,
             "suggestion": "These tables may not be directly related. Try querying them separately."},
    "E007": {"category": "SQL", "message": "EXPLAIN validation failed", "recoverable": True,
             "suggestion": "The generated query has a syntax issue. Let me try again."},
    "E008": {"category": "SQL", "message": "Row limit exceeded", "recoverable": True,
             "suggestion": "Add filters to reduce the result set."},

    # LLM errors
    "E010": {"category": "LLM", "message": "LLM request failed", "recoverable": True,
             "suggestion": "Ollama may be busy. Retrying automatically..."},
    "E011": {"category": "LLM", "message": "LLM response unparseable", "recoverable": True,
             "suggestion": "The model returned an unexpected format. Retrying..."},
    "E012": {"category": "LLM", "message": "Context window exceeded", "recoverable": False,
             "suggestion": "Your conversation is very long. Start a new chat session."},
    "E013": {"category": "LLM", "message": "Embedding service unavailable", "recoverable": True,
             "suggestion": "Check that Ollama is running with the embedding model."},

    # Input errors
    "E020": {"category": "Input", "message": "Unrecognizable input", "recoverable": True,
             "suggestion": "Please rephrase your question in plain English."},
    "E021": {"category": "Input", "message": "Potential injection detected", "recoverable": False,
             "suggestion": "Your message contains patterns that cannot be processed."},
    "E022": {"category": "Input", "message": "Input too long", "recoverable": True,
             "suggestion": "Please shorten your message to under 10,000 characters."},

    # File errors
    "E030": {"category": "File", "message": "Unsupported file type", "recoverable": True,
             "suggestion": "Supported formats: PDF, XLSX, CSV, TXT, JSON."},
    "E031": {"category": "File", "message": "File parsing failed", "recoverable": True,
             "suggestion": "The file may be corrupted or in an unexpected format."},
    "E032": {"category": "File", "message": "File too large", "recoverable": True,
             "suggestion": "Maximum file size is 50MB."},
    "E033": {"category": "File", "message": "Disk storage full", "recoverable": False,
             "suggestion": "Delete old sessions to free up space."},

    # Scope errors
    "E050": {"category": "Scope", "message": "Request outside application scope", "recoverable": True,
             "suggestion": "I can help with database queries, file analysis, and data visualization."},

    # System errors
    "E099": {"category": "System", "message": "Internal server error", "recoverable": False,
             "suggestion": "Check server logs for details."},
}


def get_error(code: str, override_message: str = None) -> dict:
    """Look up error by code, optionally overriding the message."""
    base = ERROR_REGISTRY.get(code, ERROR_REGISTRY["E099"]).copy()
    base["code"] = code
    if override_message:
        base["message"] = override_message
    return base
