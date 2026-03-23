"""SSE response payload builders — centralized event factories."""
import json


def sse_event(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"


# ── Typing indicators ─────────────────────────────────

def typing_start() -> dict:
    return {"type": "typing_start"}


def typing_end() -> dict:
    return {"type": "typing_end"}


# ── Progress steps ────────────────────────────────────

def step(step_number: int, label: str, status: str = "active") -> dict:
    return {"type": "step", "step_number": step_number, "label": label, "status": status}


def thinking(text: str) -> dict:
    return {"type": "thinking", "text": text}


# ── Text streaming ────────────────────────────────────

def text_delta(delta: str, message_id: str = "") -> dict:
    return {"type": "text_delta", "delta": delta, "message_id": message_id}


def text_done(content: str, message_id: str = "") -> dict:
    return {"type": "text_done", "content": content, "message_id": message_id}


# ── Data & query ──────────────────────────────────────

def data_event(rows: list, columns: list, row_count: int, sql: str = "",
               truncated: bool = False) -> dict:
    return {"type": "data", "rows": rows, "columns": columns,
            "row_count": row_count, "sql": sql, "truncated": truncated}


def query_plan(explanation: str, plan_json: dict = None) -> dict:
    event = {"type": "query_plan", "explanation": explanation}
    if plan_json:
        event["plan_json"] = plan_json
    return event


def viz_config(viz_type: str, config: dict, axis_mode: str = "on_the_fly") -> dict:
    return {"type": "viz_config", "viz_type": viz_type, "config": config, "axis_mode": axis_mode}


# ── Follow-ups & session ──────────────────────────────

def follow_ups(suggestions: list[dict]) -> dict:
    return {"type": "follow_ups", "suggestions": suggestions}


def session_meta(session_id: str = "", session_title: str = "") -> dict:
    return {"type": "session_meta", "session_id": session_id, "session_title": session_title}


# ── Clarification ─────────────────────────────────────

def clarification(clar_type: str, question: str, options: list[dict],
                  mode: str = "single_select", message_id: str = "", **extra) -> dict:
    event = {
        "type": "clarification",
        "message_id": message_id,
        "clarification": {
            "type": clar_type, "mode": mode,
            "question": question, "options": options,
            **extra,
        }
    }
    return event


# ── Error ─────────────────────────────────────────────

def error_event(code: str, message: str, recoverable: bool = True,
                suggestion: str = "", category: str = "System") -> dict:
    return {
        "type": "error", "code": code, "category": category,
        "message": message, "recoverable": recoverable,
        "suggestion": suggestion, "retry_after_sec": None,
    }


# ── Cancellation & done ───────────────────────────────

def cancelled(message_id: str, partial_content: str = "") -> dict:
    return {"type": "cancelled", "message_id": message_id, "partial_content": partial_content}


def done(message_id: str) -> dict:
    return {"type": "done", "message_id": message_id}
