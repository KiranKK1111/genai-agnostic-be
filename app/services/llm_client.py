"""LLM API client — chat, chat_json, chat_stream via any OpenAI-compatible server.

Includes automatic reconnection: if Ollama was unavailable at startup,
every LLM call checks connectivity first and flips the global flag once
the server becomes reachable.
"""
import httpx
import json
import logging
from typing import AsyncIterator
from app.config import get_settings

logger = logging.getLogger(__name__)


def _build_headers() -> dict:
    """Build HTTP headers, including Authorization if a token is configured."""
    settings = get_settings()
    headers = {"Content-Type": "application/json"}
    token = getattr(settings, "AI_FACTORY_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _ensure_llm_available():
    """Check LLM reachability. If it was previously unavailable, re-probe now.
    Raises RuntimeError if still unreachable."""
    from app.startup_validator import is_llm_available, set_llm_available
    if is_llm_available():
        return
    # Try to reconnect
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
        set_llm_available(True)
        logger.info("LLM reconnected — Ollama is now available")
    except Exception:
        raise RuntimeError(
            "LLM is unavailable. Start Ollama with: ollama serve"
        )


async def chat(messages: list[dict], system: str = None, temperature: float = None) -> str:
    """Single-turn LLM completion. Returns text response."""
    await _ensure_llm_available()
    settings = get_settings()
    payload = {
        "model": settings.AI_FACTORY_MODEL,
        "messages": [],
        "temperature": temperature if temperature is not None else settings.LLM_DEFAULT_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "stream": False,
    }
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].extend(messages)

    async with httpx.AsyncClient(timeout=settings.LLM_REQUEST_TIMEOUT) as client:
        resp = await client.post(
            f"{settings.AI_FACTORY_API}/chat/completions",
            json=payload,
            headers=_build_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def chat_json(messages: list[dict], system: str = None) -> dict:
    """LLM completion expecting JSON response. Works with any OpenAI-compatible server."""
    await _ensure_llm_available()
    settings = get_settings()

    all_messages = []
    json_system = system or (
        "You are a helpful assistant. Respond with valid JSON only. "
        "No explanation, no markdown, no text outside the JSON object."
    )
    all_messages.append({"role": "system", "content": json_system})
    all_messages.extend(messages)

    payload = {
        "model": settings.AI_FACTORY_MODEL,
        "messages": all_messages,
        "temperature": settings.LLM_JSON_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "stream": False,
    }

    headers = _build_headers()
    text = ""

    async with httpx.AsyncClient(timeout=settings.LLM_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.post(
                f"{settings.AI_FACTORY_API}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM chat_json failed: {e}")
            return {"error": "LLM request failed", "detail": str(e)}

    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    # Best-effort JSON extraction
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON object/array in the response
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            if start == -1:
                continue
            end = text.rfind(end_char)
            if end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue
        logger.warning(f"Failed to parse LLM JSON: {text[:200]}")
        return {"error": "Failed to parse LLM response as JSON", "raw": text}


def is_llm_error(result: dict) -> bool:
    """Check if an LLM response is an error. Use after chat_json() calls."""
    return isinstance(result, dict) and "error" in result


async def chat_stream(messages: list[dict], system: str = None) -> AsyncIterator[str]:
    """Streaming LLM completion. Yields text tokens one at a time."""
    await _ensure_llm_available()
    settings = get_settings()
    payload = {
        "model": settings.AI_FACTORY_MODEL,
        "messages": [],
        "temperature": settings.LLM_DEFAULT_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "stream": True,
    }
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].extend(messages)

    async with httpx.AsyncClient(timeout=settings.LLM_STREAM_TIMEOUT) as client:
        async with client.stream(
            "POST",
            f"{settings.AI_FACTORY_API}/chat/completions",
            json=payload,
            headers=_build_headers(),
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta
                except json.JSONDecodeError:
                    continue
