"""LLM API client — chat, chat_json, chat_stream via Ollama/OpenAI-compatible."""
import httpx
import json
import logging
from typing import AsyncIterator
from app.config import get_settings

logger = logging.getLogger(__name__)

async def chat(messages: list[dict], system: str = None, temperature: float = 0.7) -> str:
    """Single-turn LLM completion. Returns text response."""
    settings = get_settings()
    payload = {
        "model": settings.AI_FACTORY_MODEL,
        "messages": [],
        "temperature": temperature,
        "max_tokens": 2000,
        "stream": False,
    }
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].extend(messages)

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{settings.AI_FACTORY_API}/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

async def chat_json(messages: list[dict], system: str = None) -> dict:
    """LLM completion expecting JSON response. Uses Ollama JSON mode for reliable output."""
    settings = get_settings()

    # Build messages with a system prompt enforcing JSON
    all_messages = []
    json_system = system or "You are a helpful assistant. Respond with valid JSON only. No explanation, no markdown, no text outside the JSON object."
    all_messages.append({"role": "system", "content": json_system})
    all_messages.extend(messages)

    payload = {
        "model": settings.AI_FACTORY_MODEL,
        "messages": all_messages,
        "temperature": 0.1,
        "max_tokens": 2000,
        "stream": False,
        "format": "json",  # Ollama JSON mode — forces valid JSON output
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{settings.AI_FACTORY_API}/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]

    # Strip markdown code fences if present (belt-and-suspenders)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse LLM JSON: {text[:200]}")
        return {"error": "Failed to parse LLM response as JSON", "raw": text}

async def chat_stream(messages: list[dict], system: str = None) -> AsyncIterator[str]:
    """Streaming LLM completion. Yields text tokens one at a time."""
    settings = get_settings()
    payload = {
        "model": settings.AI_FACTORY_MODEL,
        "messages": [],
        "temperature": 0.7,
        "max_tokens": 2000,
        "stream": True,
    }
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].extend(messages)

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{settings.AI_FACTORY_API}/chat/completions", json=payload) as resp:
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
