"""SSE stream buffer — markdown-aware token buffering for clean SSE delivery."""
import asyncio
import json
import re
from typing import AsyncIterator

# Markdown elements that should not be split mid-token
_MARKDOWN_MARKERS = {"**", "```", "`", "*", "__", "~~"}


class StreamBuffer:
    """Buffer SSE events for controlled output delivery.
    For text_delta events, accumulates tokens until a safe flush point
    to prevent partial markdown rendering."""

    def __init__(self, flush_threshold: int = 50):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.closed = False
        self._text_buffer = ""
        self._flush_threshold = flush_threshold

    async def put(self, event_type: str, data: dict):
        """Add an event to the buffer. Text deltas are buffered for markdown safety."""
        if self.closed:
            return

        if event_type == "text_delta":
            self._text_buffer += data.get("delta", "")
            # Flush at safe markdown boundaries
            if self._should_flush():
                await self._flush_text(data.get("message_id", ""))
        else:
            # Flush any pending text before non-text events
            if self._text_buffer:
                await self._flush_text(data.get("message_id", ""))
            await self.queue.put({"type": event_type, **data})

    def _should_flush(self) -> bool:
        """Check if buffer should be flushed at a safe markdown boundary."""
        buf = self._text_buffer
        if not buf:
            return False
        # Flush at newlines
        if buf.endswith("\n"):
            return True
        # Flush at sentence-ending punctuation
        if buf.rstrip().endswith((".", "!", "?")):
            return True
        # Flush at closed markdown elements
        if buf.endswith("**") and buf.count("**") % 2 == 0:
            return True
        if buf.endswith("`") and "```" not in buf[-4:]:
            return True
        # Flush if buffer exceeds threshold (at word boundary)
        if len(buf) > self._flush_threshold and buf[-1] == " ":
            return True
        return False

    async def _flush_text(self, message_id: str = ""):
        """Flush accumulated text as a text_delta event."""
        if self._text_buffer:
            await self.queue.put({
                "type": "text_delta",
                "delta": self._text_buffer,
                "message_id": message_id,
            })
            self._text_buffer = ""

    async def close(self):
        """Flush remaining buffer and signal end of stream."""
        if self._text_buffer:
            await self._flush_text()
        self.closed = True
        await self.queue.put(None)

    async def stream(self) -> AsyncIterator[str]:
        """Yield formatted SSE events from the buffer."""
        while True:
            event = await self.queue.get()
            if event is None:
                break
            event_type = event.pop("type", "data")
            yield f"event: {event_type}\ndata: {json.dumps(event, default=str)}\n\n"
