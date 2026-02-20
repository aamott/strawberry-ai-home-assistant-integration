"""Unit tests for StrawberryHubClient.

These tests focus on deterministic logic that does not require a real Hub:
- SSE parsing behavior
- availability cache invalidation
"""

from __future__ import annotations

import json
import unittest

from custom_components.strawberry_conversation.hub_client import StrawberryHubClient


class _FakeStreamResponse:
    """Minimal fake streaming response for SSE parser tests."""

    def __init__(self, chunks: list[str]) -> None:
        """Store chunks returned by ``aiter_text``.

        Args:
            chunks: Text chunks that simulate network stream boundaries.
        """
        self._chunks = chunks

    async def aiter_text(self):
        """Yield configured chunks one-by-one."""
        for chunk in self._chunks:
            yield chunk


class TestHubClient(unittest.IsolatedAsyncioTestCase):
    """Behavior tests for hub client utility logic."""

    async def test_parse_sse_yields_events_and_done(self) -> None:
        """Parser should decode JSON events and terminate on [DONE]."""
        payload = {"type": "assistant_message", "content": "hello"}
        response = _FakeStreamResponse(
            [
                f"data: {json.dumps(payload)}\n",
                "data: [DONE]\n",
            ]
        )

        events = [
            event
            async for event in StrawberryHubClient._parse_sse(response)  # noqa: SLF001
        ]

        self.assertEqual(
            events,
            [
                payload,
                {"type": "done"},
            ],
        )

    async def test_parse_sse_emits_done_when_stream_ends_without_done(self) -> None:
        """Parser should still emit a terminal done event if stream ends early."""
        payload = {"type": "content_delta", "delta": "abc"}
        response = _FakeStreamResponse([f"data: {json.dumps(payload)}\n"])

        events = [
            event
            async for event in StrawberryHubClient._parse_sse(response)  # noqa: SLF001
        ]

        self.assertEqual(events[-1], {"type": "done"})
        self.assertEqual(events[0], payload)

    async def test_invalidate_cache_resets_availability_state(self) -> None:
        """Cache invalidation should force the next health check to re-probe."""
        client = StrawberryHubClient("http://localhost:8000", "token")
        client._last_available = True  # noqa: SLF001
        client._last_check_time = 123.0  # noqa: SLF001

        client.invalidate_cache()

        self.assertIsNone(client._last_available)  # noqa: SLF001
        self.assertEqual(client._last_check_time, 0.0)  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
