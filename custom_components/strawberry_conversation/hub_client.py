"""Lightweight async HTTP client for the Strawberry Hub API.

Handles:
- Health checks (fast probe for offline detection)
- Streaming chat completions (SSE)
- Hub availability caching
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

import httpx

from .const import (
    HUB_CONNECT_TIMEOUT,
    HUB_HEALTH_TIMEOUT,
    HUB_READ_TIMEOUT,
    OFFLINE_CACHE_TTL,
    ONLINE_CACHE_TTL,
)

logger = logging.getLogger(__name__)


class StrawberryHubClient:
    """Async HTTP client for the Strawberry Hub.

    Provides streaming chat completions and health checking with
    availability caching for fast offline detection.
    """

    def __init__(self, hub_url: str, token: str) -> None:
        """Initialize the Hub client.

        Args:
            hub_url: Base URL of the Strawberry Hub (e.g., http://192.168.1.100:8000).
            token: JWT device token for authentication.
        """
        self._hub_url = hub_url.rstrip("/")
        self._token = token
        self._client: httpx.AsyncClient | None = None

        # Availability cache
        self._last_check_time: float = 0.0
        self._last_available: bool | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the httpx async client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._hub_url,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(
                    connect=HUB_CONNECT_TIMEOUT,
                    read=HUB_READ_TIMEOUT,
                    write=10.0,
                    pool=5.0,
                ),
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if the Hub is reachable.

        Uses a fast timeout and caches the result to avoid
        repeated probes on every conversation turn.

        Returns:
            True if Hub is reachable, False otherwise.
        """
        now = time.monotonic()
        if self._last_available is not None:
            ttl = ONLINE_CACHE_TTL if self._last_available else OFFLINE_CACHE_TTL
            if (now - self._last_check_time) < ttl:
                return self._last_available

        try:
            client = self._get_client()
            response = await client.get(
                "/health",
                timeout=httpx.Timeout(
                    connect=HUB_HEALTH_TIMEOUT,
                    read=HUB_HEALTH_TIMEOUT,
                    write=HUB_HEALTH_TIMEOUT,
                    pool=HUB_HEALTH_TIMEOUT,
                ),
            )
            available = response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            available = False
        except Exception:
            logger.exception("Unexpected error during Hub health check")
            available = False

        self._last_check_time = now
        self._last_available = available

        if not available:
            logger.debug("Hub health check failed — marking offline")
        return available

    def invalidate_cache(self) -> None:
        """Force the next health_check to re-probe the Hub."""
        self._last_check_time = 0.0
        self._last_available = None

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        enable_tools: bool = True,
        session_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream chat completions from the Hub via SSE.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            enable_tools: Whether Hub should run its agent loop.
            session_id: Optional Hub session ID for continuity.

        Yields:
            Parsed SSE event dicts (content_delta, tool_call_started,
            tool_call_result, assistant_message, error, done).

        Raises:
            HubConnectionError: If the Hub is unreachable.
            HubAuthError: If authentication fails.
        """
        payload: dict[str, Any] = {
            "messages": messages,
            "enable_tools": enable_tools,
            "stream": True,
        }
        if session_id:
            payload["session_id"] = session_id

        client = self._get_client()

        try:
            async with client.stream(
                "POST",
                "/api/v1/chat/completions",
                json=payload,
            ) as response:
                if response.status_code == 401:
                    self._last_available = False
                    raise HubAuthError("Hub authentication failed (401)")

                if response.status_code != 200:
                    text = await response.aread()
                    raise HubConnectionError(
                        f"Hub returned status {response.status_code}: "
                        f"{text.decode(errors='replace')[:200]}"
                    )

                # Mark Hub as available on successful connection
                self._last_check_time = time.monotonic()
                self._last_available = True

                # Parse SSE stream
                async for event in self._parse_sse(response):
                    yield event

        except (httpx.ConnectError, httpx.TimeoutException, OSError) as err:
            self._last_available = False
            self._last_check_time = time.monotonic()
            raise HubConnectionError(f"Hub unreachable: {err}") from err

    async def chat(
        self,
        messages: list[dict[str, str]],
        enable_tools: bool = True,
    ) -> dict[str, Any]:
        """Non-streaming chat completion from the Hub.

        Args:
            messages: List of message dicts.
            enable_tools: Whether Hub should run its agent loop.

        Returns:
            Parsed JSON response dict.

        Raises:
            HubConnectionError: If the Hub is unreachable.
            HubAuthError: If authentication fails.
        """
        payload = {
            "messages": messages,
            "enable_tools": enable_tools,
            "stream": False,
        }

        client = self._get_client()

        try:
            response = await client.post(
                "/api/v1/chat/completions",
                json=payload,
            )
        except (httpx.ConnectError, httpx.TimeoutException, OSError) as err:
            self._last_available = False
            self._last_check_time = time.monotonic()
            raise HubConnectionError(f"Hub unreachable: {err}") from err

        if response.status_code == 401:
            raise HubAuthError("Hub authentication failed (401)")

        if response.status_code != 200:
            raise HubConnectionError(
                f"Hub returned status {response.status_code}: "
                f"{response.text[:200]}"
            )

        self._last_check_time = time.monotonic()
        self._last_available = True
        return response.json()

    @staticmethod
    async def _parse_sse(
        response: httpx.Response,
    ) -> AsyncIterator[dict[str, Any]]:
        """Parse Server-Sent Events from an httpx streaming response.

        Yields:
            Parsed event dicts. Yields a final {"type": "done"} when
            the stream ends or receives [DONE].
        """
        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line:
                    continue

                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        yield {"type": "done"}
                        return

                    try:
                        event = json.loads(data)
                        yield event
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse SSE data: %s", data[:100])

        # Stream ended without [DONE]
        yield {"type": "done"}


class HubConnectionError(Exception):
    """Raised when the Hub is unreachable."""


class HubAuthError(Exception):
    """Raised when Hub authentication fails."""
