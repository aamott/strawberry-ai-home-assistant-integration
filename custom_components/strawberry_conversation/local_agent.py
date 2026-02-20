"""Local agent loop for offline fallback (Phase 2).

This module runs a local tool-capable chat loop when the Hub is offline.
It uses an OpenAI-compatible chat-completions interface (OpenAI, Gemini's
compat endpoint, or Ollama) and executes Home Assistant Assist tools through
``chat_log.async_add_assistant_content``.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import httpx
from homeassistant.components import conversation
from homeassistant.helpers import llm

from .const import (
    DEFAULT_MODELS,
    DEFAULT_OLLAMA_URL,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE,
    PROVIDER_NONE,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
)

if TYPE_CHECKING:
    from homeassistant.components.conversation import ChatLog
    from homeassistant.helpers.llm import APIInstance
    from homeassistant.helpers.llm import Tool

_OPENAI_COMPAT_CONNECT_TIMEOUT = 5.0
_OPENAI_COMPAT_READ_TIMEOUT = 60.0
_OPENAI_COMPAT_WRITE_TIMEOUT = 20.0
_OPENAI_COMPAT_POOL_TIMEOUT = 5.0

type RequestCompletionCallable = Callable[
    [
        str,
        str,
        str | None,
        str | None,
        list[dict[str, Any]],
        list[dict[str, Any]],
    ],
    Awaitable[dict[str, Any]],
]

logger = logging.getLogger(__name__)


def _build_provider_request_context(
    provider: str,
    api_key: str | None,
    model: str | None,
    ollama_url: str | None,
) -> tuple[str, dict[str, str], str]:
    """Build provider URL, auth headers, and model for chat completions.

    Args:
        provider: Selected offline provider.
        api_key: Provider API key when required.
        model: Requested model name.
        ollama_url: User-provided Ollama URL.

    Returns:
        Tuple of (endpoint_url, headers, resolved_model).
    """
    resolved_model = model or DEFAULT_MODELS.get(provider, "")
    headers: dict[str, str] = {"Content-Type": "application/json"}

    if provider == PROVIDER_OPENAI:
        if not api_key:
            raise ValueError("Offline OpenAI provider requires an API key")
        headers["Authorization"] = f"Bearer {api_key}"
        return "https://api.openai.com/v1/chat/completions", headers, resolved_model

    if provider == PROVIDER_GOOGLE:
        if not api_key:
            raise ValueError("Offline Google provider requires an API key")
        headers["Authorization"] = f"Bearer {api_key}"
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            headers,
            resolved_model,
        )

    if provider == PROVIDER_OLLAMA:
        base_url = (ollama_url or DEFAULT_OLLAMA_URL).rstrip("/")
        return f"{base_url}/chat/completions", headers, resolved_model

    raise ValueError(f"Unsupported offline provider: {provider}")


def _tool_to_openai_schema(tool: Tool) -> dict[str, Any]:
    """Convert a Home Assistant tool to OpenAI tool schema format."""
    try:
        from voluptuous_openapi import convert

        parameters = convert(tool.parameters)
    except Exception:
        logger.debug("Falling back to empty schema for tool %s", tool.name)
        parameters = {"type": "object", "properties": {}}

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": parameters,
        },
    }


def _chat_log_to_model_messages(chat_log: ChatLog) -> list[dict[str, Any]]:
    """Convert HA chat content into OpenAI-compatible message format."""
    messages: list[dict[str, Any]] = []

    for content in chat_log.content:
        if isinstance(content, conversation.SystemContent):
            messages.append({"role": "system", "content": content.content or ""})
            continue

        if isinstance(content, conversation.UserContent):
            messages.append({"role": "user", "content": content.content or ""})
            continue

        if isinstance(content, conversation.AssistantContent):
            message: dict[str, Any] = {
                "role": "assistant",
                "content": content.content or "",
            }
            if content.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.tool_name,
                            "arguments": json.dumps(tool_call.tool_args),
                        },
                    }
                    for tool_call in content.tool_calls
                ]
            messages.append(message)
            continue

        if isinstance(content, conversation.ToolResultContent):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": content.tool_call_id,
                    "name": content.tool_name,
                    "content": json.dumps(content.tool_result),
                }
            )

    return messages


async def _request_openai_compatible_completion(
    provider: str,
    api_key: str | None,
    model: str | None,
    ollama_url: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat completion endpoint."""
    endpoint, headers, resolved_model = _build_provider_request_context(
        provider,
        api_key,
        model,
        ollama_url,
    )

    payload: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "temperature": 0.2,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    timeout = httpx.Timeout(
        connect=_OPENAI_COMPAT_CONNECT_TIMEOUT,
        read=_OPENAI_COMPAT_READ_TIMEOUT,
        write=_OPENAI_COMPAT_WRITE_TIMEOUT,
        pool=_OPENAI_COMPAT_POOL_TIMEOUT,
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


def _extract_openai_message(response: dict[str, Any]) -> dict[str, Any]:
    """Extract first message from OpenAI-compatible response payload."""
    choices = response.get("choices")
    if not choices:
        raise ValueError("No choices returned by offline model")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("Offline model response missing message object")
    return message


def _parse_tool_call_arguments(raw_arguments: str) -> dict[str, Any]:
    """Parse tool call argument JSON safely."""
    if not raw_arguments:
        return {}
    try:
        decoded = json.loads(raw_arguments)
    except json.JSONDecodeError:
        logger.warning("Invalid tool argument JSON from offline model: %s", raw_arguments)
        return {}
    return decoded if isinstance(decoded, dict) else {}


async def run_local_agent_loop(
    chat_log: ChatLog,
    api_instance: APIInstance | None,
    system_prompt: str,
    agent_id: str,
    offline_provider: str,
    offline_api_key: str | None,
    offline_model: str | None,
    ollama_url: str | None,
    max_iterations: int = 10,
    request_completion: RequestCompletionCallable | None = None,
) -> str | None:
    """Run a local agent loop with HA Assist tools.

    The loop repeatedly calls the selected local/remote model endpoint. If the
    model requests tools, those tool calls are executed via Home Assistant's
    native Assist tool pipeline (``chat_log.async_add_assistant_content``),
    then the loop continues with tool results in context.

    Args:
        chat_log: HA ChatLog with conversation history.
        api_instance: HA LLM APIInstance with Assist tools.
        system_prompt: System prompt for the LLM.
        agent_id: Entity ID of the conversation agent.
        offline_provider: Offline provider selected in config flow.
        offline_api_key: API key for provider when required.
        offline_model: Optional model override.
        ollama_url: Optional Ollama OpenAI-compatible base URL.
        max_iterations: Maximum tool call iterations.
        request_completion: Injectable completion function for tests.

    Returns:
        Final response content, or ``None`` if local mode cannot respond.
    """
    if offline_provider == PROVIDER_NONE:
        logger.debug("Offline provider is disabled")
        return None

    if offline_provider == PROVIDER_ANTHROPIC:
        logger.warning(
            "Anthropic offline provider is not yet implemented in the local adapter"
        )
        return None

    if api_instance is None:
        logger.warning("No Assist API instance available for local tool execution")
        return None

    if request_completion is None:
        request_completion = _request_openai_compatible_completion

    if system_prompt and (
        not chat_log.content
        or not isinstance(chat_log.content[0], conversation.SystemContent)
    ):
        chat_log.content.insert(0, conversation.SystemContent(content=system_prompt))

    tools = [_tool_to_openai_schema(tool) for tool in api_instance.tools]

    for iteration in range(max_iterations):
        logger.debug("Local loop iteration %s", iteration + 1)
        messages = _chat_log_to_model_messages(chat_log)

        try:
            raw_response = await request_completion(
                offline_provider,
                offline_api_key,
                offline_model,
                ollama_url,
                messages,
                tools,
            )
        except Exception:
            logger.exception("Offline provider request failed")
            return None

        try:
            message = _extract_openai_message(raw_response)
        except ValueError:
            logger.exception("Offline provider response was malformed")
            return None

        response_content = message.get("content") or ""
        raw_tool_calls = message.get("tool_calls") or []

        if raw_tool_calls:
            tool_inputs: list[llm.ToolInput] = []
            for raw_tool_call in raw_tool_calls:
                function = raw_tool_call.get("function", {})
                tool_name = function.get("name", "")
                if not tool_name:
                    continue

                tool_inputs.append(
                    llm.ToolInput(
                        tool_name=tool_name,
                        tool_args=_parse_tool_call_arguments(
                            function.get("arguments", "")
                        ),
                        id=raw_tool_call.get("id") or llm.ulid_now(),
                    )
                )

            if not tool_inputs and response_content:
                return response_content

            assistant_content = conversation.AssistantContent(
                agent_id=agent_id,
                content=response_content or None,
                tool_calls=tool_inputs,
            )

            async for _ in chat_log.async_add_assistant_content(assistant_content):
                # The generator handles tool execution and appends results to chat_log.
                # We only consume it here so the loop can continue with new context.
                pass
            continue

        if response_content.strip():
            return response_content

        logger.debug("Offline provider returned an empty assistant message")
        return None

    logger.warning("Local loop reached max iterations (%s)", max_iterations)
    return None
