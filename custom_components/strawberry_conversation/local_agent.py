"""Local agent loop for offline fallback (Phase 2).

This module runs a local tool-capable chat loop when the Hub is offline.
When the ``tensorzero`` package is available, it dynamically builds an
embedded TensorZero gateway from HA UI settings and routes all cloud
providers (OpenAI, Google, Anthropic) through it.  Falls back to direct
httpx calls only when TensorZero is unavailable or for Ollama-only chains.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import httpx
from homeassistant.components import conversation
from homeassistant.helpers import llm

try:
    from tensorzero import AsyncTensorZeroGateway
except Exception:  # pragma: no cover
    AsyncTensorZeroGateway = None  # type: ignore[assignment]

from .const import (
    CONF_OFFLINE_ANTHROPIC_API_KEY,
    CONF_OFFLINE_ANTHROPIC_MODEL,
    CONF_OFFLINE_BACKEND,
    CONF_OFFLINE_FALLBACK_PROVIDERS,
    CONF_OFFLINE_GOOGLE_API_KEY,
    CONF_OFFLINE_GOOGLE_MODEL,
    CONF_OFFLINE_OLLAMA_MODEL,
    CONF_OFFLINE_OPENAI_API_KEY,
    CONF_OFFLINE_OPENAI_MODEL,
    CONF_TENSORZERO_FUNCTION_NAME,
    DEFAULT_MODELS,
    DEFAULT_OLLAMA_URL,
    OFFLINE_BACKEND_AUTO,
    OFFLINE_BACKEND_OPENAI_COMPAT,
    OFFLINE_BACKEND_TENSORZERO,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE,
    PROVIDER_NONE,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    _OPENAI_COMPAT_CONNECT_TIMEOUT,
    _OPENAI_COMPAT_READ_TIMEOUT,
    _OPENAI_COMPAT_WRITE_TIMEOUT,
    _OPENAI_COMPAT_POOL_TIMEOUT,
)
from .tz_config import (
    build_credentials,
    effective_provider_chain as _effective_tz_chain,
    get_or_build_gateway,
)

if TYPE_CHECKING:
    from homeassistant.components.conversation import ChatLog
    from homeassistant.helpers.llm import APIInstance
    from homeassistant.helpers.llm import Tool

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


def _simplify_messages_for_tensorzero(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI-format tool messages to TensorZero-compatible format.

    TensorZero only accepts ``{role, content}`` fields on each message.
    After HA locally executes an Assist tool (offline mode), the chat log
    contains OpenAI-style messages with ``role: 'tool'``, ``tool_call_id``,
    and ``tool_calls`` arrays on assistant messages — none of which TZ
    understands.  This converter mirrors the spoke's approach
    (``agent_runner.py``) of folding tool results into plain user messages.

    Args:
        messages: Messages in OpenAI chat-completion format.

    Returns:
        Messages using only ``role`` and ``content`` — safe for TensorZero.
    """
    simplified: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "")

        if role == "tool":
            # Convert tool-result to a user message with a descriptive prefix
            tool_name = msg.get("name", "unknown_tool")
            content = msg.get("content", "")
            simplified.append({
                "role": "user",
                "content": (
                    f"[Tool Result: {tool_name}]\n{content}\n\n"
                    "[Now respond naturally to the user based on this result. "
                    "Do not rerun the same tool call again unless the user asks.]"
                ),
            })
            continue

        if role == "assistant":
            # Strip tool_calls — keep only role + content.
            # If content is empty but tool_calls exist, synthesize a
            # description so the LLM knows what action was taken.
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls")
            if not content and tool_calls:
                summaries: list[str] = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown")
                    args = fn.get("arguments", "")
                    summaries.append(f"{name}({args})")
                content = "[Called " + ", ".join(summaries) + "]"
            simplified.append({"role": "assistant", "content": content})
            continue

        # system, user, etc. — pass through unchanged
        simplified.append({"role": role, "content": msg.get("content") or ""})

    return simplified


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


def _extract_text_from_tz_block(block: Any) -> str:
    """Extract text from a TensorZero block object or dict."""
    if hasattr(block, "text"):
        text = getattr(block, "text", "")
        return str(text) if text else ""
    if isinstance(block, dict) and block.get("type") == "text":
        text = block.get("text", "")
        return str(text) if text else ""
    return ""


def _extract_tool_call_from_tz_block(block: Any) -> dict[str, Any] | None:
    """Extract a tool_call dict from a TensorZero block object or dict."""
    if hasattr(block, "type") and getattr(block, "type", None) == "tool_call":
        name = getattr(block, "name", None) or getattr(block, "raw_name", None)
        arguments: Any = getattr(block, "arguments", None)
        if not isinstance(arguments, dict):
            raw_arguments = getattr(block, "raw_arguments", "")
            arguments = _parse_tool_call_arguments(raw_arguments)
        if not name:
            return None
        return {
            "id": str(getattr(block, "id", "") or llm.ulid_now()),
            "type": "function",
            "function": {
                "name": str(name),
                "arguments": json.dumps(arguments),
            },
        }

    if isinstance(block, dict) and block.get("type") == "tool_call":
        name = block.get("name") or block.get("raw_name")
        if not name:
            return None
        arguments = block.get("arguments")
        if not isinstance(arguments, dict):
            arguments = _parse_tool_call_arguments(block.get("raw_arguments", ""))
        return {
            "id": str(block.get("id") or llm.ulid_now()),
            "type": "function",
            "function": {
                "name": str(name),
                "arguments": json.dumps(arguments),
            },
        }

    return None


def _extract_message_from_tz_response(response: Any) -> dict[str, Any]:
    """Convert TensorZero response object/dict to OpenAI-like message dict.

    Handles unexpected response shapes (None, strings, missing keys)
    gracefully by falling back to an empty assistant message.
    """
    try:
        if isinstance(response, dict):
            content_blocks = response.get("content") or []
        elif hasattr(response, "content"):
            content_blocks = getattr(response, "content", []) or []
        else:
            content_blocks = []

        # Guard against non-iterable content (e.g. a plain string)
        if not isinstance(content_blocks, list):
            logger.warning(
                "TZ response 'content' is not a list (got %s); "
                "treating as plain text",
                type(content_blocks).__name__,
            )
            return {
                "role": "assistant",
                "content": str(content_blocks) if content_blocks else "",
            }

        content = ""
        tool_calls: list[dict[str, Any]] = []
        for block in content_blocks:
            content += _extract_text_from_tz_block(block)
            tool_call = _extract_tool_call_from_tz_block(block)
            if tool_call:
                tool_calls.append(tool_call)

        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    except Exception:
        logger.exception(
            "Failed to parse TZ response; returning empty assistant message"
        )
        return {"role": "assistant", "content": ""}


async def _request_tensorzero_completion(
    gateway: AsyncTensorZeroGateway,
    messages: list[dict[str, Any]],
    system_prompt: str,
    function_name: str,
    credentials: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Call TensorZero embedded gateway and normalize to OpenAI-like shape.

    TensorZero only accepts ``{role, content}`` fields, so we convert
    any OpenAI-format tool messages (``role: 'tool'``, ``tool_calls``)
    to simplified user/assistant messages before calling the gateway.

    Args:
        gateway: Initialized ``AsyncTensorZeroGateway``.
        messages: Chat messages in OpenAI format.
        system_prompt: System prompt text.
        function_name: TensorZero function name.
        credentials: Dynamic API key credentials.

    Returns:
        OpenAI-compatible response dict with ``choices[0].message``.
    """
    # Convert tool messages to TZ-compatible format, then strip system role
    tz_messages = _simplify_messages_for_tensorzero(messages)
    tz_input: dict[str, Any] = {
        "messages": [m for m in tz_messages if m.get("role") != "system"]
    }
    if system_prompt:
        tz_input["system"] = system_prompt

    kwargs: dict[str, Any] = {
        "function_name": function_name,
        "input": tz_input,
    }
    if credentials:
        kwargs["credentials"] = credentials

    response = await gateway.inference(**kwargs)
    return {
        "choices": [
            {
                "message": _extract_message_from_tz_response(response),
            }
        ]
    }


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


def _provider_chain(
    primary: str,
    fallbacks: list[str] | None,
) -> list[str]:
    """Build ordered provider chain with duplicates and disabled entries removed."""
    chain: list[str] = []

    for provider in [primary, *(fallbacks or [])]:
        if provider in ("", PROVIDER_NONE):
            continue
        if provider not in chain:
            chain.append(provider)

    return chain


def _resolve_provider_api_key(
    provider: str,
    legacy_api_key: str | None,
    provider_api_keys: dict[str, str | None] | None,
) -> str | None:
    """Resolve API key for provider, preserving legacy single-key behavior."""
    if provider_api_keys and provider in provider_api_keys:
        return provider_api_keys[provider]
    return legacy_api_key


def _resolve_provider_model(
    provider: str,
    legacy_model: str | None,
    provider_models: dict[str, str | None] | None,
) -> str | None:
    """Resolve model for provider, preserving legacy single-model behavior."""
    if provider_models and provider in provider_models:
        return provider_models[provider]
    return legacy_model


async def _request_with_provider_fallback(
    provider_chain: list[str],
    legacy_api_key: str | None,
    legacy_model: str | None,
    provider_api_keys: dict[str, str | None] | None,
    provider_models: dict[str, str | None] | None,
    ollama_url: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    request_completion: RequestCompletionCallable,
) -> dict[str, Any] | None:
    """Try providers in order until one returns a valid response."""
    for provider in provider_chain:
        provider_api_key = _resolve_provider_api_key(
            provider,
            legacy_api_key,
            provider_api_keys,
        )
        provider_model = _resolve_provider_model(
            provider,
            legacy_model,
            provider_models,
        )

        try:
            return await request_completion(
                provider,
                provider_api_key,
                provider_model,
                ollama_url,
                messages,
                tools,
            )
        except Exception:
            logger.exception("Offline provider %s request failed", provider)

    return None


def _should_use_tensorzero(offline_backend: str) -> bool:
    """Decide whether to use the TensorZero dynamic gateway.

    Args:
        offline_backend: User-selected backend (auto, openai_compat, tensorzero).

    Returns:
        True if TZ should be used for this request.
    """
    if offline_backend == OFFLINE_BACKEND_OPENAI_COMPAT:
        return False
    if offline_backend == OFFLINE_BACKEND_TENSORZERO:
        if AsyncTensorZeroGateway is None:
            logger.warning(
                "TensorZero backend selected but tensorzero package is unavailable; "
                "falling back to OpenAI-compatible HTTP path"
            )
            return False
        return True
    # auto: prefer TZ when installed
    return AsyncTensorZeroGateway is not None


def _build_full_key_map(
    provider_chain: list[str],
    legacy_api_key: str | None,
    provider_api_keys: dict[str, str | None] | None,
) -> dict[str, str | None]:
    """Build a complete API-key map for every provider in the chain.

    Merges per-provider keys with the legacy shared key fallback.
    """
    result: dict[str, str | None] = {}
    for provider in provider_chain:
        result[provider] = _resolve_provider_api_key(
            provider, legacy_api_key, provider_api_keys
        )
    return result


def _build_full_model_map(
    provider_chain: list[str],
    legacy_model: str | None,
    provider_models: dict[str, str | None] | None,
) -> dict[str, str]:
    """Build a complete model map for every provider in the chain.

    Falls back to ``DEFAULT_MODELS`` when no explicit model is configured.
    """
    result: dict[str, str] = {}
    for provider in provider_chain:
        resolved = _resolve_provider_model(provider, legacy_model, provider_models)
        result[provider] = resolved or DEFAULT_MODELS.get(provider, "")
    return result


async def run_local_agent_loop(
    chat_log: ChatLog,
    api_instance: APIInstance | None,
    system_prompt: str,
    agent_id: str,
    offline_provider: str,
    offline_api_key: str | None,
    offline_model: str | None,
    ollama_url: str | None,
    fallback_providers: list[str] | None = None,
    provider_api_keys: dict[str, str | None] | None = None,
    provider_models: dict[str, str | None] | None = None,
    offline_backend: str = OFFLINE_BACKEND_AUTO,
    tensorzero_function_name: str = "chat",
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
        fallback_providers: Ordered fallback providers after the primary provider.
        provider_api_keys: Per-provider API key mapping.
        provider_models: Per-provider model mapping.
        offline_backend: Backend for local mode (``auto``, ``openai_compat``, ``tensorzero``).
        tensorzero_function_name: TensorZero function name to call when enabled.
        max_iterations: Maximum tool call iterations.
        request_completion: Injectable completion function for tests.

    Returns:
        Final response content, or ``None`` if local mode cannot respond.
    """
    provider_chain = _provider_chain(offline_provider, fallback_providers)
    if not provider_chain:
        logger.debug("No offline providers configured")
        return None

    if api_instance is None:
        logger.warning("No Assist API instance available for local tool execution")
        return None

    if request_completion is None:
        request_completion = _request_openai_compatible_completion

    # Decide whether to use the TensorZero dynamic gateway.
    use_tensorzero = _should_use_tensorzero(offline_backend)

    if system_prompt and (
        not chat_log.content
        or not isinstance(chat_log.content[0], conversation.SystemContent)
    ):
        chat_log.content.insert(0, conversation.SystemContent(content=system_prompt))

    tools = [_tool_to_openai_schema(tool) for tool in api_instance.tools]

    # Pre-build TZ gateway + credentials once before the loop.
    tz_gateway = None
    tz_credentials: dict[str, str] | None = None
    if use_tensorzero:
        resolved_keys = _build_full_key_map(provider_chain, offline_api_key, provider_api_keys)
        resolved_models = _build_full_model_map(provider_chain, offline_model, provider_models)
        tz_chain = _effective_tz_chain(provider_chain, resolved_keys)
        if tz_chain:
            try:
                tz_gateway = await get_or_build_gateway(
                    provider_chain=tz_chain,
                    provider_models=resolved_models,
                    ollama_url=ollama_url,
                    tools=tools,
                    function_name=tensorzero_function_name,
                )
                tz_credentials = build_credentials(tz_chain, resolved_keys)
            except Exception:
                logger.exception(
                    "Failed to build TensorZero gateway; falling back to httpx"
                )
                tz_gateway = None
        else:
            logger.warning("No providers with valid credentials for TensorZero")

    for iteration in range(max_iterations):
        logger.debug("Local loop iteration %d / %d", iteration + 1, max_iterations)
        messages = _chat_log_to_model_messages(chat_log)

        if tz_gateway is not None:
            try:
                raw_response = await _request_tensorzero_completion(
                    gateway=tz_gateway,
                    messages=messages,
                    system_prompt=system_prompt,
                    function_name=tensorzero_function_name,
                    credentials=tz_credentials,
                )
            except Exception:
                logger.exception(
                    "TensorZero request failed on iteration %d; "
                    "falling back to httpx providers",
                    iteration + 1,
                )
                raw_response = await _request_with_provider_fallback(
                    provider_chain=provider_chain,
                    legacy_api_key=offline_api_key,
                    legacy_model=offline_model,
                    provider_api_keys=provider_api_keys,
                    provider_models=provider_models,
                    ollama_url=ollama_url,
                    messages=messages,
                    tools=tools,
                    request_completion=request_completion,
                )
        else:
            raw_response = await _request_with_provider_fallback(
                provider_chain=provider_chain,
                legacy_api_key=offline_api_key,
                legacy_model=offline_model,
                provider_api_keys=provider_api_keys,
                provider_models=provider_models,
                ollama_url=ollama_url,
                messages=messages,
                tools=tools,
                request_completion=request_completion,
            )
        if raw_response is None:
            logger.warning(
                "All offline providers failed on iteration %d "
                "(chain=%s). Cannot continue agent loop.",
                iteration + 1,
                provider_chain,
            )
            return None

        try:
            message = _extract_openai_message(raw_response)
        except ValueError:
            logger.exception(
                "Offline provider returned malformed response on iteration %d",
                iteration + 1,
            )
            return None

        response_content = message.get("content") or ""
        raw_tool_calls = message.get("tool_calls") or []

        if raw_tool_calls:
            tool_inputs: list[llm.ToolInput] = []
            for raw_tool_call in raw_tool_calls:
                function = raw_tool_call.get("function", {})
                tool_name = function.get("name", "")
                if not tool_name:
                    logger.warning(
                        "Skipping tool call with empty name: %s",
                        raw_tool_call,
                    )
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

            # LLM returned tool_calls but none had a valid name
            if not tool_inputs:
                if response_content:
                    return response_content
                logger.warning(
                    "LLM returned %d tool call(s) but none had a valid "
                    "tool_name. Skipping tool execution.",
                    len(raw_tool_calls),
                )
                # Fall through to the empty-content check below
            else:
                logger.debug(
                    "Executing %d tool call(s): %s",
                    len(tool_inputs),
                    [t.tool_name for t in tool_inputs],
                )
                assistant_content = conversation.AssistantContent(
                    agent_id=agent_id,
                    content=response_content or None,
                    tool_calls=tool_inputs,
                )

                async for _ in chat_log.async_add_assistant_content(
                    assistant_content
                ):
                    # HA executes the tools and appends results to chat_log.
                    pass
                continue

        # Final text response — return it
        if response_content.strip():
            return response_content

        logger.warning(
            "Offline LLM returned empty content on iteration %d "
            "(no tool calls, no text). Aborting agent loop.",
            iteration + 1,
        )
        return None

    logger.warning(
        "Local agent loop reached max iterations (%d) without a final "
        "text response.",
        max_iterations,
    )
    return None


def provider_key_map_from_options(options: dict[str, Any]) -> dict[str, str | None]:
    """Build per-provider API key mapping from config entry options."""
    return {
        PROVIDER_OPENAI: options.get(CONF_OFFLINE_OPENAI_API_KEY),
        PROVIDER_GOOGLE: options.get(CONF_OFFLINE_GOOGLE_API_KEY),
        PROVIDER_ANTHROPIC: options.get(CONF_OFFLINE_ANTHROPIC_API_KEY),
    }


def provider_model_map_from_options(options: dict[str, Any]) -> dict[str, str | None]:
    """Build per-provider model mapping from config entry options."""
    return {
        PROVIDER_OPENAI: options.get(CONF_OFFLINE_OPENAI_MODEL, DEFAULT_MODELS[PROVIDER_OPENAI]),
        PROVIDER_GOOGLE: options.get(CONF_OFFLINE_GOOGLE_MODEL, DEFAULT_MODELS[PROVIDER_GOOGLE]),
        PROVIDER_ANTHROPIC: options.get(
            CONF_OFFLINE_ANTHROPIC_MODEL, DEFAULT_MODELS[PROVIDER_ANTHROPIC]
        ),
        PROVIDER_OLLAMA: options.get(CONF_OFFLINE_OLLAMA_MODEL, DEFAULT_MODELS[PROVIDER_OLLAMA]),
    }


def fallback_providers_from_options(options: dict[str, Any]) -> list[str]:
    """Read fallback provider list from options with robust type handling."""
    raw = options.get(CONF_OFFLINE_FALLBACK_PROVIDERS, [])
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


def offline_backend_from_options(options: dict[str, Any]) -> str:
    """Read offline backend choice from options with robust fallback."""
    raw = options.get(CONF_OFFLINE_BACKEND, OFFLINE_BACKEND_AUTO)
    return str(raw) if raw else OFFLINE_BACKEND_AUTO


def tensorzero_function_name_from_options(options: dict[str, Any]) -> str:
    """Read TensorZero function name from options with default."""
    raw = options.get(CONF_TENSORZERO_FUNCTION_NAME, "chat")
    value = str(raw).strip() if raw is not None else ""
    return value or "chat"
