"""Strawberry AI conversation agent entity.

Architecture:
- **Hub online (passthrough)**: User messages are forwarded to the Strawberry
  Hub, which manages the full agent loop including ALL tool calls (HA Assist
  tools are exposed to the Hub via MCP).  The HA integration only records
  the Hub's SSE events (tool calls, results, final text) into the ChatLog.
- **Hub offline (local agent loop)**: TensorZero calls the LLM directly.
  If the LLM requests HA Assist tools, those are executed locally via
  ``chat_log.async_add_assistant_content()``.  Tool results are fed back
  to TensorZero for the next LLM iteration.
"""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components import conversation
from homeassistant.components.conversation import ChatLog
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    CONF_OFFLINE_API_KEY,
    CONF_OFFLINE_MODEL,
    CONF_OFFLINE_PROVIDER,
    CONF_OLLAMA_URL,
    CONF_PROMPT,
    DOMAIN,
    MAX_TOOL_ITERATIONS,
)
from .hub_client import HubAuthError, HubConnectionError, StrawberryHubClient
from .local_agent import (
    fallback_providers_from_options,
    offline_backend_from_options,
    provider_key_map_from_options,
    provider_model_map_from_options,
    run_local_agent_loop,
    tensorzero_function_name_from_options,
)

_LOGGER = logging.getLogger(__name__)

# Error shown to user when something goes wrong
ERROR_GETTING_RESPONSE = (
    "Sorry, I had a problem getting a response. Please try again."
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Strawberry conversation entities from config subentries.

    Each 'conversation' subentry creates one StrawberryConversationEntity.

    Args:
        hass: Home Assistant instance.
        config_entry: The parent config entry.
        async_add_entities: Callback to register new entities.
    """
    entities: list[StrawberryConversationEntity] = []
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type == "conversation":
            entities.append(
                StrawberryConversationEntity(config_entry, subentry)
            )
    async_add_entities(entities)


class StrawberryConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
):
    """Strawberry AI conversation agent.

    When the Hub is reachable, proxies messages to the Hub's agent loop
    (which has access to all connected Spokes' skills). When the Hub is
    offline, runs a local agent loop using TensorZero embedded gateway
    with HA's native Assist API tools.
    """

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True

    def __init__(
        self,
        config_entry: ConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the conversation entity.

        Args:
            config_entry: Parent config entry (holds Hub client).
            subentry: Conversation subentry with options (prompt, LLM API, etc.).
        """
        self._config_entry = config_entry
        self._subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = None  # TODO: link to HA device if needed

    @property
    def supported_languages(self) -> list[str] | str:
        """Return supported languages (all languages supported)."""
        return MATCH_ALL

    @property
    def _hub_client(self) -> StrawberryHubClient:
        """Get the Hub client from the config entry's runtime data."""
        return self._config_entry.runtime_data

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: ChatLog,
    ) -> conversation.ConversationResult:
        """Handle a conversation message.

        1. Provides LLM data (Assist API tools) to the chat log.
        2. Tries the Hub first (online mode).
        3. Falls back to local agent loop if Hub is unreachable.

        Args:
            user_input: The user's conversation input.
            chat_log: The conversation's chat log for history and tool execution.

        Returns:
            ConversationResult with the assistant's response.
        """
        # Truncate chat_log to only the current session's turn
        # Assist by default persists the same chat_log across all conversations
        # For now, we want each Assist session to be independent.
        # We keep the SystemContent if it's already there, and the latest UserContent.
        system_content = next((c for c in chat_log.content if isinstance(c, conversation.SystemContent)), None)
        latest_user_content = next((c for c in reversed(chat_log.content) if isinstance(c, conversation.UserContent)), None)
        
        chat_log.content.clear()
        if system_content:
            chat_log.content.append(system_content)
        if latest_user_content:
            chat_log.content.append(latest_user_content)

        options = self._subentry.data

        # Provide HA LLM data (system prompt, Assist API tools, exposed entities)
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        # Try Hub first, fall back to local on connection failure
        try:
            available = await self._hub_client.health_check()
            if available:
                _LOGGER.debug("Hub online — using Hub agent loop")
                await self._handle_via_hub(user_input, chat_log)
            else:
                _LOGGER.debug("Hub offline — using local agent loop")
                await self._handle_locally(chat_log)
        except (HubConnectionError, HubAuthError) as err:
            _LOGGER.warning("Hub error, falling back to local: %s", err)
            self._hub_client.invalidate_cache()
            
            # Trigger a background coordinator update so the binary sensor updates immediately
            if getattr(self, "hass", None) and (coordinator := self.hass.data.get("strawberry_conversation", {}).get(self._config_entry.entry_id, {}).get("coordinator")):
                self.hass.async_create_task(coordinator.async_request_refresh())
                
            try:
                await self._handle_locally(chat_log)
            except Exception:
                _LOGGER.exception("Local agent loop also failed")
                self._add_error_response(chat_log)
        except Exception:
            _LOGGER.exception(
                "Unexpected error in conversation handler — both Hub and "
                "local agent paths failed"
            )
            self._add_error_response(chat_log)

        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def _handle_via_hub(
        self,
        user_input: conversation.ConversationInput,
        chat_log: ChatLog,
    ) -> None:
        """Route the conversation through the Strawberry Hub (passthrough).

        The Hub owns the full agent loop and executes ALL tools — including
        HA Assist tools exposed via MCP.  This method only streams the Hub's
        SSE events and records them in the ChatLog so the HA conversation
        UI can display them.  No local tool execution occurs here.

        Args:
            user_input: The user's conversation input.
            chat_log: The chat log to populate with the response.
        """
        messages = _chat_log_to_messages(chat_log)

        final_content = ""
        # Accumulate tool calls between tool_call_started and tool_call_result
        pending_tool_calls: list[dict[str, Any]] = []

        async for event in self._hub_client.chat_stream(
            messages=messages,
            enable_tools=True,
        ):
            event_type = event.get("type", "")

            if event_type == "content_delta":
                delta = event.get("delta", "")
                if delta:
                    final_content += delta

            elif event_type == "tool_call_started":
                pending_tool_calls.append(event)

            elif event_type == "tool_call_result":
                self._record_hub_tool_result(chat_log, event, pending_tool_calls)

            elif event_type == "assistant_message":
                content = event.get("content", "")
                if content:
                    final_content = content

            elif event_type == "error":
                error_msg = event.get("error", "Hub returned an error")
                _LOGGER.error("Hub stream error: %s", error_msg)
                raise HubConnectionError(error_msg)

            elif event_type == "done":
                break

        # Add the final assistant response to the chat log
        if final_content.strip():
            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(
                    agent_id=self.entity_id,
                    content=final_content,
                )
            )
        else:
            _LOGGER.warning("Hub returned empty response")
            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(
                    agent_id=self.entity_id,
                    content=ERROR_GETTING_RESPONSE,
                )
            )

    def _record_hub_tool_result(
        self,
        chat_log: ChatLog,
        result_event: dict[str, Any],
        pending_tool_calls: list[dict[str, Any]],
    ) -> None:
        """Record a Hub-executed tool call and its result in the ChatLog.

        Finds the matching ``tool_call_started`` event, writes an
        ``AssistantContent`` with the external tool call, then appends the
        ``ToolResultContent``.  This gives HA's conversation UI visibility
        into what the Hub's agent loop did.

        Args:
            chat_log: The chat log to append to.
            result_event: The ``tool_call_result`` SSE event dict.
            pending_tool_calls: Accumulated ``tool_call_started`` events.
        """
        tool_call_id = result_event.get("tool_call_id", "")
        tool_name = result_event.get("tool_name", "")

        # Pop the matching started event (if any)
        started_event: dict[str, Any] | None = None
        for i, tc in enumerate(pending_tool_calls):
            if tc.get("tool_call_id") == tool_call_id:
                started_event = pending_tool_calls.pop(i)
                break

        tool_args = (started_event or {}).get("arguments", {})

        # Record the assistant's tool call as an external call
        tool_input = llm.ToolInput(
            tool_name=tool_name,
            tool_args=tool_args if isinstance(tool_args, dict) else {},
            id=tool_call_id,
            external=True,
        )
        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=self.entity_id,
                content=None,
                tool_calls=[tool_input],
            )
        )

        # Record the tool result
        success = result_event.get("success", False)
        if success:
            tool_result = {"result": result_event.get("result", "")}
        else:
            tool_result = {"error": result_event.get("error", "unknown error")}

        chat_log.async_add_assistant_content_without_tools(
            conversation.ToolResultContent(
                agent_id=self.entity_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_result=tool_result,
            )
        )

    def _add_error_response(self, chat_log: ChatLog) -> None:
        """Add an error response to the chat log.

        Used when both Hub and local agent fail, so the user sees
        a meaningful message instead of an exception.

        Args:
            chat_log: The chat log to append the error to.
        """
        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=self.entity_id,
                content=ERROR_GETTING_RESPONSE,
            )
        )

    async def _handle_locally(self, chat_log: ChatLog) -> None:
        """Run a local agent loop when the Hub is offline.

        Uses TensorZero to call the LLM directly.  If the LLM requests
        HA Assist tools, those are executed locally via the ChatLog's
        built-in tool pipeline.  Tool results are converted to
        TensorZero-compatible messages for the next LLM iteration.

        Only used when the Hub is unreachable — the Hub is the primary
        owner of all tool execution (including HA tools via MCP).

        Args:
            chat_log: The chat log with LLM data already provided.
        """
        options = self._subentry.data
        local_response = await run_local_agent_loop(
            chat_log=chat_log,
            api_instance=chat_log.llm_api,
            system_prompt=options.get(CONF_PROMPT, ""),
            agent_id=self.entity_id,
            offline_provider=options.get(CONF_OFFLINE_PROVIDER, "none"),
            offline_api_key=options.get(CONF_OFFLINE_API_KEY),
            offline_model=options.get(CONF_OFFLINE_MODEL),
            ollama_url=options.get(CONF_OLLAMA_URL),
            fallback_providers=fallback_providers_from_options(options),
            provider_api_keys=provider_key_map_from_options(options),
            provider_models=provider_model_map_from_options(options),
            offline_backend=offline_backend_from_options(options),
            tensorzero_function_name=tensorzero_function_name_from_options(options),
            max_iterations=MAX_TOOL_ITERATIONS,
        )

        if local_response:
            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(
                    agent_id=self.entity_id,
                    content=local_response,
                )
            )
            return

        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=self.entity_id,
                content=(
                    "I'm currently unable to reach the Strawberry Hub. "
                    "Local offline mode could not complete your request. "
                    "Please verify offline provider settings and try again."
                ),
            )
        )


def _chat_log_to_messages(chat_log: ChatLog) -> list[dict[str, str]]:
    """Convert a HA ChatLog to a list of message dicts for the Hub API.

    Maps HA's content types to OpenAI-compatible message roles:
    - SystemContent → system
    - UserContent → user
    - AssistantContent → assistant
    - ToolResultContent → tool

    Args:
        chat_log: The HA conversation ChatLog.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    messages: list[dict[str, str]] = []

    for content in chat_log.content:
        if isinstance(content, conversation.SystemContent):
            messages.append({
                "role": "system",
                "content": content.content or "",
            })
        elif isinstance(content, conversation.UserContent):
            messages.append({
                "role": "user",
                "content": content.content or "",
            })
        elif isinstance(content, conversation.AssistantContent):
            if content.content:
                messages.append({
                    "role": "assistant",
                    "content": content.content,
                })
        elif isinstance(content, conversation.ToolResultContent):
            messages.append({
                "role": "tool",
                "content": str(content.tool_result) if content.tool_result else "",
            })

    return messages
