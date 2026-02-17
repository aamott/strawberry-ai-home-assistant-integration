"""Strawberry AI conversation agent entity.

Routes user messages to the Strawberry Hub when online, or falls back to a
local agent loop (TensorZero embedded + HA Assist API tools) when the Hub
is unreachable.
"""

from __future__ import annotations

import logging

from homeassistant.components import conversation
from homeassistant.components.conversation import ChatLog
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import CONF_PROMPT, DOMAIN
from .hub_client import HubAuthError, HubConnectionError, StrawberryHubClient

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
            if await self._hub_client.health_check():
                await self._handle_via_hub(user_input, chat_log)
            else:
                _LOGGER.debug("Hub offline — using local agent loop")
                await self._handle_locally(chat_log)
        except (HubConnectionError, HubAuthError) as err:
            _LOGGER.warning("Hub error, falling back to local: %s", err)
            self._hub_client.invalidate_cache()
            try:
                await self._handle_locally(chat_log)
            except Exception:
                _LOGGER.exception("Local agent loop also failed")
                self._add_error_response(chat_log)
        except Exception:
            _LOGGER.exception("Unexpected error in conversation handler")
            self._add_error_response(chat_log)

        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def _handle_via_hub(
        self,
        user_input: conversation.ConversationInput,
        chat_log: ChatLog,
    ) -> None:
        """Route the conversation through the Strawberry Hub.

        Streams SSE events from the Hub and reconstructs the response
        into the HA ChatLog.

        Args:
            user_input: The user's conversation input.
            chat_log: The chat log to populate with the response.
        """
        # Convert chat log to Hub message format
        messages = _chat_log_to_messages(chat_log)

        final_content = ""

        async for event in self._hub_client.chat_stream(
            messages=messages,
            enable_tools=True,
        ):
            event_type = event.get("type", "")

            if event_type == "content_delta":
                delta = event.get("delta", "")
                if delta:
                    final_content += delta

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

        Uses the HA Assist API tools that were already loaded into
        the chat_log by async_provide_llm_data. For Phase 1 (MVP),
        this is a simple pass-through that acknowledges offline status.

        Phase 2 will add TensorZero embedded gateway for full local
        agent loop capability.

        Args:
            chat_log: The chat log with LLM data already provided.
        """
        # Phase 1: Return a helpful offline message
        # Phase 2: Will use TensorZero embedded gateway + HA Assist tools
        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=self.entity_id,
                content=(
                    "I'm currently unable to reach the Strawberry Hub. "
                    "Offline mode with local LLM fallback is not yet "
                    "configured. Please check that the Hub is running "
                    "and try again."
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
