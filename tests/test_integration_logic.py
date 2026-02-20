"""Unit tests for Strawberry conversation routing logic.

These tests run without a Home Assistant runtime by creating a minimal mock
module tree before importing the integration module.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock


# Ensure integration root is importable as top-level package path.
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_ROOT = os.path.dirname(TESTS_DIR)
if INTEGRATION_ROOT not in sys.path:
    sys.path.insert(0, INTEGRATION_ROOT)


def _bootstrap_homeassistant_mocks() -> None:
    """Create a minimal Home Assistant module graph for isolated tests."""
    mock_hass = types.ModuleType("homeassistant")
    mock_components = types.ModuleType("homeassistant.components")
    mock_conversation = types.ModuleType("homeassistant.components.conversation")
    mock_config_entries = types.ModuleType("homeassistant.config_entries")
    mock_const = types.ModuleType("homeassistant.const")
    mock_core = types.ModuleType("homeassistant.core")
    mock_helpers = types.ModuleType("homeassistant.helpers")
    mock_entity_platform = types.ModuleType("homeassistant.helpers.entity_platform")
    mock_llm = types.ModuleType("homeassistant.helpers.llm")

    sys.modules["homeassistant"] = mock_hass
    sys.modules["homeassistant.components"] = mock_components
    sys.modules["homeassistant.components.conversation"] = mock_conversation
    sys.modules["homeassistant.config_entries"] = mock_config_entries
    sys.modules["homeassistant.const"] = mock_const
    sys.modules["homeassistant.core"] = mock_core
    sys.modules["homeassistant.helpers"] = mock_helpers
    sys.modules["homeassistant.helpers.entity_platform"] = mock_entity_platform
    sys.modules["homeassistant.helpers.llm"] = mock_llm

    mock_hass.components = mock_components
    mock_components.conversation = mock_conversation
    mock_hass.config_entries = mock_config_entries
    mock_hass.const = mock_const
    mock_hass.core = mock_core
    mock_hass.helpers = mock_helpers
    mock_helpers.entity_platform = mock_entity_platform
    mock_helpers.llm = mock_llm

    class MockConversationEntity:
        """Minimal base class placeholder."""

    class MockAbstractConversationAgent:
        """Minimal base class placeholder."""

    @dataclass
    class AssistantContent:
        """Simple replacement for HA AssistantContent."""

        agent_id: str
        content: str | None = None
        tool_calls: list[object] | None = None

    @dataclass
    class SystemContent:
        """Simple replacement for HA SystemContent."""

        content: str

    @dataclass
    class UserContent:
        """Simple replacement for HA UserContent."""

        content: str

    @dataclass
    class ToolResultContent:
        """Simple replacement for HA ToolResultContent."""

        agent_id: str
        tool_call_id: str
        tool_name: str
        tool_result: object

    class MockChatLog:
        """Simple chat log with required helper methods."""

        def __init__(self) -> None:
            self.content: list[object] = []
            self.llm_api = None

        async def async_provide_llm_data(self, *args, **kwargs) -> None:
            """No-op for test."""

        def async_add_assistant_content_without_tools(self, content: object) -> None:
            """Append content to the local log list."""
            self.content.append(content)

        async def async_add_assistant_content(self, content: object):
            """Mimic HA helper that executes non-external tool calls."""
            self.content.append(content)

            tool_calls = getattr(content, "tool_calls", None) or []
            for tool_call in tool_calls:
                tool_result = await self.llm_api.async_call_tool(tool_call)
                result_content = ToolResultContent(
                    agent_id=getattr(content, "agent_id", "conversation.strawberry_ai"),
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.tool_name,
                    tool_result=tool_result,
                )
                self.content.append(result_content)
                yield result_content

    mock_conversation.ConversationEntity = MockConversationEntity
    mock_conversation.AbstractConversationAgent = MockAbstractConversationAgent
    mock_conversation.ChatLog = MockChatLog
    mock_conversation.ConversationInput = MagicMock()
    mock_conversation.ConversationResult = MagicMock()
    mock_conversation.AssistantContent = AssistantContent
    mock_conversation.SystemContent = SystemContent
    mock_conversation.UserContent = UserContent
    mock_conversation.ToolResultContent = ToolResultContent
    mock_conversation.ConverseError = Exception
    mock_conversation.async_get_result_from_chat_log = MagicMock(
        return_value="RESULT"
    )

    mock_const.CONF_LLM_HASS_API = "llm_hass_api"
    mock_const.MATCH_ALL = "*"

    class Platform:
        """Minimal Platform enum-like object used by integration setup."""

        CONVERSATION = "conversation"

    mock_const.Platform = Platform

    mock_config_entries.ConfigEntry = MagicMock
    mock_config_entries.ConfigSubentry = MagicMock
    mock_core.HomeAssistant = MagicMock
    mock_entity_platform.AddConfigEntryEntitiesCallback = MagicMock

    @dataclass(slots=True)
    class ToolInput:
        """Minimal ToolInput replacement used by local agent logic."""

        tool_name: str
        tool_args: dict
        id: str = "tool-id"

    mock_llm.ToolInput = ToolInput
    mock_llm.ulid_now = lambda: "ulid-test"


_bootstrap_homeassistant_mocks()

try:
    from custom_components.strawberry_conversation.conversation import (
        StrawberryConversationEntity,
        _chat_log_to_messages,
    )
    from custom_components.strawberry_conversation.hub_client import (
        HubConnectionError,
        StrawberryHubClient,
    )
    from custom_components.strawberry_conversation.local_agent import (
        run_local_agent_loop,
    )
except ImportError:
    sys.path.append(INTEGRATION_ROOT)
    from custom_components.strawberry_conversation.conversation import (
        StrawberryConversationEntity,
        _chat_log_to_messages,
    )
    from custom_components.strawberry_conversation.hub_client import (
        HubConnectionError,
        StrawberryHubClient,
    )
    from custom_components.strawberry_conversation.local_agent import (
        run_local_agent_loop,
    )


class TestStrawberryConversation(unittest.IsolatedAsyncioTestCase):
    """Behavior tests for online/offline conversation routing."""

    async def asyncSetUp(self) -> None:
        """Create a fresh entity and dependency mocks for each test."""
        self.mock_config_entry = MagicMock()
        self.mock_subentry = MagicMock()
        self.mock_subentry.subentry_id = "test_subentry"
        self.mock_subentry.data = {"prompt": "test prompt"}

        self.mock_hub_client = AsyncMock(spec=StrawberryHubClient)
        self.mock_hub_client.invalidate_cache = MagicMock()
        self.mock_config_entry.runtime_data = self.mock_hub_client

        self.entity = StrawberryConversationEntity(
            self.mock_config_entry,
            self.mock_subentry,
        )
        self.entity.entity_id = "conversation.strawberry_ai"

    async def test_online_flow_uses_hub_response(self) -> None:
        """When Hub is available, final assistant text should come from Hub."""
        self.mock_hub_client.health_check.return_value = True

        async def stream_events(*args, **kwargs):
            yield {"type": "content_delta", "delta": "Hello "}
            yield {"type": "content_delta", "delta": "from Hub"}
            yield {"type": "done"}

        self.mock_hub_client.chat_stream = stream_events

        user_input = MagicMock()
        user_input.as_llm_context.return_value = {}
        user_input.extra_system_prompt = None
        chat_log = sys.modules[
            "homeassistant.components.conversation"
        ].ChatLog()

        result = await self.entity._async_handle_message(user_input, chat_log)

        self.mock_hub_client.health_check.assert_awaited_once()
        self.assertEqual(result, "RESULT")
        self.assertTrue(
            any(
                getattr(item, "content", "") == "Hello from Hub"
                for item in chat_log.content
            )
        )

    async def test_offline_flow_uses_local_message(self) -> None:
        """When Hub is offline, entity should return local fallback text."""
        self.mock_hub_client.health_check.return_value = False

        user_input = MagicMock()
        user_input.as_llm_context.return_value = {}
        user_input.extra_system_prompt = None
        chat_log = sys.modules[
            "homeassistant.components.conversation"
        ].ChatLog()

        await self.entity._async_handle_message(user_input, chat_log)

        self.mock_hub_client.health_check.assert_awaited_once()
        self.assertTrue(
            any(
                "unable to reach the Strawberry Hub"
                in str(getattr(item, "content", ""))
                for item in chat_log.content
            )
        )

    async def test_stream_failure_invalidates_cache_and_falls_back(self) -> None:
        """If Hub fails mid-request, cache should invalidate and fallback runs."""
        self.mock_hub_client.health_check.return_value = True

        async def broken_stream(*args, **kwargs):
            raise HubConnectionError("connection lost")
            yield {"type": "done"}

        self.mock_hub_client.chat_stream = broken_stream

        user_input = MagicMock()
        user_input.as_llm_context.return_value = {}
        user_input.extra_system_prompt = None
        chat_log = sys.modules[
            "homeassistant.components.conversation"
        ].ChatLog()

        await self.entity._async_handle_message(user_input, chat_log)

        self.mock_hub_client.invalidate_cache.assert_called_once()
        self.assertTrue(
            any(
                "unable to reach the Strawberry Hub"
                in str(getattr(item, "content", ""))
                for item in chat_log.content
            )
        )

    def test_chat_log_to_messages_mapping(self) -> None:
        """Chat content types should map to expected Hub message roles."""
        conversation_mod = sys.modules["homeassistant.components.conversation"]
        chat_log = conversation_mod.ChatLog()
        chat_log.content.extend(
            [
                conversation_mod.SystemContent(content="sys"),
                conversation_mod.UserContent(content="hello"),
                conversation_mod.AssistantContent(
                    agent_id="conversation.strawberry_ai",
                    content="hi",
                ),
                conversation_mod.ToolResultContent(
                    agent_id="conversation.strawberry_ai",
                    tool_call_id="call_1",
                    tool_name="HassTurnOn",
                    tool_result={"ok": True},
                ),
            ]
        )

        messages = _chat_log_to_messages(chat_log)
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "tool", "content": "{'ok': True}"},
            ],
        )


class TestLocalAgentLoop(unittest.IsolatedAsyncioTestCase):
    """Behavior tests for the Phase 2 local agent loop."""

    async def test_provider_none_returns_none(self) -> None:
        """Disabled provider should skip local requests immediately."""
        conversation_mod = sys.modules["homeassistant.components.conversation"]
        chat_log = conversation_mod.ChatLog()
        chat_log.llm_api = AsyncMock()

        result = await run_local_agent_loop(
            chat_log=chat_log,
            api_instance=chat_log.llm_api,
            system_prompt="sys",
            agent_id="conversation.strawberry_ai",
            offline_provider="none",
            offline_api_key=None,
            offline_model=None,
            ollama_url=None,
        )

        self.assertIsNone(result)

    async def test_tool_call_then_final_response(self) -> None:
        """Loop should execute tool calls and then return final text response."""
        conversation_mod = sys.modules["homeassistant.components.conversation"]
        chat_log = conversation_mod.ChatLog()
        chat_log.content.extend(
            [
                conversation_mod.SystemContent(content="You are helpful"),
                conversation_mod.UserContent(content="Turn on bedroom lamp"),
            ]
        )

        fake_tool = MagicMock()
        fake_tool.name = "HassTurnOn"
        fake_tool.description = "Turn on entity"
        fake_tool.parameters = MagicMock()

        api_instance = AsyncMock()
        api_instance.tools = [fake_tool]
        api_instance.async_call_tool.return_value = {"success": True}
        chat_log.llm_api = api_instance

        calls = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "HassTurnOn",
                                        "arguments": '{"name": "bedroom lamp"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Done! The bedroom lamp is on.",
                        }
                    }
                ]
            },
        ]

        async def fake_request(*args, **kwargs):
            return calls.pop(0)

        result = await run_local_agent_loop(
            chat_log=chat_log,
            api_instance=api_instance,
            system_prompt="You are helpful",
            agent_id="conversation.strawberry_ai",
            offline_provider="openai",
            offline_api_key="test-key",
            offline_model="gpt-4o-mini",
            ollama_url=None,
            request_completion=fake_request,
        )

        self.assertEqual(result, "Done! The bedroom lamp is on.")
        api_instance.async_call_tool.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
