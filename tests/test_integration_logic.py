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
    sys.modules["homeassistant.helpers.update_coordinator"] = types.ModuleType("homeassistant.helpers.update_coordinator")
    mock_llm = types.ModuleType("homeassistant.helpers.llm")

    sys.modules["homeassistant"] = mock_hass
    sys.modules["homeassistant.components"] = mock_components
    sys.modules["homeassistant.components.conversation"] = mock_conversation
    sys.modules["homeassistant.config_entries"] = mock_config_entries
    sys.modules["homeassistant.const"] = mock_const
    sys.modules["homeassistant.core"] = mock_core
    sys.modules["homeassistant.helpers"] = mock_helpers
    sys.modules["homeassistant.helpers.entity_platform"] = mock_entity_platform
    
    sys.modules["homeassistant.components.binary_sensor"] = types.ModuleType("homeassistant.components.binary_sensor")
    sys.modules["homeassistant.components.binary_sensor"].BinarySensorEntity = type("BinarySensorEntity", (), {})
    sys.modules["homeassistant.components.binary_sensor"].BinarySensorDeviceClass = type("BinarySensorDeviceClass", (), {"CONNECTIVITY": "connectivity"})

    sys.modules["homeassistant.helpers.update_coordinator"] = types.ModuleType("homeassistant.helpers.update_coordinator")
    sys.modules["homeassistant.helpers.update_coordinator"].DataUpdateCoordinator = type("DataUpdateCoordinator", (), {"__class_getitem__": classmethod(lambda cls, item: cls)})
    sys.modules["homeassistant.helpers.update_coordinator"].CoordinatorEntity = type("CoordinatorEntity", (), {})
    sys.modules["homeassistant.helpers.update_coordinator"].UpdateFailed = type("UpdateFailed", (Exception,), {})
    
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

    mock_const.Platform = type("Platform", (), {"CONVERSATION": "conversation", "BINARY_SENSOR": "binary_sensor"})

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
        external: bool = False

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
        fallback_providers_from_options,
        offline_backend_from_options,
        provider_key_map_from_options,
        provider_model_map_from_options,
        run_local_agent_loop,
        tensorzero_function_name_from_options,
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
        fallback_providers_from_options,
        offline_backend_from_options,
        provider_key_map_from_options,
        provider_model_map_from_options,
        run_local_agent_loop,
        tensorzero_function_name_from_options,
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

    async def test_online_flow_records_hub_tool_calls(self) -> None:
        """Hub tool_call_started/result events should appear in ChatLog."""
        self.mock_hub_client.health_check.return_value = True

        async def stream_events(*args, **kwargs):
            yield {
                "type": "tool_call_started",
                "tool_call_id": "tc_1",
                "tool_name": "search_skills",
                "arguments": {"query": "lights"},
            }
            yield {
                "type": "tool_call_result",
                "tool_call_id": "tc_1",
                "tool_name": "search_skills",
                "success": True,
                "result": "HomeAssistantSkill",
                "error": None,
            }
            yield {"type": "content_delta", "delta": "Found it!"}
            yield {"type": "done"}

        self.mock_hub_client.chat_stream = stream_events

        user_input = MagicMock()
        user_input.as_llm_context.return_value = {}
        user_input.extra_system_prompt = None
        chat_log = sys.modules[
            "homeassistant.components.conversation"
        ].ChatLog()

        await self.entity._async_handle_message(user_input, chat_log)

        # Should contain: AssistantContent with external tool call,
        # ToolResultContent, and final AssistantContent with text
        conversation_mod = sys.modules["homeassistant.components.conversation"]

        tool_call_items = [
            item
            for item in chat_log.content
            if isinstance(item, conversation_mod.AssistantContent)
            and getattr(item, "tool_calls", None)
        ]
        self.assertEqual(len(tool_call_items), 1)
        tc = tool_call_items[0].tool_calls[0]
        self.assertEqual(tc.tool_name, "search_skills")
        self.assertTrue(tc.external)

        tool_result_items = [
            item
            for item in chat_log.content
            if isinstance(item, conversation_mod.ToolResultContent)
        ]
        self.assertEqual(len(tool_result_items), 1)
        self.assertEqual(tool_result_items[0].tool_name, "search_skills")
        self.assertEqual(
            tool_result_items[0].tool_result, {"result": "HomeAssistantSkill"}
        )

        # Final text response
        text_items = [
            item
            for item in chat_log.content
            if isinstance(item, conversation_mod.AssistantContent)
            and getattr(item, "content", None)
        ]
        self.assertTrue(any("Found it!" in (i.content or "") for i in text_items))

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

    async def test_fallback_provider_is_used_when_primary_fails(self) -> None:
        """Local loop should continue to next provider if primary request fails."""
        conversation_mod = sys.modules["homeassistant.components.conversation"]
        chat_log = conversation_mod.ChatLog()
        chat_log.content.append(conversation_mod.UserContent(content="hi"))

        api_instance = AsyncMock()
        api_instance.tools = []
        chat_log.llm_api = api_instance

        call_order: list[str] = []

        async def fake_request(
            provider,
            api_key,
            model,
            ollama_url,
            messages,
            tools,
        ):
            call_order.append(provider)
            if provider == "openai":
                raise RuntimeError("openai down")
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "fallback worked",
                        }
                    }
                ]
            }

        result = await run_local_agent_loop(
            chat_log=chat_log,
            api_instance=api_instance,
            system_prompt="sys",
            agent_id="conversation.strawberry_ai",
            offline_provider="openai",
            offline_api_key="legacy",
            offline_model="legacy-model",
            ollama_url=None,
            fallback_providers=["google"],
            provider_api_keys={"openai": "openai-key", "google": "google-key"},
            provider_models={"openai": "o-model", "google": "g-model"},
            request_completion=fake_request,
        )

        self.assertEqual(result, "fallback worked")
        self.assertEqual(call_order, ["openai", "google"])

    def test_provider_maps_from_options(self) -> None:
        """Options helper functions should parse fallback and provider maps."""
        options = {
            "offline_fallback_providers": ["google", "anthropic", "ollama"],
            "offline_openai_api_key": "openai-key",
            "offline_google_api_key": "google-key",
            "offline_anthropic_api_key": "anthropic-key",
            "offline_openai_model": "o-model",
            "offline_google_model": "g-model",
            "offline_anthropic_model": "a-model",
            "offline_ollama_model": "llama3.2:latest",
        }

        self.assertEqual(
            fallback_providers_from_options(options),
            ["google", "anthropic", "ollama"],
        )
        self.assertEqual(
            provider_key_map_from_options(options)["openai"],
            "openai-key",
        )
        self.assertEqual(
            provider_key_map_from_options(options)["anthropic"],
            "anthropic-key",
        )
        self.assertEqual(
            provider_model_map_from_options(options)["google"],
            "g-model",
        )
        self.assertEqual(
            provider_model_map_from_options(options)["anthropic"],
            "a-model",
        )
        self.assertEqual(
            offline_backend_from_options(options),
            "auto",
        )
        self.assertEqual(
            tensorzero_function_name_from_options(options),
            "chat",
        )

    def test_backend_helpers_from_options(self) -> None:
        """Backend helpers should read explicit values and apply defaults."""
        options = {
            "offline_backend": "tensorzero",
            "tensorzero_function_name": "ha_chat",
        }
        self.assertEqual(offline_backend_from_options(options), "tensorzero")
        self.assertEqual(tensorzero_function_name_from_options(options), "ha_chat")
        self.assertEqual(offline_backend_from_options({}), "auto")
        self.assertEqual(tensorzero_function_name_from_options({}), "chat")


class TestTzConfig(unittest.TestCase):
    """Tests for dynamic TensorZero configuration generation."""

    def test_build_dynamic_config_creates_toml_and_tools(self) -> None:
        """build_dynamic_config should write a valid TOML and tool schemas."""
        from custom_components.strawberry_conversation.tz_config import (
            build_dynamic_config,
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "HassTurnOn",
                    "description": "Turn on an entity",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                },
            }
        ]

        config_dir, toml_path = build_dynamic_config(
            provider_chain=["openai", "google"],
            provider_models={"openai": "gpt-4o-mini", "google": "gemini-2.5-flash-lite"},
            ollama_url=None,
            tools=tools,
            function_name="chat",
        )

        import os
        self.assertTrue(os.path.isfile(toml_path))
        with open(toml_path) as f:
            content = f.read()

        # Models defined
        self.assertIn("[models.openai_model]", content)
        self.assertIn("[models.google_model]", content)
        self.assertIn('type = "openai"', content)
        self.assertIn('type = "google_ai_studio_gemini"', content)

        # Dynamic credentials
        self.assertIn('api_key_location = "dynamic::openai_api_key"', content)
        self.assertIn('api_key_location = "dynamic::google_api_key"', content)

        # Tool schema file
        tool_schema = os.path.join(config_dir, "tools", "HassTurnOn.json")
        self.assertTrue(os.path.isfile(tool_schema))

        # Function + variants
        self.assertIn("[functions.chat]", content)
        self.assertIn("[functions.chat.variants.openai_variant]", content)
        self.assertIn("[functions.chat.variants.google_variant]", content)
        self.assertIn("fallback_variants", content)

        # Cleanup
        import shutil
        shutil.rmtree(config_dir, ignore_errors=True)

    def test_build_dynamic_config_with_anthropic(self) -> None:
        """Anthropic provider should produce an 'anthropic' type model."""
        from custom_components.strawberry_conversation.tz_config import (
            build_dynamic_config,
        )

        config_dir, toml_path = build_dynamic_config(
            provider_chain=["anthropic"],
            provider_models={"anthropic": "claude-haiku-4-5"},
            ollama_url=None,
            tools=[],
            function_name="chat",
        )

        with open(toml_path) as f:
            content = f.read()

        self.assertIn("[models.anthropic_model]", content)
        self.assertIn('type = "anthropic"', content)
        self.assertIn('model_name = "claude-haiku-4-5"', content)
        self.assertIn('api_key_location = "dynamic::anthropic_api_key"', content)

        import shutil
        shutil.rmtree(config_dir, ignore_errors=True)

    def test_build_dynamic_config_ollama(self) -> None:
        """Ollama provider should use openai type with api_base and api_key_location=none."""
        from custom_components.strawberry_conversation.tz_config import (
            build_dynamic_config,
        )

        config_dir, toml_path = build_dynamic_config(
            provider_chain=["ollama"],
            provider_models={"ollama": "llama3.2:3b"},
            ollama_url="http://myhost:11434/v1",
            tools=[],
            function_name="ha_chat",
        )

        with open(toml_path) as f:
            content = f.read()

        self.assertIn('type = "openai"', content)
        self.assertIn('api_key_location = "none"', content)
        self.assertIn('api_base = "http://myhost:11434/v1/"', content)
        self.assertIn("[functions.ha_chat]", content)

        import shutil
        shutil.rmtree(config_dir, ignore_errors=True)

    def test_effective_provider_chain_filters_missing_keys(self) -> None:
        """Providers without API keys should be excluded (except Ollama)."""
        from custom_components.strawberry_conversation.tz_config import (
            effective_provider_chain,
        )

        chain = effective_provider_chain(
            ["openai", "google", "anthropic", "ollama"],
            {"openai": "key1", "google": None, "anthropic": "key3"},
        )
        self.assertEqual(chain, ["openai", "anthropic", "ollama"])

    def test_build_credentials(self) -> None:
        """build_credentials should map dynamic keys to actual API key values."""
        from custom_components.strawberry_conversation.tz_config import (
            build_credentials,
        )

        creds = build_credentials(
            ["openai", "anthropic", "ollama"],
            {"openai": "sk-123", "anthropic": "ant-456"},
        )
        self.assertEqual(creds, {
            "openai_api_key": "sk-123",
            "anthropic_api_key": "ant-456",
        })

    def test_config_hash_changes_with_different_inputs(self) -> None:
        """Config hash should change when inputs differ."""
        from custom_components.strawberry_conversation.tz_config import (
            _compute_config_hash,
        )

        h1 = _compute_config_hash(["openai"], {"openai": "gpt-4o"}, "", ["tool1"], "chat")
        h2 = _compute_config_hash(["openai"], {"openai": "gpt-4o-mini"}, "", ["tool1"], "chat")
        h3 = _compute_config_hash(["openai"], {"openai": "gpt-4o"}, "", ["tool1"], "chat")
        self.assertNotEqual(h1, h2)
        self.assertEqual(h1, h3)


if __name__ == "__main__":
    unittest.main()
