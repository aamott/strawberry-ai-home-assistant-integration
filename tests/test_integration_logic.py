import sys
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import json
import types

# --- MOCKS FOR HOME ASSISTANT ---
# We must mock these BEFORE importing any integration code

# 1. Create module objects
mock_hass = types.ModuleType("homeassistant")
mock_components = types.ModuleType("homeassistant.components")
mock_conversation = types.ModuleType("homeassistant.components.conversation")
mock_config_entries = types.ModuleType("homeassistant.config_entries")
mock_const = types.ModuleType("homeassistant.const")
mock_core = types.ModuleType("homeassistant.core")
mock_helpers = types.ModuleType("homeassistant.helpers")
mock_entity_platform = types.ModuleType("homeassistant.helpers.entity_platform")

# 2. Register them in sys.modules
sys.modules["homeassistant"] = mock_hass
sys.modules["homeassistant.components"] = mock_components
sys.modules["homeassistant.components.conversation"] = mock_conversation
sys.modules["homeassistant.config_entries"] = mock_config_entries
sys.modules["homeassistant.const"] = mock_const
sys.modules["homeassistant.core"] = mock_core
sys.modules["homeassistant.helpers"] = mock_helpers
sys.modules["homeassistant.helpers.entity_platform"] = mock_entity_platform

# 3. Link them so 'from homeassistant.components import conversation' works
mock_hass.components = mock_components
mock_components.conversation = mock_conversation
mock_hass.config_entries = mock_config_entries
mock_hass.const = mock_const
mock_hass.core = mock_core
mock_hass.helpers = mock_helpers
mock_helpers.entity_platform = mock_entity_platform

# Define mock classes/constants used in imports
class MockConversationEntity:
    pass

class MockAbstractConversationAgent:
    pass

mock_conversation.ConversationEntity = MockConversationEntity
mock_conversation.AbstractConversationAgent = MockAbstractConversationAgent

class MockChatLog:
    def __init__(self):
        self.content = []
    async def async_provide_llm_data(self, *args, **kwargs): pass
    def async_add_assistant_content_without_tools(self, content):
        self.content.append(content)

mock_conversation.ChatLog = MockChatLog
mock_conversation.ConversationInput = MagicMock()
mock_conversation.ConversationResult = MagicMock()
mock_conversation.AssistantContent = lambda agent_id, content: f"AssistantContent(agent_id={agent_id}, content={content})"
mock_conversation.SystemContent = MagicMock
mock_conversation.UserContent = MagicMock
mock_conversation.ToolResultContent = MagicMock
mock_conversation.ConverseError = Exception # Must be an exception type
mock_conversation.async_get_result_from_chat_log = MagicMock(return_value="RESULT")

mock_const.CONF_LLM_HASS_API = "llm_hass_api"
mock_const.MATCH_ALL = "*"
class MockPlatform:
    CONVERSATION = "conversation"
mock_const.Platform = MockPlatform

# Fill in other required mocks (used in decorators or base classes if any)
mock_config_entries.ConfigEntry = MagicMock
mock_config_entries.ConfigSubentry = MagicMock
mock_core.HomeAssistant = MagicMock
mock_entity_platform.AddConfigEntryEntitiesCallback = MagicMock

# --- IMPORT INTEGRATION CODE ---
# Now we can import the integration code safely
# We assume the test runner adds the correct paths
try:
    from custom_components.strawberry_conversation.conversation import StrawberryConversationEntity
    from custom_components.strawberry_conversation.const import DOMAIN
    from custom_components.strawberry_conversation.hub_client import StrawberryHubClient, HubConnectionError
except ImportError:
    # Fallback for when running from root
    import os
    sys.path.append(os.path.abspath("ha-integration"))
    from custom_components.strawberry_conversation.conversation import StrawberryConversationEntity
    from custom_components.strawberry_conversation.const import DOMAIN
    from custom_components.strawberry_conversation.hub_client import StrawberryHubClient, HubConnectionError


class TestStrawberryConversation(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mock_config_entry = MagicMock()
        self.mock_subentry = MagicMock()
        self.mock_subentry.subentry_id = "test_subentry"
        self.mock_subentry.data = {"prompt": "test prompt"}
        
        self.mock_hub_client = AsyncMock(spec=StrawberryHubClient)
        self.mock_config_entry.runtime_data = self.mock_hub_client
        
        self.entity = StrawberryConversationEntity(self.mock_config_entry, self.mock_subentry)
        self.entity._attr_unique_id = "test_subentry" # Allow simple id check
        self.entity.entity_id = "conversation.strawberry_ai"

    async def test_online_flow(self):
        """Test that messages go to Hub when it's online."""
        # Setup Hub to be online
        self.mock_hub_client.health_check.return_value = True
        
        # Setup Hub streaming response
        async def mock_stream(*args, **kwargs):
            yield {"type": "assistant_message", "content": "Hello from Hub"}
            yield {"type": "done"}
        self.mock_hub_client.chat_stream = mock_stream

        # Setup input
        user_input = MagicMock()
        chat_log = MockChatLog() # Use our simple mock
        
        # Run handler
        await self.entity._async_handle_message(user_input, chat_log)
        
        # Verify Hub was checked
        self.mock_hub_client.health_check.assert_awaited()
        
        # Verify result in chat log
        # The content in our mock is a string repr of AssistantContent
        self.assertTrue(any("Hello from Hub" in str(c) for c in chat_log.content))

    async def test_offline_flow(self):
        """Test fallback to local when Hub is offline."""
        # Setup Hub to be offline
        self.mock_hub_client.health_check.return_value = False
        
        # Setup input
        user_input = MagicMock()
        chat_log = MockChatLog()
        
        # Run handler
        await self.entity._async_handle_message(user_input, chat_log)
        
        # Verify Hub checked
        self.mock_hub_client.health_check.assert_awaited()
        
        # Verify offline message in chat log
        # Look for offline text from conversation.py
        self.assertTrue(any("unable to reach the Strawberry Hub" in str(c) for c in chat_log.content))

    async def test_active_failure_flow(self):
        """Test fallback when Hub fails during active request."""
        # Setup Hub to appear online initially
        self.mock_hub_client.health_check.return_value = True
        
        # But fail during stream
        self.mock_hub_client.chat_stream.side_effect = HubConnectionError("Connection lost")
        
        # Setup input
        user_input = MagicMock()
        chat_log = MockChatLog()
        
        # Run handler
        await self.entity._async_handle_message(user_input, chat_log)
        
        # Verify cache invalidation was called
        self.mock_hub_client.invalidate_cache.assert_called()
        
        # Verify fallback message
        self.assertTrue(any("unable to reach the Strawberry Hub" in str(c) for c in chat_log.content))

if __name__ == "__main__":
    unittest.main()
