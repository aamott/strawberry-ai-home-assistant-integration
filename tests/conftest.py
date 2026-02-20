"""Pytest test bootstrap for the standalone HA integration test suite.

This file provides lightweight module stubs so the integration package can be
imported outside a real Home Assistant runtime.
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock


def _ensure_integration_root_on_path() -> None:
    """Ensure ``ha-integration`` root is importable as a top-level package path."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    integration_root = os.path.dirname(tests_dir)
    if integration_root not in sys.path:
        sys.path.insert(0, integration_root)


def _bootstrap_homeassistant_modules() -> None:
    """Install minimal Home Assistant stubs required for module imports.

    The integration package imports Home Assistant modules at import time in
    ``custom_components.strawberry_conversation.__init__``. These stubs are
    intentionally small and only provide names needed by current tests.
    """
    if "homeassistant" in sys.modules:
        # Keep existing modules if another test already provided richer stubs.
        return

    mock_hass = types.ModuleType("homeassistant")
    mock_components = types.ModuleType("homeassistant.components")
    mock_conversation = types.ModuleType("homeassistant.components.conversation")
    mock_config_entries = types.ModuleType("homeassistant.config_entries")
    mock_const = types.ModuleType("homeassistant.const")
    mock_core = types.ModuleType("homeassistant.core")
    mock_helpers = types.ModuleType("homeassistant.helpers")
    mock_entity_platform = types.ModuleType("homeassistant.helpers.entity_platform")

    sys.modules["homeassistant"] = mock_hass
    sys.modules["homeassistant.components"] = mock_components
    sys.modules["homeassistant.components.conversation"] = mock_conversation
    sys.modules["homeassistant.config_entries"] = mock_config_entries
    sys.modules["homeassistant.const"] = mock_const
    sys.modules["homeassistant.core"] = mock_core
    sys.modules["homeassistant.helpers"] = mock_helpers
    sys.modules["homeassistant.helpers.entity_platform"] = mock_entity_platform

    mock_hass.components = mock_components
    mock_components.conversation = mock_conversation
    mock_hass.config_entries = mock_config_entries
    mock_hass.const = mock_const
    mock_hass.core = mock_core
    mock_hass.helpers = mock_helpers
    mock_helpers.entity_platform = mock_entity_platform

    class Platform:
        """Minimal enum-like placeholder for HA platform constants."""

        CONVERSATION = "conversation"

    mock_const.Platform = Platform
    mock_const.CONF_LLM_HASS_API = "llm_hass_api"
    mock_const.MATCH_ALL = "*"

    mock_config_entries.ConfigEntry = MagicMock
    mock_config_entries.ConfigSubentry = MagicMock
    mock_core.HomeAssistant = MagicMock
    mock_entity_platform.AddConfigEntryEntitiesCallback = MagicMock


_ensure_integration_root_on_path()
_bootstrap_homeassistant_modules()
