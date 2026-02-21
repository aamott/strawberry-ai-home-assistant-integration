"""Strawberry AI Conversation integration for Home Assistant.

Adds a conversation agent powered by the Strawberry Hub with automatic
offline fallback via TensorZero embedded gateway + HA's native Assist tools.
"""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import CONF_HUB_TOKEN, CONF_HUB_URL, DOMAIN, LOGGER
from .coordinator import HubStatusCoordinator
from .hub_client import StrawberryHubClient

_LOGGER = logging.getLogger(__name__)

# Platforms this integration sets up
PLATFORMS = [Platform.CONVERSATION, Platform.BINARY_SENSOR]

# Type alias for config entry with runtime data
type StrawberryConfigEntry = ConfigEntry[StrawberryHubClient]


async def async_setup_entry(
    hass: HomeAssistant,
    entry: StrawberryConfigEntry,
) -> bool:
    """Set up Strawberry Conversation from a config entry.

    Creates the Hub client and verifies connectivity. If the Hub is
    unreachable at setup time, we still allow setup (offline mode will
    handle it), but log a warning.

    Args:
        hass: Home Assistant instance.
        entry: The config entry being set up.

    Returns:
        True if setup succeeded.
    """
    hub_url = entry.data[CONF_HUB_URL]
    hub_token = entry.data[CONF_HUB_TOKEN]

    # Create the Hub client (stored as runtime_data on the entry)
    client = StrawberryHubClient(hub_url=hub_url, token=hub_token)

    # Check Hub connectivity — warn but don't block setup
    try:
        available = await client.health_check()
        if not available:
            LOGGER.warning(
                "Strawberry Hub at %s is not reachable. "
                "Conversation agent will operate in offline mode until "
                "the Hub becomes available.",
                hub_url,
            )
    except Exception:
        LOGGER.warning(
            "Could not check Strawberry Hub health at %s. "
            "Will retry on first conversation request.",
            hub_url,
        )

    # Store client as runtime data
    entry.runtime_data = client

    # Set up coordinator for connection status
    coordinator = HubStatusCoordinator(hass, client)
    await coordinator.async_config_entry_first_refresh()

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {
        "coordinator": coordinator,
    }

    # Forward setup to conversation platform
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(
    hass: HomeAssistant,
    entry: StrawberryConfigEntry,
) -> bool:
    """Unload a Strawberry Conversation config entry.

    Args:
        hass: Home Assistant instance.
        entry: The config entry being unloaded.

    Returns:
        True if unload succeeded.
    """
    # Close the Hub client
    client: StrawberryHubClient = entry.runtime_data
    await client.close()

    # Unload platforms
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        hass.data[DOMAIN].pop(entry.entry_id)
    return unload_ok
