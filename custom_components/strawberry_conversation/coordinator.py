"""Data update coordinator for the Strawberry Conversation integration."""

from __future__ import annotations

import logging
from datetime import timedelta

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import (
    DataUpdateCoordinator,
    UpdateFailed,
)

from .const import DOMAIN
from .hub_client import StrawberryHubClient

_LOGGER = logging.getLogger(__name__)


class HubStatusCoordinator(DataUpdateCoordinator[bool]):
    """Class to manage fetching Hub connection status."""

    def __init__(self, hass: HomeAssistant, client: StrawberryHubClient) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=60),
        )
        self.client = client

    async def _async_update_data(self) -> bool:
        """Fetch the latest status from the Hub."""
        try:
            self.client.invalidate_cache()
            return await self.client.health_check()
        except Exception as err:
            raise UpdateFailed(f"Error checking Hub connection: {err}") from err
