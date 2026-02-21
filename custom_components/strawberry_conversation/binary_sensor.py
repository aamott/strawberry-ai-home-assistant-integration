"""Binary sensor platform for Strawberry AI."""

from __future__ import annotations

import logging

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
)

from . import StrawberryConfigEntry
from .const import DEFAULT_TITLE, DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: StrawberryConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Strawberry binary sensors from a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]

    async_add_entities([StrawberryHubConnectionSensor(coordinator, entry)])


class StrawberryHubConnectionSensor(CoordinatorEntity, BinarySensorEntity):
    """Binary sensor for the Hub connection status."""

    _attr_device_class = BinarySensorDeviceClass.CONNECTIVITY
    _attr_has_entity_name = True
    _attr_name = "Hub Connection"

    def __init__(self, coordinator: DataUpdateCoordinator, entry: ConfigEntry) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_hub_connection"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.title or DEFAULT_TITLE,
            "manufacturer": "Strawberry AI",
        }

    @property
    def is_on(self) -> bool:
        """Return true if the Hub is connected."""
        return self.coordinator.data
