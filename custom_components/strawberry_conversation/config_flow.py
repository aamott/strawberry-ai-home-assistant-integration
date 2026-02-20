"""Config flow for Strawberry AI Conversation integration."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    CONF_HUB_TOKEN,
    CONF_HUB_URL,
    CONF_OFFLINE_ANTHROPIC_API_KEY,
    CONF_OFFLINE_ANTHROPIC_MODEL,
    CONF_OFFLINE_API_KEY,
    CONF_OFFLINE_FALLBACK_PROVIDERS,
    CONF_OFFLINE_GOOGLE_API_KEY,
    CONF_OFFLINE_GOOGLE_MODEL,
    CONF_OFFLINE_MODEL,
    CONF_OFFLINE_OLLAMA_MODEL,
    CONF_OFFLINE_OPENAI_API_KEY,
    CONF_OFFLINE_OPENAI_MODEL,
    CONF_OFFLINE_PROVIDER,
    CONF_OLLAMA_URL,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_HUB_URL,
    DEFAULT_MODELS,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TITLE,
    DOMAIN,
    OFFLINE_PROVIDERS,
    PROVIDER_NONE,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    PROVIDER_GOOGLE,
    PROVIDER_ANTHROPIC,
    RECOMMENDED_CONVERSATION_OPTIONS,
)
from .hub_client import StrawberryHubClient

_LOGGER = logging.getLogger(__name__)

_PROVIDER_API_KEY_FIELD_MAP = {
    PROVIDER_OPENAI: CONF_OFFLINE_OPENAI_API_KEY,
    PROVIDER_GOOGLE: CONF_OFFLINE_GOOGLE_API_KEY,
    PROVIDER_ANTHROPIC: CONF_OFFLINE_ANTHROPIC_API_KEY,
}

_PROVIDER_MODEL_FIELD_MAP = {
    PROVIDER_OPENAI: CONF_OFFLINE_OPENAI_MODEL,
    PROVIDER_GOOGLE: CONF_OFFLINE_GOOGLE_MODEL,
    PROVIDER_ANTHROPIC: CONF_OFFLINE_ANTHROPIC_MODEL,
    PROVIDER_OLLAMA: CONF_OFFLINE_OLLAMA_MODEL,
}

# Schema for the initial Hub connection step
STEP_HUB_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_HUB_URL, default=DEFAULT_HUB_URL): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
        vol.Required(CONF_HUB_TOKEN): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
    }
)


async def _validate_hub_connection(
    hass: HomeAssistant,
    hub_url: str,
    hub_token: str,
) -> dict[str, str]:
    """Validate the Hub connection and return errors dict.

    Args:
        hass: Home Assistant instance.
        hub_url: Hub base URL.
        hub_token: Device JWT token.

    Returns:
        Empty dict if valid, or dict with error keys.
    """
    client = StrawberryHubClient(hub_url=hub_url, token=hub_token)
    try:
        available = await client.health_check()
        if not available:
            return {"base": "cannot_connect"}
    except Exception:
        _LOGGER.exception("Unexpected error validating Hub connection")
        return {"base": "unknown"}
    finally:
        await client.close()
    return {}


class StrawberryConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Strawberry AI Conversation."""

    VERSION = 1
    MINOR_VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial Hub connection step.

        Collects Hub URL and device token, validates connectivity,
        and creates the config entry with a default conversation subentry.
        """
        errors: dict[str, str] = {}

        if user_input is not None:
            # Prevent duplicate entries for the same Hub URL
            self._async_abort_entries_match(
                {CONF_HUB_URL: user_input[CONF_HUB_URL]}
            )

            # Validate connection
            errors = await _validate_hub_connection(
                self.hass,
                user_input[CONF_HUB_URL],
                user_input[CONF_HUB_TOKEN],
            )

            if not errors:
                return self.async_create_entry(
                    title=DEFAULT_TITLE,
                    data=user_input,
                    subentries=[
                        {
                            "subentry_type": "conversation",
                            "data": RECOMMENDED_CONVERSATION_OPTIONS,
                            "title": DEFAULT_CONVERSATION_NAME,
                            "unique_id": None,
                        },
                    ],
                )

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_HUB_DATA_SCHEMA,
            errors=errors,
        )

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            "conversation": ConversationSubentryFlow,
        }


class ConversationSubentryFlow(ConfigSubentryFlow):
    """Flow for managing conversation agent subentries."""

    last_rendered_recommended = False

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == "user"

    async def async_step_set_options(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Set conversation options.

        Handles both new subentry creation and reconfiguration.
        Shows/hides advanced options based on the 'recommended' toggle.
        """
        errors: dict[str, str] = {}

        if user_input is None:
            # Load defaults or existing options
            if self._is_new:
                options = RECOMMENDED_CONVERSATION_OPTIONS.copy()
            else:
                options = self._get_reconfigure_subentry().data.copy()

            self.last_rendered_recommended = bool(
                options.get(CONF_RECOMMENDED, False)
            )
        else:
            # Process submitted form
            if user_input[CONF_RECOMMENDED] == self.last_rendered_recommended:
                # Clean up empty optional fields
                if not user_input.get(CONF_LLM_HASS_API):
                    user_input.pop(CONF_LLM_HASS_API, None)
                if not user_input.get(CONF_OFFLINE_API_KEY):
                    user_input.pop(CONF_OFFLINE_API_KEY, None)
                if not user_input.get(CONF_OFFLINE_FALLBACK_PROVIDERS):
                    user_input.pop(CONF_OFFLINE_FALLBACK_PROVIDERS, None)

                for field in _PROVIDER_API_KEY_FIELD_MAP.values():
                    if not user_input.get(field):
                        user_input.pop(field, None)

                for field in _PROVIDER_MODEL_FIELD_MAP.values():
                    if not user_input.get(field):
                        user_input.pop(field, None)

                if user_input.get(CONF_OFFLINE_PROVIDER) == PROVIDER_NONE:
                    user_input.pop(CONF_OFFLINE_API_KEY, None)
                    user_input.pop(CONF_OFFLINE_MODEL, None)
                    user_input.pop(CONF_OLLAMA_URL, None)
                    user_input.pop(CONF_OFFLINE_FALLBACK_PROVIDERS, None)
                    for field in _PROVIDER_API_KEY_FIELD_MAP.values():
                        user_input.pop(field, None)
                    for field in _PROVIDER_MODEL_FIELD_MAP.values():
                        user_input.pop(field, None)

                if self._is_new:
                    return self.async_create_entry(
                        title=user_input.pop(CONF_NAME),
                        data=user_input,
                    )
                return self.async_update_and_abort(
                    self._get_entry(),
                    self._get_reconfigure_subentry(),
                    data=user_input,
                )

            # Toggle changed — re-render with new visibility
            self.last_rendered_recommended = user_input[CONF_RECOMMENDED]
            options = user_input

        schema = _build_conversation_schema(
            self.hass, self._is_new, options
        )
        return self.async_show_form(
            step_id="set_options",
            data_schema=vol.Schema(schema),
            errors=errors,
        )

    # Wire up both new and reconfigure flows to the same handler
    async_step_reconfigure = async_step_set_options
    async_step_user = async_step_set_options


def _build_conversation_schema(
    hass: HomeAssistant,
    is_new: bool,
    options: dict[str, Any],
) -> dict:
    """Build the voluptuous schema for conversation options.

    Args:
        hass: Home Assistant instance.
        is_new: Whether this is a new subentry (shows name field).
        options: Current option values for suggested defaults.

    Returns:
        Schema dict for the config flow form.
    """
    # Get available HA LLM APIs for the "Control Home Assistant" selector
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(label=api.name, value=api.id)
        for api in llm.async_get_apis(hass)
    ]

    # Resolve suggested LLM APIs
    suggested_llm_apis = options.get(CONF_LLM_HASS_API)
    if isinstance(suggested_llm_apis, str):
        suggested_llm_apis = [suggested_llm_apis]

    schema: dict = {}

    # Name field (only for new subentries)
    if is_new:
        schema[vol.Required(
            CONF_NAME,
            default=options.get(CONF_NAME, DEFAULT_CONVERSATION_NAME),
        )] = str

    # System prompt
    schema[vol.Optional(
        CONF_PROMPT,
        description={
            "suggested_value": options.get(CONF_PROMPT, "")
        },
    )] = TemplateSelector()

    # Control Home Assistant (LLM API selection)
    schema[vol.Optional(
        CONF_LLM_HASS_API,
        description={"suggested_value": suggested_llm_apis},
    )] = SelectSelector(
        SelectSelectorConfig(options=hass_apis, multiple=True)
    )

    # Recommended settings toggle
    schema[vol.Required(
        CONF_RECOMMENDED,
        default=options.get(CONF_RECOMMENDED, False),
    )] = bool

    # If recommended, skip advanced options
    if options.get(CONF_RECOMMENDED):
        return schema

    # Offline provider options
    provider_options = [
        SelectOptionDict(label=p.title(), value=p)
        for p in OFFLINE_PROVIDERS
    ]
    fallback_provider_options = [
        SelectOptionDict(label=p.title(), value=p)
        for p in OFFLINE_PROVIDERS
        if p != PROVIDER_NONE
    ]
    schema[vol.Optional(
        CONF_OFFLINE_PROVIDER,
        description={
            "suggested_value": options.get(CONF_OFFLINE_PROVIDER, PROVIDER_NONE)
        },
        default=PROVIDER_NONE,
    )] = SelectSelector(
        SelectSelectorConfig(
            mode=SelectSelectorMode.DROPDOWN,
            options=provider_options,
        )
    )

    schema[vol.Optional(
        CONF_OFFLINE_FALLBACK_PROVIDERS,
        description={
            "suggested_value": options.get(CONF_OFFLINE_FALLBACK_PROVIDERS, [])
        },
    )] = SelectSelector(
        SelectSelectorConfig(
            mode=SelectSelectorMode.DROPDOWN,
            options=fallback_provider_options,
            multiple=True,
        )
    )

    # Legacy shared API key/model remain visible for backward compatibility.
    schema[vol.Optional(
        CONF_OFFLINE_API_KEY,
        description={
            "suggested_value": options.get(CONF_OFFLINE_API_KEY, "")
        },
    )] = TextSelector(
        TextSelectorConfig(type=TextSelectorType.PASSWORD)
    )

    schema[vol.Optional(
        CONF_OFFLINE_MODEL,
        description={
            "suggested_value": options.get(CONF_OFFLINE_MODEL, "")
        },
    )] = str

    schema[vol.Optional(
        CONF_OFFLINE_OPENAI_API_KEY,
        description={
            "suggested_value": options.get(CONF_OFFLINE_OPENAI_API_KEY, "")
        },
    )] = TextSelector(
        TextSelectorConfig(type=TextSelectorType.PASSWORD)
    )
    schema[vol.Optional(
        CONF_OFFLINE_OPENAI_MODEL,
        description={
            "suggested_value": options.get(
                CONF_OFFLINE_OPENAI_MODEL,
                DEFAULT_MODELS[PROVIDER_OPENAI],
            )
        },
        default=DEFAULT_MODELS[PROVIDER_OPENAI],
    )] = str

    schema[vol.Optional(
        CONF_OFFLINE_GOOGLE_API_KEY,
        description={
            "suggested_value": options.get(CONF_OFFLINE_GOOGLE_API_KEY, "")
        },
    )] = TextSelector(
        TextSelectorConfig(type=TextSelectorType.PASSWORD)
    )
    schema[vol.Optional(
        CONF_OFFLINE_GOOGLE_MODEL,
        description={
            "suggested_value": options.get(
                CONF_OFFLINE_GOOGLE_MODEL,
                DEFAULT_MODELS[PROVIDER_GOOGLE],
            )
        },
        default=DEFAULT_MODELS[PROVIDER_GOOGLE],
    )] = str

    schema[vol.Optional(
        CONF_OFFLINE_ANTHROPIC_API_KEY,
        description={
            "suggested_value": options.get(CONF_OFFLINE_ANTHROPIC_API_KEY, "")
        },
    )] = TextSelector(
        TextSelectorConfig(type=TextSelectorType.PASSWORD)
    )
    schema[vol.Optional(
        CONF_OFFLINE_ANTHROPIC_MODEL,
        description={
            "suggested_value": options.get(
                CONF_OFFLINE_ANTHROPIC_MODEL,
                DEFAULT_MODELS[PROVIDER_ANTHROPIC],
            )
        },
        default=DEFAULT_MODELS[PROVIDER_ANTHROPIC],
    )] = str

    schema[vol.Optional(
        CONF_OFFLINE_OLLAMA_MODEL,
        description={
            "suggested_value": options.get(
                CONF_OFFLINE_OLLAMA_MODEL,
                DEFAULT_MODELS[PROVIDER_OLLAMA],
            )
        },
        default=DEFAULT_MODELS[PROVIDER_OLLAMA],
    )] = str

    schema[vol.Optional(
        CONF_OLLAMA_URL,
        description={
            "suggested_value": options.get(
                CONF_OLLAMA_URL,
                DEFAULT_OLLAMA_URL,
            )
        },
        default=DEFAULT_OLLAMA_URL,
    )] = TextSelector(
        TextSelectorConfig(type=TextSelectorType.URL)
    )

    return schema
