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
from homeassistant.data_entry_flow import section
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
    CONF_OFFLINE_BACKEND,
    CONF_OFFLINE_FALLBACK_PROVIDERS,
    CONF_OFFLINE_GOOGLE_API_KEY,
    CONF_OFFLINE_GOOGLE_MODEL,
    CONF_OFFLINE_OLLAMA_MODEL,
    CONF_OFFLINE_OPENAI_API_KEY,
    CONF_OFFLINE_OPENAI_MODEL,
    CONF_OFFLINE_PROVIDER,
    CONF_OLLAMA_URL,
    CONF_PROMPT,
    CONF_ADVANCED_FALLBACK,
    CONF_TENSORZERO_FUNCTION_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_HUB_URL,
    DEFAULT_MODELS,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TITLE,
    DOMAIN,
    OFFLINE_BACKEND_AUTO,
    OFFLINE_BACKENDS,
    OFFLINE_PROVIDERS,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE,
    PROVIDER_NONE,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
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

            self.last_rendered_recommended = options.get(CONF_ADVANCED_FALLBACK, False)

        if user_input is not None:
            # If the user toggled the advanced settings, re-render the form
            if user_input.get(CONF_ADVANCED_FALLBACK) != self.last_rendered_recommended:
                # Clean up empty optional fields
                if not user_input.get(CONF_LLM_HASS_API):
                    user_input.pop(CONF_LLM_HASS_API, None)

                for field in _PROVIDER_API_KEY_FIELD_MAP.values():
                    if not user_input.get(field):
                        user_input.pop(field, None)

                for field in _PROVIDER_MODEL_FIELD_MAP.values():
                    if not user_input.get(field):
                        user_input.pop(field, None)

                if user_input.get(CONF_OFFLINE_PROVIDER) == PROVIDER_NONE:
                    user_input.pop(CONF_OFFLINE_BACKEND, None)
                    user_input.pop(CONF_TENSORZERO_FUNCTION_NAME, None)
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
            self.last_rendered_recommended = user_input.get(CONF_ADVANCED_FALLBACK, False)
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


def _build_routing_section(
    options: dict[str, Any],
    backend_options: list[SelectOptionDict],
    provider_options: list[SelectOptionDict],
    fallback_provider_options: list[SelectOptionDict],
) -> vol.Schema:
    return section(
        vol.Schema({
            vol.Optional(
                CONF_OFFLINE_BACKEND,
                description={
                    "suggested_value": options.get(CONF_OFFLINE_BACKEND, OFFLINE_BACKEND_AUTO)
                },
                default=OFFLINE_BACKEND_AUTO,
            ): SelectSelector(
                SelectSelectorConfig(
                    mode=SelectSelectorMode.DROPDOWN,
                    options=backend_options,
                )
            ),
            vol.Optional(
                CONF_TENSORZERO_FUNCTION_NAME,
                description={
                    "suggested_value": options.get(CONF_TENSORZERO_FUNCTION_NAME, "chat")
                },
                default="chat",
            ): str,
            vol.Optional(
                CONF_OFFLINE_PROVIDER,
                description={
                    "suggested_value": options.get(CONF_OFFLINE_PROVIDER, PROVIDER_NONE)
                },
                default=PROVIDER_NONE,
            ): SelectSelector(
                SelectSelectorConfig(
                    mode=SelectSelectorMode.DROPDOWN,
                    options=provider_options,
                )
            ),
            vol.Optional(
                CONF_OFFLINE_FALLBACK_PROVIDERS,
                description={
                    "suggested_value": options.get(CONF_OFFLINE_FALLBACK_PROVIDERS, [])
                },
            ): SelectSelector(
                SelectSelectorConfig(
                    mode=SelectSelectorMode.DROPDOWN,
                    options=fallback_provider_options,
                    multiple=True,
                )
            ),
        }),
        {"collapsed": False}
    )


def _build_openai_section(options: dict[str, Any]) -> vol.Schema:
    return section(
        vol.Schema({
            vol.Optional(
                CONF_OFFLINE_OPENAI_API_KEY,
                description={
                    "suggested_value": options.get(CONF_OFFLINE_OPENAI_API_KEY, "")
                },
            ): TextSelector(
                TextSelectorConfig(type=TextSelectorType.PASSWORD)
            ),
            vol.Optional(
                CONF_OFFLINE_OPENAI_MODEL,
                description={
                    "suggested_value": options.get(
                        CONF_OFFLINE_OPENAI_MODEL,
                        DEFAULT_MODELS[PROVIDER_OPENAI],
                    )
                },
                default=DEFAULT_MODELS[PROVIDER_OPENAI],
            ): str,
        }),
        {"collapsed": True}
    )


def _build_google_section(options: dict[str, Any]) -> vol.Schema:
    return section(
        vol.Schema({
            vol.Optional(
                CONF_OFFLINE_GOOGLE_API_KEY,
                description={
                    "suggested_value": options.get(CONF_OFFLINE_GOOGLE_API_KEY, "")
                },
            ): TextSelector(
                TextSelectorConfig(type=TextSelectorType.PASSWORD)
            ),
            vol.Optional(
                CONF_OFFLINE_GOOGLE_MODEL,
                description={
                    "suggested_value": options.get(
                        CONF_OFFLINE_GOOGLE_MODEL,
                        DEFAULT_MODELS[PROVIDER_GOOGLE],
                    )
                },
                default=DEFAULT_MODELS[PROVIDER_GOOGLE],
            ): str,
        }),
        {"collapsed": True}
    )


def _build_anthropic_section(options: dict[str, Any]) -> vol.Schema:
    return section(
        vol.Schema({
            vol.Optional(
                CONF_OFFLINE_ANTHROPIC_API_KEY,
                description={
                    "suggested_value": options.get(CONF_OFFLINE_ANTHROPIC_API_KEY, "")
                },
            ): TextSelector(
                TextSelectorConfig(type=TextSelectorType.PASSWORD)
            ),
            vol.Optional(
                CONF_OFFLINE_ANTHROPIC_MODEL,
                description={
                    "suggested_value": options.get(
                        CONF_OFFLINE_ANTHROPIC_MODEL,
                        DEFAULT_MODELS[PROVIDER_ANTHROPIC],
                    )
                },
                default=DEFAULT_MODELS[PROVIDER_ANTHROPIC],
            ): str,
        }),
        {"collapsed": True}
    )


def _build_ollama_section(options: dict[str, Any]) -> vol.Schema:
    return section(
        vol.Schema({
            vol.Optional(
                CONF_OLLAMA_URL,
                description={
                    "suggested_value": options.get(CONF_OLLAMA_URL, DEFAULT_OLLAMA_URL)
                },
                default=DEFAULT_OLLAMA_URL,
            ): TextSelector(
                TextSelectorConfig(type=TextSelectorType.URL)
            ),
            vol.Optional(
                CONF_OFFLINE_OLLAMA_MODEL,
                description={
                    "suggested_value": options.get(
                        CONF_OFFLINE_OLLAMA_MODEL,
                        DEFAULT_MODELS[PROVIDER_OLLAMA],
                    )
                },
                default=DEFAULT_MODELS[PROVIDER_OLLAMA],
            ): str,
        }),
        {"collapsed": True}
    )


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

    # Advanced Fallback toggle
    schema[vol.Required(
        CONF_ADVANCED_FALLBACK,
        default=options.get(CONF_ADVANCED_FALLBACK, False),
    )] = bool

    # If not advanced, skip advanced options
    if not options.get(CONF_ADVANCED_FALLBACK):
        return schema

    # General Offline Routing
    backend_options = [
        SelectOptionDict(label=b.replace("_", " ").title(), value=b)
        for b in OFFLINE_BACKENDS
    ]
    provider_options = [
        SelectOptionDict(label=p.title(), value=p)
        for p in OFFLINE_PROVIDERS
    ]
    fallback_provider_options = [
        SelectOptionDict(label=p.title(), value=p)
        for p in OFFLINE_PROVIDERS
        if p != PROVIDER_NONE
    ]

    schema[vol.Required("routing_section")] = _build_routing_section(
        options, backend_options, provider_options, fallback_provider_options
    )
    schema[vol.Required("openai_section")] = _build_openai_section(options)
    schema[vol.Required("google_section")] = _build_google_section(options)
    schema[vol.Required("anthropic_section")] = _build_anthropic_section(options)
    schema[vol.Required("ollama_section")] = _build_ollama_section(options)

    return schema
