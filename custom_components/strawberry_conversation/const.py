"""Constants for the Strawberry Conversation integration."""

import logging

DOMAIN = "strawberry_conversation"
LOGGER = logging.getLogger(__name__)

# Config keys
CONF_HUB_URL = "hub_url"
CONF_HUB_TOKEN = "hub_token"
CONF_OFFLINE_PROVIDER = "offline_provider"
CONF_OFFLINE_API_KEY = "offline_api_key"
CONF_OFFLINE_MODEL = "offline_model"
CONF_OLLAMA_URL = "ollama_url"
CONF_PROMPT = "prompt"
CONF_RECOMMENDED = "recommended"

# Defaults
DEFAULT_HUB_URL = "http://localhost:8000"
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
DEFAULT_TITLE = "Strawberry AI"
DEFAULT_CONVERSATION_NAME = "Strawberry Conversation"

# Offline LLM provider choices
PROVIDER_GOOGLE = "google"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OLLAMA = "ollama"
PROVIDER_NONE = "none"

OFFLINE_PROVIDERS = [
    PROVIDER_NONE,
    PROVIDER_GOOGLE,
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_OLLAMA,
]

# Default models per provider
DEFAULT_MODELS = {
    PROVIDER_GOOGLE: "gemini-2.5-flash-lite",
    PROVIDER_OPENAI: "gpt-4o-mini",
    PROVIDER_ANTHROPIC: "claude-sonnet-4-20250514",
    PROVIDER_OLLAMA: "llama3.2:3b",
}

# Timeouts (seconds)
HUB_CONNECT_TIMEOUT = 2.0
HUB_READ_TIMEOUT = 60.0
HUB_HEALTH_TIMEOUT = 1.5
OFFLINE_CACHE_TTL = 30  # seconds to cache Hub offline status
ONLINE_CACHE_TTL = 60  # seconds to cache Hub online status

# Agent loop
MAX_TOOL_ITERATIONS = 10

# Recommended conversation subentry options
RECOMMENDED_CONVERSATION_OPTIONS = {
    CONF_RECOMMENDED: True,
}
