"""Constants for the Strawberry Conversation integration."""

import logging

DOMAIN = "strawberry_conversation"
LOGGER = logging.getLogger(__name__)

# Config keys
CONF_HUB_URL = "hub_url"
CONF_HUB_TOKEN = "hub_token"
CONF_OFFLINE_PROVIDER = "offline_provider"
CONF_OFFLINE_FALLBACK_PROVIDERS = "offline_fallback_providers"
CONF_OFFLINE_API_KEY = "offline_api_key"
CONF_OFFLINE_OPENAI_API_KEY = "offline_openai_api_key"
CONF_OFFLINE_GOOGLE_API_KEY = "offline_google_api_key"
CONF_OFFLINE_ANTHROPIC_API_KEY = "offline_anthropic_api_key"
CONF_OFFLINE_MODEL = "offline_model"
CONF_OFFLINE_OPENAI_MODEL = "offline_openai_model"
CONF_OFFLINE_GOOGLE_MODEL = "offline_google_model"
CONF_OFFLINE_ANTHROPIC_MODEL = "offline_anthropic_model"
CONF_OFFLINE_OLLAMA_MODEL = "offline_ollama_model"
CONF_OFFLINE_BACKEND = "offline_backend"
CONF_TENSORZERO_FUNCTION_NAME = "tensorzero_function_name"
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

OFFLINE_BACKEND_AUTO = "auto"
OFFLINE_BACKEND_OPENAI_COMPAT = "openai_compat"
OFFLINE_BACKEND_TENSORZERO = "tensorzero"

OFFLINE_PROVIDERS = [
    PROVIDER_NONE,
    PROVIDER_GOOGLE,
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_OLLAMA,
]

OFFLINE_BACKENDS = [
    OFFLINE_BACKEND_AUTO,
    OFFLINE_BACKEND_OPENAI_COMPAT,
    OFFLINE_BACKEND_TENSORZERO,
]

# Default models per provider
DEFAULT_MODELS = {
    PROVIDER_GOOGLE: "gemini-2.5-flash-lite",
    PROVIDER_OPENAI: "gpt-4o-mini",
    PROVIDER_ANTHROPIC: "claude-haiku-4-5",
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
