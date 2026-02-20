"""Dynamic TensorZero configuration generator for offline fallback.

Builds a ``tensorzero.toml`` and tool-parameter JSON schemas on the fly from
Home Assistant UI settings, then manages the lifecycle of the cached embedded
``AsyncTensorZeroGateway``.

The generated config uses ``dynamic::`` credential locations so API keys are
passed at inference time rather than baked into environment variables.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from typing import Any

from .const import (
    DEFAULT_MODELS,
    DEFAULT_OLLAMA_URL,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
)

try:
    from tensorzero import AsyncTensorZeroGateway
except Exception:  # pragma: no cover
    AsyncTensorZeroGateway = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ── TensorZero provider mapping ──────────────────────────────────────────────

# Maps HA provider name → TensorZero provider ``type`` field.
_PROVIDER_TZ_TYPE: dict[str, str] = {
    PROVIDER_OPENAI: "openai",
    PROVIDER_GOOGLE: "google_ai_studio_gemini",
    PROVIDER_ANTHROPIC: "anthropic",
    PROVIDER_OLLAMA: "openai",  # Ollama exposes an OpenAI-compatible API
}

# Maps HA provider name → dynamic credential key used in inference calls.
_PROVIDER_CREDENTIAL_KEY: dict[str, str] = {
    PROVIDER_OPENAI: "openai_api_key",
    PROVIDER_GOOGLE: "google_api_key",
    PROVIDER_ANTHROPIC: "anthropic_api_key",
}

# ── Module-level cached gateway state ────────────────────────────────────────

_cached_gateway: AsyncTensorZeroGateway | None = None
_cached_config_hash: str = ""
_cached_config_dir: str | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _escape_toml_string(value: str) -> str:
    """Escape a string for embedding in a TOML double-quoted value."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _compute_config_hash(
    provider_chain: list[str],
    provider_models: dict[str, str],
    ollama_url: str,
    tool_names: list[str],
    function_name: str,
) -> str:
    """Compute a SHA-256 hash of config inputs to detect changes."""
    data = json.dumps(
        {
            "providers": provider_chain,
            "models": provider_models,
            "ollama_url": ollama_url,
            "tools": sorted(tool_names),
            "function_name": function_name,
        },
        sort_keys=True,
    )
    return hashlib.sha256(data.encode()).hexdigest()


# ── TOML section builders ───────────────────────────────────────────────────


def _build_model_section(
    provider: str,
    model_name: str,
    ollama_url: str | None = None,
) -> str:
    """Build TOML ``[models.*]`` + ``[models.*.providers.*]`` for one provider.

    Args:
        provider: HA provider key (e.g. ``openai``, ``google``).
        model_name: LLM model identifier.
        ollama_url: Base URL when provider is Ollama.

    Returns:
        Multi-line TOML fragment.
    """
    tz_type = _PROVIDER_TZ_TYPE[provider]
    model_id = f"{provider}_model"
    provider_id = f"{provider}_provider"

    lines = [
        f"[models.{model_id}]",
        f'routing = ["{provider_id}"]',
        "",
        f"[models.{model_id}.providers.{provider_id}]",
        f'type = "{tz_type}"',
        f'model_name = "{_escape_toml_string(model_name)}"',
    ]

    if provider == PROVIDER_OLLAMA:
        # Ollama needs no API key and a custom base URL.
        lines.append('api_key_location = "none"')
        base = (ollama_url or DEFAULT_OLLAMA_URL).rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        lines.append(f'api_base = "{base}/"')
    elif provider in _PROVIDER_CREDENTIAL_KEY:
        cred_key = _PROVIDER_CREDENTIAL_KEY[provider]
        lines.append(f'api_key_location = "dynamic::{cred_key}"')

    lines.append("")
    return "\n".join(lines)


def _build_tool_section(
    tool_name: str,
    description: str,
    schema_path: str,
) -> str:
    """Build TOML ``[tools.*]`` section for one HA tool.

    Args:
        tool_name: Tool identifier.
        description: Human-readable tool description.
        schema_path: Absolute path to the JSON-Schema parameter file.

    Returns:
        Multi-line TOML fragment.
    """
    return (
        f'[tools."{_escape_toml_string(tool_name)}"]\n'
        f'description = "{_escape_toml_string(description)}"\n'
        f'parameters = "{_escape_toml_string(schema_path)}"\n'
    )


def _build_function_section(
    function_name: str,
    provider_chain: list[str],
    tool_names: list[str],
) -> str:
    """Build TOML function, variant, and experimentation sections.

    All variants are placed in ``fallback_variants`` for deterministic
    sequential fallback matching the user-configured provider order.

    Args:
        function_name: TensorZero function name (e.g. ``chat``).
        provider_chain: Ordered providers.
        tool_names: Tool names to attach to the function.

    Returns:
        Multi-line TOML fragment.
    """
    tools_str = ", ".join(f'"{t}"' for t in tool_names)

    lines = [
        f"[functions.{function_name}]",
        'type = "chat"',
    ]
    if tool_names:
        lines.append(f"tools = [{tools_str}]")
    lines.append("")

    # One variant per provider
    variant_names: list[str] = []
    for provider in provider_chain:
        variant_name = f"{provider}_variant"
        variant_names.append(variant_name)
        lines.extend(
            [
                f"[functions.{function_name}.variants.{variant_name}]",
                'type = "chat_completion"',
                f'model = "{provider}_model"',
                "",
            ]
        )

    # Sequential fallback: all variants in fallback_variants
    if variant_names:
        fallback_str = ", ".join(f'"{v}"' for v in variant_names)
        lines.extend(
            [
                f"[functions.{function_name}.experimentation]",
                'type = "uniform"',
                f"fallback_variants = [{fallback_str}]",
                "",
            ]
        )

    return "\n".join(lines)


# ── Public API ───────────────────────────────────────────────────────────────


def effective_provider_chain(
    provider_chain: list[str],
    provider_api_keys: dict[str, str | None],
) -> list[str]:
    """Filter provider chain to providers with usable credentials.

    Ollama is always kept (no API key needed). Cloud providers are kept
    only when an API key is available.

    Args:
        provider_chain: Full ordered provider list.
        provider_api_keys: Per-provider API key mapping.

    Returns:
        Filtered ordered list.
    """
    effective: list[str] = []
    for provider in provider_chain:
        if provider == PROVIDER_OLLAMA:
            effective.append(provider)
        elif provider in _PROVIDER_CREDENTIAL_KEY:
            if provider_api_keys.get(provider):
                effective.append(provider)
            else:
                logger.warning(
                    "Skipping provider %s in TZ config: no API key configured",
                    provider,
                )
        else:
            logger.warning("Unknown provider %s, skipping in TZ config", provider)
    return effective


def build_dynamic_config(
    provider_chain: list[str],
    provider_models: dict[str, str],
    ollama_url: str | None,
    tools: list[dict[str, Any]],
    function_name: str = "chat",
) -> tuple[str, str]:
    """Build a dynamic TensorZero config directory on disk.

    Creates a temp directory containing ``tensorzero.toml`` and per-tool
    JSON-Schema files under ``tools/``.

    Args:
        provider_chain: Ordered list of providers (already filtered).
        provider_models: Per-provider model name mapping.
        ollama_url: Ollama base URL (for Ollama provider).
        tools: HA tools in OpenAI schema format.
        function_name: TensorZero function name.

    Returns:
        Tuple of ``(config_dir_path, toml_file_path)``.
    """
    config_dir = tempfile.mkdtemp(prefix="tz_ha_")
    tools_dir = os.path.join(config_dir, "tools")
    os.makedirs(tools_dir, exist_ok=True)

    sections: list[str] = [
        "[gateway]",
        "observability.enabled = false",
        "",
    ]

    # Model sections
    for provider in provider_chain:
        model = provider_models.get(provider) or DEFAULT_MODELS.get(provider, "")
        sections.append(_build_model_section(provider, model, ollama_url))

    # Tool sections + JSON schema files
    tool_names: list[str] = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        if not name:
            continue
        tool_names.append(name)

        parameters = func.get("parameters", {"type": "object", "properties": {}})
        schema_path = os.path.join(tools_dir, f"{name}.json")
        with open(schema_path, "w") as fh:
            json.dump(parameters, fh)

        sections.append(
            _build_tool_section(name, func.get("description", ""), schema_path)
        )

    # Function section
    sections.append(
        _build_function_section(function_name, provider_chain, tool_names)
    )

    toml_content = "\n".join(sections)
    toml_path = os.path.join(config_dir, "tensorzero.toml")
    with open(toml_path, "w") as fh:
        fh.write(toml_content)

    logger.debug("Generated dynamic TZ config at %s", toml_path)
    return config_dir, toml_path


def build_credentials(
    provider_chain: list[str],
    provider_api_keys: dict[str, str | None],
) -> dict[str, str]:
    """Build ``credentials`` dict for ``gateway.inference()``.

    Maps each provider's dynamic credential key to the actual API key
    value from HA config.

    Args:
        provider_chain: Ordered providers.
        provider_api_keys: Per-provider API key mapping.

    Returns:
        Dict of ``{dynamic_key: api_key_value}``.
    """
    credentials: dict[str, str] = {}
    for provider in provider_chain:
        if provider in _PROVIDER_CREDENTIAL_KEY:
            api_key = provider_api_keys.get(provider)
            if api_key:
                credentials[_PROVIDER_CREDENTIAL_KEY[provider]] = api_key
    return credentials


async def get_or_build_gateway(
    provider_chain: list[str],
    provider_models: dict[str, str],
    ollama_url: str | None,
    tools: list[dict[str, Any]],
    function_name: str = "chat",
) -> AsyncTensorZeroGateway:
    """Return a cached gateway, or build a new one if config has changed.

    The gateway is rebuilt whenever the provider chain, models, tools, or
    Ollama URL change (detected via SHA-256 of the serialised parameters).

    Args:
        provider_chain: Ordered providers (already filtered).
        provider_models: Per-provider model names.
        ollama_url: Ollama base URL.
        tools: HA tools in OpenAI schema format.
        function_name: TensorZero function name.

    Returns:
        Ready-to-use ``AsyncTensorZeroGateway``.

    Raises:
        RuntimeError: If the ``tensorzero`` package is not installed.
    """
    global _cached_gateway, _cached_config_hash, _cached_config_dir  # noqa: PLW0603

    if AsyncTensorZeroGateway is None:
        raise RuntimeError("tensorzero package is not installed")

    tool_names = [
        t.get("function", {}).get("name", "")
        for t in tools
        if t.get("function", {}).get("name")
    ]

    config_hash = _compute_config_hash(
        provider_chain,
        {p: provider_models.get(p, "") for p in provider_chain},
        ollama_url or "",
        tool_names,
        function_name,
    )

    if _cached_gateway is not None and _cached_config_hash == config_hash:
        return _cached_gateway

    # Clean up previous temp directory
    if _cached_config_dir and os.path.isdir(_cached_config_dir):
        shutil.rmtree(_cached_config_dir, ignore_errors=True)

    config_dir, toml_path = build_dynamic_config(
        provider_chain=provider_chain,
        provider_models=provider_models,
        ollama_url=ollama_url,
        tools=tools,
        function_name=function_name,
    )

    logger.info(
        "Building TensorZero embedded gateway from dynamic config: %s", toml_path
    )

    _cached_gateway = await AsyncTensorZeroGateway.build_embedded(
        config_file=toml_path,
        async_setup=True,
    )
    _cached_config_hash = config_hash
    _cached_config_dir = config_dir

    return _cached_gateway


def reset_gateway_cache() -> None:
    """Reset the cached gateway (useful for tests and reconfiguration)."""
    global _cached_gateway, _cached_config_hash, _cached_config_dir  # noqa: PLW0603

    if _cached_config_dir and os.path.isdir(_cached_config_dir):
        shutil.rmtree(_cached_config_dir, ignore_errors=True)

    _cached_gateway = None
    _cached_config_hash = ""
    _cached_config_dir = None
