# Strawberry AI Conversation for Home Assistant

A custom Home Assistant integration that adds a **conversation agent** powered
by the [Strawberry AI Assistant](https://github.com/aamott/strawberry-assistant)
Hub.

## Features

- **Hub-powered conversations** — Routes messages through the Strawberry Hub,
  which has access to all connected Spokes' skills (weather, music, system
  control, etc.)
- **Home Assistant control** — Optionally expose HA's Assist API so the AI can
  control your smart home devices
- **Offline fallback** (Phase 2) — When the Hub is unreachable, falls back to a
  local LLM via TensorZero with HA's native Assist tools
- **Streaming responses** — Real-time streaming from Hub to HA's conversation UI
- **HACS compatible** — Install via the Home Assistant Community Store

## Installation

### HACS (Recommended)

1. Open HACS in your Home Assistant
2. Click **Integrations** → **⋮** → **Custom repositories**
3. Add this repository URL and select **Integration** as the category
4. Install **Strawberry AI Conversation**
5. Restart Home Assistant

### Manual

1. Copy `custom_components/strawberry_conversation/` to your HA
   `custom_components/` directory
2. Restart Home Assistant

## Configuration

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for **Strawberry AI Conversation**
3. Enter your Hub URL and device token
4. Configure conversation options (system prompt, HA control, offline provider)

### Getting a Device Token

1. Register a device with your Strawberry Hub (see Hub documentation)
2. Copy the JWT device token
3. Paste it into the integration config flow

## Architecture

```
User ──► HA Assist Pipeline ──► StrawberryConversationEntity
                                        │
                              ┌─────────┴──────────┐
                              │                     │
                         Hub Online?           Hub Offline?
                              │                     │
                    ┌─────────▼──────────┐ ┌────────▼─────────┐
                    │ Hub Agent Loop     │ │ Local Agent Loop  │
                    │ (all Spoke skills) │ │ (TensorZero +     │
                    │                    │ │  HA Assist tools)  │
                    └────────────────────┘ └───────────────────┘
```

When online, the Hub's agent loop has full access to skills across all
connected Spokes (weather, music, system control, etc.). When offline,
the local agent uses TensorZero's embedded gateway to call an LLM directly
and can only control HA entities via the native Assist API.

## Requirements

- Home Assistant 2024.7.0+
- Strawberry Hub running and accessible on the network
- A registered device token from the Hub

## Development

See [Design Doc](../docs/plans/ha-assist-integration.md) for detailed
architecture, sequence diagrams, and phased implementation plan.
