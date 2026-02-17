"""Local agent loop for offline fallback (Phase 2).

When the Strawberry Hub is unreachable, this module provides a local
agent loop using the TensorZero embedded gateway with HA's native
Assist API tools.

Phase 2 implementation will include:
- TensorZero embedded gateway initialization
- HA llm.Tool → TensorZero tool format conversion
- Agent loop with tool call execution via ChatLog
- Streaming delta support
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.components.conversation import ChatLog
    from homeassistant.helpers.llm import APIInstance

logger = logging.getLogger(__name__)


async def run_local_agent_loop(
    chat_log: ChatLog,
    api_instance: APIInstance | None,
    system_prompt: str,
    agent_id: str,
    max_iterations: int = 10,
) -> str | None:
    """Run a local agent loop with TensorZero + HA Assist tools.

    This is a Phase 2 feature. Currently returns None to signal
    that local mode is not yet available.

    Args:
        chat_log: HA ChatLog with conversation history.
        api_instance: HA LLM APIInstance with Assist tools.
        system_prompt: System prompt for the LLM.
        agent_id: Entity ID of the conversation agent.
        max_iterations: Maximum tool call iterations.

    Returns:
        Final response content, or None if not available.
    """
    # Phase 2: Initialize TensorZero embedded gateway
    # Phase 2: Convert api_instance.tools to TensorZero tool format
    # Phase 2: Run agent loop with tool execution via chat_log
    logger.warning(
        "Local agent loop not yet implemented (Phase 2). "
        "Hub must be available for conversation."
    )
    return None
