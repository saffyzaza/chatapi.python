"""Centralized Agent defaults and 429-aware crew kickoff helper."""
import logging
import time
import asyncio

from src.config import get_settings

logger = logging.getLogger(__name__)


def _patch_gemini_429_backoff() -> None:
    """Monkey-patch CrewAI Agent to sleep on 429 before retrying."""
    try:
        from crewai.agent.core import Agent as CrewAgent
    except ImportError:
        logger.debug("crewai.agent.core.Agent not found — skipping 429 backoff patch")
        return

    if getattr(CrewAgent, "_429_patched", False):
        return

    _original_handle = CrewAgent._handle_execution_error
    _original_handle_async = CrewAgent._handle_execution_error_async

    def _get_delay(agent_instance):
        base_delay = get_settings().GEMINI_RETRY_DELAY
        attempt = getattr(agent_instance, "_times_executed", 0)
        delay = min(base_delay * (2 ** attempt), 300)
        logger.warning(
            "429 RESOURCE_EXHAUSTED — sleeping %ds before retry (attempt %d)",
            delay, attempt + 1,
        )
        return delay

    def _patched_handle(self, e, task, context, tools):
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            time.sleep(_get_delay(self))
        return _original_handle(self, e, task, context, tools)

    async def _patched_handle_async(self, e, task, context, tools):
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            await asyncio.sleep(_get_delay(self))
        return await _original_handle_async(self, e, task, context, tools)

    CrewAgent._handle_execution_error = _patched_handle
    CrewAgent._handle_execution_error_async = _patched_handle_async
    CrewAgent._429_patched = True
    logger.info("CrewAI 429 backoff patch applied")


_patch_gemini_429_backoff()


def agent_retry_kwargs() -> dict:
    s = get_settings()
    return {"max_retry_limit": s.GEMINI_RETRY_LIMIT}


def kickoff_with_retry(crew, max_attempts: int = 3):
    """Run crew.kickoff() with automatic retry on 429 RESOURCE_EXHAUSTED."""
    delay = get_settings().GEMINI_RETRY_DELAY
    for attempt in range(1, max_attempts + 1):
        try:
            return crew.kickoff()
        except Exception as e:
            err = str(e)
            if ("429" in err or "RESOURCE_EXHAUSTED" in err) and attempt < max_attempts:
                logger.warning(
                    "429 on attempt %d/%d — sleeping %ds", attempt, max_attempts, delay
                )
                time.sleep(delay)
            else:
                raise
