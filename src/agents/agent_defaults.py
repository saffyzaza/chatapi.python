"""Centralized Agent defaults and 429/503-aware crew kickoff helper."""
import logging
import random
import time
import asyncio

from src.config import get_settings

logger = logging.getLogger(__name__)


def _configure_litellm_retry() -> None:
    """Set litellm global retry so ALL LLM calls (including crewai.flow.runtime)
    retry on 429/503 before surfacing the error to callers."""
    try:
        import litellm
        litellm.num_retries = 3
        litellm.retry_on_failure = True
        # ServiceUnavailableError (503) and RateLimitError (429) are retried by litellm
        # when num_retries > 0; this covers crewai.flow.runtime.call_llm_native_tools
        logger.info("litellm global retry configured: num_retries=3, retry_on_failure=True")
    except Exception as exc:
        logger.debug("Could not configure litellm retry: %s", exc)


_configure_litellm_retry()


def _patch_gemini_429_backoff() -> None:
    """Monkey-patch CrewAI Agent to sleep on 429/503 before retrying."""
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
        # ⚠️ cap ลดจาก 300s → 45s — 300s (5 นาที) ต่อ retry เดียวไม่เหมาะกับแชท
        # interactive ที่ผู้ใช้รอผลอยู่หน้าจอ ดู GEMINI_RETRY_DELAY ใน config.py
        delay = min(base_delay * (2 ** attempt), 45)
        # ⚠️ Jitter กันหลาย request/worker คำนวณ delay ตรงกันเป๊ะแล้ว retry
        # พร้อมกันเป็นชุด (thundering herd) — ยิ่งซ้ำเติม 429 quota-per-minute ให้
        # เกินซ้ำไปอีกรอบ ทั้งที่ระบบตั้งใจจะ "หลบ" ช่วงโควตาตันอยู่แล้ว
        jittered_delay = delay + random.uniform(0, delay * 0.5)
        logger.warning(
            "429 RESOURCE_EXHAUSTED — sleeping %.1fs before retry (attempt %d)",
            jittered_delay, attempt + 1,
        )
        return jittered_delay

    def _patched_handle(self, e, task, context, tools):
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "503" in err or "UNAVAILABLE" in err:
            time.sleep(_get_delay(self))
        return _original_handle(self, e, task, context, tools)

    async def _patched_handle_async(self, e, task, context, tools):
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "503" in err or "UNAVAILABLE" in err:
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


_RETRYABLE_ERRORS = (
    "429",
    "RESOURCE_EXHAUSTED",
    "503",
    "UNAVAILABLE",
    "None or empty",
    "Invalid response from LLM",
)


def kickoff_with_retry(crew, max_attempts: int = 3):
    """Run crew.kickoff() with automatic retry on 429 RESOURCE_EXHAUSTED
    and Gemini None/empty response errors.

    If called from within a running event loop (e.g. FastAPI endpoint),
    executes in a separate thread to avoid the sync-in-async error.
    """
    import concurrent.futures

    delay = get_settings().GEMINI_RETRY_DELAY

    def _run():
        for attempt in range(1, max_attempts + 1):
            try:
                return crew.kickoff()
            except Exception as e:
                err = str(e)
                is_rate_limit = "429" in err or "RESOURCE_EXHAUSTED" in err
                is_overloaded = "503" in err or "UNAVAILABLE" in err
                is_retryable = is_rate_limit or is_overloaded or any(m in err for m in _RETRYABLE_ERRORS)
                if is_retryable and attempt < max_attempts:
                    if is_rate_limit:
                        sleep_secs = delay
                    elif is_overloaded:
                        sleep_secs = min(30 * attempt, 90)
                    else:
                        sleep_secs = 5
                    # ⚠️ Jitter เหตุผลเดียวกับใน _get_delay ด้านบน — กันหลาย
                    # pipeline ที่โดน 429/503 พร้อมกันแล้ว retry ประสานเวลากันเป๊ะ
                    sleep_secs += random.uniform(0, sleep_secs * 0.5)
                    logger.warning(
                        "Retryable error on attempt %d/%d (%s) — sleeping %.1fs",
                        attempt, max_attempts, err[:80], sleep_secs,
                    )
                    time.sleep(sleep_secs)
                else:
                    raise

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            return future.result()
    else:
        return _run()
