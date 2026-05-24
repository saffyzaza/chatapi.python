"""Redis cache for ThaiJo PDF summaries."""
import hashlib
import logging
import os

logger = logging.getLogger("python-ai.thaijo_cache")

_redis_client = None


def _get_client():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        from redis import Redis
        from redis.exceptions import RedisError
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = Redis.from_url(url, decode_responses=True)
        client.ping()
        _redis_client = client
        logger.info("ThaiJo Redis cache connected: %s", url)
    except Exception as e:
        logger.warning("ThaiJo Redis unavailable — cache disabled: %s", e)
        _redis_client = None
    return _redis_client


def _key(pdf_url: str) -> str:
    digest = hashlib.sha256(pdf_url.encode()).hexdigest()
    return f"thaijo_pdf:{digest}"


def get_cached_summary(pdf_url: str) -> str | None:
    client = _get_client()
    if not client:
        return None
    try:
        return client.get(_key(pdf_url))
    except Exception as e:
        logger.warning("Redis get failed: %s", e)
        return None


def save_cached_summary(pdf_url: str, summary: str) -> None:
    client = _get_client()
    if not client:
        return
    try:
        client.set(_key(pdf_url), summary, ex=60 * 60 * 24 * 7)  # 7 วัน
    except Exception as e:
        logger.warning("Redis set failed: %s", e)
