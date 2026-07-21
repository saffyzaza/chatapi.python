"""Session history store — Redis-backed, shared across all workers."""
import json
import logging
from typing import Any

import redis as redis_lib

from src.config import get_settings

logger = logging.getLogger(__name__)

_MAX_HISTORY_TURNS = 6
_TTL_SECONDS = 60 * 60 * 24  # 24 hours

_redis_client: redis_lib.Redis | None = None


def _get_redis() -> redis_lib.Redis:
    global _redis_client
    if _redis_client is None:
        s = get_settings()
        _redis_client = redis_lib.from_url(s.REDIS_URL, decode_responses=True)
    return _redis_client


def get_history(session_id: str) -> list[dict[str, str]]:
    if not session_id:
        return []
    try:
        data = _get_redis().get(f"session:{session_id}")
        return json.loads(data) if data else []
    except Exception:
        logger.warning("Redis unavailable — returning empty history")
        return []


def append_history(session_id: str, role: str, text: str) -> None:
    if not session_id:
        return
    try:
        r = _get_redis()
        history = get_history(session_id)
        history.append({"role": role, "text": text})
        if len(history) > _MAX_HISTORY_TURNS * 2:
            history = history[-(_MAX_HISTORY_TURNS * 2):]
        r.set(f"session:{session_id}", json.dumps(history, ensure_ascii=False), ex=_TTL_SECONDS)
    except Exception:
        logger.warning("Redis unavailable — history not saved")


def build_history_context(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    lines = []
    for msg in history:
        role_label = "ผู้ใช้" if msg.get("role") == "user" else "AI"
        lines.append(f"{role_label}: {msg.get('text', '').strip()}")
    return "ประวัติการสนทนาก่อนหน้า:\n" + "\n".join(lines)


def get_verified_file_ids(session_id: str) -> set[str]:
    """file_id ทั้งหมดที่เคยผ่าน relevance gate แล้วในเซสชันนี้ (ข้าม gate ซ้ำได้)."""
    if not session_id:
        return set()
    try:
        data = _get_redis().get(f"session:{session_id}:verified_files")
        return set(json.loads(data)) if data else set()
    except Exception:
        logger.warning("Redis unavailable — returning empty verified-file set")
        return set()


def mark_file_verified(session_id: str, file_id: str) -> None:
    if not session_id or not file_id:
        return
    try:
        r = _get_redis()
        verified = get_verified_file_ids(session_id)
        verified.add(file_id)
        r.set(f"session:{session_id}:verified_files", json.dumps(list(verified)), ex=_TTL_SECONDS)
    except Exception:
        logger.warning("Redis unavailable — verified file not saved")
