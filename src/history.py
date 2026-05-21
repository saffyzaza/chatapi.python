"""Session history store — เก็บประวัติการสนทนาต่อ session."""
from typing import Any

_session_history: dict[str, list[dict[str, str]]] = {}
_MAX_HISTORY_TURNS = 6


def get_history(session_id: str) -> list[dict[str, str]]:
    return _session_history.get(session_id, [])


def append_history(session_id: str, role: str, text: str) -> None:
    history = _session_history.setdefault(session_id, [])
    history.append({"role": role, "text": text})
    if len(history) > _MAX_HISTORY_TURNS * 2:
        _session_history[session_id] = history[-(_MAX_HISTORY_TURNS * 2):]


def build_history_context(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    lines = []
    for msg in history:
        role_label = "ผู้ใช้" if msg.get("role") == "user" else "AI"
        lines.append(f"{role_label}: {msg.get('text', '').strip()}")
    return "ประวัติการสนทนาก่อนหน้า:\n" + "\n".join(lines)
