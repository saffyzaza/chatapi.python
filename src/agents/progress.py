"""Agent Progress Tracking — queue-based SSE event system."""
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Literal
from contextlib import contextmanager

_progress_queues: dict[str, queue.Queue] = {}
_lock = threading.Lock()


@dataclass
class AgentProgress:
    request_id: str
    agent_name: str
    agent_icon: str
    status: Literal["running", "done", "error"]
    message: str
    elapsed_seconds: float = 0.0
    order: int = 0


ACCIDENT_CHAT_PIPELINE_AGENTS = [
    {"name": "Accident SQL Agent", "icon": "🗄️", "order": 0},
    {"name": "Accident Answer Writer", "icon": "📋", "order": 1},
]

ZONE10_POLICY_PIPELINE_AGENTS = [
    {"name": "Zone 10 SQL Data Fetcher", "icon": "🗄️", "order": 0},
    {"name": "Zone 10 RTI Policy Analyst", "icon": "🔬", "order": 1},
    {"name": "Zone 10 RTI Policy Report Writer", "icon": "📋", "order": 2},
]

_ALL_AGENTS = ACCIDENT_CHAT_PIPELINE_AGENTS + ZONE10_POLICY_PIPELINE_AGENTS


def create_progress_queue(request_id: str) -> queue.Queue:
    with _lock:
        q = queue.Queue()
        _progress_queues[request_id] = q
        return q


def get_progress_queue(request_id: str) -> Optional[queue.Queue]:
    with _lock:
        return _progress_queues.get(request_id)


def remove_progress_queue(request_id: str) -> None:
    with _lock:
        _progress_queues.pop(request_id, None)


def emit_progress(
    request_id: str,
    agent_name: str,
    status: Literal["running", "done", "error"],
    message: str = "",
    elapsed_seconds: float = 0.0,
) -> None:
    q = get_progress_queue(request_id)
    if not q:
        return
    meta = next((a for a in _ALL_AGENTS if a["name"] == agent_name), None)
    event = AgentProgress(
        request_id=request_id,
        agent_name=agent_name,
        agent_icon=meta["icon"] if meta else "🤖",
        status=status,
        message=message,
        elapsed_seconds=elapsed_seconds,
        order=meta["order"] if meta else 0,
    )
    q.put(event)


@contextmanager
def track_agent(request_id: str, agent_name: str):
    start = time.time()
    emit_progress(request_id, agent_name, "running", "กำลังทำงาน...")
    try:
        yield
        elapsed = time.time() - start
        emit_progress(request_id, agent_name, "done", "เสร็จสิ้น", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        emit_progress(request_id, agent_name, "error", str(e)[:100], elapsed)
        raise
