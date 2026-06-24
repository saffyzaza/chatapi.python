"""Tools Router — SSE streaming endpoints for 4 new chat modes.

Endpoints:
  POST /api/compare  — เปรียบเทียบ 2 CSV datasets
  POST /api/report   — สร้างรายงาน Thai Markdown จาก CSV
  POST /api/workplan — สร้างแผนงาน HTML (Pure LLM, no CSV)
  POST /api/database — วิเคราะห์ไฟล์ที่แนบมาจาก MinIO
"""
import asyncio
import json
import threading
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.schemas.tools import CompareRequest, ReportRequest, WorkplanRequest, DatabaseRequest

router = APIRouter(tags=["tools"])


# ── Shared SSE helper ──────────────────────────────────────────────────────────

def _stream_response(queue: asyncio.Queue) -> StreamingResponse:
    async def stream():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ── POST /api/compare ──────────────────────────────────────────────────────────

def _build_history_context(session_id: str, client_history: list | None) -> str:
    """Merge client-sent history with Redis history, return formatted context string."""
    from src.history import get_history, build_history_context
    raw = client_history or get_history(session_id)
    if raw and raw[-1].get("role") in ("user", "human"):
        raw = raw[:-1]
    return build_history_context(raw)


def _thread_compare(
    prompt: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    session_id: str = "",
    history_context: str = "",
) -> None:
    from src.agents.compare_agent import run_compare_pipeline
    try:
        run_compare_pipeline(
            prompt=prompt,
            queue=queue,
            loop=loop,
            session_id=session_id,
            history_context=history_context,
        )
    except Exception as exc:
        asyncio.run_coroutine_threadsafe(
            queue.put({"type": "error", "message": str(exc)}), loop
        )
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)


@router.post("/api/compare")
async def compare_datasets(request: CompareRequest) -> StreamingResponse:
    """Stream comparison analysis pipeline between two CSV datasets."""
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()
    client_history = [{"role": m["role"], "text": m.get("text", "")} for m in (request.history or [])]
    history_context = _build_history_context(request.sessionId, client_history or None)

    threading.Thread(
        target=_thread_compare,
        args=(request.prompt, queue, loop, request.sessionId, history_context),
        daemon=True,
    ).start()

    return _stream_response(queue)


# ── POST /api/report ───────────────────────────────────────────────────────────

def _thread_report(
    prompt: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    session_id: str = "",
    history_context: str = "",
) -> None:
    from src.agents.report_agent import run_report_pipeline
    try:
        run_report_pipeline(
            prompt=prompt,
            queue=queue,
            loop=loop,
            session_id=session_id,
            history_context=history_context,
        )
    except Exception as exc:
        asyncio.run_coroutine_threadsafe(
            queue.put({"type": "error", "message": str(exc)}), loop
        )
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)


@router.post("/api/report")
async def generate_report(request: ReportRequest) -> StreamingResponse:
    """Stream comprehensive Thai Markdown report generation from a CSV dataset."""
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()
    client_history = [{"role": m["role"], "text": m.get("text", "")} for m in (request.history or [])]
    history_context = _build_history_context(request.sessionId, client_history or None)

    threading.Thread(
        target=_thread_report,
        args=(request.prompt, queue, loop, request.sessionId, history_context),
        daemon=True,
    ).start()

    return _stream_response(queue)


# ── POST /api/workplan ─────────────────────────────────────────────────────────

def _thread_workplan(
    prompt: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    session_id: str = "",
    doc_type: str = "workplan",
    history_context: str = "",
) -> None:
    from src.agents.workplan_agent import run_workplan_pipeline
    try:
        run_workplan_pipeline(
            prompt=prompt,
            queue=queue,
            loop=loop,
            session_id=session_id,
            doc_type=doc_type,
            history_context=history_context,
        )
    except Exception as exc:
        asyncio.run_coroutine_threadsafe(
            queue.put({"type": "error", "message": str(exc)}), loop
        )
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)


@router.post("/api/workplan")
async def generate_workplan(request: WorkplanRequest) -> StreamingResponse:
    """Stream Thai work plan HTML generation (pure LLM, no CSV needed)."""
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()
    client_history = [{"role": m["role"], "text": m.get("text", "")} for m in (request.history or [])]
    history_context = _build_history_context(request.sessionId, client_history or None)

    threading.Thread(
        target=_thread_workplan,
        args=(request.prompt, queue, loop, request.sessionId, request.doc_type, history_context),
        daemon=True,
    ).start()

    return _stream_response(queue)


# ── POST /api/database ─────────────────────────────────────────────────────────

def _thread_database(
    prompt: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    session_id: str = "",
    attached_files: list[dict] = [],
    history_context: str = "",
) -> None:
    from src.agents.database_agent import run_database_pipeline
    try:
        run_database_pipeline(
            prompt=prompt,
            queue=queue,
            loop=loop,
            session_id=session_id,
            attached_files=attached_files,
            history_context=history_context,
        )
    except Exception as exc:
        asyncio.run_coroutine_threadsafe(
            queue.put({"type": "error", "message": str(exc)}), loop
        )
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)


@router.post("/api/database")
async def analyze_database_file(request: DatabaseRequest) -> StreamingResponse:
    """Stream analysis pipeline for user-attached MinIO files."""
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()
    client_history = [{"role": m["role"], "text": m.get("text", "")} for m in (request.history or [])]
    history_context = _build_history_context(request.sessionId, client_history or None)

    threading.Thread(
        target=_thread_database,
        args=(request.prompt, queue, loop, request.sessionId, request.attached_files, history_context),
        daemon=True,
    ).start()

    return _stream_response(queue)
