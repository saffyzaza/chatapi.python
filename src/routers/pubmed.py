"""PubMed Router — SSE streaming endpoint for medical literature search."""
import asyncio
import json
import threading
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.schemas.pubmed import PubMedRequest

router = APIRouter(tags=["pubmed"])


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


def _thread_pipeline(
    prompt: str, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop,
    session_id: str = "", retmax: int = 10,
) -> None:
    from src.agents.pubmed_agent import run_pubmed_pipeline

    # ⚠️ ดึงความจำการสนทนาจาก session เดียวกับหน้าแชทหลัก เหมือน thaijo router —
    # ผู้ใช้อาจสลับมาที่ PubMed จากบทสนทนาเดิม (sessionId เดียวกัน)
    history_context = ""
    if session_id:
        from src.history import get_history, build_history_context
        history_context = build_history_context(get_history(session_id))

    try:
        run_pubmed_pipeline(
            prompt=prompt, queue=queue, loop=loop,
            session_id=session_id, retmax=retmax,
            history_context=history_context,
        )
    except Exception as exc:
        asyncio.run_coroutine_threadsafe(
            queue.put({"type": "error", "message": str(exc)}), loop
        )
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)


@router.post("/api/pubmed")
async def pubmed_search(request: PubMedRequest) -> StreamingResponse:
    """Stream PubMed research pipeline (live NCBI E-utilities API)."""
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    threading.Thread(
        target=_thread_pipeline,
        args=(request.prompt, queue, loop, request.sessionId, request.retmax),
        daemon=True,
    ).start()

    return _stream_response(queue)
