"""ThaiJo Router — SSE streaming endpoints for research journal generation."""
import asyncio
import json
import threading
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.schemas.thaijo import ThaiJoRequest, ThaiJoGenerateRequest, ThaiJoTopicsRequest

router = APIRouter(tags=["thaijo"])


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


def _thread_pipeline(prompt: str, queue: asyncio.Queue,
                     loop: asyncio.AbstractEventLoop,
                     session_id: str = "", use_mock: bool = False,
                     doc_type: str = "policy") -> None:
    from src.agents.thaijo_agent import run_thaijo_pipeline
    try:
        run_thaijo_pipeline(
            prompt=prompt, queue=queue, loop=loop,
            session_id=session_id, use_mock=use_mock, doc_type=doc_type,
        )
    except Exception as exc:
        asyncio.run_coroutine_threadsafe(
            queue.put({"type": "error", "message": str(exc)}), loop
        )
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)


# ── POST /api/thaijo — live search ────────────────────────────────────────────

@router.post("/api/thaijo")
async def thaijo_search(request: ThaiJoRequest) -> StreamingResponse:
    """Stream ThaiJo research pipeline (live ThaiJo API)."""
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    threading.Thread(
        target=_thread_pipeline,
        args=(request.prompt, queue, loop, request.sessionId, False, request.doc_type),
        daemon=True,
    ).start()

    return _stream_response(queue)


# ── POST /api/thaijo/report — generate report from fetched articles ───────────

@router.post("/api/thaijo/report")
async def thaijo_generate_report(request: ThaiJoGenerateRequest) -> StreamingResponse:
    """Generate structured report (policy / plan / workplan) from pre-fetched articles."""
    from src.agents.thaijo_agent import run_thaijo_report_pipeline

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def run() -> None:
        try:
            run_thaijo_report_pipeline(
                query=request.query,
                articles_text=request.articles_text,
                queue=queue, loop=loop,
                doc_type=request.doc_type,
                session_id=request.sessionId,
                topic_plan=request.topic_plan,
            )
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "error", "message": str(exc)}), loop
            )
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    threading.Thread(target=run, daemon=True).start()
    return _stream_response(queue)


# ── POST /api/thaijo/topics ─────────────────────────────────────────────────

@router.post("/api/thaijo/topics")
async def thaijo_plan_topics(request: ThaiJoTopicsRequest):
    """Generate AI-suggested topic headings for a report from pre-fetched articles."""
    from src.agents.thaijo_agent import run_topic_planner
    topics = run_topic_planner(
        query=request.query,
        articles_text=request.articles_text,
        doc_type=request.doc_type,
    )
    return {"topics": topics}


# ── POST /api/thaijo/demo — mock data ─────────────────────────────────────────

@router.post("/api/thaijo/demo")
async def thaijo_demo() -> StreamingResponse:
    """Stream ThaiJo demo pipeline using built-in mock articles (no API key needed)."""
    from src.agents.thaijo_agent import _DEMO_PROMPT

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    threading.Thread(
        target=_thread_pipeline,
        args=(_DEMO_PROMPT, queue, loop, "demo", True),
        daemon=True,
    ).start()

    return _stream_response(queue)
