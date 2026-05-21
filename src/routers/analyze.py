"""Analyze router — SSE streaming pipeline for health domain Q&A."""
import asyncio
import json
import os
import threading
from typing import Any

from crewai import Agent, LLM, Crew, Task
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.agents.router import route_domain, route_with_web_search
from src.agents.csv_pipeline import run_pipeline, _run_agent
from src.history import get_history, append_history, build_history_context
from src.schemas.analyze import AnalyzeRequest

router = APIRouter(tags=["analyze"])


def _get_llm() -> LLM:
    return LLM(model="gemini/gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))


def _orchestrate(
    prompt: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    session_id: str = "",
    client_history: list[dict[str, Any]] | None = None,
    mode: str = "normal",
) -> None:
    """Full pipeline entry point — runs in a background thread."""
    def put(ev: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(queue.put(ev), loop)

    def finish() -> None:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    try:
        # Merge history
        raw_history = client_history or get_history(session_id)
        if raw_history and raw_history[-1].get("role") == "user":
            raw_history = raw_history[:-1]
        history_context = build_history_context(raw_history)
        history_section = f"{history_context}\n\n" if history_context else ""

        if session_id:
            append_history(session_id, "user", prompt)

        # ── Tavily mode: ให้ Router ตัดสินใจว่าต้องค้น web หรือตอบจากความรู้ ────
        if mode == "tavily":
            put({"type": "agent_start", "step": "router", "agentName": "Router Agent"})
            decision, domain = route_with_web_search(prompt, history_context)
            put({
                "type": "agent_done",
                "step": "router",
                "agentName": "Router Agent",
                "result": f"เลือก: {decision}",
                "domain": {"code": decision, "nameTh": "ค้นหาทั่วไป" if decision == "tavily" else (domain.name_th if domain else ""), "nameEn": "Web Search" if decision == "tavily" else (domain.name_en if domain else "")},
            })

            if decision == "tavily":
                from src.agents.tavily_pipeline import run_tavily_pipeline
                put({"type": "agent_start", "step": "reasoning", "agentName": "Reasoning Narrator"})
                reasoning = f"ค้นหาข้อมูลจากอินเทอร์เน็ตด้วย Tavily"
                put({"type": "agent_done", "step": "reasoning", "agentName": "Reasoning Narrator", "result": reasoning})
                run_tavily_pipeline(prompt=prompt, queue=queue, loop=loop,
                                    session_id=session_id, history_section=history_section, reasoning=reasoning)
            else:
                # ตอบจากความรู้ — ใช้ pipeline ปกติ
                put({"type": "agent_start", "step": "reasoning", "agentName": "Reasoning Narrator"})
                narrator = Agent(
                    role=f"Reasoning Narrator — {domain.name_en}",
                    goal=f"อธิบายขั้นตอนการตอบคำถามด้าน{domain.name_th}",
                    backstory=domain.expertise,
                    llm=_get_llm(), verbose=False, max_iter=3,
                )
                reasoning = _run_agent(narrator,
                    f"คำถาม: {prompt}\nDomain: {domain.name_th}\nอธิบายสั้นๆ ว่าจะตอบอย่างไร ตอบเป็นภาษาไทย",
                    "คำอธิบายสั้นๆ เป็นภาษาไทย")
                put({"type": "agent_done", "step": "reasoning", "agentName": "Reasoning Narrator", "result": reasoning})
                run_pipeline(prompt=prompt, queue=queue, loop=loop, domain=domain,
                             history_context=history_context, history_section=history_section,
                             session_id=session_id, reasoning=reasoning)
            return

        # ── Normal mode: domain routing ───────────────────────────────────────

        # STEP 0: Router
        put({"type": "agent_start", "step": "router", "agentName": "Router Agent"})
        domain = route_domain(prompt, history_context)
        put({
            "type": "agent_done",
            "step": "router",
            "agentName": "Router Agent",
            "result": f"Domain: {domain.code} — {domain.name_th} ({domain.name_en})",
            "domain": {"code": domain.code, "nameTh": domain.name_th, "nameEn": domain.name_en},
        })

        # STEP 1: Reasoning Narrator
        put({"type": "agent_start", "step": "reasoning", "agentName": "Reasoning Narrator"})
        narrator = Agent(
            role=f"Reasoning Narrator — {domain.name_en}",
            goal=f"อธิบายขั้นตอนการวิเคราะห์ข้อมูลด้าน{domain.name_th}ให้ผู้ใช้เข้าใจ",
            backstory=domain.expertise,
            llm=_get_llm(),
            verbose=False,
            max_iter=5,
        )
        reasoning_prompt = (
            f"{history_section}"
            f"คำถาม: {prompt}\n"
            f"Domain: {domain.name_th}"
            + (" (คำถามทั่วไป ไม่ต้องค้นหาไฟล์ข้อมูล)\n\nอธิบายสั้น ๆ ว่าคุณจะตอบจากความรู้ด้านสาธารณสุขโดยตรง"
               if domain.code == "d0"
               else "\n\nอธิบายว่าจะหาข้อมูลอะไร วิเคราะห์อย่างไร และคาดว่าจะพบอะไร")
            + " ตอบเป็นภาษาไทย"
        )
        reasoning = _run_agent(narrator, reasoning_prompt, "คำอธิบายขั้นตอนการวิเคราะห์เป็นภาษาไทย")
        put({"type": "agent_done", "step": "reasoning", "agentName": "Reasoning Narrator", "result": reasoning})

        # STEP 2+: Domain-specific pipeline
        run_pipeline(
            prompt=prompt,
            queue=queue,
            loop=loop,
            domain=domain,
            history_context=history_context,
            history_section=history_section,
            session_id=session_id,
            reasoning=reasoning,
        )

    except Exception as exc:
        put({"type": "error", "message": str(exc)})
    finally:
        finish()


async def _handle_analyze(request: AnalyzeRequest) -> StreamingResponse:
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    client_history = (
        [{"role": m.role, "text": m.text} for m in request.history]
        if request.history else None
    )

    thread = threading.Thread(
        target=_orchestrate,
        args=(request.prompt, queue, loop),
        kwargs={"session_id": request.sessionId, "client_history": client_history, "mode": request.mode},
        daemon=True,
    )
    thread.start()

    async def stream():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    return await _handle_analyze(request)


@router.post("/api/chat")
async def chat(request: AnalyzeRequest):
    return await _handle_analyze(request)
