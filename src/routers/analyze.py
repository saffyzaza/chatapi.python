"""Analyze router — SSE streaming pipeline for health domain Q&A."""
import asyncio
import json
import os
import threading
from typing import Any

# จำกัด 5 AI pipelines พร้อมกันต่อ worker (4 workers = 20 concurrent รวม)
_AI_SEMAPHORE = threading.BoundedSemaphore(5)

from crewai import Agent, LLM, Crew, Task
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.agents.router import route_domain, route_with_web_search, route_multi_domain
from src.agents.csv_pipeline import run_pipeline, _run_agent
from src.agents.multi_csv_pipeline import run_multi_pipeline
from src.agents.thaijo_agent import run_thaijo_pipeline
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

        # ── Memory Agent: แปลง follow-up question ให้ครบถ้วน ─────────────────
        if history_context:
            from src.agents.question_resolver import resolve_question
            put({"type": "agent_start", "step": "memory", "agentName": "Memory Agent"})
            resolved, was_changed = resolve_question(
                prompt, history_context, os.getenv("GEMINI_API_KEY", "")
            )
            if was_changed:
                put({
                    "type": "agent_done", "step": "memory", "agentName": "Memory Agent",
                    "result": f"ปรับคำถาม: {resolved}",
                })
                prompt = resolved  # ← downstream agents ทั้งหมดใช้ resolved prompt
            else:
                put({
                    "type": "agent_done", "step": "memory", "agentName": "Memory Agent",
                    "result": "คำถามชัดเจน ไม่ต้องปรับ",
                })

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

        # ── Normal mode: multi-domain aware routing ───────────────────────────

        # STEP 0: Router (detects single vs multi-domain)
        put({"type": "agent_start", "step": "router", "agentName": "Router Agent"})
        domains, is_multi = route_multi_domain(prompt, history_context)
        domain = domains[0]
        domain_names_th = " + ".join(d.name_th for d in domains)
        domain_names_en = " + ".join(d.name_en for d in domains)
        put({
            "type": "agent_done",
            "step": "router",
            "agentName": "Router Agent",
            "result": f"{'Multi-Domain' if is_multi else 'Domain'}: {domain_names_th}",
            "domain": {
                "code": "multi" if is_multi else domain.code,
                "nameTh": domain_names_th,
                "nameEn": domain_names_en,
            },
        })

        # STEP 1: Reasoning Narrator
        put({"type": "agent_start", "step": "reasoning", "agentName": "Reasoning Narrator"})
        narrator = Agent(
            role=f"Reasoning Narrator — {'Multi-Domain' if is_multi else domain.name_en}",
            goal=(
                f"อธิบายแผนการวิเคราะห์ข้ามหลาย domain ({domain_names_th}) ให้ชัดเจนและเข้าใจง่าย"
                if is_multi
                else f"อธิบายแผนการวิเคราะห์ข้อมูล{domain.name_th}อย่างชัดเจน"
            ),
            backstory=(
                "คุณเป็นนักวิเคราะห์ข้อมูลสาธารณสุขที่อธิบายกระบวนการคิดได้ชัดเจน "
                "คุณบอกว่าจะวิเคราะห์อะไร ใช้ข้อมูลอะไร และคาดหวังผลลัพธ์อะไร "
                "เพื่อให้ผู้ใช้เข้าใจก่อนดูผลการวิเคราะห์"
            ),
            llm=_get_llm(),
            verbose=False,
            max_iter=3,
        )
        if is_multi:
            reasoning_prompt = (
                f"{history_section}"
                f"คำถาม: {prompt}\n"
                f"Domains ที่จะวิเคราะห์: {domain_names_th}\n\n"
                "อธิบายแผนการวิเคราะห์ในรูปแบบนี้ (3-5 ประโยค):\n"
                "1. ทำไมคำถามนี้ถึงต้องใช้ข้อมูลจากหลาย domain\n"
                "2. จะค้นหาไฟล์ข้อมูลอะไรบ้างจากแต่ละ domain\n"
                "3. จะ merge ข้อมูลอย่างไรและคำนวณ composite score อย่างไร\n"
                "4. คาดว่าจะพบ pattern หรือ Red Zone ลักษณะใด\n"
                "ตอบเป็นภาษาไทย กระชับ ไม่เกิน 5 ประโยค"
            )
        else:
            reasoning_prompt = (
                f"{history_section}"
                f"คำถาม: {prompt}\n"
                f"Domain: {domain.name_th}\n\n"
                + ("อธิบายสั้นๆ ว่าจะตอบจากความรู้ด้านสาธารณสุขโดยตรง ไม่เกิน 3 ประโยค"
                   if domain.code == "d0"
                   else "จะค้นหาบทความวิจัยจาก ThaiJo แล้วสังเคราะห์สร้าง Journal Report อัตโนมัติ"
                   if domain.code == "dt"
                   else
                   "อธิบายแผนการวิเคราะห์ (3-4 ประโยค):\n"
                   "1. จะค้นหาไฟล์ข้อมูลอะไร\n"
                   "2. จะ filter/aggregate ข้อมูลอย่างไร\n"
                   "3. คาดว่าจะพบผลลัพธ์ลักษณะใด")
                + "\nตอบเป็นภาษาไทย กระชับ"
            )
        reasoning = _run_agent(narrator, reasoning_prompt, "คำอธิบายแผนการวิเคราะห์เป็นภาษาไทย กระชับ")
        put({"type": "agent_done", "step": "reasoning", "agentName": "Reasoning Narrator", "result": reasoning})

        # ── ThaiJo Research pipeline ──────────────────────────────────────────
        if domain.code == "dt" or mode == "thaijo":
            run_thaijo_pipeline(prompt=prompt, queue=queue, loop=loop, session_id=session_id)
            return

        # mode=multi forces multi-domain pipeline regardless of router decision
        if mode == "multi":
            is_multi = True

        # STEP 2+: Multi-domain or single-domain pipeline
        if is_multi:
            run_multi_pipeline(
                prompt=prompt,
                queue=queue,
                loop=loop,
                domains=domains,
                history_context=history_context,
                history_section=history_section,
                session_id=session_id,
                reasoning=reasoning,
            )
        else:
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
        _AI_SEMAPHORE.release()


async def _handle_analyze(request: AnalyzeRequest) -> StreamingResponse:
    if not _AI_SEMAPHORE.acquire(blocking=False):
        async def busy_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'ระบบกำลังประมวลผลเต็มความสามารถ กรุณารอสักครู่แล้วลองใหม่'}, ensure_ascii=False)}\n\n"
        return StreamingResponse(busy_stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

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
