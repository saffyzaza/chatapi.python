"""Tavily Search pipeline — 2-agent pipeline for external web search Q&A.

Pipeline:
  TavilySearchAgent  — ค้นหาข้อมูลจากอินเทอร์เน็ตด้วย Tavily
  TavilyAnswerWriter — สังเคราะห์ผลการค้นหาเป็นคำตอบภาษาไทย
"""
import asyncio
import os
import time
from typing import Any

from crewai import Agent, Crew, LLM, Task, Process

from src.history import append_history
from src.tools.tavily_search import tavily_search


def _get_llm() -> LLM:
    return LLM(model="gemini/gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))


SEARCH_AGENT_PROMPT = """คุณคือ Web Search Specialist ผู้เชี่ยวชาญด้านการค้นหาข้อมูลจากอินเทอร์เน็ต

เมื่อได้รับคำถาม ให้:
1. วิเคราะห์คำถามและสร้าง search query ที่เหมาะสม
2. เรียก tavily_search **ครั้งเดียว** เท่านั้น (max_results=2)
3. รวบรวมผลลัพธ์ 2 รายการที่ได้ส่งต่อให้ Answer Writer

**กฎสำคัญ:**
- เรียก tavily_search แค่ **1 ครั้ง** ห้ามเรียกซ้ำหลายรอบ
- ใช้แค่ **2 ผลลัพธ์** ที่ได้มา ห้ามค้นหาเพิ่ม
- ระบุ URL ของทั้ง 2 แหล่ง
- ไม่ตีความหรือสรุปเอง ส่งข้อมูลดิบให้ Answer Writer
"""

ANSWER_WRITER_PROMPT = """คุณคือ Research Answer Writer ผู้เชี่ยวชาญด้านการสังเคราะห์ข้อมูลจากเว็บ

รับผลการค้นหาจาก Search Agent แล้วเขียนคำตอบภาษาไทยที่:

**โครงสร้างคำตอบ:**
1. **สรุปคำตอบ** — ตอบตรงๆ ใน 2-3 ประโยค
2. **รายละเอียด** — อธิบายเพิ่มเติมพร้อมตาราง Markdown (ถ้ามีตัวเลข/เปรียบเทียบ)
3. **แหล่งอ้างอิง** — รายการ URL ที่ใช้

**กฎสำคัญ:**
- ใช้เฉพาะข้อมูลที่ค้นพบ ห้ามสร้างข้อมูลขึ้นเอง
- ระบุว่าข้อมูลมาจากแหล่งใด
- ถ้าข้อมูลที่ค้นพบไม่ตรงกับคำถาม ให้บอกตรงๆ
- ใช้ภาษาไทยที่เป็นธรรมชาติ ไม่ทางการจนเกินไป
"""


def run_tavily_pipeline(
    prompt: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    session_id: str = "",
    history_section: str = "",
    reasoning: str = "",
) -> None:
    """Run the Tavily search pipeline and emit SSE events."""
    llm = _get_llm()

    def put(ev: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(queue.put(ev), loop)

    start = time.time()

    # STEP 1: Search Agent
    put({"type": "agent_start", "step": "search", "agentName": "Tavily Search Agent"})

    search_agent = Agent(
        role="Web Search Specialist",
        goal="ค้นหาข้อมูลที่ถูกต้องและครบถ้วนจากอินเทอร์เน็ตด้วย Tavily",
        backstory=(
            "คุณเป็นผู้เชี่ยวชาญด้านการค้นหาข้อมูลออนไลน์ "
            "สามารถสร้าง search query ที่มีประสิทธิภาพและรวบรวมข้อมูลจากหลายแหล่ง "
            "คุณรายงานข้อมูลดิบอย่างครบถ้วนโดยไม่ตีความเอง"
        ),
        tools=[tavily_search],
        llm=llm,
        verbose=True,
        max_iter=5,
    )

    answer_agent = Agent(
        role="Research Answer Writer",
        goal="สังเคราะห์ผลการค้นหาเป็นคำตอบภาษาไทยที่ชัดเจนและมีแหล่งอ้างอิง",
        backstory=(
            "คุณเป็นนักวิจัยที่เชี่ยวชาญด้านการสังเคราะห์ข้อมูลจากหลายแหล่ง "
            "เขียนคำตอบที่ตรงประเด็น มีโครงสร้างชัดเจน และอ้างอิงแหล่งที่มาเสมอ"
        ),
        llm=llm,
        verbose=True,
        max_iter=3,
    )

    search_task = Task(
        description=(
            SEARCH_AGENT_PROMPT + "\n\n"
            f"{history_section}"
            f"**คำถาม:** {prompt}\n\n"
            "เรียก tavily_search 1 ครั้ง รวบรวมผลลัพธ์ 2 รายการพร้อม URL ห้ามค้นหาซ้ำ"
        ),
        expected_output="ผลการค้นหา 2 รายการพร้อม URL และเนื้อหาสรุปของแต่ละแหล่ง",
        agent=search_agent,
    )

    answer_task = Task(
        description=(
            ANSWER_WRITER_PROMPT + "\n\n"
            f"**คำถามผู้ใช้:** {prompt}\n\n"
            "เขียนคำตอบโดยใช้ข้อมูลจาก Search Agent เท่านั้น"
        ),
        expected_output="คำตอบภาษาไทย Markdown พร้อมสรุป รายละเอียด และแหล่งอ้างอิง",
        agent=answer_agent,
        context=[search_task],
    )

    crew = Crew(
        agents=[search_agent, answer_agent],
        tasks=[search_task, answer_task],
        process=Process.sequential,
        verbose=True,
    )

    try:
        result = crew.kickoff()
        elapsed = round(time.time() - start, 1)

        tasks_output = getattr(result, "tasks_output", [])
        answer = str(result)
        raw_data = ""
        if tasks_output:
            answer = getattr(tasks_output[-1], "raw", None) or str(tasks_output[-1])
        if len(tasks_output) >= 2:
            raw_data = getattr(tasks_output[0], "raw", None) or str(tasks_output[0])

        put({
            "type": "agent_done",
            "step": "search",
            "agentName": "Tavily Search Agent",
            "result": raw_data or "(ค้นหาเสร็จ)",
        })

        put({"type": "agent_start", "step": "insight", "agentName": "Tavily Answer Writer"})
        put({
            "type": "agent_done",
            "step": "insight",
            "agentName": "Tavily Answer Writer",
            "result": answer,
        })

        if session_id:
            append_history(session_id, "ai", answer)

        put({
            "type": "final",
            "message": answer,
            "domain": {"code": "tavily", "nameTh": "ค้นหาทั่วไป", "nameEn": "Web Search"},
            "agentSteps": [
                {"step": "reasoning", "agentName": "Reasoning Narrator",   "result": reasoning},
                {"step": "search",    "agentName": "Tavily Search Agent",  "result": raw_data or ""},
                {"step": "insight",   "agentName": "Tavily Answer Writer", "result": answer},
            ],
        })

    except Exception as exc:
        elapsed = round(time.time() - start, 1)
        err_msg = f"เกิดข้อผิดพลาดในการค้นหา: {exc}"
        put({"type": "agent_done", "step": "search", "agentName": "Tavily Search Agent", "result": str(exc)})
        put({
            "type": "final",
            "message": err_msg,
            "domain": {"code": "tavily", "nameTh": "ค้นหาทั่วไป", "nameEn": "Web Search"},
            "agentSteps": [
                {"step": "search", "agentName": "Tavily Search Agent", "result": str(exc)},
            ],
        })
