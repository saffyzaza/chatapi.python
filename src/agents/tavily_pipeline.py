"""Tavily Search pipeline — 2-agent pipeline for external web search Q&A.

Pipeline:
  TavilySearchAgent  — ค้นหาข้อมูลจากอินเทอร์เน็ตด้วย Tavily
  TavilyAnswerWriter — สังเคราะห์ผลการค้นหาเป็นคำตอบภาษาไทย
"""
import asyncio
import os
import re
import time
from typing import Any

from crewai import Agent, Crew, LLM, Task, Process

from src.config import get_settings
from src.history import append_history
from src.tools.tavily_search import tavily_search


def _get_llm() -> LLM:
    return LLM(model="gemini/gemini-2.5-flash-lite", api_key=os.getenv("GEMINI_API_KEY"))


_URL_RE = re.compile(r"https?://[^\s)\]]+")


def _ensure_all_sources(answer: str, raw_data: str) -> str:
    """แนบ "แหล่งอ้างอิง" ให้ครบทุก URL ที่ได้จากการค้นคว้า.

    ⚠️ Answer Writer (LLM) มักสรุปแล้ว "อ้างอิงแค่บางแหล่ง" ทำให้ผู้ใช้เห็น
    reference น้อยกว่าที่ค้นเจอจริง — ดึง URL ทั้งหมดจากผลดิบ (raw_data ของ
    Search Agent ซึ่งมีรายการแหล่งครบ) แล้วเติมอันที่ยังไม่ถูกอ้างในคำตอบ
    ให้ครบ เพื่อไม่ให้แหล่งอ้างอิงตกหล่น
    """
    if not raw_data:
        return answer
    all_urls = list(dict.fromkeys(_URL_RE.findall(raw_data)))  # unique, keep order
    if not all_urls:
        return answer
    missing = [u for u in all_urls if u not in answer]
    if not missing:
        return answer
    already = len(all_urls) - len(missing)
    extra = "\n".join(f"- {u}" for u in missing)
    return (
        f"{answer.rstrip()}\n\n"
        f"**แหล่งอ้างอิงเพิ่มเติมจากการค้นคว้า** (อีก {len(missing)} แหล่ง "
        f"นอกเหนือจาก {already} แหล่งที่อ้างไว้ด้านบน):\n{extra}"
    )


def _search_agent_prompt() -> str:
    """สร้าง prompt สำหรับ Search Agent ที่ใช้ Tavily Search (10 ผลลัพธ์ เน้นไทย)"""
    n = get_settings().TAVILY_MAX_RESULTS
    return f"""คุณคือ Web Search Specialist ผู้เชี่ยวชาญด้านการค้นหาข้อมูลจากอินเทอร์เน็ต

เมื่อได้รับคำถาม ให้:
1. วิเคราะห์คำถามแล้วสร้าง search query ที่เหมาะสม **เป็นภาษาไทยเสมอ**
   (เพื่อให้ได้แหล่งข้อมูลไทย ไม่ใช่เฉพาะภาษาอังกฤษ) — ใช้คำสำคัญภาษาไทย
   ตามหัวข้อ/จังหวัด/หน่วยงานที่ถาม
2. เรียก tavily_search **ครั้งเดียว** เท่านั้น (จะได้ {n} ผลลัพธ์)
3. ส่งผลลัพธ์ทั้ง {n} รายการพร้อม URL ต่อให้ Answer Writer

**กฎสำคัญ:**
- **query ต้องเป็นภาษาไทย** — ห้ามแปลเป็นอังกฤษก่อนค้น
- เรียก tavily_search แค่ **1 ครั้ง** ห้ามเรียกซ้ำหลายรอบ
- ส่งต่อผลลัพธ์และ URL **ทั้ง {n} แหล่ง** ห้ามตัดทิ้ง
- ไม่ตีความหรือสรุปเอง ส่งข้อมูลดิบให้ Answer Writer
"""

ANSWER_WRITER_PROMPT = """คุณคือ Research Answer Writer ผู้เชี่ยวชาญด้านการสังเคราะห์ข้อมูลจากเว็บ

รับผลการค้นหาจาก Search Agent (หลายแหล่ง) แล้วเขียนคำตอบภาษาไทยที่ **ครบถ้วนและมีเนื้อหาแน่น**:

**โครงสร้างคำตอบ:**
1. **สรุปคำตอบ** — ตอบตรงประเด็นใน 3-4 ประโยค
2. **ตัวเลข/สถิติที่พบ** — CRITICAL ห้ามข้ามข้อนี้: ไล่อ่านทุกแหล่งแล้ว "คัดลอกตัวเลขทุกตัว"
   ที่เจอ (จำนวน/ร้อยละ/เป้าหมาย/อัตรา/ปีงบประมาณ ฯลฯ) มาเป็น bullet list แยกทีละตัวเลข
   คำต่อคำตามต้นฉบับ ห้ามปัดเศษ ห้ามเปลี่ยนหน่วย พร้อมบริบทสั้นๆ + แหล่งที่มา เช่น:
   "- เป้าหมายลดผู้เสียชีวิตทั่วประเทศปี 2570: เหลือ 8,474 คน (12 คน/แสนประชากร) — Source: ..."
   "- เป้าหมายลดผู้เสียชีวิตจังหวัดศรีสะเกษปี 2570: ลด 1 คน, บาดเจ็บสาหัสลด 1 คน — Source: ..."
   ถ้าค้นทุกแหล่งแล้วไม่พบตัวเลขเลยจริงๆ ให้เขียนตรงๆ ว่า "ไม่พบข้อมูลเชิงตัวเลขจากแหล่งที่ค้นพบ"
   ห้ามเว้นข้อนี้ไปเฉยๆ
3. **รายละเอียด** — อธิบายให้ละเอียด โดย **สังเคราะห์ข้อมูลจากทุกแหล่งที่ค้นพบ**
   ให้ครบ (ไม่ใช่หยิบมาแค่ 2-3 แหล่ง) จัดเป็นหัวข้อย่อย/bullet ตามประเด็น
   พร้อมตาราง Markdown เมื่อมีตัวเลข/สถิติ/การเปรียบเทียบ และระบุแหล่งที่มา
   กำกับแต่ละประเด็น — **ต้องดึงตัวเลขจากข้อ 2 กลับมาใส่ในย่อหน้าด้วยเสมอ** ห้ามเขียน
   คลุมเครือแบบ "มีการระบุเป้าหมายเฉพาะ" หรือ "มีตัวชี้วัดที่ชัดเจน" โดยไม่บอกตัวเลขจริง —
   ถ้าไม่มีตัวเลขให้เขียนอธิบายเชิงคุณภาพตรงๆ แทน อย่าแกล้งทำเป็นมีตัวเลขแล้วไม่บอก
4. **แหล่งอ้างอิง** — รายการ URL ที่ใช้ทั้งหมด

**กฎสำคัญ:**
- **ห้าม "พารากราฟ" ตัวเลข/สถิติที่พบให้กลายเป็นประโยคทั่วไปที่ไม่มีตัวเลข** — ถ้าแหล่งข้อมูล
  บอกว่า "ลดผู้เสียชีวิต 1 คน" ต้องเขียนตัวเลข "1 คน" ซ้ำในคำตอบ ห้ามเขียนแค่ "มีการกำหนด
  เป้าหมายเฉพาะ" เฉยๆ — การละเลขทิ้งแบบนี้คือข้อผิดพลาดร้ายแรงที่สุดของงานนี้
- **ใช้ประโยชน์จากทุกแหล่งที่ค้นพบให้มากที่สุด** — ดึงข้อเท็จจริง/ตัวเลข/มุมมองที่
  แตกต่างจากแต่ละแหล่งมาประกอบกัน อย่าสรุปสั้นจนตกข้อมูลสำคัญ
- ใช้เฉพาะข้อมูลที่ค้นพบ ห้ามสร้างข้อมูลขึ้นเอง
- ระบุว่าข้อมูลแต่ละส่วนมาจากแหล่งใด
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
            _search_agent_prompt() + "\n\n"
            f"{history_section}"
            f"**คำถาม:** {prompt}\n\n"
            f"เรียก tavily_search 1 ครั้ง แล้วส่งต่อรายงานและแหล่งอ้างอิงทั้งหมดที่ได้ ห้ามค้นหาซ้ำ"
        ),
        expected_output="รายงานเชิงลึกจาก Tavily Research พร้อมแหล่งอ้างอิง (URL) ครบถ้วน",
        agent=search_agent,
    )

    answer_task = Task(
        description=(
            ANSWER_WRITER_PROMPT + "\n\n"
            f"**คำถามผู้ใช้:** {prompt}\n\n"
            f"เขียนคำตอบโดยใช้ข้อมูลจาก Search Agent เท่านั้น "
            f"สังเคราะห์จากรายงานและทุกแหล่งอ้างอิงให้ครบถ้วน อย่าตอบสั้นจนตกข้อมูลสำคัญ"
        ),
        expected_output=(
            "คำตอบภาษาไทย Markdown ที่มีเนื้อหาแน่น: สรุปคำตอบ + bullet list ตัวเลข/สถิติ"
            " ทุกตัวที่พบ (คำต่อคำ ไม่พารากราฟทิ้ง) + รายละเอียดที่สังเคราะห์จากทุกแหล่ง"
            " (มีหัวข้อย่อย/ตารางเมื่อเหมาะสม และดึงตัวเลขกลับมาใส่ในย่อหน้าด้วย) + แหล่งอ้างอิงครบ"
        ),
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

        # เติมแหล่งอ้างอิงที่ Answer Writer ตกหล่นให้ครบทุก URL ที่ค้นเจอ
        answer = _ensure_all_sources(answer, raw_data)

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
