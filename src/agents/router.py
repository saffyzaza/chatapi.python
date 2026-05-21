"""Router Agent — classifies user questions into health domains d0–d8."""
import os
import re

from crewai import Agent, Crew, LLM, Task

from src.domains import Domain, DOMAINS, DOMAIN_LIST_TEXT


def _get_llm() -> LLM:
    return LLM(model="gemini/gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))


def route_domain(prompt: str, history_context: str = "") -> Domain:
    """Run the Router Agent and return the matched Domain."""
    router = Agent(
        role="Router Agent",
        goal="วิเคราะห์คำถามแล้วเลือก domain ที่เหมาะสมที่สุด",
        backstory=(
            "คุณเป็น Router Agent ที่เชี่ยวชาญการจำแนกคำถามสุขภาพว่าเกี่ยวกับ domain ใด "
            "คุณตอบเฉพาะรหัส domain เช่น d1, d2 เท่านั้น ห้ามตอบอย่างอื่น "
            "หากคำถามปัจจุบันเป็น follow-up ให้ใช้ domain เดิมจากประวัติการสนทนา"
        ),
        llm=_get_llm(),
        verbose=False,
        max_iter=3,
    )

    history_section = f"{history_context}\n\n" if history_context else ""
    task = Task(
        description=(
            f"{history_section}"
            f"คำถามล่าสุด: {prompt}\n\n"
            f"Domain ที่มี:\n{DOMAIN_LIST_TEXT}\n\n"
            "เลือก domain ที่เหมาะสมที่สุด 1 อัน ตอบเฉพาะรหัส เช่น d2"
        ),
        expected_output="รหัส domain เช่น d2",
        agent=router,
    )
    crew = Crew(agents=[router], tasks=[task], verbose=False)

    try:
        result = str(crew.kickoff()).strip()
    except Exception:
        result = "d0"

    m = re.search(r'\b(d[0-8])\b', result.lower())
    code = m.group(1) if m else "d0"
    return DOMAINS[code]


def route_with_web_search(prompt: str, history_context: str = "") -> tuple[str, Domain | None]:
    """Router ที่รู้จัก 'tavily' — ให้ LLM ตัดสินใจเองว่าต้องค้น web หรือตอบจากความรู้

    Returns:
        ("tavily", None)         — ต้องค้นข้อมูลจากอินเทอร์เน็ต
        ("d0", DOMAINS["d0"])   — ตอบจากความรู้ทั่วไป
        ("dN", DOMAINS["dN"])   — domain เฉพาะทาง
    """
    router = Agent(
        role="Smart Router Agent",
        goal="วิเคราะห์คำถามแล้วตัดสินใจว่าต้องค้นข้อมูลจากอินเทอร์เน็ตหรือตอบจากความรู้",
        backstory=(
            "คุณเป็น Smart Router ที่เข้าใจว่าคำถามไหนต้องการข้อมูลล่าสุดจากเว็บ "
            "และคำถามไหนตอบได้จากความรู้ทั่วไป "
            "คุณตอบเพียงรหัสเดียว ห้ามอธิบายเพิ่ม"
        ),
        llm=_get_llm(),
        verbose=False,
        max_iter=3,
    )

    history_section = f"{history_context}\n\n" if history_context else ""
    task = Task(
        description=(
            f"{history_section}"
            f"คำถาม: {prompt}\n\n"
            "เลือก 1 ตัวเลือก:\n"
            "- tavily  → คำถามต้องการข้อมูลล่าสุด/ข่าว/เหตุการณ์ปัจจุบัน/ข้อมูลเฉพาะจากเว็บ\n"
            "- d0      → ตอบได้จากความรู้ทั่วไป (คณิตศาสตร์ ความหมาย คำอธิบาย ฯลฯ)\n"
            f"Domain อื่นที่มี:\n{DOMAIN_LIST_TEXT}\n\n"
            "ตอบเพียงรหัสเดียว เช่น tavily หรือ d0 หรือ d2"
        ),
        expected_output="รหัสเดียว เช่น tavily หรือ d0",
        agent=router,
    )
    crew = Crew(agents=[router], tasks=[task], verbose=False)

    try:
        result = str(crew.kickoff()).strip().lower()
    except Exception:
        result = "d0"

    if "tavily" in result:
        return "tavily", None

    m = re.search(r'\b(d[0-8])\b', result)
    code = m.group(1) if m else "d0"
    return code, DOMAINS[code]
