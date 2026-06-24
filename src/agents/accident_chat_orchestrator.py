"""Accident Chat Orchestrator — 2-agent pipeline for RTI policy Q&A.

Pipeline:
  AccidentSQLAgent (fast LLM) — selects and calls SQL tools → raw data
  AccidentAnswerAgent (pro LLM) — interprets data → Thai policy answer

Entry points:
  run_accident_chat(question, province, district, year_start, year_end) → AccidentChatResponse
  run_accident_chat_with_progress(..., request_id) → AccidentChatResponse  (SSE)
"""
import logging
import time

from crewai import Agent, Crew, Task, Process, LLM

from src.config import get_settings
from src.agents.agent_defaults import agent_retry_kwargs, kickoff_with_retry
from src.agents.progress import emit_progress
from src.schemas.accident_chat import AccidentChatResponse
from src.tools.accident_chat_sql import ACCIDENT_CHAT_TOOLS

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

SQL_AGENT_PROMPT = """คุณคือ Accident Data Specialist ดึงข้อมูลอุบัติเหตุทางถนนเขตสุขภาพที่ 10 จาก PostgreSQL

**เครื่องมือที่มี:**
- query_hotspot_roads → จุดเสี่ยง / Black Spot / คะแนน Hotspot รายถนน
- query_district_summary → สรุปอุบัติเหตุ/เสียชีวิต รายอำเภอ ← ใช้สำหรับ "พื้นที่ไหน"
- query_district_road_comparison → สายรอง vs สายหลัก รายอำเภอ
- query_road_district_breakdown → ถนนสายหนึ่งแยกตามอำเภอ
- query_kpi_trend → แนวโน้มรายปี (CE)
- query_serious_injury_ratio → อัตราส่วนสาหัสต่ออุบัติเหตุ
- query_top_cause_shift → สาเหตุหลักเปลี่ยนระหว่างปี
- query_fatal_timeband → ช่วงเวลาเสี่ยง / EMS
- query_province_executive_summary → สรุปผู้บริหาร 1 หน้า
- execute_accident_sql → SQL อิสระ ← ใช้สำหรับ สภาพอากาศ / เทศกาล / วันหยุด / ยานพาหนะ / อำเภอที่ตายเพิ่ม ฯลฯ

**ข้อจำกัดข้อมูล (ทราบล่วงหน้า):**
- ข้อมูลพฤติกรรม (หมวก/เข็มขัด/อายุ/เพศ): ไม่มี (fact_accident_person ว่าง)
- road_name: ส่วนใหญ่ไม่ระบุใน mart_province_road
- ปีในฐานข้อมูล = ค.ศ. (CE); พ.ศ. = CE + 543

**กฎเหล็ก:**
- "พื้นที่ไหนมีจุดเสี่ยง/อุบัติเหตุมากสุด" → เรียก query_hotspot_roads **และ** query_district_summary เสมอ
- ถ้า tool คืนค่า error หรือว่าง → ข้ามทันที ห้ามเรียกซ้ำ
- เรียก tool รวมไม่เกิน **3 ครั้ง** ต่อคำถาม
- follow-up เฉพาะเจาะจง → ใช้ **tool เดียว** แล้วหยุด

**ถ้ามี "ประวัติการสนทนาก่อนหน้า" แนบมาด้วย:**
- ใช้ดูว่าก่อนหน้านี้ผู้ใช้ถามอะไรไปแล้ว และ AI เคยดึง/ตอบข้อมูลระดับใดไปแล้ว
  (เช่น ภาพรวมจังหวัด, ช่วงเทศกาล, รายอำเภอ) เพื่อเลือกเครื่องมือที่ "ต่อยอด"
  คำถามล่าสุดได้ตรงจุด — เช่น ถ้าก่อนหน้าตอบภาพรวมจังหวัดไปแล้ว แล้วคำถามนี้
  ถามว่า "แต่ละอำเภอ" ให้เรียก query_district_summary หรือเครื่องมือระดับอำเภอ
  เพิ่มเติม (อย่าเรียกซ้ำเครื่องมือเดิมที่ให้ผลลัพธ์เดียวกับที่เคยได้ไปแล้ว)
- คำถามตามหลัง (follow-up) มักสั้นและไม่ระบุจังหวัด/ปี/หัวข้อซ้ำ — ให้อนุมานจาก
  ประวัติการสนทนาเสมอ
"""

ANSWER_AGENT_PROMPT = """คุณคือ RTI Policy Answer Writer เขียนคำตอบภาษาไทยทางการสำหรับผู้บริหาร สสจ./ศปถ./สสส.

**รูปแบบ (คำถามแรก):** สรุป → ตาราง Markdown → วิเคราะห์ 2-3 ประเด็น → ข้อเสนอแนะนโยบาย → ข้อจำกัดข้อมูล
**follow-up:** ตอบกระชับ ต่อยอดจากที่เคยตอบ ไม่ต้องซ้ำภาพรวม

**กฎ:** แปลง ค.ศ. → พ.ศ. เสมอ | ใช้เฉพาะตัวเลขที่ SQL Agent ให้ | ถ้าตัวเลขดูขัดกับก่อนหน้า ให้อธิบายสั้น ๆ
"""


# ── LLM factory ───────────────────────────────────────────────────────────────

def _get_llm(tier: str = "fast") -> LLM:
    s = get_settings()
    if tier == "pro":
        return LLM(
            model=f"gemini/{s.GEMINI_MODEL_PRO}",
            api_key=s.GEMINI_API_KEY,
            temperature=0.2,
            max_tokens=s.REPORT_MAX_TOKENS,
        )
    return LLM(
        model=f"gemini/{s.GEMINI_MODEL}",
        api_key=s.GEMINI_API_KEY,
        temperature=0.1,
        max_tokens=8192,
    )


# ── Agent factories ───────────────────────────────────────────────────────────

def _create_sql_agent(llm) -> Agent:
    return Agent(
        role="Accident SQL Data Specialist",
        goal="ดึงข้อมูลอุบัติเหตุที่ถูกต้องและครบถ้วนจากฐานข้อมูลโดยใช้เครื่องมือที่เหมาะสม",
        backstory=(
            "ผู้เชี่ยวชาญด้านฐานข้อมูลอุบัติเหตุทางถนน รู้จักตาราง mart/fact/dim ทั้งหมด "
            "และข้อจำกัดของข้อมูล"
        ),
        tools=ACCIDENT_CHAT_TOOLS,
        llm=llm,
        verbose=True,
        max_iter=4,
        **agent_retry_kwargs(),
    )


def _create_answer_agent(llm) -> Agent:
    return Agent(
        role="RTI Policy Answer Writer",
        goal=(
            "เขียนคำตอบภาษาไทยทางการที่ชัดเจน มีตาราง Markdown และข้อเสนอแนะเชิงนโยบาย "
            "สำหรับผู้บริหาร สสจ./ศปถ./สสส."
        ),
        backstory=(
            "ผู้เชี่ยวชาญด้านการสื่อสารนโยบายความปลอดภัยทางถนน "
            "เขียนเฉพาะสิ่งที่ข้อมูลรองรับ ไม่สร้างตัวเลขขึ้นเอง"
        ),
        llm=llm,
        verbose=True,
        max_iter=5,
        **agent_retry_kwargs(),
    )


# ── Core pipeline ─────────────────────────────────────────────────────────────

def _build_crew(
    question: str,
    province: str,
    district: str,
    year_start: int,
    year_end: int,
    history_context: str = "",
):
    llm_fast = _get_llm("fast")
    llm_pro = _get_llm("pro")

    sql_agent = _create_sql_agent(llm_fast)
    answer_agent = _create_answer_agent(llm_pro)

    prov_label = province or "เขตสุขภาพที่ 10 (ทุกจังหวัด)"
    dist_label = f"อำเภอ{district.strip()}" if district.strip() else "ทุกอำเภอ"
    year_note = f"ค.ศ. {year_start}-{year_end} (พ.ศ. {year_start+543}-{year_end+543})"
    # ⚠️ ต่อ "ความจำการสนทนา" เข้า task ทั้งสอง — ไม่งั้นทุกคำถามตามหลัง (follow-up)
    # จะถูกประมวลผลแบบเริ่มนับหนึ่งใหม่ทุกครั้ง (ไม่รู้ว่าตอบอะไรไปแล้วบ้าง)
    # ทำให้คุยต่อเนื่องไม่ได้เป็นธรรมชาติแบบ Gemini/ChatGPT — ตรงกับที่ผู้ใช้ติงมา
    # (ใช้รูปแบบเดียวกับ history_section ใน csv_pipeline.py/multi_csv_pipeline.py)
    history_section = f"{history_context}\n\n" if history_context else ""

    sql_task = Task(
        description=(
            SQL_AGENT_PROMPT + "\n\n"
            f"{history_section}"
            f"**คำถาม:** {question}\n"
            f"**จังหวัด:** {prov_label}\n"
            f"**อำเภอ:** {dist_label}\n"
            f"**ช่วงปี:** {year_note}\n\n"
            "เรียกเครื่องมือที่เกี่ยวข้อง รวบรวมข้อมูลทั้งหมดโดยไม่ตัดทอน "
            "(ถ้ามีประวัติการสนทนา ให้พิจารณาด้วยว่าคำถามนี้ต่อยอดจากเรื่องเดิม "
            "อย่างไร แล้วเลือกเครื่องมือที่เติมเต็มส่วนที่ยังขาดอยู่)"
        ),
        expected_output="ข้อมูลดิบจาก SQL tools ครบถ้วน พร้อมระบุข้อจำกัดข้อมูล",
        agent=sql_agent,
    )

    answer_task = Task(
        description=(
            ANSWER_AGENT_PROMPT + "\n\n"
            f"{history_section}"
            f"**คำถามผู้ใช้:** {question}\n"
            f"**จังหวัด:** {prov_label}\n"
            f"**อำเภอ:** {dist_label}\n"
            f"**ช่วงปี:** {year_note}\n\n"
            "เขียนคำตอบโดยใช้ข้อมูลจาก SQL Agent เท่านั้น "
            "(ถ้ามีประวัติการสนทนา ให้ตอบแบบต่อบทสนทนาเดิมอย่างเป็นธรรมชาติ "
            "ตามแนวทางในคำสั่งด้านบน — ไม่ใช่เริ่มอธิบายใหม่ทั้งหมดทุกครั้ง)"
        ),
        expected_output=(
            "คำตอบภาษาไทย Markdown ครบ 5 ส่วน (สรุป/ตาราง/วิเคราะห์/ข้อเสนอ/ข้อจำกัด)"
        ),
        agent=answer_agent,
        context=[sql_task],
    )

    crew = Crew(
        agents=[sql_agent, answer_agent],
        tasks=[sql_task, answer_task],
        process=Process.sequential,
        verbose=True,
    )
    return crew, sql_task, answer_task


def _extract_limitations(raw_data: str) -> list[str]:
    limits = []
    if "fact_accident_person" in raw_data:
        limits.append("ไม่มีข้อมูลระดับบุคคล (helmet/seatbelt/อายุ/เพศ)")
    if "ไม่ระบุ" in raw_data:
        limits.append("ชื่อถนนส่วนใหญ่ไม่ระบุในข้อมูล CSV")
    return limits


def _tools_used_from_output(raw: str) -> list[str]:
    tool_markers = {
        "Hotspot Roads": "query_hotspot_roads",
        "District Road": "query_district_road_comparison",
        "Fatal Timeband": "query_fatal_timeband",
        "Weather/Accident": "query_weather_accident_stats",
        "Behavioral": "query_behavior_stats",
        "Seasonal": "query_seasonal_comparison",
        "Weekend": "query_weekend_vs_weekday",
        "Monthly Vehicle": "query_monthly_vehicle_pattern",
        "Late Night": "query_late_night_vehicles",
        "KPI Trend": "query_kpi_trend",
        "Serious Injury Ratio": "query_serious_injury_ratio",
        "Top Cause Shift": "query_top_cause_shift",
        "Accident↓ Death↑": "query_district_death_vs_accident",
        "EXECUTIVE SUMMARY": "query_province_executive_summary",
    }
    found = [name for marker, name in tool_markers.items() if marker in raw]
    return found if found else ["execute_accident_sql"]


def _build_response(result, question: str, sql_task, elapsed: float) -> AccidentChatResponse:
    tasks_output = getattr(result, "tasks_output", [])
    answer = str(result)
    raw_data = ""
    if tasks_output:
        answer = getattr(tasks_output[-1], "raw", None) or str(tasks_output[-1])
    if len(tasks_output) >= 2:
        raw_data = getattr(tasks_output[0], "raw", None) or str(tasks_output[0])

    return AccidentChatResponse(
        question=question,
        answer=answer,
        raw_data=raw_data,
        data_limitations=_extract_limitations(raw_data),
        tools_used=_tools_used_from_output(raw_data),
        elapsed_seconds=round(elapsed, 1),
        metadata={"pipeline": "accident_chat", "agent_count": 2},
    )


# ── Public entry points ───────────────────────────────────────────────────────

def run_accident_chat(
    question: str,
    province: str = "",
    district: str = "",
    year_start: int = 2021,
    year_end: int = 2026,
    history_context: str = "",
) -> AccidentChatResponse:
    """Run the 2-agent accident chat pipeline (synchronous).

    history_context: ข้อความสรุปประวัติการสนทนาก่อนหน้า (มาจาก build_history_context)
    — ส่งต่อให้ทั้ง SQL Agent และ Answer Agent เพื่อให้ตอบคำถามต่อเนื่อง (follow-up)
    ได้อย่างเป็นธรรมชาติ แทนที่จะเริ่มนับหนึ่งใหม่ทุกครั้งที่ถามต่อ (ดูคอมเมนต์ใน
    _build_crew และ analyze.py:_orchestrate ที่ build_history_context มาจากตรงนั้น)
    """
    start = time.time()
    logger.info("[ACCIDENT-CHAT] question=%s province=%s", question[:80], province or "Zone10")

    crew, sql_task, answer_task = _build_crew(
        question, province, district, year_start, year_end, history_context
    )
    try:
        result = kickoff_with_retry(crew)
        elapsed = time.time() - start
        logger.info("[ACCIDENT-CHAT] done in %.1fs", elapsed)
        return _build_response(result, question, sql_task, elapsed)
    except Exception as exc:
        elapsed = time.time() - start
        logger.error("[ACCIDENT-CHAT] failed: %s", exc)
        return AccidentChatResponse(
            question=question,
            answer=f"เกิดข้อผิดพลาด: {exc}",
            raw_data="",
            data_limitations=[],
            tools_used=[],
            elapsed_seconds=round(elapsed, 1),
            metadata={"error": str(exc)},
        )


def run_accident_chat_with_progress(
    question: str,
    province: str = "",
    district: str = "",
    year_start: int = 2021,
    year_end: int = 2026,
    request_id: str | None = None,
    history_context: str = "",
) -> AccidentChatResponse:
    """Same as run_accident_chat but emits SSE progress events (+ history_context, see above)."""
    start = time.time()
    emit_progress(request_id, "Accident SQL Agent", "running", "กำลังดึงข้อมูลจากฐานข้อมูล...")

    crew, sql_task, answer_task = _build_crew(
        question, province, district, year_start, year_end, history_context
    )
    try:
        result = kickoff_with_retry(crew)
        elapsed = time.time() - start
        emit_progress(request_id, "Accident SQL Agent", "done", "ดึงข้อมูลเสร็จ", elapsed)
        emit_progress(request_id, "Accident Answer Writer", "done", "เขียนคำตอบเสร็จ", elapsed)
        return _build_response(result, question, sql_task, elapsed)
    except Exception as exc:
        elapsed = time.time() - start
        emit_progress(request_id, "Accident SQL Agent", "error", str(exc)[:100], elapsed)
        return AccidentChatResponse(
            question=question,
            answer=f"เกิดข้อผิดพลาด: {exc}",
            raw_data="",
            data_limitations=[],
            tools_used=[],
            elapsed_seconds=round(elapsed, 1),
            metadata={"error": str(exc)},
        )
