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

SQL_AGENT_PROMPT = """คุณคือ Accident Data Specialist ผู้เชี่ยวชาญด้านการดึงข้อมูลอุบัติเหตุทางถนน
สำหรับเขตสุขภาพที่ 10 จากฐานข้อมูล PostgreSQL

เมื่อได้รับคำถาม ให้:
1. วิเคราะห์ว่าคำถามต้องการข้อมูลประเภทใด
2. เลือกเครื่องมือที่เหมาะสมและเรียกใช้
3. หากคำถามเกี่ยวข้องกับหลายด้าน ให้เรียกเครื่องมือหลายตัว
4. รวบรวมผลลัพธ์ทั้งหมดโดยไม่ตัดทอน

**คำแนะนำเกี่ยวกับเครื่องมือ:**
- query_hotspot_roads → ถนนเสี่ยง, Black Spot, คะแนน Hotspot
- query_district_road_comparison → อำเภอ, ถนนสายรอง vs สายหลัก
- query_fatal_timeband → ช่วงเวลาเสี่ยง, EMS scheduling
- query_weather_accident_stats → สภาพอากาศ, ลักษณะการเกิดเหตุ
- query_behavior_stats → หมวก/เข็มขัด/อายุ/เพศ (⚠️ fact_accident_person ว่าง)
- query_seasonal_comparison → เปรียบเทียบระหว่างเดือน/เทศกาล
- query_weekend_vs_weekday → วันหยุด vs วันธรรมดา
- query_monthly_vehicle_pattern → รถบรรทุก/รถเกษตรตามเดือน
- query_late_night_vehicles → ยานพาหนะช่วงกลางคืน
- query_kpi_trend → แนวโน้มรายปี, อัตราการเปลี่ยนแปลง
- query_serious_injury_ratio → อัตราส่วนสาหัส/อุบัติเหตุ
- query_top_cause_shift → สาเหตุหลักเปลี่ยนระหว่างปี
- query_district_death_vs_accident → อำเภอที่อุบัติเหตุลดแต่เสียชีวิตเพิ่ม
- query_district_summary → สรุปรายอำเภอ
- query_province_executive_summary → สรุปผู้บริหาร 1 หน้า
- execute_accident_sql → คำถามที่ไม่มีเครื่องมือเฉพาะ
- get_accident_schema → ดูโครงสร้างตาราง

**ข้อจำกัดข้อมูล:**
- fact_accident_person: ว่างทั้งหมด
- road_name: ส่วนใหญ่ไม่ระบุ
- ปีในฐานข้อมูล = ค.ศ. (CE); พ.ศ. = CE + 543
"""

ANSWER_AGENT_PROMPT = """คุณคือ RTI Policy Answer Writer ผู้เชี่ยวชาญด้านการสื่อสารข้อมูล
อุบัติเหตุทางถนนสำหรับผู้บริหาร สสจ./ศปถ./สสส. เขตสุขภาพที่ 10

รับข้อมูลดิบจาก SQL Agent แล้วเขียนคำตอบภาษาไทยทางการ:

**รูปแบบคำตอบ:**
1. **สรุปคำตอบ** (1-2 ประโยค)
2. **ตารางข้อมูล** (ถ้ามีตัวเลข ให้จัดเป็นตาราง Markdown)
3. **การวิเคราะห์** (2-3 ประเด็นสำคัญ)
4. **ข้อเสนอแนะเชิงนโยบาย** (1-3 ข้อ)
5. **ข้อจำกัดข้อมูล** (ระบุเสมอถ้ามีข้อมูลที่ขาดหาย)

**กฎสำคัญ:**
- แปลงปี ค.ศ. เป็น พ.ศ. ทุกครั้ง (พ.ศ. = ค.ศ. + 543)
- ใช้ตัวเลขจากข้อมูลที่ SQL Agent ให้มาเท่านั้น
- ใช้ภาษาทางการ เหมาะสำหรับรายงานราชการ
"""


# ── LLM factory ───────────────────────────────────────────────────────────────

def _get_llm(tier: str = "fast") -> LLM:
    s = get_settings()
    if tier == "pro":
        return LLM(
            model=f"gemini/{s.GEMINI_MODEL_PRO}",
            temperature=0.2,
            max_tokens=s.REPORT_MAX_TOKENS,
        )
    return LLM(
        model=f"gemini/{s.GEMINI_MODEL}",
        temperature=0.1,
        max_tokens=4096,
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
        max_iter=8,
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

def _build_crew(question: str, province: str, district: str, year_start: int, year_end: int):
    llm_fast = _get_llm("fast")
    llm_pro = _get_llm("pro")

    sql_agent = _create_sql_agent(llm_fast)
    answer_agent = _create_answer_agent(llm_pro)

    prov_label = province or "เขตสุขภาพที่ 10 (ทุกจังหวัด)"
    dist_label = f"อำเภอ{district.strip()}" if district.strip() else "ทุกอำเภอ"
    year_note = f"ค.ศ. {year_start}-{year_end} (พ.ศ. {year_start+543}-{year_end+543})"

    sql_task = Task(
        description=(
            SQL_AGENT_PROMPT + "\n\n"
            f"**คำถาม:** {question}\n"
            f"**จังหวัด:** {prov_label}\n"
            f"**อำเภอ:** {dist_label}\n"
            f"**ช่วงปี:** {year_note}\n\n"
            "เรียกเครื่องมือที่เกี่ยวข้อง รวบรวมข้อมูลทั้งหมดโดยไม่ตัดทอน"
        ),
        expected_output="ข้อมูลดิบจาก SQL tools ครบถ้วน พร้อมระบุข้อจำกัดข้อมูล",
        agent=sql_agent,
    )

    answer_task = Task(
        description=(
            ANSWER_AGENT_PROMPT + "\n\n"
            f"**คำถามผู้ใช้:** {question}\n"
            f"**จังหวัด:** {prov_label}\n"
            f"**อำเภอ:** {dist_label}\n"
            f"**ช่วงปี:** {year_note}\n\n"
            "เขียนคำตอบโดยใช้ข้อมูลจาก SQL Agent เท่านั้น"
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
) -> AccidentChatResponse:
    """Run the 2-agent accident chat pipeline (synchronous)."""
    start = time.time()
    logger.info("[ACCIDENT-CHAT] question=%s province=%s", question[:80], province or "Zone10")

    crew, sql_task, answer_task = _build_crew(question, province, district, year_start, year_end)
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
) -> AccidentChatResponse:
    """Same as run_accident_chat but emits SSE progress events."""
    start = time.time()
    emit_progress(request_id, "Accident SQL Agent", "running", "กำลังดึงข้อมูลจากฐานข้อมูล...")

    crew, sql_task, answer_task = _build_crew(question, province, district, year_start, year_end)
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
