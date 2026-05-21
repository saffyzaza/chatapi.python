"""Zone 10 Accident Policy pipeline orchestrator.

Sequential 3-agent crew:
  Zone10SqlFetcher → Zone10PolicyAnalyst → Zone10ReportWriter

Entry point: run_zone10_analysis(provinces, questions, year_range) → dict
"""
import json
import logging
import re
import time
from datetime import datetime

from crewai import Crew, Task, Process, LLM

from src.config import get_settings
from src.agents.agent_defaults import kickoff_with_retry
from src.agents.accident_policy_agent import (
    SQL_FETCHER_PROMPT,
    POLICY_ANALYST_PROMPT,
    REPORT_WRITER_PROMPT,
    create_zone10_sql_fetcher,
    create_zone10_policy_analyst,
    create_zone10_report_writer,
    ZONE10_TOOLS,
)
from src.tools.zone10_accident import ZONE10_PROVINCES

logger = logging.getLogger(__name__)


def _get_llm(tier: str = "fast") -> LLM:
    s = get_settings()
    if tier == "pro":
        return LLM(
            model=f"gemini/{s.GEMINI_MODEL_PRO}",
            temperature=0.3,
            max_tokens=s.REPORT_MAX_TOKENS,
        )
    return LLM(
        model=f"gemini/{s.GEMINI_MODEL}",
        temperature=0.2,
        max_tokens=4096,
    )


def run_zone10_analysis(
    provinces: list[str] | None = None,
    questions: list[str] | None = None,
    year_range: list[int] | None = None,
) -> dict:
    """Run the Zone 10 RTI accident policy analysis pipeline.

    Args:
        provinces: List of Thai province names (defaults to all 5 Zone 10 provinces).
        questions: List of question IDs e.g. ["Q1","Q3"] or ["all"] (default: all).
        year_range: [start_year, end_year] CE (default: [2021, 2026]).

    Returns:
        dict with keys: zone, provinces, policy_brief, sections, charts, metadata
    """
    start_time = time.time()

    if not provinces:
        provinces = list(ZONE10_PROVINCES)
    if not year_range or len(year_range) < 2:
        year_range = [2021, 2026]

    provinces_str = ",".join(provinces)
    today = datetime.now().strftime("%Y-%m-%d")

    llm_fast = _get_llm("fast")
    llm_pro = _get_llm("pro")

    fetcher = create_zone10_sql_fetcher(llm_fast)
    analyst = create_zone10_policy_analyst(llm_pro)
    writer = create_zone10_report_writer(llm_pro)

    fetch_task = Task(
        description=(
            SQL_FETCHER_PROMPT + "\n\n"
            f"**จังหวัดเป้าหมาย:** {provinces_str}\n"
            f"**ช่วงปี:** {year_range[0]}–{year_range[1]} (CE)\n"
            "เรียกเครื่องมือทั้ง 7 ตัว ส่งค่า provinces ตามที่ระบุ"
        ),
        expected_output=(
            "ผลลัพธ์จากเครื่องมือทั้ง 7 ตัว รวมไว้ในรูปแบบ Q1: ... Q7: ... ครบถ้วน"
        ),
        agent=fetcher,
    )

    analyst_task = Task(
        description=(
            POLICY_ANALYST_PROMPT + "\n\n"
            f"**จังหวัด:** {provinces_str}\n"
            f"**ช่วงปี:** {year_range[0]}–{year_range[1]}\n"
            f"**วันที่วิเคราะห์:** {today}\n\n"
            "วิเคราะห์ข้อมูลทั้งหมดจาก SQL Fetcher แล้วผลิต JSON ตามโครงสร้างที่กำหนด"
        ),
        expected_output=(
            "JSON object ที่มี keys: hotspot, human_behavior, environment, kpi, haddon_matrix"
        ),
        agent=analyst,
        context=[fetch_task],
    )

    report_task = Task(
        description=(
            REPORT_WRITER_PROMPT + "\n\n"
            f"**จังหวัด:** {provinces_str}\n"
            f"**ช่วงปี:** {year_range[0]}–{year_range[1]}\n"
            f"**วันที่จัดทำ:** {today}\n\n"
            "เขียนรายงานนโยบายฉบับสมบูรณ์ 5 ส่วน ในรูปแบบ Markdown ภาษาทางการ"
        ),
        expected_output=(
            "รายงานนโยบาย Markdown ครบ 5 ส่วน พร้อมตาราง KPI และข้อเสนอแนะ 3 ระยะ"
        ),
        agent=writer,
        context=[analyst_task],
    )

    crew = Crew(
        agents=[fetcher, analyst, writer],
        tasks=[fetch_task, analyst_task, report_task],
        process=Process.sequential,
        verbose=True,
    )

    logger.info("[ZONE10-POLICY] provinces=%s year_range=%s", provinces_str, year_range)

    try:
        result = kickoff_with_retry(crew)
        elapsed = time.time() - start_time
        logger.info("Zone 10 crew completed in %.1fs", elapsed)
        return _build_response(result, provinces, year_range, elapsed, [fetch_task, analyst_task, report_task])
    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("Zone 10 crew failed after %.1fs: %s", elapsed, exc)
        return {
            "zone": "เขตสุขภาพที่ 10",
            "provinces": provinces,
            "policy_brief": f"เกิดข้อผิดพลาดในการวิเคราะห์: {exc}",
            "sections": {},
            "charts": [],
            "metadata": {
                "error": str(exc),
                "elapsed_seconds": round(elapsed, 1),
                "pipeline": "zone10_accident_policy",
            },
        }


def _build_response(result, provinces, year_range, elapsed, all_tasks) -> dict:
    tasks_output = getattr(result, "tasks_output", [])

    policy_brief = str(result)
    if tasks_output:
        last_out = tasks_output[-1]
        policy_brief = getattr(last_out, "raw", None) or str(last_out)

    sections: dict = {}
    if len(tasks_output) >= 2:
        analyst_raw = getattr(tasks_output[-2], "raw", None) or str(tasks_output[-2])
        sections = _parse_analyst_json(analyst_raw)

    return {
        "zone": "เขตสุขภาพที่ 10",
        "provinces": provinces,
        "policy_brief": policy_brief,
        "sections": sections,
        "charts": _extract_chart_candidates(sections),
        "metadata": {
            "elapsed_seconds": round(elapsed, 1),
            "agent_count": 3,
            "pipeline": "zone10_accident_policy",
            "provinces_analyzed": provinces,
            "year_range": year_range,
        },
    }


def _parse_analyst_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if any(k in parsed for k in ("hotspot", "human_behavior", "environment", "kpi")):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return {"_raw": raw}


def _extract_chart_candidates(sections: dict) -> list[dict]:
    charts = []
    kpi = sections.get("kpi", {})
    if isinstance(kpi.get("yearly_trend"), list) and kpi["yearly_trend"]:
        charts.append({
            "type": "line",
            "title": "แนวโน้มอุบัติเหตุรายปี — เขตสุขภาพที่ 10",
            "data": {"labels": [], "datasets": []},
            "source": "Q6",
        })
    if isinstance(kpi.get("festival_risk"), list) and kpi["festival_risk"]:
        charts.append({
            "type": "bar",
            "title": "ความเสี่ยงรายเดือน (เทศกาล)",
            "data": {"labels": [], "datasets": []},
            "source": "Q7",
        })
    return charts


def run_zone10_data_only(provinces: list[str] | None = None) -> dict:
    """Return raw SQL tool results for all 7 questions without running LLM agents."""
    from src.tools.zone10_accident import (
        _query_top_roads,
        _query_time_bands,
        _query_motorcycle_severity,
        _query_car_serious_injuries,
        _query_environment_risk,
        _query_yearly_kpi,
        _query_monthly_risk,
    )

    if not provinces:
        provinces = list(ZONE10_PROVINCES)
    provinces_str = ",".join(provinces)

    results: dict[str, str] = {}
    errors: dict[str, str] = {}

    for label, fn in [
        ("Q1_hotspot_roads", _query_top_roads),
        ("Q2_time_bands", _query_time_bands),
        ("Q3_motorcycle_severity", _query_motorcycle_severity),
        ("Q4_car_injuries", _query_car_serious_injuries),
        ("Q5_environment_risk", _query_environment_risk),
        ("Q6_yearly_kpi", _query_yearly_kpi),
        ("Q7_monthly_risk", _query_monthly_risk),
    ]:
        try:
            if label == "Q1_hotspot_roads":
                results[label] = fn(provinces_str, 10)
            else:
                results[label] = fn(provinces_str)
        except Exception as exc:
            errors[label] = str(exc)

    return {
        "zone": "เขตสุขภาพที่ 10",
        "provinces": provinces,
        "questions": results,
        "errors": errors,
        "metadata": {"pipeline": "zone10_data_only", "provinces_queried": provinces},
    }
