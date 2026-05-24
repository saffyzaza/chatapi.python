"""Error Monitor Agent — reads aggregated error logs and generates Thai-language analysis.

Usage:
    from src.agents.error_monitor_agent import run_error_monitor
    report = run_error_monitor(days=7)
"""
import json
import os

from crewai import Agent, Crew, LLM, Task

from src.tools.error_logger import read_all_errors, aggregate_errors


def _get_llm() -> LLM:
    return LLM(model="gemini/gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))


def run_error_monitor(days: int = 7) -> dict:
    """Read error logs for the last N days and return structured report + LLM analysis.

    Returns:
        {
            "aggregate": {...},   # raw counts
            "report": "...",      # Thai markdown from LLM
            "entries": [...]      # raw entries (newest first)
        }
    """
    entries = read_all_errors(days=days)
    agg = aggregate_errors(entries)

    if not entries:
        return {
            "aggregate": agg,
            "report": "ไม่พบ error log ในช่วง {} วันที่ผ่านมา".format(days),
            "entries": [],
        }

    # Build compact summary for LLM — avoid sending all entries raw
    recent_samples = entries[:10]
    agg_text = json.dumps(agg, ensure_ascii=False, indent=2)
    samples_text = json.dumps(recent_samples, ensure_ascii=False, indent=2)

    analyst = Agent(
        role="Agent Error Monitor",
        goal="วิเคราะห์ error log ของ AI Agent pipeline และสรุปปัญหาพร้อมข้อเสนอแนะ",
        backstory=(
            "คุณเป็น DevOps / MLOps specialist ที่เชี่ยวชาญการวิเคราะห์ log ของระบบ Multi-Agent AI "
            "คุณอ่าน error pattern แล้วสรุปสาเหตุหลัก ผลกระทบ และแนวทางแก้ไขเป็นภาษาไทย"
        ),
        llm=_get_llm(),
        verbose=False,
        max_iter=3,
    )

    task = Task(
        description=(
            f"วิเคราะห์ error log ของ AI Agent ใน {days} วันที่ผ่านมา\n\n"
            f"สถิติรวม:\n{agg_text}\n\n"
            f"ตัวอย่าง error ล่าสุด ({len(recent_samples)} รายการ):\n{samples_text}\n\n"
            "เขียนรายงานภาษาไทย โครงสร้าง:\n\n"
            "## สรุปภาพรวม Error\n"
            "จำนวน error ทั้งหมด, ช่วงเวลา, error type หลัก\n\n"
            "## ตาราง Error ตามประเภท\n"
            "| Error Type | จำนวน | ความหมาย | ความรุนแรง |\n"
            "|---|---|---|---|\n"
            "...\n\n"
            "## Agent / Step ที่มีปัญหาบ่อยที่สุด\n"
            "รายการ agent และ step ที่ error มากที่สุด\n\n"
            "## สาเหตุหลักที่พบ\n"
            "วิเคราะห์ root cause ของแต่ละ error type\n\n"
            "## ข้อเสนอแนะการแก้ไข\n"
            "แนวทางแก้ไขเรียงตามความสำคัญ\n\n"
            "IMPORTANT: ต้องมีตาราง markdown เสมอ"
        ),
        expected_output="รายงาน error analysis ภาษาไทยพร้อมตาราง markdown",
        agent=analyst,
    )

    crew = Crew(agents=[analyst], tasks=[task], verbose=False)
    try:
        report = str(crew.kickoff()).strip()
    except Exception as exc:
        report = f"[Error Monitor ล้มเหลว: {exc}]\n\nสถิติดิบ:\n{agg_text}"

    return {
        "aggregate": agg,
        "report": report,
        "entries": entries,
    }
