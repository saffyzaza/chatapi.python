"""Workplan Agent — Pure LLM pipeline to generate Thai work plans as streaming HTML.

Flow:
  [Step 1] Plan Analyzer — CrewAI agent วิเคราะห์วัตถุประสงค์ กลุ่มเป้าหมาย บริบท
  [Step 2] Plan Writer   — litellm streaming (gemini-2.5-pro) สร้าง HTML work plan

SSE events → queue:
  {"type": "agent_start",    "step": "analyzer",  "agentName": "Plan Analyzer"}
  {"type": "agent_done",     "step": "analyzer",  "result": "...plan structure..."}
  {"type": "agent_start",    "step": "generator", "agentName": "Plan Writer"}
  {"type": "generator_chunk","html": "...chunk..."}   ← streamed while generating
  {"type": "agent_done",     "step": "generator", "result": "สร้าง HTML work plan สำเร็จ (N chars)"}
  {"type": "final",          "reportHtml": "...", "reportTitle": "...", "agentSteps": [...]}
"""
import asyncio
import os
import re
from typing import Any

import litellm
from crewai import Agent, LLM

from src.agents.csv_pipeline import _get_llm, _run_agent
from src.agents.thaijo_prompts import JOURNAL_CSS
from src.tools.error_logger import log_agent_error


# ── Workplan CSS — extends JOURNAL_CSS with workplan-specific styles ──────────

_WORKPLAN_CSS = JOURNAL_CSS + """
  .cover-title { font-size: 22px; font-weight: 700; text-align: center; margin: 24px 0 12px; color: #1a3c6e; line-height: 1.4; }
  .cover-subtitle { font-size: 15px; text-align: center; color: #444; margin-bottom: 8px; }
  .cover-agency { font-size: 14px; text-align: center; color: #555; margin-bottom: 6px; }
  .cover-year { font-size: 13px; text-align: center; color: #666; margin-top: 16px; }
  .cover-logo { text-align: center; font-size: 48px; margin-bottom: 16px; }
  .principle-box { background: #f0f4ff; border-left: 4px solid #1a3c6e; padding: 14px 18px; margin: 12px 0; border-radius: 4px; }
  .objective-list { list-style: none; padding-left: 0; margin: 10px 0; }
  .objective-list li { padding: 6px 0 6px 24px; position: relative; font-size: 13px; line-height: 1.7; }
  .objective-list li::before { content: "✓"; position: absolute; left: 0; color: #1a6b3c; font-weight: 700; }
  .activity-table { width: 100%; border-collapse: collapse; font-size: 12px; margin: 14px 0; }
  .activity-table th { background: #1a3c6e; color: #fff; padding: 8px 10px; text-align: left; }
  .activity-table td { border: 1px solid #ccc; padding: 6px 10px; vertical-align: top; }
  .activity-table tr:nth-child(even) td { background: #f5f8fc; }
  .budget-box { background: #fff8e1; border: 1px solid #f9c74f; border-radius: 6px; padding: 14px 18px; margin: 12px 0; }
  .budget-total { font-size: 16px; font-weight: 700; color: #b85c00; margin-top: 8px; }
  .kpi-list { list-style: decimal; padding-left: 1.4em; font-size: 13px; line-height: 1.8; }
  .responsible-box { background: #f0fff4; border: 1px solid #aad5b8; border-radius: 6px; padding: 12px 16px; margin: 10px 0; }
  .gantt-bar { display: inline-block; background: #1a3c6e; height: 14px; border-radius: 3px; vertical-align: middle; }
"""


def _fallback_workplan_html(prompt: str, doc_type: str) -> str:
    """Fallback HTML if streaming fails."""
    doc_labels = {"workplan": "แผนงาน", "plan": "แผนยุทธศาสตร์", "policy": "แผนนโยบาย"}
    label = doc_labels.get(doc_type, "แผนงาน")
    return f"""<!DOCTYPE html><html lang="th"><head><meta charset="UTF-8">
<style>{_WORKPLAN_CSS}</style></head><body>
<div class="page">
  <div class="cover-logo">📋</div>
  <div class="cover-title">{label}: {prompt}</div>
  <h3 class="section-heading">หลักการและเหตุผล</h3>
  <div class="principle-box">
    <p class="body-para">แผนงานนี้จัดทำขึ้นเพื่อตอบสนองต่อ: {prompt}</p>
  </div>
  <p class="body-para">[กำลังสร้างแผนงาน — กรุณาลองอีกครั้ง]</p>
</div></body></html>"""


def _build_workplan_prompt(prompt: str, plan_structure: str, doc_type: str) -> str:
    """Build the HTML generation prompt for the workplan."""
    doc_labels = {
        "workplan": ("แผนงาน (Work Plan)", "แผนปฏิบัติการ"),
        "plan": ("แผนยุทธศาสตร์ (Strategic Plan)", "แผนยุทธศาสตร์"),
        "policy": ("แผนนโยบาย (Policy Plan)", "แผนนโยบาย"),
    }
    doc_label, doc_short = doc_labels.get(doc_type, ("แผนงาน", "แผนงาน"))

    return f"""คุณคือผู้เชี่ยวชาญการเขียนแผนงานสาธารณสุขไทยระดับมืออาชีพ
สร้าง {doc_label} ฉบับสมบูรณ์จากหัวข้อ: {prompt}

โครงสร้างที่วิเคราะห์ไว้:
{plan_structure}

สร้าง HTML ที่มีโครงสร้างดังนี้ (3-5 หน้า A4 แยกด้วย <div class="page">):

หน้า 1 — หน้าปก:
  <div class="cover-logo">📋</div>
  <div class="cover-title">ชื่อ{doc_short}เต็ม</div>
  <div class="cover-subtitle">ชื่อย่อหรือรหัสแผน</div>
  <div class="cover-agency">หน่วยงานรับผิดชอบ</div>
  <div class="cover-year">ปีงบประมาณ พ.ศ. ...</div>
  <span class="page-num">1</span>

หน้า 2 — หลักการเหตุผลและวัตถุประสงค์:
  <h3 class="section-heading">หลักการและเหตุผล</h3>
  <div class="principle-box">2-3 ย่อหน้า อธิบายที่มาและความสำคัญ</div>
  <h3 class="section-heading">วัตถุประสงค์</h3>
  <ul class="objective-list">
    <li>วัตถุประสงค์ที่ 1...</li>
    <li>วัตถุประสงค์ที่ 2...</li>
  </ul>
  <h3 class="section-heading">กลุ่มเป้าหมาย</h3>
  <p class="body-para">ระบุกลุ่มเป้าหมาย จำนวน และพื้นที่</p>
  <span class="page-num">2</span>

หน้า 3 — กิจกรรมและระยะเวลา:
  <h3 class="section-heading">กิจกรรมและระยะเวลาดำเนินการ</h3>
  <table class="activity-table">
    <tr><th>ที่</th><th>กิจกรรม</th><th>ระยะเวลา</th><th>ผู้รับผิดชอบ</th><th>งบประมาณ (บาท)</th></tr>
    <!-- สร้างแถวสำหรับแต่ละกิจกรรม (อย่างน้อย 5-8 กิจกรรม) -->
  </table>
  <span class="page-num">3</span>

หน้า 4 — งบประมาณและตัวชี้วัด:
  <h3 class="section-heading">งบประมาณโดยประมาณ</h3>
  <div class="budget-box">
    <table class="activity-table"> ตารางแยกหมวดงบประมาณ </table>
    <div class="budget-total">งบประมาณรวม: X,XXX,XXX บาท</div>
  </div>
  <h3 class="section-heading">ตัวชี้วัดความสำเร็จ</h3>
  <ol class="kpi-list">
    <li>ตัวชี้วัดที่ 1 พร้อมค่าเป้าหมาย</li>
    <li>ตัวชี้วัดที่ 2 พร้อมค่าเป้าหมาย</li>
  </ol>
  <span class="page-num">4</span>

หน้า 5 — ผู้รับผิดชอบและการติดตาม:
  <h3 class="section-heading">ผู้รับผิดชอบ</h3>
  <div class="responsible-box">ระบุชื่อตำแหน่ง/หน่วยงาน</div>
  <h3 class="section-heading">การติดตามและประเมินผล</h3>
  <p class="body-para">วิธีการติดตามและประเมินผล รายงานผลทุก ... เดือน</p>
  <h3 class="section-heading">ผลที่คาดว่าจะได้รับ</h3>
  <ul class="objective-list">
    <li>ผลที่คาดว่าจะได้รับที่ 1</li>
    <li>ผลที่คาดว่าจะได้รับที่ 2</li>
  </ul>
  <span class="page-num">5</span>

CSS ที่ต้องใส่ใน <style>:
{_WORKPLAN_CSS}

กฎบังคับ:
1. ตอบเป็น HTML เท่านั้น เริ่มด้วย <!DOCTYPE html> ห้ามมี markdown หรือ ``` ก่อนหรือหลัง HTML
2. ห้ามใช้ markdown — ใช้ HTML class ตามที่กำหนด
3. ตาราง activity ต้องมีอย่างน้อย 5-8 กิจกรรมที่เป็นรูปธรรม
4. ตัวเลขงบประมาณต้องสมจริง ระบุหน่วย (บาท)
5. ตัวชี้วัดต้องมีค่าเป้าหมายที่วัดได้

ตอบเป็น HTML เท่านั้น เริ่มด้วย <!DOCTYPE html>:"""


def run_workplan_pipeline(
    prompt: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    session_id: str = "",
    doc_type: str = "workplan",
) -> None:
    """Run the work plan generation pipeline (pure LLM, no CSV needed).

    Emits SSE events via queue/loop including streaming HTML chunks.
    """
    llm = _get_llm()

    def put(ev: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(queue.put(ev), loop)

    agent_steps: list[dict] = []

    # ── STEP 1: Plan Analyzer ──────────────────────────────────────────────────
    put({"type": "agent_start", "step": "analyzer", "agentName": "Plan Analyzer"})
    analyzer = Agent(
        role="Plan Analyzer — Thai Public Health Work Plan Specialist",
        goal=(
            "วิเคราะห์หัวข้อและสร้างโครงสร้างแผนงานที่ครบถ้วน "
            "ระบุวัตถุประสงค์ กลุ่มเป้าหมาย กิจกรรม และตัวชี้วัดที่ชัดเจน"
        ),
        backstory=(
            "คุณเป็นผู้เชี่ยวชาญการวางแผนงานสาธารณสุขระดับจังหวัดและเขตสุขภาพ "
            "คุณมีประสบการณ์ในการจัดทำแผนงาน งบประมาณ และตัวชี้วัดตามระบบ BSC "
            "คุณวิเคราะห์โจทย์อย่างรอบด้าน คำนึงถึงบริบท กลุ่มเป้าหมาย และทรัพยากรที่มี"
        ),
        llm=llm,
        verbose=False,
        max_iter=5,
    )

    doc_labels = {"workplan": "แผนปฏิบัติการ", "plan": "แผนยุทธศาสตร์", "policy": "แผนนโยบาย"}
    doc_label = doc_labels.get(doc_type, "แผนงาน")

    plan_structure = _run_agent(
        analyzer,
        (
            f"หัวข้อ: {prompt}\n"
            f"ประเภทเอกสาร: {doc_label}\n\n"
            f"วิเคราะห์และสรุปโครงสร้าง{doc_label}นี้ (ไม่เกิน 20 บรรทัด):\n"
            "1. ชื่อแผนงานที่เหมาะสม\n"
            "2. หลักการเหตุผล (2-3 ประเด็น)\n"
            "3. วัตถุประสงค์ (3-5 ข้อ)\n"
            "4. กลุ่มเป้าหมาย (ระบุให้ชัดเจน)\n"
            "5. กิจกรรมหลัก (5-8 กิจกรรม พร้อมระยะเวลา)\n"
            "6. งบประมาณโดยประมาณ (แยกหมวด)\n"
            "7. ตัวชี้วัดความสำเร็จ (3-5 ตัวชี้วัด พร้อมค่าเป้าหมาย)\n"
            "8. ผู้รับผิดชอบหลัก"
        ),
        f"โครงสร้าง{doc_label}สั้นๆ ไม่เกิน 20 บรรทัด พร้อมรายละเอียดที่สำคัญ",
        step="analyzer", session_id=session_id,
    )
    put({"type": "agent_done", "step": "analyzer", "agentName": "Plan Analyzer",
         "result": plan_structure, "docType": doc_type})
    agent_steps.append({"step": "analyzer", "agentName": "Plan Analyzer",
                        "result": plan_structure})

    # ── STEP 2: Plan Writer (streaming HTML) ───────────────────────────────────
    put({"type": "agent_start", "step": "generator", "agentName": "Plan Writer"})

    gen_prompt = _build_workplan_prompt(prompt, plan_structure, doc_type)
    html_parts: list[str] = []
    stream_error = ""

    try:
        stream = litellm.completion(
            model="gemini/gemini-2.5-pro",
            api_key=os.getenv("GEMINI_API_KEY"),
            messages=[{"role": "user", "content": gen_prompt}],
            stream=True,
            temperature=0.3,
        )
        for chunk in stream:
            delta = ""
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta.content or ""
            if delta:
                html_parts.append(delta)
                put({"type": "generator_chunk", "html": delta})
    except Exception as exc:
        stream_error = str(exc)
        log_agent_error(
            stream_error,
            agent_name="Plan Writer",
            step="generator",
            domain="workplan",
            prompt=prompt,
        )

    full_html = "".join(html_parts).strip()

    # Strip markdown fences if model wrapped output
    if full_html.startswith("```"):
        full_html = re.sub(r"^```[a-z]*\n?", "", full_html)
        full_html = re.sub(r"\n?```$", "", full_html).strip()

    if not full_html or "<html" not in full_html:
        full_html = _fallback_workplan_html(prompt, doc_type)
        put({"type": "generator_chunk", "html": full_html})

    done_msg = f"สร้าง HTML work plan สำเร็จ ({len(full_html)} ตัวอักษร)"
    put({"type": "agent_done", "step": "generator", "agentName": "Plan Writer",
         "result": done_msg})
    agent_steps.append({"step": "generator", "agentName": "Plan Writer",
                        "result": done_msg})

    # ── FINAL EVENT ────────────────────────────────────────────────────────────
    put({
        "type":        "final",
        "message":     f"สร้าง{doc_labels.get(doc_type, 'แผนงาน')}สำเร็จ",
        "reportHtml":  full_html,
        "reportTitle": prompt,
        "agentSteps":  agent_steps,
    })
