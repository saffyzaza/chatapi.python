"""Report Agent — finds a CSV dataset and generates a comprehensive Thai Markdown report.

Flow:
  [Step 1] File Finder   — ค้นหา dataset
  [Step 2] Schema Analyst — อ่าน schema
  [Step 3] Data Analyst   — สร้าง pandas code เพื่อวิเคราะห์ข้อมูล
  [Step 4] Python Executor — รันโค้ด
  [Step 5] Report Writer  — เขียนรายงาน Markdown ภาษาไทยครบถ้วน

SSE events → queue:
  {"type": "agent_start", "step": "file_finder",   "agentName": "File Finder"}
  {"type": "agent_done",  "step": "file_finder",   "result": "..."}
  {"type": "agent_start", "step": "schema",         "agentName": "Schema Analyst"}
  {"type": "agent_done",  "step": "schema",         "result": "..."}
  {"type": "agent_start", "step": "data_analyst",   "agentName": "Data Analyst"}
  {"type": "agent_done",  "step": "data_analyst",   "result": "..."}
  {"type": "agent_start", "step": "executor",       "agentName": "Python Executor"}
  {"type": "agent_done",  "step": "executor",       "result": "..."}
  {"type": "agent_start", "step": "report_writer",  "agentName": "Report Writer"}
  {"type": "agent_done",  "step": "report_writer",  "result": "..."}
  {"type": "final",       "message": "...(full Markdown)...", "agentSteps": [...]}
"""
import asyncio
import os
import re
from typing import Any

from crewai import Agent, LLM

from src.tools.minio import (
    list_csv_files,
    list_csv_files_impl,
    read_csv_schema_impl,
    exec_python,
    resolve_file_id,
    fallback_find_file,
)
from src.agents.csv_pipeline import (
    _get_llm,
    _run_agent,
    _extract_code,
    _is_agent_error,
    _is_exec_error,
    _find_code_issues,
)
from src.agents.prompt_profile import (
    ANALYST_CORE_POLICY,
    CODE_GENERATOR_CORE_POLICY,
    INSIGHT_RESPONSE_BLUEPRINT,
    MISSING_DATA_POLICY,
    join_prompt,
)
from src.tools.error_logger import log_agent_error


def run_report_pipeline(
    prompt: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    session_id: str = "",
) -> None:
    """Run the comprehensive report generation pipeline.

    Emits SSE events via queue/loop.
    Final event message contains the full Markdown report.
    """
    llm = _get_llm()

    def put(ev: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(queue.put(ev), loop)

    agent_steps: list[dict] = []

    # ── STEP 1: File Finder ────────────────────────────────────────────────────
    put({"type": "agent_start", "step": "file_finder", "agentName": "File Finder"})
    finder = Agent(
        role="File Finder — ค้นหา dataset สำหรับรายงาน",
        goal="ค้นหาไฟล์ CSV ที่เกี่ยวข้องกับคำถามและคืนค่า file ID ที่ถูกต้อง",
        backstory=(
            "คุณเป็นผู้เชี่ยวชาญการค้นหา dataset สาธารณสุข "
            "คุณต้องใช้ tool list_csv_files เพื่อดูรายการไฟล์ "
            "แล้วคืนค่า ENTIRE line รวมถึง [ID:...] ที่แน่ชัด"
        ),
        tools=[list_csv_files],
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    file_result = _run_agent(
        finder,
        (
            f"คำถาม: {prompt}\n\n"
            "ขั้นตอน:\n"
            "1. เรียก list_csv_files(prefix='') เพื่อดูไฟล์ทั้งหมด\n"
            "2. เลือกไฟล์ที่ตรงกับคำถามมากที่สุด\n"
            "3. ตอบเฉพาะ 1 บรรทัด ในรูปแบบ: [ID:xxxxxx] filename.csv\n"
            "   โดย [ID:xxxxxx] คือ ID จริงที่ได้จาก tool (ห้ามเปลี่ยน)"
        ),
        "Selected file: exactly one line in format [ID:xxxxxx] filename.csv",
        step="file_finder", session_id=session_id,
    )
    if _is_agent_error(file_result):
        file_result = fallback_find_file(prompt, "")
    put({"type": "agent_done", "step": "file_finder", "agentName": "File Finder",
         "result": file_result})
    agent_steps.append({"step": "file_finder", "agentName": "File Finder",
                        "result": file_result})

    resolved_file_id = resolve_file_id(file_result)
    if not resolved_file_id:
        fallback_listing = list_csv_files_impl("")
        if fallback_listing and not fallback_listing.startswith("No") and not fallback_listing.startswith("Error"):
            first_line = fallback_listing.split("\n")[0]
            resolved_file_id = resolve_file_id(first_line)
            if resolved_file_id:
                file_result = first_line

    if not resolved_file_id:
        fallback_listing = list_csv_files_impl("")
        sample_lines = []
        if fallback_listing and not fallback_listing.startswith("No") and not fallback_listing.startswith("Error"):
            sample_lines = [ln for ln in fallback_listing.split("\n") if ln.strip()][:5]
        candidate_text = "\n".join(f"- {ln}" for ln in sample_lines) if sample_lines else "- ไม่พบไฟล์ตัวอย่าง"
        message = (
            "ยังหาไฟล์สำหรับสร้างรายงานไม่เจอ จึงยังวิเคราะห์ต่อไม่ได้อย่างมั่นใจ\n\n"
            "ไฟล์ตัวอย่างที่พบ:\n"
            f"{candidate_text}\n\n"
            "กรุณาระบุคำค้นให้ชัดขึ้น เช่น ปี/จังหวัด/ตัวชี้วัด"
        )
        put({"type": "final", "message": message, "agentSteps": agent_steps})
        return

    # ── STEP 2: Schema Analyst ─────────────────────────────────────────────────
    put({"type": "agent_start", "step": "schema", "agentName": "Schema Analyst"})
    schema_result = ""
    if resolved_file_id:
        schema_result = read_csv_schema_impl(resolved_file_id)
    if not schema_result or schema_result.startswith("Error"):
        schema_result = f"[ไม่สามารถอ่าน schema ของ file_id={resolved_file_id}]"
    put({"type": "agent_done", "step": "schema", "agentName": "Schema Analyst",
         "result": schema_result})
    agent_steps.append({"step": "schema", "agentName": "Schema Analyst",
                        "result": schema_result})

    # ── STEP 3: Data Analyst ───────────────────────────────────────────────────
    put({"type": "agent_start", "step": "data_analyst", "agentName": "Data Analyst"})
    data_analyst = Agent(
        role="Data Analyst — Comprehensive Report Code Generator",
        goal=(
            "สร้าง Python/Pandas code ที่รันได้ทันที เพื่อวิเคราะห์ข้อมูลอย่างครบถ้วน "
            "สำหรับนำไปเขียนรายงานสรุปผู้บริหาร"
        ),
        backstory=join_prompt(
            "คุณเป็น Python/Pandas expert ที่เชี่ยวชาญการวิเคราะห์ข้อมูลสาธารณสุข "
            "คุณสร้างโค้ดที่ครอบคลุม: สถิติพื้นฐาน แนวโน้มรายปี ranking พื้นที่ "
            "และการกระจายตัวของข้อมูล เพื่อให้นักวิเคราะห์ใช้เขียนรายงานได้",
            CODE_GENERATOR_CORE_POLICY,
        ),
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    code_result = _run_agent(
        data_analyst,
        (
            f"คำถาม/หัวข้อรายงาน: {prompt}\n"
            f"file_id (ใช้ค่านี้เท่านั้น): '{resolved_file_id}'\n"
            f"Schema:\n{schema_result}\n\n"
            "==== กฎบังคับ (ห้ามละเมิด) ====\n"
            f"1. บรรทัดแรก: df = load_csv('{resolved_file_id}')\n"
            "2. ห้าม redefine load_csv / import minio / ใช้ pd.read_csv()\n"
            "3. pd.set_option('display.max_rows', 100) ก่อน print\n\n"
            "==== การวิเคราะห์สำหรับรายงาน ====\n"
            "4. สถิติพื้นฐานภาพรวม (sum, mean, max, min พร้อม label)\n"
            "5. แนวโน้มรายปี (ถ้ามี column ปี)\n"
            "6. Top 10 พื้นที่/จังหวัด (ถ้ามี column พื้นที่)\n"
            "7. การกระจายตัว (ค่าสูงสุด ค่าต่ำสุด ค่าเฉลี่ย)\n"
            "8. print หัวข้อก่อนทุก section\n"
            "9. ใช้ print(df.to_string(index=False)) เพื่อแสดงครบ\n"
            "Wrap code in ```python ... ```\n\n"
            f"{CODE_GENERATOR_CORE_POLICY}"
        ),
        "Working Python code with comprehensive analysis output for report writing",
        step="data_analyst", session_id=session_id,
    )
    put({"type": "agent_done", "step": "data_analyst", "agentName": "Data Analyst",
         "result": code_result})
    agent_steps.append({"step": "data_analyst", "agentName": "Data Analyst",
                        "result": code_result})

    # ── STEP 4: Python Executor ────────────────────────────────────────────────
    put({"type": "agent_start", "step": "executor", "agentName": "Python Executor"})
    code = _extract_code(code_result)

    if _is_agent_error(code):
        exec_output = f"[ข้ามการรัน — code generation ล้มเหลว]\n{code_result}"
        code = ""
    else:
        required_lines = [f"df = load_csv('{resolved_file_id}')"]
        code_issues = _find_code_issues(code, required_lines)
        if code_issues:
            repair_result = _run_agent(
                data_analyst,
                (
                    f"คำถาม: {prompt}\n"
                    f"file_id: '{resolved_file_id}'\n"
                    f"Schema:\n{schema_result}\n\n"
                    f"โค้ดปัจจุบัน:\n```python\n{code}\n```\n\n"
                    f"Contract violations:\n{chr(10).join(f'- {i}' for i in code_issues)}\n\n"
                    "แก้โค้ดให้ผ่านกฎ:\n"
                    f"1. ต้องมีบรรทัด df = load_csv('{resolved_file_id}')\n"
                    "2. ห้าม import/use Minio\n"
                    "3. ห้ามใช้ pd.read_csv\n"
                    "4. ห้าม redefine helpers\n"
                    "Wrap code in ```python ... ```"
                ),
                "Repaired Python code that passes contract checks",
                step="data_analyst_contract_repair", session_id=session_id,
            )
            repaired_code = _extract_code(repair_result)
            repaired_issues = _find_code_issues(repaired_code, required_lines)
            if not repaired_issues:
                code = repaired_code
                code_result = repair_result
            else:
                exec_output = f"[ข้ามการรัน — โค้ดยังผิดกติกา] issues: {', '.join(repaired_issues)}"
                code = ""

        if code:
            code = re.sub(r"load_csv\(['\"][^'\"]*['\"]\)", f"load_csv('{resolved_file_id}')", code)
            exec_output = exec_python(code)

        if code and _is_exec_error(exec_output):
            retry_result = _run_agent(
                data_analyst,
                (
                    f"คำถาม: {prompt}\n"
                    f"file_id: '{resolved_file_id}'\n"
                    f"Schema:\n{schema_result}\n\n"
                    f"โค้ดเดิมที่ error:\n```python\n{code}\n```\n"
                    f"Error:\n{exec_output}\n\n"
                    "แก้ไขโค้ดให้รันได้:\n"
                    f"1. บรรทัดแรก: df = load_csv('{resolved_file_id}')\n"
                    "2. ห้าม redefine load_csv\n"
                    "3. ตรวจสอบชื่อ column ให้ตรงกับ schema\n"
                    "Wrap code in ```python ... ```"
                ),
                "Fixed Python code that runs without errors",
                step="data_analyst_retry", session_id=session_id,
            )
            retry_code = _extract_code(retry_result)
            retry_code = re.sub(r"load_csv\(['\"][^'\"]*['\"]\)", f"load_csv('{resolved_file_id}')", retry_code)
            retry_output = exec_python(retry_code)
            if not _is_exec_error(retry_output) or len(retry_output) > len(exec_output):
                code = retry_code
                exec_output = retry_output

    put({"type": "agent_done", "step": "executor", "agentName": "Python Executor",
         "code": code, "result": exec_output})
    agent_steps.append({"step": "executor", "agentName": "Python Executor",
                        "result": exec_output, "code": code})

    # ── STEP 5: Report Writer ──────────────────────────────────────────────────
    put({"type": "agent_start", "step": "report_writer", "agentName": "Report Writer"})
    writer = Agent(
        role="Report Writer — Thai Public Health Analyst",
        goal=(
            "เขียนรายงานสรุปผู้บริหารภาษาไทยที่ครบถ้วน มีโครงสร้างชัดเจน "
            "จากผลการวิเคราะห์ข้อมูลจริง"
        ),
        backstory=join_prompt(
            "คุณเป็นนักวิเคราะห์นโยบายสาธารณสุขอาวุโสที่เชี่ยวชาญการเขียนรายงาน "
            "คุณเขียนรายงานเชิงวิชาการที่อ่านง่าย มีตาราง markdown ข้อมูลจริง "
            "และข้อเสนอแนะเชิงนโยบายที่ปฏิบัติได้จริง "
            "คุณใช้เฉพาะข้อมูลจาก Execution Result — ห้ามสร้างข้อมูลสมมติ",
            ANALYST_CORE_POLICY,
        ),
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    report = _run_agent(
        writer,
        (
            f"หัวข้อรายงาน: {prompt}\n"
            f"Dataset: {file_result}\n\n"
            f"ผลการวิเคราะห์ข้อมูล (Execution Result):\n{exec_output}\n\n"
            "==== กฎเหล็ก — ห้ามละเมิด ====\n"
            "1. ใช้เฉพาะข้อมูลจาก Execution Result ด้านบน\n"
            "2. ห้ามสร้างชื่อจังหวัดสมมติหรือตัวเลขที่ไม่มีในผลลัพธ์\n"
            "3. ถ้า Execution มี error → ระบุข้อจำกัดในรายงาน\n\n"
            "==== โครงสร้างรายงาน (ต้องครบทุก section) ====\n"
            "## บทสรุปผู้บริหาร\n"
            "(3-4 ประโยค สรุปประเด็นสำคัญที่สุด)\n\n"
            "## สถานการณ์ปัจจุบัน\n"
            "(อธิบายสภาพปัจจุบันจากข้อมูล 2-3 ย่อหน้า)\n\n"
            "## ตารางสถิติสำคัญ\n"
            "(ตาราง markdown จากข้อมูลจริง — ต้องมีตาราง)\n\n"
            "## แนวโน้มและข้อสังเกต\n"
            "(bullet points แนวโน้ม ความแตกต่าง จุดที่ควรสนใจ)\n\n"
            "## ข้อเสนอแนะ\n"
            "(มาตรการเชิงนโยบายและปฏิบัติที่เฉพาะเจาะจง เป็น bullet list)\n\n"
            f"{INSIGHT_RESPONSE_BLUEPRINT}\n\n"
            f"{MISSING_DATA_POLICY}"
        ),
        "รายงาน Markdown ภาษาไทยที่ครบถ้วนทุก section พร้อมตาราง markdown",
        step="report_writer", session_id=session_id,
    )
    put({"type": "agent_done", "step": "report_writer", "agentName": "Report Writer",
         "result": report})
    agent_steps.append({"step": "report_writer", "agentName": "Report Writer",
                        "result": report})

    # ── FINAL EVENT ────────────────────────────────────────────────────────────
    put({
        "type":       "final",
        "message":    report,
        "agentSteps": agent_steps,
    })
