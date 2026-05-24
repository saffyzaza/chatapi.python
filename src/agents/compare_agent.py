"""Compare Agent — finds 2 CSV datasets and generates a comparison analysis pipeline.

Flow:
  [Step 1] File Finder A   — ค้นหา dataset แรก
  [Step 2] File Finder B   — ค้นหา dataset ที่สอง
  [Step 3] Schema Analyst  — อ่าน schema ทั้งสองไฟล์
  [Step 4] Code Generator  — สร้าง pandas code เพื่อเปรียบเทียบ (join, diff, statistics)
  [Step 5] Python Executor — รันโค้ด
  [Step 6] Insight Analyst — สรุป insight ภาษาไทย

SSE events → queue:
  {"type": "agent_start", "step": "file_finder",   "agentName": "File Finder A"}
  {"type": "agent_done",  "step": "file_finder",   "result": "..."}
  {"type": "agent_start", "step": "file_finder_b", "agentName": "File Finder B"}
  {"type": "agent_done",  "step": "file_finder_b", "result": "..."}
  {"type": "agent_start", "step": "schema",         "agentName": "Schema Analyst"}
  {"type": "agent_done",  "step": "schema",         "result": "..."}
  {"type": "agent_start", "step": "code_gen",       "agentName": "Code Generator"}
  {"type": "agent_done",  "step": "code_gen",       "result": "..."}
  {"type": "agent_start", "step": "executor",       "agentName": "Python Executor"}
  {"type": "agent_done",  "step": "executor",       "result": "..."}
  {"type": "agent_start", "step": "insight",        "agentName": "Insight Analyst"}
  {"type": "agent_done",  "step": "insight",        "result": "..."}
  {"type": "final",       "message": "...", "agentSteps": [...]}
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
    _sanitize_generated_code,
    _age_scope_repair_hints,
    _strip_csv_extension_mentions,
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


def run_compare_pipeline(
    prompt: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    session_id: str = "",
) -> None:
    """Run a comparison pipeline between two CSV datasets.

    Emits SSE events via queue/loop.
    """
    llm = _get_llm()

    def put(ev: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(queue.put(ev), loop)

    agent_steps: list[dict] = []

    # ── STEP 1: File Finder A ──────────────────────────────────────────────────
    put({"type": "agent_start", "step": "file_finder", "agentName": "File Finder A"})
    finder_a = Agent(
        role="File Finder A — ค้นหา dataset แรก",
        goal="ค้นหาไฟล์ CSV ชุดแรกที่เกี่ยวข้องกับคำถามและคืนค่า file ID ที่ถูกต้อง",
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
    file_result_a = _run_agent(
        finder_a,
        (
            f"คำถาม: {prompt}\n\n"
            "ขั้นตอน:\n"
            "1. เรียก list_csv_files(prefix='') เพื่อดูไฟล์ทั้งหมด\n"
            "2. เลือกไฟล์ชุดแรกที่ตรงกับคำถามมากที่สุด (dataset หลัก/ปีแรก/กลุ่มแรก)\n"
            "3. ตอบเฉพาะ 1 บรรทัด ในรูปแบบ: [ID:xxxxxx] filename.csv\n"
            "   โดย [ID:xxxxxx] คือ ID จริงที่ได้จาก tool (ห้ามเปลี่ยน)"
        ),
        "Selected file A: exactly one line in format [ID:xxxxxx] filename.csv",
        step="file_finder", session_id=session_id,
    )
    if _is_agent_error(file_result_a):
        file_result_a = fallback_find_file(prompt, "")
    put({"type": "agent_done", "step": "file_finder", "agentName": "File Finder A",
         "result": file_result_a})
    agent_steps.append({"step": "file_finder", "agentName": "File Finder A",
                        "result": file_result_a})

    resolved_id_a = resolve_file_id(file_result_a)
    if not resolved_id_a:
        fallback = list_csv_files_impl("")
        if fallback and not fallback.startswith("No") and not fallback.startswith("Error"):
            first_line = fallback.split("\n")[0]
            resolved_id_a = resolve_file_id(first_line)
            if resolved_id_a:
                file_result_a = first_line

    # ── STEP 2: File Finder B ──────────────────────────────────────────────────
    put({"type": "agent_start", "step": "file_finder_b", "agentName": "File Finder B"})
    finder_b = Agent(
        role="File Finder B — ค้นหา dataset ที่สอง",
        goal="ค้นหาไฟล์ CSV ชุดที่สองสำหรับการเปรียบเทียบและคืนค่า file ID ที่ถูกต้อง",
        backstory=(
            "คุณเป็นผู้เชี่ยวชาญการค้นหา dataset สาธารณสุข "
            "คุณต้องหาไฟล์ที่แตกต่างจากชุดแรกเพื่อนำมาเปรียบเทียบ "
            "เช่น ต่างปี ต่างพื้นที่ ต่างกลุ่มโรค หรือต่าง indicator "
            "คุณต้องใช้ tool list_csv_files เพื่อดูรายการไฟล์ "
            "แล้วคืนค่า ENTIRE line รวมถึง [ID:...] ที่แน่ชัด"
        ),
        tools=[list_csv_files],
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    file_result_b = _run_agent(
        finder_b,
        (
            f"คำถาม: {prompt}\n"
            f"ไฟล์ชุดแรกที่เลือกแล้ว: {file_result_a}\n\n"
            "ขั้นตอน:\n"
            "1. เรียก list_csv_files(prefix='') เพื่อดูไฟล์ทั้งหมด\n"
            "2. เลือกไฟล์ชุดที่สองสำหรับเปรียบเทียบ (ต้องไม่ใช่ไฟล์เดิม)\n"
            "   เช่น ไฟล์ปีอื่น พื้นที่อื่น หรือ indicator อื่นที่เกี่ยวข้องกับคำถาม\n"
            "3. ตอบเฉพาะ 1 บรรทัด ในรูปแบบ: [ID:xxxxxx] filename.csv\n"
            "   โดย [ID:xxxxxx] คือ ID จริงที่ได้จาก tool (ห้ามเปลี่ยน)"
        ),
        "Selected file B: exactly one line in format [ID:xxxxxx] filename.csv",
        step="file_finder_b", session_id=session_id,
    )
    if _is_agent_error(file_result_b):
        file_result_b = fallback_find_file(prompt, "")
    put({"type": "agent_done", "step": "file_finder_b", "agentName": "File Finder B",
         "result": file_result_b})
    agent_steps.append({"step": "file_finder_b", "agentName": "File Finder B",
                        "result": file_result_b})

    resolved_id_b = resolve_file_id(file_result_b)

    if not resolved_id_a or not resolved_id_b:
        available = list_csv_files_impl("")
        samples = []
        if available and not available.startswith("No") and not available.startswith("Error"):
            samples = [ln for ln in available.split("\n") if ln.strip()][:5]
        sample_text = "\n".join(f"- {ln}" for ln in samples) if samples else "- ไม่พบไฟล์ตัวอย่าง"
        message = (
            "ยังเลือกไฟล์เปรียบเทียบให้ครบ 2 ชุดไม่ได้ จึงยังวิเคราะห์ต่อไม่ได้อย่างมั่นใจ\n\n"
            "ไฟล์ตัวอย่างที่พบ:\n"
            f"{sample_text}\n\n"
            "กรุณาระบุหัวข้อให้ชัดขึ้น เช่น ปี/จังหวัด/ตัวชี้วัด ที่ต้องการเปรียบเทียบ"
        )
        put({"type": "final", "message": message, "agentSteps": agent_steps})
        return

    # ── STEP 3: Schema Analyst ─────────────────────────────────────────────────
    put({"type": "agent_start", "step": "schema", "agentName": "Schema Analyst"})

    schema_a = ""
    schema_b = ""
    if resolved_id_a:
        schema_a = read_csv_schema_impl(resolved_id_a)
    if resolved_id_b:
        schema_b = read_csv_schema_impl(resolved_id_b)

    schema_result = (
        f"=== Schema ของไฟล์ A: {file_result_a} ===\n{schema_a}\n\n"
        f"=== Schema ของไฟล์ B: {file_result_b} ===\n{schema_b}"
    )
    put({"type": "agent_done", "step": "schema", "agentName": "Schema Analyst",
         "result": schema_result})
    agent_steps.append({"step": "schema", "agentName": "Schema Analyst",
                        "result": schema_result})

    # ── STEP 4: Code Generator ─────────────────────────────────────────────────
    put({"type": "agent_start", "step": "code_gen", "agentName": "Code Generator"})
    generator = Agent(
        role="Python Code Generator — Data Comparison Specialist",
        goal="สร้าง Python/Pandas code ที่รันได้ทันที เพื่อเปรียบเทียบข้อมูลระหว่างสองชุดและ print ผลลัพธ์ชัดเจน",
        backstory=join_prompt(
            "คุณเป็น Python/Pandas expert ที่เชี่ยวชาญการเปรียบเทียบข้อมูล "
            "คุณสร้างโค้ดที่วิเคราะห์ความแตกต่าง แนวโน้ม และสถิติเปรียบเทียบ "
            "โค้ดต้องรันได้ทันทีและ output ต้องพร้อมให้ AI วิเคราะห์ต่อได้โดยไม่ต้องเดา",
            CODE_GENERATOR_CORE_POLICY,
        ),
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    code_result = _run_agent(
        generator,
        (
            f"คำถาม: {prompt}\n"
            f"file_id A (ใช้ค่านี้เท่านั้น): '{resolved_id_a}'\n"
            f"file_id B (ใช้ค่านี้เท่านั้น): '{resolved_id_b}'\n\n"
            f"Schema:\n{schema_result}\n\n"
            "==== กฎบังคับ (ห้ามละเมิด) ====\n"
            f"1. บรรทัดแรก: df_a = load_csv('{resolved_id_a}')\n"
            f"2. บรรทัดที่สอง: df_b = load_csv('{resolved_id_b}')\n"
            "3. ห้าม redefine load_csv / import minio / ใช้ pd.read_csv()\n"
            "4. pd.set_option('display.max_rows', 100) ก่อน print\n\n"
            "==== การวิเคราะห์เปรียบเทียบ ====\n"
            "5. เปรียบเทียบสถิติพื้นฐาน (mean, max, min, sum) ของทั้งสองชุด\n"
            "6. หา common columns และ join/merge ถ้าเป็นไปได้\n"
            "7. แสดง diff หรือ % change ระหว่างสองชุด\n"
            "7.1 ถ้าคำถามระบุปี/พื้นที่/ช่วงอายุ ต้อง filter ให้ตรงตามที่ถามก่อนคำนวณ\n"
            "7.2 ต้อง print section '=== SCOPE CHECK ===' ระบุช่วงที่ถามและช่วงที่ใช้จริง\n"
            "8. print หัวข้อก่อนทุก section เช่น print('=== เปรียบเทียบสถิติ ===')\n"
            "9. ใช้ print(df.to_string(index=False)) เพื่อแสดงครบ\n"
            "Wrap code in ```python ... ```\n\n"
            f"{CODE_GENERATOR_CORE_POLICY}"
        ),
        "Working Python code that compares two datasets with clear labeled output",
        step="code_gen", session_id=session_id,
    )
    put({"type": "agent_done", "step": "code_gen", "agentName": "Code Generator",
         "result": code_result})
    agent_steps.append({"step": "code_gen", "agentName": "Code Generator",
                        "result": code_result})

    # ── STEP 5: Python Executor ────────────────────────────────────────────────
    put({"type": "agent_start", "step": "executor", "agentName": "Python Executor"})
    code = _extract_code(code_result)

    if _is_agent_error(code):
        exec_output = f"[ข้ามการรัน — code generation ล้มเหลว]\n{code_result}"
        code = ""
    else:
        required_lines = [
            f"df_a = load_csv('{resolved_id_a}')",
            f"df_b = load_csv('{resolved_id_b}')",
        ]
        sanitized_code = _sanitize_generated_code(code, required_lines, prompt)
        code_issues = _find_code_issues(sanitized_code, required_lines, prompt)
        if not code_issues:
            code = sanitized_code

        if code_issues:
            age_scope_hints = _age_scope_repair_hints(prompt, code_issues)
            repair_result = _run_agent(
                generator,
                (
                    f"คำถาม: {prompt}\n"
                    f"file_id A: '{resolved_id_a}'\n"
                    f"file_id B: '{resolved_id_b}'\n"
                    f"Schema:\n{schema_result}\n\n"
                    f"โค้ดปัจจุบัน:\n```python\n{code}\n```\n\n"
                    f"Contract violations:\n{chr(10).join(f'- {i}' for i in code_issues)}\n\n"
                    f"{age_scope_hints}\n"
                    "แก้โค้ดให้ผ่านกฎ:\n"
                    f"1. ต้องมีบรรทัด df_a = load_csv('{resolved_id_a}')\n"
                    f"2. ต้องมีบรรทัด df_b = load_csv('{resolved_id_b}')\n"
                    "3. ห้าม import/use Minio\n"
                    "4. ห้ามใช้ pd.read_csv\n"
                    "5. ห้าม redefine helpers\n"
                    "6. ต้องมี '=== SCOPE CHECK ===' และยืนยันช่วงปี/พื้นที่/ช่วงอายุที่ถาม\n"
                    "7. ถ้าไม่มีคอลัมน์อายุตรงเป้า ให้คำนวณ estimate และติดป้ายช่วงอายุเป้าหมาย\n"
                    "Wrap code in ```python ... ```"
                ),
                "Repaired Python code that passes contract checks",
                step="code_contract_repair", session_id=session_id,
            )
            repaired_code = _sanitize_generated_code(_extract_code(repair_result), required_lines, prompt)
            repaired_issues = _find_code_issues(repaired_code, required_lines, prompt)
            if not repaired_issues:
                code = repaired_code
                code_result = repair_result
            else:
                exec_output = f"[ข้ามการรัน — โค้ดยังผิดกติกา] issues: {', '.join(repaired_issues)}"
                code = ""

        if code:
            code = re.sub(
                r"load_csv\(['\"][^'\"]*['\"]\)",
                f"load_csv('{resolved_id_a}')",
                code,
                count=1,
            )
            code = re.sub(
                r"load_csv\(['\"][^'\"]*['\"]\)",
                f"load_csv('{resolved_id_b}')",
                code,
                count=1,
            )
            exec_output = exec_python(code)

        if code and _is_exec_error(exec_output):
            # Retry with error context
            retry_result = _run_agent(
                generator,
                (
                    f"คำถาม: {prompt}\n"
                    f"file_id A: '{resolved_id_a}'\n"
                    f"file_id B: '{resolved_id_b}'\n"
                    f"Schema:\n{schema_result}\n\n"
                    f"โค้ดเดิมที่ error:\n```python\n{code}\n```\n"
                    f"Error:\n{exec_output}\n\n"
                    "แก้ไขโค้ดให้รันได้:\n"
                    f"1. บรรทัดแรก: df_a = load_csv('{resolved_id_a}')\n"
                    f"2. บรรทัดที่สอง: df_b = load_csv('{resolved_id_b}')\n"
                    "3. ห้าม redefine load_csv\n"
                    "4. ตรวจสอบชื่อ column ให้ตรงกับ schema\n"
                    "Wrap code in ```python ... ```"
                ),
                "Fixed Python code that runs without errors",
                step="code_gen_retry", session_id=session_id,
            )
            retry_code = _sanitize_generated_code(_extract_code(retry_result), required_lines, prompt)
            retry_output = exec_python(retry_code)
            if not _is_exec_error(retry_output) or len(retry_output) > len(exec_output):
                code = retry_code
                exec_output = retry_output

    put({"type": "agent_done", "step": "executor", "agentName": "Python Executor",
         "code": code, "result": exec_output})
    agent_steps.append({"step": "executor", "agentName": "Python Executor",
                        "result": exec_output, "code": code})

    # ── STEP 6: Insight Analyst ────────────────────────────────────────────────
    put({"type": "agent_start", "step": "insight", "agentName": "Insight Analyst"})
    analyst = Agent(
        role="Insight Analyst — Data Comparison Expert",
        goal="วิเคราะห์ผลลัพธ์การเปรียบเทียบและเขียนรายงานภาษาไทยที่ชัดเจนและเป็นประโยชน์",
        backstory=join_prompt(
            "คุณเป็นนักวิเคราะห์ข้อมูลสาธารณสุขที่รายงานผลจากข้อมูลจริงเท่านั้น "
            "คุณไม่สร้างข้อมูลหรือตัวเลขที่ไม่มีในผลลัพธ์ "
            "คุณเชี่ยวชาญการตีความความแตกต่างและแนวโน้มระหว่างสองชุดข้อมูล",
            ANALYST_CORE_POLICY,
        ),
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    insight = _run_agent(
        analyst,
        (
            f"คำถาม: {prompt}\n\n"
            f"ไฟล์ที่เปรียบเทียบ:\n"
            f"  A: {file_result_a}\n"
            f"  B: {file_result_b}\n\n"
            f"ผลการรันโค้ด (Execution Result):\n{exec_output}\n\n"
            "==== กฎเหล็ก — ห้ามละเมิด ====\n"
            "1. ใช้เฉพาะข้อมูลจาก Execution Result ด้านบน\n"
            "2. ห้ามสร้างชื่อจังหวัดสมมติหรือตัวเลขที่ไม่มีในผลลัพธ์\n"
            "3. ถ้า Execution มี error → อธิบาย error + สรุปสิ่งที่ทราบได้\n"
            "4. ต้องยืนยันว่าผลลัพธ์ครอบคลุมปี/พื้นที่/ช่วงอายุที่ผู้ใช้ถามครบหรือไม่\n\n"
            "==== โครงสร้างรายงาน ====\n"
            "## สรุปภาพรวมการเปรียบเทียบ\n"
            "อธิบายว่าเปรียบเทียบอะไรกับอะไร และผลโดยรวม (2-3 ประโยค)\n\n"
            "## ตารางเปรียบเทียบ\n"
            "ตาราง markdown แสดงค่าสำคัญของทั้งสองชุดข้อมูล\n\n"
            "## ความแตกต่างที่สำคัญ\n"
            "- bullet points อธิบายความแตกต่าง % change แนวโน้ม\n\n"
            "## ข้อเสนอแนะ\n"
            "มาตรการหรือแนวทางที่เหมาะสมจากผลการเปรียบเทียบ\n\n"
            f"{INSIGHT_RESPONSE_BLUEPRINT}\n\n"
            f"{MISSING_DATA_POLICY}"
        ),
        "รายงาน insight ภาษาไทยที่มีตาราง markdown เปรียบเทียบสองชุดข้อมูลจากข้อมูลจริง",
        step="insight", session_id=session_id,
    )
    insight = _strip_csv_extension_mentions(insight)
    put({"type": "agent_done", "step": "insight", "agentName": "Insight Analyst",
         "result": insight})
    agent_steps.append({"step": "insight", "agentName": "Insight Analyst",
                        "result": insight})

    # ── FINAL EVENT ────────────────────────────────────────────────────────────
    put({
        "type":       "final",
        "message":    insight,
        "agentSteps": agent_steps,
    })
