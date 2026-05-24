"""Database Agent — loads an attached file from MinIO and answers user's question about it.

Flow:
  [Step 1] Schema Analyst  — อ่าน schema ของไฟล์ที่แนบมา
  [Step 2] Code Generator  — สร้าง pandas code เพื่อตอบคำถาม
  [Step 3] Python Executor — รันโค้ด
  [Step 4] Insight Analyst — อธิบายผลลัพธ์ภาษาไทย

SSE events → queue:
  {"type": "agent_start", "step": "schema",   "agentName": "Schema Analyst"}
  {"type": "agent_done",  "step": "schema",   "result": "..."}
  {"type": "agent_start", "step": "code_gen", "agentName": "Code Generator"}
  {"type": "agent_done",  "step": "code_gen", "result": "..."}
  {"type": "agent_start", "step": "executor", "agentName": "Python Executor"}
  {"type": "agent_done",  "step": "executor", "result": "..."}
  {"type": "agent_start", "step": "insight",  "agentName": "Insight Analyst"}
  {"type": "agent_done",  "step": "insight",  "result": "..."}
  {"type": "final",       "message": "...", "agentSteps": [...]}
"""
import asyncio
import base64
import os
import re
from typing import Any

import litellm
from crewai import Agent, LLM

from src.tools.minio import (
    read_csv_schema_impl,
    read_file_bytes_impl,
    read_file_text_impl,
    read_file_extension_impl,
    exec_python,
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

_TEXT_TYPES  = {"pdf", "doc", "docx", "txt", "md"}
_DATA_TYPES  = {"csv", "xlsx", "xls"}
_IMAGE_TYPES = {"jpg", "jpeg", "png", "gif", "webp", "bmp"}
_IMAGE_MIME  = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "png": "image/png",  "gif":  "image/gif",
    "webp":"image/webp", "bmp":  "image/bmp",
}


def run_database_pipeline(
    prompt: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    session_id: str = "",
    attached_files: list[dict] = [],
) -> None:
    """Run the database analysis pipeline for user-attached files.

    Args:
        prompt: คำถามของผู้ใช้
        queue: asyncio.Queue สำหรับ SSE events
        loop: asyncio event loop
        session_id: session identifier
        attached_files: list of {"id": str, "name": str} — use id as minio object key
    """
    llm = _get_llm()

    def put(ev: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(queue.put(ev), loop)

    agent_steps: list[dict] = []

    # Validate attached files
    if not attached_files:
        put({
            "type": "final",
            "message": "ไม่พบไฟล์ที่แนบมา กรุณาแนบไฟล์ CSV ก่อนถามคำถาม",
            "agentSteps": [],
        })
        return

    # Use the first attached file (primary dataset)
    primary_file = attached_files[0]
    file_id = primary_file.get("id", "")
    file_name = primary_file.get("name", file_id)

    if not file_id:
        put({
            "type": "final",
            "message": "ไม่สามารถระบุ ID ของไฟล์ที่แนบมาได้ กรุณาลองอีกครั้ง",
            "agentSteps": [],
        })
        return

    # ── Detect file type ─────────────────────────────────────────────────────
    ext = read_file_extension_impl(file_id)
    if not ext:
        dot = file_name.rfind(".")
        ext = file_name[dot + 1:].lower() if dot >= 0 else "csv"
    is_image_file = ext in _IMAGE_TYPES
    is_text_file  = ext in _TEXT_TYPES

    # ── IMAGE: Gemini Vision Q&A ──────────────────────────────────────────────
    if is_image_file:
        put({"type": "agent_start", "step": "schema", "agentName": "Image Analyst"})
        try:
            img_bytes = read_file_bytes_impl(file_id)
            img_b64   = base64.b64encode(img_bytes).decode()
            mime      = _IMAGE_MIME.get(ext, "image/jpeg")
            put({"type": "agent_done", "step": "schema", "agentName": "Image Analyst",
                 "result": f"โหลดรูปภาพสำเร็จ ({len(img_bytes):,} bytes)"})
            agent_steps.append({"step": "schema", "agentName": "Image Analyst",
                                "result": f"โหลดรูปภาพ {file_name}"})

            put({"type": "agent_start", "step": "insight", "agentName": "Vision Agent"})
            resp = litellm.completion(
                model="gemini/gemini-2.0-flash",
                api_key=os.getenv("GEMINI_API_KEY"),
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": f"ตอบเป็นภาษาไทย\n\nคำถาม: {prompt}"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{mime};base64,{img_b64}"
                        }},
                    ],
                }],
                temperature=0.3,
            )
            insight = resp.choices[0].message.content or "ไม่สามารถวิเคราะห์รูปภาพได้"
        except Exception as exc:
            log_agent_error(str(exc), agent_name="Vision Agent",
                            step="insight", domain="database", prompt=file_id)
            insight = f"เกิดข้อผิดพลาดในการวิเคราะห์รูปภาพ: {exc}"

        insight = _strip_csv_extension_mentions(insight)

        put({"type": "agent_done", "step": "insight", "agentName": "Vision Agent",
             "result": insight[:400]})
        agent_steps.append({"step": "insight", "agentName": "Vision Agent",
                            "result": insight[:400]})
        put({"type": "final", "message": insight, "agentSteps": agent_steps})
        return

    # ── STEP 1: File Reader / Schema Analyst ───────────────────────────────────
    step_name = "File Reader" if is_text_file else "Schema Analyst"
    put({"type": "agent_start", "step": "schema", "agentName": step_name})

    if is_text_file:
        schema_result = read_file_text_impl(file_id, ext)
    elif ext in ("xlsx", "xls"):
        schema_result = read_file_text_impl(file_id, ext)
    else:
        schema_result = read_csv_schema_impl(file_id)

    if not schema_result or schema_result.startswith("Error"):
        schema_result = f"[ไม่สามารถอ่านไฟล์ '{file_name}' (ID: {file_id}, type: {ext})]"

    put({"type": "agent_done", "step": "schema", "agentName": step_name,
         "result": schema_result[:500] + ("…" if len(schema_result) > 500 else "")})
    agent_steps.append({"step": "schema", "agentName": step_name,
                        "result": schema_result[:500]})

    # If there are additional attached files, read their schemas too
    extra_schemas = ""
    if len(attached_files) > 1:
        for extra_file in attached_files[1:]:
            extra_id = extra_file.get("id", "")
            extra_name = extra_file.get("name", extra_id)
            if extra_id:
                extra_schema = read_csv_schema_impl(extra_id)
                extra_schemas += f"\n\n=== Schema ของไฟล์ '{extra_name}' (ID: {extra_id}) ===\n{extra_schema}"

    full_schema = (
        f"=== Schema ของไฟล์หลัก '{file_name}' (ID: {file_id}) ===\n{schema_result}"
        + extra_schemas
    )

    if is_text_file or ext in ("xlsx", "xls"):
        # ── PDF / DOCX / XLSX: Direct Q&A (no code execution) ─────────────────
        put({"type": "agent_start", "step": "insight", "agentName": "Document Analyst"})
        analyst = Agent(
            role="Document Analyst — Thai Language Expert",
            goal="ตอบคำถามจากเนื้อหาไฟล์ที่ผู้ใช้แนบมาอย่างตรงประเด็น ชัดเจน ภาษาไทย",
            backstory=join_prompt(
                "คุณเป็นผู้เชี่ยวชาญการวิเคราะห์เอกสาร ตอบโดยใช้ข้อมูลจริงจากไฟล์เท่านั้น "
                "ไม่สร้างข้อมูลหรือตัวเลขที่ไม่มีในเนื้อหา",
                ANALYST_CORE_POLICY,
            ),
            llm=llm, verbose=False, max_iter=5,
        )
        insight = _run_agent(
            analyst,
            (
                f"คำถาม: {prompt}\n"
                f"ไฟล์: '{file_name}' (ประเภท: {ext.upper()})\n\n"
                f"เนื้อหาไฟล์:\n{schema_result[:8000]}\n\n"
                "กฎ: ใช้เฉพาะข้อมูลจากเนื้อหาด้านบน ห้ามสร้างข้อมูลเพิ่มเติม\n\n"
                "โครงสร้างคำตอบ:\n"
                "## คำตอบ\n"
                "(ตอบตรงๆ 2-3 ประโยค)\n\n"
                "## รายละเอียด\n"
                "(อ้างอิงข้อมูลจากไฟล์ที่เกี่ยวข้อง)\n\n"
                "## สรุป\n"
                "(bullet points ประเด็นสำคัญ)\n\n"
                f"{INSIGHT_RESPONSE_BLUEPRINT}\n\n"
                f"{MISSING_DATA_POLICY}"
            ),
            "คำตอบภาษาไทยที่ตรงประเด็นจากเนื้อหาไฟล์",
            step="insight", session_id=session_id,
        )
        insight = _strip_csv_extension_mentions(insight)
        put({"type": "agent_done", "step": "insight", "agentName": "Document Analyst",
             "result": insight})
        agent_steps.append({"step": "insight", "agentName": "Document Analyst",
                            "result": insight})

    else:
        # ── CSV: Code Gen → Execute → Insight ─────────────────────────────────
        put({"type": "agent_start", "step": "code_gen", "agentName": "Code Generator"})
        generator = Agent(
            role="Python Code Generator — User File Analyst",
            goal="สร้าง Python/Pandas code ที่รันได้ทันที เพื่อวิเคราะห์ไฟล์ CSV ตามคำถาม",
            backstory=join_prompt(
                "คุณเป็น Python/Pandas expert อ่าน schema แล้วสร้างโค้ดที่ตอบคำถามผู้ใช้ "
                "output ชัดเจน มี label ครบถ้วน",
                CODE_GENERATOR_CORE_POLICY,
            ),
            llm=llm, verbose=False, max_iter=5,
        )

        load_instructions = f"1. บรรทัดแรก: df = load_csv('{file_id}')"
        if len(attached_files) > 1:
            for i, extra_file in enumerate(attached_files[1:], 2):
                extra_id = extra_file.get("id", "")
                if extra_id:
                    load_instructions += f"\n   บรรทัดที่ {i}: df{i} = load_csv('{extra_id}')"

        code_result = _run_agent(
            generator,
            (
                f"คำถาม: {prompt}\n"
                f"ไฟล์หลัก: '{file_name}' (file_id: '{file_id}')\n\n"
                f"Schema:\n{full_schema}\n\n"
                "==== กฎบังคับ ====\n"
                f"{load_instructions}\n"
                "- ห้าม redefine load_csv / import minio / ใช้ pd.read_csv()\n"
                "- pd.set_option('display.max_rows', 100) ก่อน print\n"
                "- print หัวข้อก่อนทุก section\n"
                "- ถ้าคำถามระบุปี/พื้นที่/ช่วงอายุ ต้อง filter ให้ตรงตามที่ถามก่อนคำนวณ\n"
                "- ต้อง print section '=== SCOPE CHECK ===' ระบุช่วงที่ถามและช่วงที่ใช้จริง\n"
                "- ถ้าไม่พบคอลัมน์ตรงกับช่วงอายุ/กลุ่มเป้าหมาย ให้คำนวณค่าประมาณจากคอลัมน์ใกล้เคียงและพิมพ์ section '=== ESTIMATION METHOD ==='\n"
                "Wrap code in ```python ... ```\n\n"
                f"{CODE_GENERATOR_CORE_POLICY}"
            ),
            "Working Python code that answers user's question",
            step="code_gen", session_id=session_id,
        )
        put({"type": "agent_done", "step": "code_gen", "agentName": "Code Generator",
             "result": code_result})
        agent_steps.append({"step": "code_gen", "agentName": "Code Generator",
                            "result": code_result})

        put({"type": "agent_start", "step": "executor", "agentName": "Python Executor"})
        code = _extract_code(code_result)
        if _is_agent_error(code):
            exec_output = f"[code generation ล้มเหลว]\n{code_result}"
            code = ""
        else:
            required_lines = [f"df = load_csv('{file_id}')"]
            for i, extra_file in enumerate(attached_files[1:], 2):
                extra_id = extra_file.get("id", "")
                if extra_id:
                    required_lines.append(f"df{i} = load_csv('{extra_id}')")

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
                        f"ไฟล์หลัก: '{file_id}'\n"
                        f"Schema:\n{full_schema}\n\n"
                        f"โค้ดปัจจุบัน:\n```python\n{code}\n```\n\n"
                        f"Contract violations:\n{chr(10).join(f'- {i}' for i in code_issues)}\n\n"
                        f"{age_scope_hints}\n"
                        "แก้โค้ดให้ผ่านกฎ:\n"
                        f"1. ต้องมีบรรทัดโหลดไฟล์ตามนี้:\n{chr(10).join(required_lines)}\n"
                        "2. ห้าม import/use Minio\n"
                        "3. ห้ามใช้ pd.read_csv\n"
                        "4. ห้าม redefine helper functions\n"
                        "5. ต้องมี '=== SCOPE CHECK ===' และยืนยันช่วงปี/พื้นที่/ช่วงอายุที่ถาม\n"
                        "6. ถ้าไม่มีคอลัมน์อายุตรงเป้า ให้คำนวณ estimate และติดป้ายช่วงอายุเป้าหมาย\n"
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
                code = re.sub(r"load_csv\(['\"][^'\"]*['\"]\)", f"load_csv('{file_id}')", code, count=1)
                exec_output = exec_python(code)
                if _is_exec_error(exec_output):
                    retry_result = _run_agent(
                        generator,
                        (
                            f"คำถาม: {prompt}\nfile_id: '{file_id}'\nSchema:\n{full_schema}\n\n"
                            f"โค้ดเดิม error:\n```python\n{code}\n```\nError:\n{exec_output}\n\n"
                            f"แก้ไข — บรรทัดแรก: df = load_csv('{file_id}') Wrap in ```python```"
                        ),
                        "Fixed Python code", step="code_gen_retry", session_id=session_id,
                    )
                    retry_code = _sanitize_generated_code(_extract_code(retry_result), required_lines, prompt)
                    retry_code = re.sub(r"load_csv\(['\"][^'\"]*['\"]\)", f"load_csv('{file_id}')", retry_code, count=1)
                    retry_output = exec_python(retry_code)
                    if not _is_exec_error(retry_output) or len(retry_output) > len(exec_output):
                        code, exec_output = retry_code, retry_output

        put({"type": "agent_done", "step": "executor", "agentName": "Python Executor",
             "code": code, "result": exec_output})
        agent_steps.append({"step": "executor", "agentName": "Python Executor",
                            "result": exec_output, "code": code})

        put({"type": "agent_start", "step": "insight", "agentName": "Insight Analyst"})
        analyst = Agent(
            role="Insight Analyst — CSV Data Expert",
            goal="วิเคราะห์ผลลัพธ์จาก CSV และเขียนคำตอบภาษาไทยที่ชัดเจน",
            backstory=join_prompt(
                "คุณรายงานผลจากข้อมูลจริงเท่านั้น ไม่สร้างตัวเลขที่ไม่มีในผลลัพธ์",
                ANALYST_CORE_POLICY,
            ),
            llm=llm, verbose=False, max_iter=5,
        )
        insight = _run_agent(
            analyst,
            (
                f"คำถาม: {prompt}\nไฟล์: '{file_name}'\n\n"
                f"ผลการรันโค้ด:\n{exec_output}\n\n"
                "กฎ: ใช้เฉพาะข้อมูลจาก Execution Result\n\n"
                "กฎเพิ่ม: ต้องยืนยันว่าผลลัพธ์ครอบคลุมช่วงปี/พื้นที่/ช่วงอายุที่ผู้ใช้ถามครบหรือไม่\n\n"
                "## คำตอบ\n(2-3 ประโยค)\n\n"
                "## ตารางข้อมูล\n(markdown จากผลลัพธ์จริง)\n\n"
                "## ข้อสังเกตเพิ่มเติม\n- bullet points\n\n"
                f"{INSIGHT_RESPONSE_BLUEPRINT}\n\n"
                f"{MISSING_DATA_POLICY}"
            ),
            "คำตอบภาษาไทยที่ตรงประเด็น พร้อมตาราง markdown",
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
