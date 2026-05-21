"""CSV analysis pipeline — 6-step agent pipeline for health domains d0, d2–d8."""
import asyncio
import os
import re
import time
from typing import Any

from crewai import Agent, Crew, LLM, Task

from src.domains import Domain
from src.history import append_history
from src.tools.minio import (
    list_csv_files,
    read_csv_schema,
    execute_python_code,
    list_csv_files_impl,
    resolve_file_id,
    fallback_find_file,
    read_csv_schema_impl,
    exec_python,
)

# ── LLM ──────────────────────────────────────────────────────────────────────

def _get_llm() -> LLM:
    return LLM(model="gemini/gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_agent(
    agent: Agent,
    description: str,
    expected: str,
    max_retries: int = 2,
    # Optional context for error logging
    step: str = "",
    domain: str = "",
    session_id: str = "",
) -> str:
    """Run a single-agent CrewAI task with retry on empty/None response.

    All failures are persisted to error_logs/ via log_agent_error().
    """
    from src.tools.error_logger import log_agent_error

    task = Task(description=description, expected_output=expected, agent=agent)
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    prompt_snippet = description[:250]

    for attempt in range(max_retries + 1):
        try:
            result = str(crew.kickoff()).strip()
            if result and result != "None":
                return result
            if attempt < max_retries:
                time.sleep(2)
        except Exception as exc:
            err_str = str(exc)
            if ("None or empty" in err_str or "empty" in err_str.lower()) and attempt < max_retries:
                time.sleep(3)
                continue
            error_msg = f"[Agent error: {exc}]"
            log_agent_error(
                error_message=error_msg,
                agent_name=agent.role,
                step=step,
                domain=domain,
                prompt=prompt_snippet,
                session_id=session_id,
                attempt=attempt,
            )
            return error_msg

    error_msg = f"[Agent error: empty response after {max_retries} retries]"
    log_agent_error(
        error_message=error_msg,
        agent_name=agent.role,
        step=step,
        domain=domain,
        prompt=prompt_snippet,
        session_id=session_id,
        attempt=max_retries,
    )
    return error_msg


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _is_agent_error(text: str) -> bool:
    """True when the text is an agent failure message, not real output."""
    t = (text or "").strip()
    return t.startswith("[Agent error:") or not t or t == "None"


def _is_auth_error(text: str) -> bool:
    """True when the failure is an API key / quota issue (non-retryable)."""
    t = (text or "").lower()
    return "403" in t or "permission_denied" in t or "api key" in t or "leaked" in t or "quota" in t


def _is_exec_error(output: str) -> bool:
    """True when exec_python returned a runtime/timeout error."""
    if not output:
        return False
    return (
        output.startswith("Error:")
        or "timed out" in output.lower()
        or ("STDERR" in output and ("error" in output.lower() or "Error" in output))
    )


def _log_exec_error(
    output: str,
    code: str = "",
    step: str = "executor",
    domain: str = "",
    session_id: str = "",
    attempt: int = 0,
) -> None:
    """Log Python executor errors (timeout / STDERR / crash) to error_logs/."""
    if not _is_exec_error(output):
        return
    from src.tools.error_logger import log_agent_error
    log_agent_error(
        error_message=output[:600],
        agent_name="Python Executor",
        step=step,
        domain=domain,
        prompt=f"[code]\n{(code or '')[:200]}",
        session_id=session_id,
        attempt=attempt,
    )


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    prompt: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    domain: Domain,
    history_context: str,
    history_section: str,
    session_id: str = "",
    reasoning: str = "",
) -> None:
    """Run the full CSV analysis pipeline for a given domain.

    Emits SSE events via queue/loop. Covers:
      d0 — general knowledge (no CSV)
      d1 — accident agent (PostgreSQL, handled by caller)
      d2–d8 — CSV pipeline: File Finder → Schema → Code Gen → Executor → Insight
    """
    llm = _get_llm()

    def put(ev: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(queue.put(ev), loop)

    # ── d0: General knowledge ─────────────────────────────────────────────────
    if domain.code == "d0":
        put({"type": "agent_start", "step": "insight", "agentName": "Insight Analyst Agent"})
        analyst = Agent(
            role="Insight Analyst Agent — General Health Advisor",
            goal="ตอบคำถามด้านสุขภาพและสาธารณสุขทั่วไปเป็นภาษาไทยอย่างถูกต้องและเป็นประโยชน์",
            backstory=domain.expertise,
            llm=llm,
            verbose=False,
            max_iter=5,
        )
        insight = _run_agent(
            analyst,
            (
                f"{history_section}"
                f"คำถาม: {prompt}\n\n"
                "ตอบคำถามนี้โดยใช้ความรู้ด้านสุขภาพและสาธารณสุขทั่วไป "
                "หากเป็น follow-up ให้ต่อเนื่องจากบทสนทนาก่อนหน้า "
                "อธิบายอย่างละเอียด ชัดเจน และเป็นประโยชน์ต่อผู้ใช้ ตอบเป็นภาษาไทย"
            ),
            "คำตอบที่ชัดเจนและเป็นประโยชน์เป็นภาษาไทย",
        )
        put({"type": "agent_done", "step": "insight", "agentName": "Insight Analyst Agent", "result": insight})
        if session_id:
            append_history(session_id, "ai", insight)
        put({
            "type": "final",
            "message": insight,
            "domain": {"code": domain.code, "nameTh": domain.name_th, "nameEn": domain.name_en},
            "agentSteps": [
                {"step": "router",    "agentName": "Router Agent",          "result": f"{domain.code} — {domain.name_th}"},
                {"step": "reasoning", "agentName": "Reasoning Narrator",    "result": reasoning},
                {"step": "insight",   "agentName": "Insight Analyst Agent", "result": insight},
            ],
        })
        return

    # ── d1: Accident Agent (PostgreSQL) ───────────────────────────────────────
    if domain.code == "d1":
        from src.agents.accident_chat_orchestrator import run_accident_chat

        put({"type": "agent_start", "step": "sql_agent", "agentName": "Accident SQL Agent"})
        acc_result = run_accident_chat(question=prompt)
        put({"type": "agent_done", "step": "sql_agent", "agentName": "Accident SQL Agent",
             "result": acc_result.raw_data or "(ดึงข้อมูลเสร็จ)"})

        put({"type": "agent_start", "step": "insight", "agentName": "Accident Answer Writer"})
        put({"type": "agent_done", "step": "insight", "agentName": "Accident Answer Writer",
             "result": acc_result.answer})

        if session_id:
            append_history(session_id, "ai", acc_result.answer)
        put({
            "type": "final",
            "message": acc_result.answer,
            "domain": {"code": domain.code, "nameTh": domain.name_th, "nameEn": domain.name_en},
            "agentSteps": [
                {"step": "router",    "agentName": "Router Agent",          "result": f"{domain.code} — {domain.name_th}"},
                {"step": "reasoning", "agentName": "Reasoning Narrator",    "result": reasoning},
                {"step": "sql_agent", "agentName": "Accident SQL Agent",    "result": acc_result.raw_data or ""},
                {"step": "insight",   "agentName": "Accident Answer Writer", "result": acc_result.answer},
            ],
        })
        return

    # ── d2–d8: CSV pipeline ───────────────────────────────────────────────────

    # STEP 2: File Finder
    put({"type": "agent_start", "step": "file_finder", "agentName": "File Finder Agent"})
    finder = Agent(
        role=f"File Finder Agent — {domain.name_en}",
        goal=f"ค้นหาไฟล์ CSV ที่เกี่ยวข้องกับ{domain.name_th} และคืนค่า file ID ที่ถูกต้อง",
        backstory=(
            f"{domain.expertise}\n"
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
            f"คำถาม: {prompt}\n"
            f"Domain: {domain.name_th}\n\n"
            f"ขั้นตอน:\n"
            f"1. เรียก list_csv_files(prefix='{domain.folder_prefix}') เพื่อดูไฟล์ใน domain\n"
            f"2. ถ้าไม่พบ ให้เรียก list_csv_files(prefix='') เพื่อดูทั้งหมด\n"
            f"3. เลือกไฟล์ที่ตรงกับคำถามมากที่สุด\n"
            f"4. ตอบเฉพาะ 1 บรรทัด ในรูปแบบ: [ID:xxxxxx] filename.csv\n"
            f"   โดย [ID:xxxxxx] คือ ID จริงที่ได้จาก tool (ห้ามเปลี่ยน)"
        ),
        "Selected file: exactly one line in format [ID:xxxxxx] filename.csv",
        step="file_finder", domain=domain.code, session_id=session_id,
    )
    if file_result.startswith("[Agent error:") or not file_result.strip():
        file_result = fallback_find_file(prompt, domain.folder_prefix)
    put({"type": "agent_done", "step": "file_finder", "agentName": "File Finder Agent", "result": file_result})

    resolved_file_id = resolve_file_id(file_result)
    if not resolved_file_id:
        fallback_listing = list_csv_files_impl(domain.folder_prefix or "")
        if fallback_listing and not fallback_listing.startswith("No") and not fallback_listing.startswith("Error"):
            first_line = fallback_listing.split("\n")[0]
            resolved_file_id = resolve_file_id(first_line)
            if resolved_file_id:
                file_result = first_line

    # STEP 3: Schema Analyst
    put({"type": "agent_start", "step": "schema", "agentName": "Schema Analyst Agent"})
    schema_result: str = ""
    if resolved_file_id:
        direct_schema = read_csv_schema_impl(resolved_file_id)
        try:
            import json
            if "columns" in json.loads(direct_schema):
                schema_result = direct_schema
        except Exception:
            pass

    if not schema_result:
        schema_agent = Agent(
            role=f"Schema Analyst Agent -- {domain.name_en}",
            goal=f"Analyze dataset schema using file_id={resolved_file_id}",
            backstory=(f"{domain.expertise}\nUse read_csv_schema with file_id='{resolved_file_id}' only."),
            tools=[read_csv_schema],
            llm=llm,
            verbose=False,
            max_iter=5,
        )
        schema_result = _run_agent(
            schema_agent,
            (
                f"File ID to use: '{resolved_file_id}'\n\n"
                f"Call: read_csv_schema(file_path='{resolved_file_id}')\n"
                "Summarize columns, data types, and sample rows. "
                f"Remember: file_id='{resolved_file_id}' must also be used in Python code."
            ),
            "Dataset schema with columns, dtypes, sample rows, and file_id",
            step="schema", domain=domain.code, session_id=session_id,
        )
        if schema_result.startswith("[Agent error:") and resolved_file_id:
            schema_result = read_csv_schema_impl(resolved_file_id)
    put({"type": "agent_done", "step": "schema", "agentName": "Schema Analyst Agent", "result": schema_result})

    # STEP 4: Python Code Generator
    put({"type": "agent_start", "step": "code_gen", "agentName": "Python Code Generator"})
    generator = Agent(
        role=f"Python Code Generator — {domain.name_en}",
        goal=(
            f"สร้าง Python/Pandas code ที่รันได้ทันที วิเคราะห์ข้อมูล {domain.name_th} "
            f"และ print ผลลัพธ์ชัดเจนพร้อมชื่อจังหวัด/พื้นที่จริง"
        ),
        backstory=(
            f"{domain.expertise}\n"
            "คุณเป็น Python/Pandas expert ที่เขียนโค้ดถูกต้อง clean และรันได้ทันที "
            "คุณใส่ใจเรื่อง output format: ชื่อจังหวัดต้องแสดงครบ ตัวเลขมี label ชัดเจน "
            "ผลลัพธ์ต้องพร้อมให้ AI วิเคราะห์ต่อได้โดยไม่ต้องเดา"
        ),
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    code_result = _run_agent(
        generator,
        (
            f"Question: {prompt}\n"
            f"file_id (ใช้ค่านี้เท่านั้น): '{resolved_file_id}'\n"
            f"Schema:\n{schema_result}\n\n"
            "==== กฎบังคับ (ห้ามละเมิด) ====\n"
            f"1. บรรทัดแรก: df = load_csv('{resolved_file_id}')\n"
            "2. ห้าม redefine load_csv / import minio / ใช้ pd.read_csv()\n"
            f"3. ใช้ file_id='{resolved_file_id}' เท่านั้น\n\n"
            "==== Output Format บังคับ ====\n"
            "4. บรรทัดที่ 2 ของโค้ด: pd.set_option('display.max_rows', 100)\n"
            "5. ก่อน print ทุก section ให้ print หัวข้อ เช่น print('=== สรุปรายจังหวัด ===')\n"
            "6. ชื่อจังหวัด/พื้นที่ต้องแสดงเป็น text ครบ — ห้าม print index ตัวเลขอย่างเดียว\n"
            "7. ถ้ามี column จังหวัด/อำเภอ ต้องใส่เป็น column แรกในทุกตาราง\n"
            "8. print(df.to_string(index=False)) แทน print(df) เพื่อแสดงครบ\n\n"
            "==== การวิเคราะห์ ====\n"
            "9. Filter/aggregate ตามคำถาม\n"
            "10. จัดอันดับ Top 10 (มากไปน้อย) ถ้าถามหาพื้นที่สูงสุด\n"
            "11. print สถิติภาพรวม (mean, max, min) พร้อม label\n"
            "Wrap code in ```python ... ```"
        ),
        "Working Python code with clear labeled output including province names",
        step="code_gen", domain=domain.code, session_id=session_id,
    )
    put({"type": "agent_done", "step": "code_gen", "agentName": "Python Code Generator", "result": code_result})

    # STEP 5: Python Executor
    put({"type": "agent_start", "step": "executor", "agentName": "Python Executor"})
    code = _extract_code(code_result)

    # Guard: don't execute if code gen returned an agent error
    if _is_agent_error(code):
        auth_hint = " (API key ถูก report ว่า leaked — กรุณาสร้าง key ใหม่)" if _is_auth_error(code_result) else ""
        exec_output = f"[ข้ามการรัน — code generation ล้มเหลว{auth_hint}]\n{code_result}"
        code = ""
    else:
        code = re.sub(r"load_csv\(['\"][^'\"]*['\"]\)", f"load_csv('{resolved_file_id}')", code)
        exec_output = exec_python(code)
        # Log executor error (timeout / STDERR / crash)
        _log_exec_error(exec_output, code, "executor", domain.code, session_id, attempt=0)

        if _is_exec_error(exec_output):
            retry_result = _run_agent(
                generator,
                (
                    f"Question: {prompt}\n"
                    f"file_id: '{resolved_file_id}'\n"
                    f"Schema:\n{schema_result}\n\n"
                    f"Previous code that produced an error:\n```python\n{code}\n```\n"
                    f"Error output:\n{exec_output}\n\n"
                    "Fix the code so it works correctly:\n"
                    f"1. First line must be: df = load_csv('{resolved_file_id}')\n"
                    "2. DO NOT redefine load_csv\n"
                    "3. Check column names match the schema exactly\n"
                    "4. Fix the logic error shown in the error message\n"
                    "Wrap code in ```python ... ```"
                ),
                "Fixed Python code that runs without errors",
                step="code_gen_retry", domain=domain.code, session_id=session_id,
            )
            retry_code = _extract_code(retry_result)
            retry_code = re.sub(r"load_csv\(['\"][^'\"]*['\"]\)", f"load_csv('{resolved_file_id}')", retry_code)
            retry_output = exec_python(retry_code)
            # Log retry executor error too
            _log_exec_error(retry_output, retry_code, "executor_retry", domain.code, session_id, attempt=1)
            if not _is_exec_error(retry_output) or len(retry_output) > len(exec_output):
                code = retry_code
                exec_output = retry_output
            code_result = retry_result

    put({"type": "agent_done", "step": "executor", "agentName": "Python Executor",
         "code": code, "result": exec_output})

    # STEP 6: Insight Analyst
    put({"type": "agent_start", "step": "insight", "agentName": "Insight Analyst Agent"})
    analyst = Agent(
        role=f"Insight Analyst — {domain.name_en}",
        goal=(
            f"วิเคราะห์ผลลัพธ์จากข้อมูล {domain.name_th} และเขียนรายงานภาษาไทย "
            "โดยใช้เฉพาะข้อมูลจริงจาก Execution Result เท่านั้น"
        ),
        backstory=(
            f"{domain.expertise}\n"
            "คุณเป็นนักวิเคราะห์ข้อมูลสาธารณสุขที่รายงานผลจากข้อมูลจริงเท่านั้น "
            "คุณไม่สร้างข้อมูล ชื่อจังหวัด หรือตัวเลขที่ไม่มีในผลลัพธ์ "
            "ถ้าข้อมูลไม่พอ คุณบอกตรงๆ ว่า 'ข้อมูลไม่เพียงพอสำหรับการวิเคราะห์นี้'"
        ),
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    insight = _run_agent(
        analyst,
        (
            f"คำถาม: {prompt}\n"
            f"Domain: {domain.name_th}\n\n"
            f"ผลการรันโค้ด (Execution Result):\n{exec_output}\n\n"
            "==== กฎเหล็ก — ห้ามละเมิด ====\n"
            "1. ใช้เฉพาะข้อมูลจาก Execution Result ด้านบน\n"
            "2. ห้ามสร้างชื่อจังหวัดสมมติ เช่น 'จังหวัด ก.' / 'จังหวัด ข.' — ต้องใช้ชื่อจริงเท่านั้น\n"
            "3. ห้ามสร้างตัวเลขที่ไม่มีในผลลัพธ์\n"
            "4. ถ้า Execution มี error → อธิบาย error + สรุปจากสิ่งที่รู้ได้ อย่าสร้างตารางสมมติ\n"
            "5. ถ้าไม่มีชื่อจังหวัดในผลลัพธ์ → บอกว่า 'ข้อมูลไม่ระบุพื้นที่เฉพาะเจาะจง'\n\n"
            "==== โครงสร้างรายงาน ====\n"
            "## สรุปภาพรวม\n"
            "อธิบายว่าพบอะไรจากข้อมูล (2-3 ประโยค)\n\n"
            "## ตารางข้อมูลสำคัญ\n"
            "ตาราง markdown จากข้อมูลจริงในผลลัพธ์:\n"
            "| จังหวัด/พื้นที่ | ตัวชี้วัด 1 | ตัวชี้วัด 2 | ... |\n"
            "|---|---|---|---|\n"
            "| [ชื่อจริงจากผลลัพธ์] | [ค่าจริง] | ... |\n\n"
            "## ข้อสังเกตสำคัญ\n"
            "- bullet points อธิบาย trend, ความแตกต่าง, จุดน่าสนใจ\n\n"
            "## ข้อเสนอแนะ\n"
            "มาตรการหรือแนวทางที่เหมาะสม"
        ),
        "รายงาน insight ภาษาไทยที่มีตาราง markdown จากข้อมูลจริงและ bullet observations",
        step="insight", domain=domain.code, session_id=session_id,
    )
    put({"type": "agent_done", "step": "insight", "agentName": "Insight Analyst Agent", "result": insight})

    if session_id:
        append_history(session_id, "ai", insight)

    put({
        "type": "final",
        "message": insight,
        "domain": {"code": domain.code, "nameTh": domain.name_th, "nameEn": domain.name_en},
        "agentSteps": [
            {"step": "router",      "agentName": "Router Agent",         "result": f"{domain.code} -- {domain.name_th}"},
            {"step": "reasoning",   "agentName": "Reasoning Narrator",   "result": reasoning},
            {"step": "file_finder", "agentName": "File Finder Agent",    "result": file_result},
            {"step": "schema",      "agentName": "Schema Analyst Agent", "result": schema_result},
            {"step": "code_gen",    "agentName": "Python Code Generator", "result": code_result, "code": code},
            {"step": "executor",    "agentName": "Python Executor",      "result": exec_output, "code": code},
            {"step": "insight",     "agentName": "Insight Analyst Agent", "result": insight},
        ],
    })
