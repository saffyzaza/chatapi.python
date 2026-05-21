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

def _run_agent(agent: Agent, description: str, expected: str, max_retries: int = 2) -> str:
    """Run a single-agent CrewAI task with retry on empty/None response."""
    task = Task(description=description, expected_output=expected, agent=agent)
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
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
            return f"[Agent error: {exc}]"
    return f"[Agent error: empty response after {max_retries} retries]"


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


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
        )
        if schema_result.startswith("[Agent error:") and resolved_file_id:
            schema_result = read_csv_schema_impl(resolved_file_id)
    put({"type": "agent_done", "step": "schema", "agentName": "Schema Analyst Agent", "result": schema_result})

    # STEP 4: Python Code Generator
    put({"type": "agent_start", "step": "code_gen", "agentName": "Python Code Generator"})
    generator = Agent(
        role=f"Python Code Generator -- {domain.name_en}",
        goal=(
            f"Generate Python/Pandas code to analyze {domain.name_th} data "
            f"using load_csv('{resolved_file_id}') only"
        ),
        backstory=(
            f"{domain.expertise}\n"
            "You are a Python/Pandas expert who writes correct, clean, immediately runnable code. "
            "You use only the file_id provided. Never use any other filename or path."
        ),
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    code_result = _run_agent(
        generator,
        (
            f"Question: {prompt}\n"
            f"Correct file_id (use this ONLY): '{resolved_file_id}'\n"
            f"Schema:\n{schema_result}\n\n"
            "MANDATORY RULES:\n"
            f"1. First line of code MUST be: df = load_csv('{resolved_file_id}')\n"
            "2. DO NOT define or redefine load_csv -- it is already provided\n"
            "3. DO NOT import minio or define any database connection\n"
            "4. DO NOT use pd.read_csv() with a filename -- only use load_csv()\n"
            f"5. Use file_id='{resolved_file_id}' -- never use any other string\n"
            "6. Filter and aggregate data to answer the question\n"
            "7. Print all results clearly\n"
            "Wrap code in ```python ... ```"
        ),
        "Working Python code block where first line is df = load_csv(file_id)",
    )
    put({"type": "agent_done", "step": "code_gen", "agentName": "Python Code Generator", "result": code_result})

    # STEP 5: Python Executor
    put({"type": "agent_start", "step": "executor", "agentName": "Python Executor"})
    code = _extract_code(code_result)
    code = re.sub(r"load_csv\(['\"][^'\"]*['\"]\)", f"load_csv('{resolved_file_id}')", code)
    exec_output = exec_python(code)

    if "STDERR" in exec_output and ("Error" in exec_output or "error" in exec_output):
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
        )
        retry_code = _extract_code(retry_result)
        retry_code = re.sub(r"load_csv\(['\"][^'\"]*['\"]\)", f"load_csv('{resolved_file_id}')", retry_code)
        retry_output = exec_python(retry_code)
        if "STDERR" not in retry_output or len(retry_output) > len(exec_output):
            code = retry_code
            exec_output = retry_output
            code_result = retry_result

    put({"type": "agent_done", "step": "executor", "agentName": "Python Executor",
         "code": code, "result": exec_output})

    # STEP 6: Insight Analyst
    put({"type": "agent_start", "step": "insight", "agentName": "Insight Analyst Agent"})
    analyst = Agent(
        role=f"Insight Analyst Agent -- {domain.name_en}",
        goal=f"Analyze results and summarize insights from {domain.name_th} data in Thai",
        backstory=domain.expertise,
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    insight = _run_agent(
        analyst,
        (
            f"Question: {prompt}\n"
            f"Domain: {domain.name_th}\n"
            f"Execution result:\n{exec_output}\n\n"
            "Write a full insight report in Thai with this structure:\n"
            "1. A brief summary paragraph explaining what the data shows.\n"
            "2. ONE markdown table summarizing the key numbers from the execution result.\n"
            "   Format: | Column1 | Column2 | ... |\n"
            "           |---------|---------|-----|\n"
            "           | value   | value   | ... |\n"
            "3. Key observations as bullet points (trends, gender diff, district comparison, etc.)\n"
            "4. Recommendations or caveats if relevant.\n"
            "IMPORTANT: Always include the markdown table even if the data is simple. "
            "If execution had errors, explain the cause and still try to summarize what is known."
        ),
        "Insight report in Thai with markdown table + bullet observations",
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
