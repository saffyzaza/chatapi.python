"""CSV analysis pipeline — 6-step agent pipeline for health domains d0, d2–d8."""
import asyncio
import os
import re
import time
from typing import Any

from crewai import Agent, Crew, LLM, Task

from src.domains import Domain
from src.history import append_history
from src.agents.prompt_profile import (
    ANALYST_CORE_POLICY,
    CODE_GENERATOR_CORE_POLICY,
    INSIGHT_RESPONSE_BLUEPRINT,
    MISSING_DATA_POLICY,
    join_prompt,
)
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


def _strip_csv_extension_mentions(text: str) -> str:
    """Remove file-extension token '.csv' from user-facing text."""
    if not text:
        return text
    cleaned = re.sub(r"(?i)\.csv\b", "", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned


def _skip_function_block(lines: list[str], start: int) -> int:
    """Skip the body of a helper function we do not allow redefining."""
    base_indent = len(lines[start]) - len(lines[start].lstrip())
    i = start + 1
    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()
        if not stripped:
            i += 1
            continue
        indent = len(ln) - len(ln.lstrip())
        if indent <= base_indent and not ln.lstrip().startswith("@"):
            break
        i += 1
    return i


def _sanitize_generated_code(code: str, required_lines: list[str], prompt: str = "") -> str:
    """Deterministically fix common contract violations before LLM-based repair."""
    if not code:
        return ""

    lines = code.splitlines()
    cleaned: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Remove direct MinIO imports from generated code.
        if re.search(r"^from\s+minio\s+import\b|^import\s+minio\b", stripped, flags=re.IGNORECASE):
            i += 1
            continue

        # Remove redefinitions of helper functions injected by the executor preamble.
        if re.match(r"^\s*def\s+(load_csv|pct_rank|composite_score)\s*\(", line, flags=re.IGNORECASE):
            i = _skip_function_block(lines, i)
            continue

        # Rewrite forbidden reads to use helper loader.
        line = re.sub(r"\bpd\.read_csv\s*\(", "load_csv(", line, flags=re.IGNORECASE)
        cleaned.append(line)
        i += 1

    normalized = "\n".join(cleaned)

    required_vars: dict[str, str] = {}
    for req in required_lines:
        m = re.match(r"\s*([A-Za-z_]\w*)\s*=\s*load_csv\s*\(", req)
        if m:
            required_vars[m.group(1)] = req.strip()

    if required_vars:
        filtered: list[str] = []
        for ln in normalized.splitlines():
            stripped = ln.strip()
            should_drop = False
            for var_name in required_vars:
                if re.match(
                    rf"^\s*{re.escape(var_name)}\s*=\s*(?:load_csv|pd\.read_csv)\s*\(",
                    stripped,
                    flags=re.IGNORECASE,
                ):
                    should_drop = True
                    break
            if not should_drop:
                filtered.append(ln)

        body = "\n".join(filtered).strip()
        normalized = f"{'\n'.join(required_lines)}\n{body}".strip()

    target_ages = _extract_age_ranges(prompt)
    if target_ages and "SCOPE CHECK" not in normalized:
        ranges_text = ", ".join(f"{lo}-{hi}" for lo, hi in target_ages)
        scope_block = (
            "print('=== SCOPE CHECK ===')\n"
            f"print('requested_age_range: {ranges_text}')\n"
            "print('actual_scope_used: see selected columns/filters in this script')\n"
        )
        normalized = f"{scope_block}\n{normalized}".strip()

    return normalized


def _extract_age_ranges(text: str) -> list[tuple[int, int]]:
    """Extract age ranges like 12-18, 12_18, 12 to 18 from text."""
    if not text:
        return []
    ranges: list[tuple[int, int]] = []
    patterns = [
        r"(?<!\d)(\d{1,2})\s*[-–]\s*(\d{1,2})(?!\d)",
        r"(?<!\d)(\d{1,2})\s*_\s*(\d{1,2})(?!\d)",
        r"(?<!\d)(\d{1,2})\s*(?:to|ถึง)\s*(\d{1,2})(?!\d)",
    ]
    for pattern in patterns:
        for a, b in re.findall(pattern, text, flags=re.IGNORECASE):
            lo = min(int(a), int(b))
            hi = max(int(a), int(b))
            ranges.append((lo, hi))
    deduped: list[tuple[int, int]] = []
    for age_range in ranges:
        if age_range not in deduped:
            deduped.append(age_range)
    return deduped


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _has_target_age_label(text: str, lo: int, hi: int) -> bool:
    patterns = [
        rf"{lo}\s*[-–]\s*{hi}",
        rf"{lo}\s*_\s*{hi}",
        rf"{lo}\s*(?:to|ถึง)\s*{hi}",
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _has_estimation_signal(text: str) -> bool:
    return bool(re.search(r"ESTIMATION METHOD|estimate|proxy|ประมาณ", text, flags=re.IGNORECASE))


def _find_code_issues(code: str, required_lines: list[str], prompt: str = "") -> list[str]:
    """Return contract violations for generated code before execution."""
    if not code.strip():
        return ["empty_code"]

    issues: list[str] = []
    # Ignore comment-only lines to reduce false positives from explanatory comments.
    scan_code = "\n".join(ln for ln in code.splitlines() if not ln.strip().startswith("#"))
    forbidden_patterns = {
        r"\bfrom\s+minio\s+import\b": "forbidden_import_minio",
        r"\bimport\s+minio\b": "forbidden_import_minio",
        r"\bMinio\s*\(": "forbidden_minio_client",
        r"\bpd\.read_csv\s*\(": "forbidden_pd_read_csv",
        r"\bdef\s+load_csv\s*\(": "forbidden_redefine_load_csv",
        r"\bdef\s+pct_rank\s*\(": "forbidden_redefine_pct_rank",
        r"\bdef\s+composite_score\s*\(": "forbidden_redefine_composite_score",
    }

    for pattern, label in forbidden_patterns.items():
        if re.search(pattern, scan_code, flags=re.IGNORECASE):
            issues.append(label)

    normalized = "\n".join(line.strip() for line in code.splitlines() if line.strip())
    for required in required_lines:
        if required not in normalized:
            issues.append(f"missing_required_line:{required}")

    target_ages = _extract_age_ranges(prompt)
    if target_ages:
        if "SCOPE CHECK" not in scan_code:
            issues.append("missing_scope_check")

        for lo, hi in target_ages:
            if not _has_target_age_label(scan_code, lo, hi):
                issues.append(f"missing_target_age_label:{lo}-{hi}")

        code_ages = _extract_age_ranges(scan_code)
        if code_ages:
            has_target_overlap = any(
                _ranges_overlap(target, observed)
                for target in target_ages
                for observed in code_ages
            )
            if not has_target_overlap and not _has_estimation_signal(scan_code):
                issues.append("missing_estimation_method")

    # Keep issue list stable and deduplicated.
    return list(dict.fromkeys(issues))


def _age_scope_repair_hints(prompt: str, issues: list[str]) -> str:
    """Build concrete repair instructions when target age-scope constraints are violated."""
    target_ages = _extract_age_ranges(prompt)
    if not target_ages:
        return ""

    age_scope_issues = {
        "missing_scope_check",
        "missing_estimation_method",
    }
    has_age_scope_issue = any(
        (issue in age_scope_issues) or issue.startswith("missing_target_age_label:")
        for issue in issues
    )
    if not has_age_scope_issue:
        return ""

    ranges_text = ", ".join(f"{lo}-{hi}" for lo, hi in target_ages)
    primary_lo, primary_hi = target_ages[0]
    primary_label = f"{primary_lo}-{primary_hi}"

    return (
        "Age-scope mandatory fixes:\n"
        f"- Target age range from user query: {ranges_text}\n"
        "- Add section '=== SCOPE CHECK ===' and print requested vs actual scope (year/area/age).\n"
        f"- Output must contain explicit target-age label '{primary_label}'.\n"
        "- If no direct target-age columns exist, compute proxy estimate as follows:\n"
        "  1) detect nearest age-range columns via regex (e.g. 6-14, 15-19)\n"
        "  2) if lower+upper ranges exist, use weighted linear interpolation by midpoint\n"
        "  3) otherwise use nearest-neighbor proxy from closest range\n"
        "- Add section '=== ESTIMATION METHOD ===' and print source columns + formula used.\n"
    )


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
            role="Health Advisor",
            goal="ตอบคำถามด้านสุขภาพเป็นภาษาไทยอย่างกระชับ เป็นธรรมชาติ และเป็นประโยชน์",
            backstory=(
                "คุณเป็นผู้เชี่ยวชาญด้านสุขภาพและสาธารณสุขที่สื่อสารได้เป็นธรรมชาติ "
                "ตอบตรงประเด็น ภาษาเข้าใจง่าย ไม่ต้องทำรายงานทางการ "
                "หากเป็นการสนทนาต่อเนื่องให้ต่อบทสนทนาได้เลย"
            ),
            llm=llm,
            verbose=False,
            max_iter=5,
        )
        insight = _run_agent(
            analyst,
            (
                f"{history_section}"
                f"คำถาม: {prompt}\n\n"
                "ตอบเป็นภาษาไทย กระชับ เป็นธรรมชาติ ตรงประเด็น "
                "ไม่ต้องใส่หัวข้อ สรุปผู้บริหาร แหล่งข้อมูล หรือโครงสร้างรายงานทางการ "
                "ตอบเหมือนผู้เชี่ยวชาญด้านสุขภาพคุยกับคนทั่วไปโดยตรง"
            ),
            "คำตอบที่ชัดเจนและเป็นประโยชน์เป็นภาษาไทย",
        )
        insight = _strip_csv_extension_mentions(insight)
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
        answer = _strip_csv_extension_mentions(acc_result.answer)
        put({"type": "agent_done", "step": "insight", "agentName": "Accident Answer Writer",
             "result": answer})

        if session_id:
            append_history(session_id, "ai", answer)
        put({
            "type": "final",
            "message": answer,
            "domain": {"code": domain.code, "nameTh": domain.name_th, "nameEn": domain.name_en},
            "agentSteps": [
                {"step": "router",    "agentName": "Router Agent",          "result": f"{domain.code} — {domain.name_th}"},
                {"step": "reasoning", "agentName": "Reasoning Narrator",    "result": reasoning},
                {"step": "sql_agent", "agentName": "Accident SQL Agent",    "result": acc_result.raw_data or ""},
                {"step": "insight",   "agentName": "Accident Answer Writer", "result": answer},
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
            f"{history_section}"
            f"คำถาม: {prompt}\n"
            f"Domain: {domain.name_th}\n\n"
            f"ขั้นตอน:\n"
            f"1. เรียก list_csv_files(prefix='{domain.folder_prefix}') เพื่อดูไฟล์ใน domain\n"
            f"2. ถ้าไม่พบ ให้เรียก list_csv_files(prefix='') เพื่อดูทั้งหมด\n"
            f"3. เลือกไฟล์ที่ตรงกับคำถามมากที่สุด (ใช้บริบทการสนทนาก่อนหน้าประกอบด้วย)\n"
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

    if not resolved_file_id:
        fallback_listing = list_csv_files_impl(domain.folder_prefix or "")
        if not fallback_listing or fallback_listing.startswith("No") or fallback_listing.startswith("Error"):
            fallback_listing = list_csv_files_impl("")
        sample_lines = []
        if fallback_listing and not fallback_listing.startswith("No") and not fallback_listing.startswith("Error"):
            sample_lines = [ln for ln in fallback_listing.split("\n") if ln.strip()][:5]

        # ── Fallback to general AI when no file found ─────────────────────
        put({"type": "agent_start", "step": "insight", "agentName": "Insight Analyst Agent"})
        analyst = Agent(
            role="Health Advisor",
            goal="ตอบคำถามด้านสุขภาพเป็นภาษาไทยอย่างกระชับ เป็นธรรมชาติ และเป็นประโยชน์",
            backstory=(
                "คุณเป็นผู้เชี่ยวชาญด้านสุขภาพและสาธารณสุขที่สื่อสารได้เป็นธรรมชาติ "
                "ตอบตรงประเด็น ภาษาเข้าใจง่าย ไม่ต้องทำรายงานทางการ"
            ),
            llm=llm,
            verbose=False,
            max_iter=5,
        )
        insight = _run_agent(
            analyst,
            (
                f"{history_section}"
                f"คำถาม: {prompt}\n\n"
                "ตอบเป็นภาษาไทย กระชับ เป็นธรรมชาติ ตรงประเด็น "
                "ไม่ต้องใส่หัวข้อ สรุปผู้บริหาร แหล่งข้อมูล หรือโครงสร้างรายงานทางการ "
                "ตอบเหมือนผู้เชี่ยวชาญด้านสุขภาพคุยกับคนทั่วไปโดยตรง"
            ),
            "คำตอบที่ชัดเจนและเป็นประโยชน์เป็นภาษาไทย",
        )
        insight = _strip_csv_extension_mentions(insight)
        put({"type": "agent_done", "step": "insight", "agentName": "Insight Analyst Agent", "result": insight})
        if session_id:
            append_history(session_id, "ai", insight)
        put({
            "type": "final",
            "message": insight,
            "domain": {"code": domain.code, "nameTh": domain.name_th, "nameEn": domain.name_en},
            "agentSteps": [
                {"step": "router",      "agentName": "Router Agent",          "result": f"{domain.code} — {domain.name_th}"},
                {"step": "reasoning",   "agentName": "Reasoning Narrator",    "result": reasoning},
                {"step": "file_finder", "agentName": "File Finder Agent",     "result": file_result or "ไม่พบไฟล์ — ใช้ความรู้ทั่วไปแทน"},
                {"step": "insight",     "agentName": "Insight Analyst Agent", "result": insight},
            ],
        })
        return

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
            f"อ่านคำถามให้เข้าใจก่อน แล้วสร้าง Python/Pandas code "
            "ที่ดึงข้อมูลตรงกับสิ่งที่ถาม รันได้ทันที output ชัดเจน"
        ),
        backstory=join_prompt(
            domain.expertise,
            "คุณเป็น Python/Pandas expert ที่เข้าใจ context คำถาม "
            "ก่อนเขียนโค้ด คุณถามตัวเองว่า: ถามโรคอะไร? พื้นที่ไหน? ปีไหน? "
            "ต้องการ aggregate แบบไหน? แล้วเขียนโค้ดให้ตอบตรงนั้น "
            "output ต้องมีชื่อจริง ตัวเลขมี label อ่านแล้วเข้าใจทันที",
            CODE_GENERATOR_CORE_POLICY,
        ),
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    code_result = _run_agent(
        generator,
        (
            f"{history_section}"
            f"คำถาม: {prompt}\n"
            f"file_id: '{resolved_file_id}'\n"
            f"Schema:\n{schema_result}\n\n"
            "==== กฎบังคับ ====\n"
            f"1. บรรทัดแรก: df = load_csv('{resolved_file_id}')\n"
            "2. บรรทัดที่ 2: pd.set_option('display.max_rows', 200)\n"
            "3. ห้าม redefine load_csv / import minio / ใช้ pd.read_csv()\n"
            "4. ชื่อจังหวัด/พื้นที่ต้องแสดงเป็น text ครบในทุก output\n"
            "5. ใช้ print(df.to_string(index=False)) แทน print(df)\n"
            "6. drop_duplicates() ก่อน aggregate ทุกครั้ง\n\n"
            "==== เขียนโค้ดให้ตอบตรงคำถาม ====\n"
            "7. วิเคราะห์ว่าคำถามถามอะไร:\n"
            "   ถามข้อมูลทั่วไป → filter ตามเงื่อนไข + แสดงตาราง + สถิติสรุป\n"
            "   ถามรายอำเภอ/รายพื้นที่ → groupby อำเภอ แสดงครบทุกอำเภอ\n"
            "   ถามอันดับ/สูงสุด/ต่ำสุด → sort + head(N) ตามที่ถาม\n"
            "   ถามแนวโน้ม/เปรียบเทียบ → groupby ปี หรือ merge ตามที่เหมาะ\n"
            "7.0 ถ้าคำถามระบุ ปี/จังหวัด/อำเภอ/ช่วงอายุ ต้อง filter เฉพาะช่วงที่ถามก่อนคำนวณ\n"
            "    - ห้ามรวมปีนอกช่วงที่ผู้ใช้ถาม\n"
            "    - ต้อง print section '=== SCOPE CHECK ===' ระบุช่วงที่ผู้ใช้ถามและช่วงที่ใช้จริง\n"
            "7.1 ถ้าไม่พบคอลัมน์ตรงกับช่วงอายุ/กลุ่มเป้าหมายที่ถาม ให้คำนวณประมาณการจากคอลัมน์ใกล้เคียงในไฟล์เดียวกัน\n"
            "    - มี 2 ช่วงคร่อมเป้าหมาย: ใช้ interpolation ด้วย midpoint\n"
            "    - มีเพียง 1 ช่วง: ใช้ nearest-neighbor proxy\n"
            "    - ต้อง print section '=== ESTIMATION METHOD ===' และระบุว่าเป็นค่าประมาณ\n"
            "8. print หัวข้อก่อนทุก section: print('=== [หัวข้อ] ===')\n"
            "9. ถ้ามีแถวซ้ำ print('แถวก่อน dedup:', len(df), '→ หลัง:', len(df_dedup))\n"
            "10. ก่อน print ตารางหลัก ให้ rename คอลัมน์เป็นชื่อภาษาไทยอ่านได้สมบูรณ์:\n"
            "    ตัวอย่าง: 'เริ่มอ้วน_%_12-18' → 'ร้อยละเด็กเริ่มอ้วน ช่วง 12-18 ปี (ประมาณ)'\n"
            "    ใช้ df_display = df_result.rename(columns={...}) แล้ว print df_display\n"
            "11. round ตัวเลขทศนิยมทั้งหมดเป็น 2 ตำแหน่งก่อน print: df_display = df_display.round(2)\n"
            "Wrap code in ```python ... ```\n\n"
            f"{CODE_GENERATOR_CORE_POLICY}"
        ),
        "Python code ที่รันได้ ตอบตรงคำถาม output มีชื่อจริงและ label ชัดเจน",
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
        required_lines = [f"df = load_csv('{resolved_file_id}')"]
        sanitized_code = _sanitize_generated_code(code, required_lines, prompt)
        code_issues = _find_code_issues(sanitized_code, required_lines, prompt)
        if not code_issues:
            code = sanitized_code

        if code_issues:
            age_scope_hints = _age_scope_repair_hints(prompt, code_issues)
            repair_result = _run_agent(
                generator,
                (
                    f"Question: {prompt}\n"
                    f"file_id: '{resolved_file_id}'\n"
                    f"Schema:\n{schema_result}\n\n"
                    f"Current code:\n```python\n{code}\n```\n\n"
                    f"Contract violations:\n{chr(10).join(f'- {i}' for i in code_issues)}\n\n"
                    f"{age_scope_hints}\n"
                    "Fix code to satisfy all constraints:\n"
                    f"1. First required line: df = load_csv('{resolved_file_id}')\n"
                    "2. Do not import/use Minio directly\n"
                    "3. Do not use pd.read_csv\n"
                    "4. Do not redefine helper functions\n"
                    "5. Keep output labels clear and keep business logic intact\n"
                    "6. Respect requested scope (year/area/age) and include '=== SCOPE CHECK ==='\n"
                    "7. If requested age range has no direct column, compute estimate and label output with target age\n"
                    "Wrap code in ```python ... ```"
                ),
                "Repaired Python code that passes all contract checks",
                step="code_contract_repair", domain=domain.code, session_id=session_id,
            )
            repaired_code = _sanitize_generated_code(_extract_code(repair_result), required_lines, prompt)
            repaired_issues = _find_code_issues(repaired_code, required_lines, prompt)
            if not repaired_issues:
                code = repaired_code
                code_result = repair_result
            else:
                exec_output = (
                    "[ข้ามการรัน — โค้ดยังผิดกติกาหลังพยายามแก้]\n"
                    f"issues: {', '.join(repaired_issues)}"
                )
                code = ""

        if code:
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
                retry_code = _sanitize_generated_code(_extract_code(retry_result), required_lines, prompt)
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
        role=f"Senior Public Health Data Analyst — {domain.name_en}",
        goal=(
            f"อ่านคำถามให้เข้าใจก่อน แล้วเรียบเรียงรายงานระดับทางการสำหรับผู้บริหาร สสจ. "
            "ที่อ่านแล้วตัดสินใจเชิงนโยบายได้ทันที"
        ),
        backstory=join_prompt(
            domain.expertise,
            "คุณเป็นนักวิเคราะห์ข้อมูลสาธารณสุขอาวุโสที่วิเคราะห์ตรงประเด็นและมั่นใจในตัวเลข "
            "คุณตอบโดยขึ้นต้นด้วยสิ่งที่ค้นพบจากข้อมูลเสมอ ไม่ขึ้นต้นด้วยข้อแก้ตัว "
            "ถ้าใช้ค่าประมาณ คุณระบุว่า 'ค่าประมาณจาก X' แทนที่จะบอกว่า 'ไม่มีข้อมูลตรง' "
            "คุณใช้เฉพาะตัวเลขจาก Execution Result และตอบให้ครบทุก dimension ที่ถาม",
            ANALYST_CORE_POLICY,
        ),
        llm=llm,
        verbose=False,
        max_iter=5,
    )
    insight = _run_agent(
        analyst,
        (
            f"คำถามของผู้ใช้: {prompt}\n"
            f"Domain: {domain.name_th}\n\n"
            f"Execution Result:\n{exec_output}\n\n"
            "==== แนวทางการวิเคราะห์ (ปรับตามบริบทคำถาม) ====\n\n"
            "ขั้นที่ 1 — ทำความเข้าใจคำถาม:\n"
            "  ถามถึงอะไร? (โรค / พื้นที่ / ปี / ตัวชี้วัด)\n"
            "  ระดับข้อมูลไหน? (จังหวัด / อำเภอ / ภาพรวม)\n"
            "  ต้องการรู้อะไร? (สถานการณ์ / แนวโน้ม / เปรียบเทียบ / พื้นที่เสี่ยง)\n\n"
            "ขั้นที่ 2 — เลือกโครงสร้างรายงานให้เหมาะสม:\n"
            "  • ถามข้อมูลทั่วไป → สรุปภาพรวม + ตาราง + ข้อสังเกตสั้นๆ\n"
            "  • ถามรายอำเภอ → ตารางรายอำเภอเรียงจากมากไปน้อย\n"
            "    + อำเภอที่น่าเป็นห่วง 3 อันดับแรกพร้อมเหตุผลตัวเลข\n"
            "    + ข้อเสนอแนะเชิงปฏิบัติในระดับอำเภอ\n"
            "  • ถามแนวโน้ม → ตารางตามปี + อธิบายทิศทางการเปลี่ยนแปลง\n"
            "  • ถามพื้นที่เสี่ยง → Red Zone table + Pattern + ข้อเสนอแนะ\n\n"
            "ขั้นที่ 3 — เขียนรายงาน:\n"
            "  • ใช้ตาราง markdown จากข้อมูลจริงเสมอ ชื่อคอลัมน์ต้องเป็นภาษาไทยอ่านได้สมบูรณ์ ไม่ใช่ชื่อตัวแปรดิบ\n"
            "  • ตัวเลขในตารางแสดง 2 ทศนิยม พร้อมหน่วย (%) ถ้าเป็นร้อยละ\n"
            "  • ระบุแหล่งข้อมูลที่ใช้วิเคราะห์\n"
            "  • ถ้าไม่มีข้อมูลตามที่ถาม → แจ้งตรงๆ อย่าสร้างข้อมูลสมมติ\n"
            "  • ถ้า Execution มี error → อธิบายและสรุปเท่าที่ข้อมูลมี\n"
            "  • ใช้เฉพาะตัวเลขและชื่อจริงจาก Execution Result\n"
            "  • ต้องอธิบายให้ครบ: ข้อมูลมาจากไหน, ใช้ปีไหน, วิธีคำนวณ (ถ้ามีสูตรให้ใช้ LaTeX block — วางสูตรบรรทัดเดียวโดดๆ เช่น\n\n$$\\hat{v} = ...$$\n\nห้ามวางสูตรกลางประโยค), และความหมายคอลัมน์ในตาราง\n"
            "  • ห้ามขึ้นต้นสรุปด้วยประโยคแก้ตัว — ให้เริ่มด้วยสิ่งที่ค้นพบจากข้อมูลทันที\n"
            "  • เขียนในภาษาทางการระดับรายงานราชการ เหมาะสำหรับผู้บริหาร สสจ.\n\n"
            "ขั้นที่ 4 — ตรวจความตรงคำขอ (mandatory):\n"
            "  • ยืนยันให้ชัดว่าผลลัพธ์ครอบคลุมช่วงปี/พื้นที่/ช่วงอายุที่ผู้ใช้ถาม\n"
            "  • ถ้าครอบคลุมไม่ครบ ให้ระบุส่วนที่ขาดในส่วน 'ข้อจำกัด' เท่านั้น ไม่นำมาขึ้นต้นรายงาน\n\n"
            f"{INSIGHT_RESPONSE_BLUEPRINT}\n\n"
            f"{MISSING_DATA_POLICY}"
        ),
        "รายงานทางการภาษาไทยสำหรับผู้บริหาร สสจ. ตอบตรงประเด็น มีตาราง ชื่อคอลัมน์อ่านได้ สูตร LaTeX และข้อเสนอแนะเชิงนโยบาย",
        step="insight", domain=domain.code, session_id=session_id,
    )
    insight = _strip_csv_extension_mentions(insight)
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
