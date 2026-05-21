"""Multi-domain CSV pipeline — cross-domain Red Zone / pattern analysis.

Improvements vs single-domain pipeline:
  1. Geographic Key Detector   — keyword-based column detection for merge
  2. Domain Coverage Validator — ensures every domain has ≥1 file selected
  3. (routing handled in router.py via keyword override)
  4. (composite_score helper injected via minio_preamble)
  5. (mode=multi handled in routers/analyze.py)
  6. Per-file Schema Progress  — emits progress event per file, not batch

Pipeline order:
  Multi-File Finder → Domain Coverage Validator →
  Multi-Schema (per-file progress) → Geo Key Detector →
  Code Generator (with merge recipe) → Executor → Cross-Domain Insight
"""
import asyncio
import json
import re
from typing import Any

from crewai import Agent

from src.domains import Domain
from src.history import append_history
from src.tools.minio import (
    list_csv_files,
    list_csv_files_impl,
    resolve_file_id,
    read_csv_schema_impl,
    exec_python,
)
from src.agents.csv_pipeline import (
    _get_llm, _run_agent, _extract_code,
    _is_agent_error, _is_auth_error,
    _is_exec_error, _log_exec_error,
)

MAX_FILES = 5

# ── Geographic keyword vocabulary ─────────────────────────────────────────────

_GEO_SYNONYMS = [
    "จังหวัด", "province", "changwat", "provine", "จ.",
    "อำเภอ", "district", "amphoe", "amphur",
    "เขต", "zone", "พื้นที่", "area",
    "hospcode", "สถานพยาบาล", "รพ.",
]

_THAI_PROVINCE_SAMPLES = [
    "กรุงเทพ", "อุบล", "ขอนแก่น", "เชียงใหม่", "อุดร",
    "นครราชสีมา", "มุกดาหาร", "ยโสธร", "ศรีสะเกษ", "อำนาจเจริญ",
    "นครพนม", "สกลนคร", "บึงกาฬ",
]


# ── Step 1: Geographic Key Detector ──────────────────────────────────────────

def _detect_geo_keys(schemas_info: list[dict]) -> dict[str, str]:
    """Pure keyword detection of the geographic merge-key column per DataFrame.

    Priority 1: column name contains a geo synonym.
    Priority 2: sample values contain known Thai province names.
    Returns mapping like {"df1": "จังหวัด", "df2": "province"}.
    """
    mapping: dict[str, str] = {}
    for info in schemas_info:
        df_key = f"df{info['index']}"
        cols: list[str] = info.get("cols", [])

        # Priority 1: column name match
        for col in cols:
            col_norm = col.lower().replace(" ", "").replace("_", "")
            for kw in _GEO_SYNONYMS:
                kw_norm = kw.lower().replace(" ", "").replace("_", "")
                if kw_norm in col_norm or col_norm in kw_norm:
                    mapping[df_key] = col
                    break
            if df_key in mapping:
                break

        # Priority 2: sample value match
        if df_key not in mapping:
            for row in (info.get("sample") or []):
                for col, val in (row or {}).items():
                    if isinstance(val, str) and any(p in val for p in _THAI_PROVINCE_SAMPLES):
                        mapping[df_key] = col
                        break
                if df_key in mapping:
                    break

    return mapping


def _build_merge_recipe(geo_keys: dict[str, str]) -> str:
    """Convert geo_keys map into code-generator instructions."""
    if not geo_keys:
        return "# ไม่พบ geographic key — ให้วิเคราะห์แต่ละ DataFrame แยกกัน"

    values = list(geo_keys.values())
    canonical = max(set(values), key=values.count)  # most-common column name

    lines = ["# Geographic key ที่ตรวจพบ:"]
    renames: list[str] = []
    for df_key, col in geo_keys.items():
        lines.append(f"#   {df_key}: column = '{col}'")
        if col != canonical:
            renames.append(f"{df_key} = {df_key}.rename(columns={{'{col}': '{canonical}'}})")

    lines.append(f"# Canonical merge key: '{canonical}'")
    if renames:
        lines.append("# Rename ก่อน merge:")
        lines.extend(f"# {r}" for r in renames)
        lines.append(f"# merge: pd.merge(df1, df2, on='{canonical}', how='outer')")
    else:
        lines.append(f"# merge: pd.merge(df1, df2, on='{canonical}', how='outer')")

    return "\n".join(lines)


# ── Step 2: Domain Coverage Validator ─────────────────────────────────────────

def _enforce_domain_coverage(
    selected_files: list[tuple[str, str]],
    domains: list[Domain],
    prompt: str,
) -> list[tuple[str, str]]:
    """Guarantee at least 1 file per domain.

    For each domain with no matching file, force-injects the best keyword match.
    If at MAX_FILES capacity, replaces the lowest-scoring existing file.
    """
    result = list(selected_files)

    for domain in domains:
        prefix = domain.folder_prefix
        covered = any(prefix.lower() in line.lower() for _, line in result)
        if covered:
            continue

        # Domain not represented — find the best file from its prefix
        listing = list_csv_files_impl(prefix)
        if not listing or listing.startswith("No") or listing.startswith("Error"):
            listing = list_csv_files_impl("")  # widen to all files

        candidates = _keyword_select(prompt, listing, 1)
        for candidate in candidates:
            fid = resolve_file_id(candidate)
            if fid and not any(f == fid for f, _ in result):
                if len(result) >= MAX_FILES:
                    result[-1] = (fid, candidate)   # replace last (lowest scored)
                else:
                    result.append((fid, candidate))
                break

    return result


# ── Generic helpers ────────────────────────────────────────────────────────────

def _keyword_select(prompt: str, combined_text: str, max_n: int) -> list[str]:
    lines = [ln.strip() for ln in combined_text.split("\n") if ln.strip() and "[ID:" in ln]
    if not lines:
        return []
    words = set(re.sub(r"[^\w\s]", " ", prompt.lower()).split())

    def score(line: str) -> int:
        ll = line.lower()
        return sum(1 for w in words if len(w) > 2 and w in ll)

    return sorted(lines, key=score, reverse=True)[:max_n]


def _parse_file_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.split("\n") if ln.strip() and "[ID:" in ln]


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_multi_pipeline(
    prompt: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    domains: list[Domain],
    history_context: str,
    history_section: str,
    session_id: str = "",
    reasoning: str = "",
) -> None:
    """Stream a cross-domain analysis pipeline via SSE queue."""
    llm = _get_llm()

    def put(ev: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(queue.put(ev), loop)

    domain_names_th = " + ".join(d.name_th for d in domains)
    domain_names_en = " + ".join(d.name_en for d in domains)
    domain_prefixes = [d.folder_prefix for d in domains if d.folder_prefix]

    # ── STEP 1a: Multi-File Finder (agent with list_csv_files tool) ───────────
    put({"type": "agent_start", "step": "file_finder", "agentName": "Multi-File Finder Agent"})

    prefix_calls = "\n".join(f"  - list_csv_files(prefix='{p}')" for p in domain_prefixes) \
                   or "  - list_csv_files(prefix='')"

    finder = Agent(
        role="Multi-Domain File Finder Agent",
        goal="ค้นหาไฟล์ CSV จาก MinIO ด้วย tool list_csv_files แล้วเลือกไฟล์ที่เกี่ยวข้องสูงสุด 5 ไฟล์",
        backstory=(
            "คุณเป็นผู้เชี่ยวชาญเลือกไฟล์ข้อมูลสำหรับการวิเคราะห์ข้ามสาขา "
            "คุณต้องเรียก tool list_csv_files เพื่อดูรายการจริงจาก MinIO ก่อนเสมอ "
            "จากนั้นเลือกไฟล์ที่ครอบคลุมทุก domain และเกี่ยวข้องกับคำถามมากที่สุด "
            "ตอบเป็นรายการ แต่ละบรรทัดต้องมี [ID:...] จากผล tool เท่านั้น"
        ),
        tools=[list_csv_files],
        llm=llm,
        verbose=False,
        max_iter=8,
    )

    file_result = _run_agent(
        finder,
        (
            f"คำถาม: {prompt}\n"
            f"Domains ที่ต้องการ: {domain_names_th}\n\n"
            "ขั้นตอนบังคับ:\n"
            f"1. เรียก tool ต่อไปนี้เพื่อดูไฟล์ใน MinIO:\n{prefix_calls}\n"
            "2. ถ้าไม่พบไฟล์ ให้เรียก list_csv_files(prefix='') เพื่อดูทุกไฟล์\n"
            f"3. เลือกไฟล์ที่เกี่ยวข้องกับ '{domain_names_th}' มากที่สุด ไม่เกิน {MAX_FILES} ไฟล์\n"
            "4. ต้องเลือกให้ครอบคลุมทุก domain\n"
            "5. ตอบเป็นรายการ แต่ละบรรทัด: [ID:xxxxxx] ชื่อไฟล์\n"
            "   ห้ามสร้าง ID ใหม่ — ใช้ [ID:...] จาก tool เท่านั้น"
        ),
        f"รายการไฟล์ (≤{MAX_FILES} ไฟล์) แต่ละบรรทัด: [ID:xxxxxx] ชื่อไฟล์",
        step="file_finder", domain=domain_names_en, session_id=session_id,
    )

    selected_lines = _parse_file_lines(file_result)[:MAX_FILES]

    # Fallback: keyword scoring over full MinIO listing
    if len(selected_lines) < 2:
        all_text = list_csv_files_impl("")
        selected_lines = _keyword_select(prompt, all_text, MAX_FILES)

    # Resolve to (file_id, display_line) — de-duplicate
    selected_files: list[tuple[str, str]] = []
    for line in selected_lines:
        fid = resolve_file_id(line)
        if fid and not any(f == fid for f, _ in selected_files):
            selected_files.append((fid, line))

    # ── STEP 1b: Domain Coverage Validator ───────────────────────────────────
    selected_files = _enforce_domain_coverage(selected_files, domains, prompt)

    file_summary = "\n".join(f"  • {line}" for _, line in selected_files)
    put({
        "type": "agent_done",
        "step": "file_finder",
        "agentName": "Multi-File Finder Agent",
        "result": file_summary or "(ไม่พบไฟล์)",
        "fileCount": len(selected_files),
    })

    if not selected_files:
        put({
            "type": "final",
            "message": "ไม่พบไฟล์ CSV ที่เกี่ยวข้อง กรุณาตรวจสอบว่ามีข้อมูลอยู่ใน MinIO",
            "domain": {"code": "multi", "nameTh": domain_names_th, "nameEn": domain_names_en},
            "agentSteps": [
                {"step": "router",      "agentName": "Multi-Domain Router",     "result": f"Domains: {domain_names_th}"},
                {"step": "reasoning",   "agentName": "Reasoning Narrator",      "result": reasoning},
                {"step": "file_finder", "agentName": "Multi-File Finder Agent", "result": "ไม่พบไฟล์"},
            ],
        })
        return

    # ── STEP 2: Multi-Schema Analyst (Step 6: per-file progress) ─────────────
    put({"type": "agent_start", "step": "schema", "agentName": "Multi-Schema Analyst",
         "total": len(selected_files)})

    schemas_info: list[dict] = []
    schema_parts: list[str] = []
    total = len(selected_files)

    for i, (file_id, display_line) in enumerate(selected_files, 1):
        # Step 6: emit per-file progress event before reading
        put({
            "type": "agent_progress",
            "step": "schema",
            "agentName": "Multi-Schema Analyst",
            "current": i,
            "total": total,
            "file": display_line.strip(),
        })

        raw = read_csv_schema_impl(file_id)
        try:
            data = json.loads(raw)
            cols = data.get("columns", [])
            sample = data.get("sample", [{}])
            shape = data.get("shape", [])
            schemas_info.append({"index": i, "file_id": file_id, "cols": cols, "sample": sample})
            schema_parts.append(
                f"**df{i}** — `load_csv('{file_id}')`\n"
                f"  ชื่อไฟล์: {data.get('file_name', file_id)}\n"
                f"  Shape: {shape}\n"
                f"  Columns: {cols}\n"
                f"  ตัวอย่างข้อมูล: {sample[0] if sample else {}}"
            )
        except Exception:
            # Partial failure: record empty schema and continue
            schemas_info.append({"index": i, "file_id": file_id, "cols": [], "sample": []})
            schema_parts.append(
                f"**df{i}** — `load_csv('{file_id}')`\n"
                f"  ⚠️ อ่าน schema ไม่สำเร็จ: {raw[:120]}"
            )

    schema_summary = "\n\n".join(schema_parts)
    put({"type": "agent_done", "step": "schema", "agentName": "Multi-Schema Analyst",
         "result": schema_summary})

    # ── STEP 3: Geographic Key Detector (Step 1: pure keyword, no LLM) ───────
    geo_keys = _detect_geo_keys(schemas_info)
    merge_recipe = _build_merge_recipe(geo_keys)

    put({
        "type": "agent_done",
        "step": "geo_keys",
        "agentName": "Geographic Key Detector",
        "result": merge_recipe,
        "geoKeys": geo_keys,
    })

    # ── STEP 4: Multi-DataFrame Code Generator ────────────────────────────────
    put({"type": "agent_start", "step": "code_gen", "agentName": "Multi-DataFrame Code Generator"})

    load_block = "\n".join(f"df{i} = load_csv('{fid}')" for i, (fid, _) in enumerate(selected_files, 1))
    n = len(selected_files)

    generator = Agent(
        role="Multi-DataFrame Python Code Generator",
        goal=(
            f"สร้าง Python/Pandas code ที่ merge {n} DataFrame และ print ผลลัพธ์ชัดเจน "
            "พร้อมชื่อจังหวัดจริง composite score และ ranking Red Zone"
        ),
        backstory=(
            "คุณเป็น Python/Pandas expert ที่เชี่ยวชาญการวิเคราะห์ข้อมูลสาธารณสุขจากหลาย dataset "
            "คุณใช้ pct_rank() และ composite_score() ที่มีให้อยู่แล้ว "
            "คุณให้ความสำคัญกับ output ที่ชัดเจน: ชื่อจังหวัดต้องแสดงครบ ตัวเลขมี label "
            "เพื่อให้ AI วิเคราะห์ต่อได้โดยไม่ต้องสร้างข้อมูลขึ้นมาเอง"
        ),
        llm=llm,
        verbose=False,
        max_iter=5,
    )

    code_result = _run_agent(
        generator,
        (
            f"คำถาม: {prompt}\n"
            f"Domains: {domain_names_th}\n\n"
            f"Schemas:\n{schema_summary}\n\n"
            f"Geographic Keys (ตรวจพบอัตโนมัติ):\n{merge_recipe}\n\n"
            "==== กฎบังคับ (ห้ามละเมิด) ====\n"
            f"1. โหลดข้อมูล (ห้ามเปลี่ยน file_id):\n{load_block}\n\n"
            "2. ห้าม import minio / redefine load_csv / redefine pct_rank / redefine composite_score\n"
            "3. ห้ามใช้ pd.read_csv() — ใช้ load_csv() เท่านั้น\n"
            "4. ใช้ชื่อ column จาก schema เท่านั้น — ห้ามเดาชื่อ column\n\n"
            "==== Output Format บังคับ ====\n"
            "5. บรรทัดหลัง load: pd.set_option('display.max_rows', 100)\n"
            "6. ก่อน print ทุก section ใส่หัวข้อ เช่น print('=== Top 10 Red Zone ===')\n"
            "7. ชื่อจังหวัด/พื้นที่ต้องแสดงเป็น text ครบในทุกตาราง\n"
            "8. ใช้ print(df.to_string(index=False)) เพื่อแสดง DataFrame ครบ\n\n"
            "==== ขั้นตอนการวิเคราะห์ ====\n"
            "a. โหลด + strip whitespace จาก geo column ทุกตัว\n"
            "b. Rename geo columns ตาม merge recipe\n"
            "c. Aggregate แต่ละ df รายจังหวัด (groupby) ก่อน merge\n"
            "d. Merge ด้วย outer join บน geo key\n"
            "e. composite_score: score = composite_score(df[col1], df[col2], ...)\n"
            "f. sort_values('score', ascending=False) และ print Top 10\n"
            "g. print สรุปรายจังหวัดแต่ละ domain แยกกัน\n\n"
            "ห่อโค้ดทั้งหมดใน ```python ... ```"
        ),
        f"Python code merging {n} DataFrames with labeled output and real province names",
        step="code_gen", domain=domain_names_en, session_id=session_id,
    )
    put({"type": "agent_done", "step": "code_gen",
         "agentName": "Multi-DataFrame Code Generator", "result": code_result})

    # ── STEP 5: Python Executor ───────────────────────────────────────────────
    put({"type": "agent_start", "step": "executor", "agentName": "Python Executor"})

    code = _extract_code(code_result)

    # Guard: abort execution when code gen failed (e.g. 403 API key error)
    if _is_agent_error(code):
        auth_hint = " (API key ถูก report ว่า leaked — กรุณาสร้าง key ใหม่)" if _is_auth_error(code_result) else ""
        exec_output = f"[ข้ามการรัน — code generation ล้มเหลว{auth_hint}]\n{code_result}"
        code = ""
    else:
        # Multi-file pipeline: use longer timeout (5 files × network I/O)
        exec_output = exec_python(code, timeout=180)
        _log_exec_error(exec_output, code, "executor", domain_names_en, session_id, attempt=0)

        # Retry once on runtime error — pass geo_keys explicitly
        if _is_exec_error(exec_output):
            retry_result = _run_agent(
                generator,
                (
                    f"คำถาม: {prompt}\n"
                    f"Schemas:\n{schema_summary}\n"
                    f"Geographic Keys:\n{merge_recipe}\n\n"
                    f"โค้ดเดิมที่มี error:\n```python\n{code}\n```\n"
                    f"Error:\n{exec_output}\n\n"
                    "แก้ไขโค้ด:\n"
                    f"1. โหลดข้อมูล:\n{load_block}\n"
                    "2. ตรวจชื่อ column ให้ตรงกับ schema\n"
                    "3. ถ้า KeyError → ใช้ชื่อ column ที่ถูกต้องจาก schema\n"
                    "4. ถ้า merge error → ใช้ left_on/right_on แทน on=\n"
                    "5. ถ้า column หายไป → ข้าม column นั้น อย่า crash\n"
                    "ห่อโค้ดใน ```python ... ```"
                ),
                "Fixed Python code without errors",
                step="code_gen_retry", domain=domain_names_en, session_id=session_id,
            )
            retry_code = _extract_code(retry_result)
            if not _is_agent_error(retry_code):
                retry_output = exec_python(retry_code, timeout=180)
                _log_exec_error(retry_output, retry_code, "executor_retry", domain_names_en, session_id, attempt=1)
                if not _is_exec_error(retry_output) or len(retry_output) > len(exec_output):
                    code, exec_output, code_result = retry_code, retry_output, retry_result

    put({
        "type": "agent_done",
        "step": "executor",
        "agentName": "Python Executor",
        "code": code,
        "result": exec_output,
    })

    # ── STEP 6: Cross-Domain Insight Analyst ─────────────────────────────────
    put({"type": "agent_start", "step": "insight", "agentName": "Cross-Domain Insight Analyst"})

    insight_agent = Agent(
        role="Cross-Domain Insight Analyst",
        goal=(
            f"วิเคราะห์ผลลัพธ์จริงจากข้อมูล {domain_names_th} "
            "เพื่อระบุ Red Zone และ pattern ข้ามหลาย domain โดยใช้ข้อมูลจริงเท่านั้น"
        ),
        backstory=(
            "คุณเป็นนักวิเคราะห์ข้อมูลสาธารณสุขระดับเขตที่เชี่ยวชาญการมองภาพรวมข้ามหลายมิติ "
            "คุณเชื่อมโยงข้อมูลจากหลาย domain เพื่อหา pattern และ 'วงจร' ที่ซ่อนอยู่ "
            "คุณรายงานเฉพาะข้อมูลจริงจาก Execution Result — ไม่สร้างตัวเลขหรือชื่อจังหวัดสมมติ "
            "ถ้าข้อมูลไม่พอ คุณบอกตรงๆ และอธิบายสิ่งที่ทราบได้จากข้อมูลที่มี"
        ),
        llm=llm,
        verbose=False,
        max_iter=5,
    )

    insight = _run_agent(
        insight_agent,
        (
            f"คำถาม: {prompt}\n"
            f"Domains: {domain_names_th}\n"
            f"ไฟล์ที่ใช้:\n{file_summary}\n\n"
            f"ผลการรันโค้ด (Execution Result):\n{exec_output}\n\n"
            "==== กฎเหล็ก — ห้ามละเมิด ====\n"
            "1. ใช้เฉพาะข้อมูลจาก Execution Result ด้านบน\n"
            "2. ห้ามสร้างชื่อจังหวัดสมมติ เช่น 'จังหวัด ก.' 'จังหวัด ข.' 'Province A' — ต้องใช้ชื่อจริงเท่านั้น\n"
            "3. ห้ามสร้างตัวเลข composite score หรือ % ที่ไม่มีในผลลัพธ์\n"
            "4. ถ้า Execution มี error → อธิบาย error + สรุปจากข้อมูลบางส่วนที่ได้ ไม่ต้องสร้างตารางสมมติ\n"
            "5. ถ้าไม่มีชื่อจังหวัดในผลลัพธ์ → ระบุว่า 'ข้อมูลจากการรันโค้ดไม่ระบุจังหวัดเฉพาะ'\n\n"
            "==== โครงสร้างรายงาน ====\n\n"
            "## สรุปภาพรวม\n"
            "อธิบายสิ่งที่พบจาก Execution Result (2-4 ประโยค)\n\n"
            "## ตาราง Red Zone / พื้นที่เสี่ยง\n"
            "ตาราง markdown จากข้อมูลจริงในผลลัพธ์:\n"
            "| จังหวัด | Composite Score | [metric1] | [metric2] | ... |\n"
            "|---|---|---|---|---|\n"
            "| [ชื่อจริงจาก output] | [ค่าจริง] | ... |\n"
            "หมายเหตุ: ถ้า output ไม่มี composite score ให้ใช้ค่าที่มีในผลลัพธ์แทน\n\n"
            "## Pattern / วงจรที่พบ\n"
            "อธิบาย pattern ที่เห็นจากข้อมูล: เช่น พื้นที่ที่มีปัญหาซ้ำซ้อนหลาย domain\n\n"
            "## Red Zone ที่น่าเป็นห่วงที่สุด\n"
            "ระบุชื่อจังหวัด/พื้นที่จริงที่ควรได้รับความสนใจ พร้อมตัวเลขอ้างอิงจากผลลัพธ์\n\n"
            "## ข้อเสนอแนะเชิงนโยบาย\n"
            "มาตรการที่เหมาะสมสำหรับพื้นที่ Red Zone ที่ระบุ"
        ),
        "รายงาน insight ภาษาไทยพร้อมตาราง Red Zone จากข้อมูลจริง ชื่อจังหวัดจริง และข้อเสนอแนะ",
        step="insight", domain=domain_names_en, session_id=session_id,
    )
    put({"type": "agent_done", "step": "insight",
         "agentName": "Cross-Domain Insight Analyst", "result": insight})

    if session_id:
        append_history(session_id, "ai", insight)

    put({
        "type": "final",
        "message": insight,
        "domain": {"code": "multi", "nameTh": domain_names_th, "nameEn": domain_names_en},
        "agentSteps": [
            {"step": "router",      "agentName": "Multi-Domain Router",            "result": f"Domains: {domain_names_th}"},
            {"step": "reasoning",   "agentName": "Reasoning Narrator",             "result": reasoning},
            {"step": "file_finder", "agentName": "Multi-File Finder Agent",        "result": file_summary},
            {"step": "geo_keys",    "agentName": "Geographic Key Detector",        "result": merge_recipe},
            {"step": "schema",      "agentName": "Multi-Schema Analyst",           "result": schema_summary},
            {"step": "code_gen",    "agentName": "Multi-DataFrame Code Generator", "result": code_result, "code": code},
            {"step": "executor",    "agentName": "Python Executor",                "result": exec_output, "code": code},
            {"step": "insight",     "agentName": "Cross-Domain Insight Analyst",   "result": insight},
        ],
    })
