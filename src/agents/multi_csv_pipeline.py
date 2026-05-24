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
    _sanitize_generated_code,
    _age_scope_repair_hints,
    _strip_csv_extension_mentions,
    _is_agent_error, _is_auth_error,
    _is_exec_error, _log_exec_error,
    _find_code_issues,
)
from src.agents.prompt_profile import (
    ANALYST_CORE_POLICY,
    CODE_GENERATOR_CORE_POLICY,
    INSIGHT_RESPONSE_BLUEPRINT,
    MISSING_DATA_POLICY,
    join_prompt,
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
    target_ages = _extract_age_ranges(prompt)

    def score(line: str) -> int:
        ll = line.lower()
        base = sum(1 for w in words if len(w) > 2 and w in ll)

        if not target_ages:
            return base

        line_ages = _extract_age_ranges(line)
        if not line_ages:
            return base

        age_bonus = 0
        for target in target_ages:
            best = 0
            for found in line_ages:
                overlap = _range_overlap(target, found)
                if overlap > 0:
                    best = max(best, 20 + overlap)
                else:
                    dist = _range_distance(target, found)
                    best = max(best, max(1, 10 - dist))
            age_bonus += best

        return base + age_bonus

    return sorted(lines, key=score, reverse=True)[:max_n]


def _parse_file_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.split("\n") if ln.strip() and "[ID:" in ln]


def _extract_top_n(prompt: str, default_n: int = 10) -> int:
    m = re.search(r"(?:top|อันดับ)\s*(\d{1,2})", prompt.lower())
    if m:
        try:
            n = int(m.group(1))
            return max(3, min(n, 30))
        except Exception:
            return default_n
    return default_n


def _extract_age_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for a, b in re.findall(r"(?<!\d)(\d{1,2})\s*[-–]\s*(\d{1,2})(?!\d)", text):
        lo = min(int(a), int(b))
        hi = max(int(a), int(b))
        ranges.append((lo, hi))

    deduped: list[tuple[int, int]] = []
    for age_range in ranges:
        if age_range not in deduped:
            deduped.append(age_range)
    return deduped


def _range_overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return max(0, hi - lo + 1)


def _range_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    if _range_overlap(a, b) > 0:
        return 0
    if a[1] < b[0]:
        return b[0] - a[1]
    return a[0] - b[1]


def _infer_focus(prompt: str) -> dict[str, Any]:
    p = prompt.lower()
    intents: list[str] = []

    if any(k in p for k in ["เปรียบเทียบ", "เทียบ", "compare", "vs"]):
        intents.append("comparison")
    if any(k in p for k in ["แนวโน้ม", "trend", "ย้อนหลัง", "รายปี", "ช่วงเวลา"]):
        intents.append("trend")
    if any(k in p for k in ["red zone", "พื้นที่เสี่ยง", "เสี่ยงสูง", "จุดเสี่ยง"]):
        intents.append("red_zone")
    if any(k in p for k in ["อันดับ", "top", "สูงสุด", "ต่ำสุด"]):
        intents.append("ranking")
    if any(k in p for k in ["ความสัมพันธ์", "สัมพันธ์", "correlation"]):
        intents.append("relationship")

    if not intents:
        intents.append("general")

    years = re.findall(r"\b(20\d{2}|25\d{2})\b", prompt)
    provinces = re.findall(r"จังหวัด\s*([\wก-๙.-]+)", prompt)
    districts = re.findall(r"(?:อำเภอ|เขต)\s*([\wก-๙.-]+)", prompt)
    age_ranges = _extract_age_ranges(prompt)

    return {
        "intents": intents,
        "top_n": _extract_top_n(prompt),
        "years": list(dict.fromkeys(years)),
        "provinces": list(dict.fromkeys(provinces)),
        "districts": list(dict.fromkeys(districts)),
        "age_ranges": age_ranges,
    }


def _focus_brief(focus: dict[str, Any]) -> str:
    age_ranges = focus.get("age_ranges", [])
    age_text = [f"{lo}-{hi}" for lo, hi in age_ranges] if age_ranges else "not specified"
    return "\n".join([
        f"- intent: {', '.join(focus.get('intents', []))}",
        f"- top_n: {focus.get('top_n', 10)}",
        f"- years: {focus.get('years', []) or 'not specified'}",
        f"- provinces: {focus.get('provinces', []) or 'not specified'}",
        f"- districts: {focus.get('districts', []) or 'not specified'}",
        f"- age_ranges: {age_text}",
    ])


def _build_column_hints(prompt: str, schemas_info: list[dict]) -> str:
    words = [w for w in re.sub(r"[^\wก-๙\s]", " ", prompt.lower()).split() if len(w) > 1]
    if not words:
        return "- no keyword hints"

    hints: list[str] = []
    for info in schemas_info:
        cols: list[str] = info.get("cols", [])
        scored: list[tuple[int, str]] = []
        for col in cols:
            col_l = str(col).lower()
            score = sum(1 for w in words if w in col_l or col_l in w)
            if score > 0:
                scored.append((score, col))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_cols = [c for _, c in scored[:6]]
        if top_cols:
            hints.append(f"- df{info.get('index')}: {top_cols}")

    return "\n".join(hints) if hints else "- no strongly matched columns"


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
    focus = _infer_focus(prompt)

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
    focus_spec = _focus_brief(focus)
    column_hints = _build_column_hints(prompt, schemas_info)

    generator = Agent(
        role="Multi-DataFrame Python Code Generator",
        goal=(
            f"สร้าง Python/Pandas code ที่ merge {n} DataFrame และ print ผลลัพธ์ชัดเจน "
            "ให้ตอบตาม Target Spec ของคำถามด้วยชื่อพื้นที่จริงและตัวเลขที่ตรวจสอบได้"
        ),
        backstory=join_prompt(
            "คุณเป็น Python/Pandas expert ที่เชี่ยวชาญการวิเคราะห์ข้อมูลสาธารณสุขจากหลาย dataset "
            "คุณใช้ pct_rank() และ composite_score() ที่มีให้อยู่แล้ว "
            "คุณให้ความสำคัญกับ output ที่ชัดเจน: ชื่อจังหวัดต้องแสดงครบ ตัวเลขมี label "
            "เพื่อให้ AI วิเคราะห์ต่อได้โดยไม่ต้องสร้างข้อมูลขึ้นมาเอง",
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
            f"Domains: {domain_names_th}\n\n"
            f"Schemas:\n{schema_summary}\n\n"
            f"Geographic Keys (ตรวจพบอัตโนมัติ):\n{merge_recipe}\n\n"
            f"Target Spec จากคำถาม (ต้องตอบตามนี้):\n{focus_spec}\n\n"
            f"Candidate columns ที่น่าจะตรงโจทย์:\n{column_hints}\n\n"
            "==== กฎบังคับ (ห้ามละเมิด) ====\n"
            f"1. โหลดข้อมูล (ห้ามเปลี่ยน file_id):\n{load_block}\n\n"
            "2. ห้าม import minio / redefine load_csv / redefine pct_rank / redefine composite_score\n"
            "3. ห้ามใช้ pd.read_csv() — ใช้ load_csv() เท่านั้น\n"
            "4. ใช้ชื่อ column จาก schema เท่านั้น — ห้ามเดาชื่อ column\n\n"
            "4.1 ต้องเริ่มจากระบุคอลัมน์ที่เลือกใช้จริงจาก Candidate columns\n"
            "4.2 ถ้าไม่พบคอลัมน์ที่ตรงโจทย์ ให้ fallback เป็นคอลัมน์ที่ใกล้เคียงที่สุดและพิมพ์เหตุผล\n\n"
            "4.3 ถ้าโจทย์ระบุช่วงอายุ (เช่น 12-18) แต่ไม่มีคอลัมน์ตรงตัว ให้คำนวณประมาณการจากช่วงใกล้เคียงในไฟล์เดียวกัน\n"
            "     - ถ้ามี 2 ช่วงที่คร่อมช่วงเป้าหมาย ให้ใช้ interpolation ด้วย midpoint\n"
            "     - ถ้ามีได้แค่ 1 ช่วง ให้ใช้ nearest-neighbor proxy\n"
            "     - ต้อง print ว่าเป็น ESTIMATE และบอกช่วงอายุที่ใช้คำนวณ\n\n"
            "4.4 ถ้าคำถามระบุปี/จังหวัด/อำเภอ/ช่วงอายุ ต้อง filter ให้ตรงก่อน aggregate/merge\n"
            "     - ห้ามใช้ปีนอกช่วงที่ผู้ใช้ถามในตารางผลลัพธ์หลัก\n"
            "     - ต้อง print section '=== SCOPE CHECK ===' ระบุช่วงที่ถามและช่วงที่ใช้จริง\n\n"
            "==== Output Format บังคับ ====\n"
            "5. บรรทัดหลัง load: pd.set_option('display.max_rows', 100)\n"
            "6. ก่อน print ทุก section ใส่หัวข้อ เช่น print('=== [หัวข้อ] ===')\n"
            "7. ชื่อจังหวัด/พื้นที่ต้องแสดงเป็น text ครบในทุกตาราง\n"
            "8. ใช้ print(df.to_string(index=False)) เพื่อแสดง DataFrame ครบ\n\n"
            "==== ขั้นตอนการวิเคราะห์ ====\n"
            "a0. เริ่มด้วย print('=== Analysis Plan ===') และพิมพ์สิ่งที่จะหาให้ตรง Target Spec\n"
            "a. โหลด + strip whitespace จาก geo column ทุกตัว\n"
            "b. Rename geo columns ตาม merge recipe\n"
            "c. Aggregate แต่ละ df รายจังหวัด (groupby) ก่อน merge\n"
            "d. Merge ด้วย outer join บน geo key\n"
            "e. composite_score: score = composite_score(df[col1], df[col2], ...)\n"
            "f. ถ้า intent มี ranking/red_zone ให้ sort_values('score', ascending=False) และ print Top N จาก top_n\n"
            "g. ถ้า intent มี trend ให้แสดงแนวโน้มรายปี (เมื่อมีคอลัมน์ปี)\n"
            "h. ถ้า intent มี comparison ให้สร้างตารางเปรียบเทียบตัวชี้วัดหลัก\n"
            "i. print สรุปรายจังหวัดแต่ละ domain แยกกัน\n"
            "j. ถ้าใช้ค่าประมาณ ให้มี section '=== ESTIMATION METHOD ===' อธิบายสูตรและคอลัมน์ที่ใช้\n"
            "k. ก่อน print ตารางหลัก ให้ rename คอลัมน์เป็นชื่อภาษาไทยอ่านได้สมบูรณ์:\n"
            "   - ตัวอย่าง: 'เริ่มอ้วน_%_12-18' → 'ร้อยละเด็กเริ่มอ้วน ช่วง 12-18 ปี (ประมาณ)'\n"
            "   - ตัวอย่าง: 'อ้วน_%_6-14' → 'ร้อยละเด็กอ้วน ช่วง 6-14 ปี'\n"
            "   - ใช้ df_display = df_result.rename(columns={...}) แล้ว print df_display\n"
            "l. round ตัวเลขทศนิยมทั้งหมดเป็น 2 ตำแหน่งก่อน print: df_display = df_display.round(2)\n\n"
            "ห่อโค้ดทั้งหมดใน ```python ... ```\n\n"
            f"{CODE_GENERATOR_CORE_POLICY}"
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
        required_lines = [f"df{i} = load_csv('{fid}')" for i, (fid, _) in enumerate(selected_files, 1)]
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
                    f"Schemas:\n{schema_summary}\n\n"
                    f"Target Spec:\n{focus_spec}\n\n"
                    f"โค้ดปัจจุบัน:\n```python\n{code}\n```\n\n"
                    f"Contract violations:\n{chr(10).join(f'- {i}' for i in code_issues)}\n\n"
                    f"{age_scope_hints}\n"
                    "แก้โค้ดให้ผ่านกฎ:\n"
                    f"1. ต้องมีบรรทัดโหลดไฟล์ครบดังนี้:\n{load_block}\n"
                    "2. ห้าม import/use Minio โดยตรง\n"
                    "3. ห้ามใช้ pd.read_csv\n"
                    "4. ห้าม redefine helpers\n"
                    "5. รักษา logic ตาม Target Spec\n"
                    "6. ต้องมี '=== SCOPE CHECK ===' และยืนยันช่วงปี/พื้นที่/ช่วงอายุที่ถาม\n"
                    "7. ถ้าไม่มีคอลัมน์อายุตรงเป้า ให้คำนวณ estimate และติดป้ายช่วงอายุเป้าหมาย\n"
                    "Wrap code in ```python ... ```"
                ),
                "Repaired Python code that passes contract checks",
                step="code_contract_repair", domain=domain_names_en, session_id=session_id,
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
            # Multi-file pipeline: use longer timeout (5 files × network I/O)
            exec_output = exec_python(code, timeout=180)
            _log_exec_error(exec_output, code, "executor", domain_names_en, session_id, attempt=0)
        # Retry once on runtime error — pass geo_keys explicitly
        if code and _is_exec_error(exec_output):
            retry_result = _run_agent(
                generator,
                (
                    f"คำถาม: {prompt}\n"
                    f"Schemas:\n{schema_summary}\n"
                    f"Geographic Keys:\n{merge_recipe}\n\n"
                    f"Target Spec:\n{focus_spec}\n\n"
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
            retry_code = _sanitize_generated_code(_extract_code(retry_result), required_lines, prompt)
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
            "และเรียบเรียงรายงานระดับทางการสำหรับผู้บริหาร สสจ. ที่อ่านแล้วตัดสินใจเชิงนโยบายได้ทันที"
        ),
        backstory=join_prompt(
            "คุณเป็นนักวิเคราะห์ข้อมูลสาธารณสุขระดับเขตที่มั่นใจในการวิเคราะห์และตอบตรงประเด็น "
            "คุณเริ่มรายงานด้วยสิ่งที่ค้นพบจากข้อมูลเสมอ ไม่ขึ้นต้นด้วยข้อแก้ตัวหรือข้อจำกัด "
            "ถ้าใช้ค่าประมาณ คุณระบุทันทีในตารางว่า 'ค่าประมาณจาก X' แล้ววิเคราะห์ต่อได้เลย "
            "คุณใช้เฉพาะข้อมูลจาก Execution Result และไม่สร้างตัวเลขหรือชื่อสมมติ",
            ANALYST_CORE_POLICY,
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
            f"Target Spec ที่ต้องตอบให้ครบ:\n{focus_spec}\n\n"
            f"ผลการรันโค้ด (Execution Result):\n{exec_output}\n\n"
            "==== กฎเหล็ก — ห้ามละเมิด ====\n"
            "1. ใช้เฉพาะข้อมูลจาก Execution Result ด้านบน\n"
            "2. ห้ามสร้างชื่อจังหวัดสมมติ เช่น 'จังหวัด ก.' 'จังหวัด ข.' 'Province A' — ต้องใช้ชื่อจริงเท่านั้น\n"
            "3. ห้ามสร้างตัวเลข composite score หรือ % ที่ไม่มีในผลลัพธ์\n"
            "4. ถ้า Execution มี error → อธิบาย error + สรุปจากข้อมูลบางส่วนที่ได้ ไม่ต้องสร้างตารางสมมติ\n"
            "5. ถ้าไม่มีชื่อจังหวัดในผลลัพธ์ → ระบุว่า 'ข้อมูลจากการรันโค้ดไม่ระบุจังหวัดเฉพาะ'\n"
            "6. ถ้าผลลัพธ์เป็นค่าประมาณ (ESTIMATE/PROXY) ให้ติดป้าย *(ประมาณ)* ในหัวคอลัมน์ตารางทันที แล้ววิเคราะห์ต่อได้เลย ห้ามสร้างย่อหน้าแยกต่างหากก่อนตาราง\n"
            "7. ระบุให้ครบว่าใช้ข้อมูลจากไฟล์/ชุดข้อมูลใดบ้าง และใช้ช่วงปีใดบ้าง ในหัวข้อ ## แหล่งข้อมูล\n"
            "8. อธิบายวิธีคำนวณใน ## วิธีคำนวณ (1-3 บรรทัดก็พอ)\n"
            "9. อธิบายความหมายคอลัมน์ใต้ตาราง (1-2 บรรทัด)\n"
            "10. ถ้าครอบคลุมไม่ครบ ให้ระบุในส่วน ## ข้อจำกัด เท่านั้น ไม่นำมาขึ้นต้นรายงาน\n\n"
            "==== แนวทางการเขียนรายงาน (สำคัญ) ====\n"
            "คนอ่านคือผู้อำนวยการ สสจ. — ต้องการรายงานทางการที่อ่านแล้วใช้ตัดสินใจได้ทันที จึงเขียนระดับ:\n"
            "- สรุปผู้บริหาร: ย่อหน้า 3-5 ประโยคภาษาทางการ เริ่มด้วยตัวเลขหรือ pattern ที่พบทันที ห้ามขึ้นต้น 'เนื่องจากไม่มีข้อมูล'\n"
            "- ใช้โครงสร้างจาก INSIGHT_RESPONSE_BLUEPRINT เพียงชุดเดียว ห้ามสร้างหัวข้อซ้ำ\n"
            "- ชื่อคอลัมน์ในตารางต้องเป็นภาษาไทยอ่านได้สมบูรณ์ เช่น 'ร้อยละเด็กเริ่มอ้วน ช่วง 12-18 ปี (ประมาณ)' ไม่ใช่ชื่อตัวแปรดิบ\n"
            "- ตัวเลขในตารางแสดง 2 ทศนิยม พร้อมหน่วย (%) ถ้าเป็นร้อยละ\n"
            "- วิธีคำนวณ: ถ้ามีสูตรให้ใช้ LaTeX block วางบรรทัดเดียวโดดๆ เสมอ เช่น:\n\n$$\\hat{{v}} = v_1 + \\frac{{(v_2 - v_1) \\times (t - t_1)}}{{t_2 - t_1}}$$\n\n(ปรับตัวแปรตามจริง) — ห้ามวาง $$...$$ กลางประโยคหรือท้ายบรรทัดธรรมดา\n"
            "- Insight และข้อเสนอแนะเขียนเป็นย่อหน้าสมบูรณ์ มีตัวเลขอ้างอิง ระบุหน่วยงานที่เกี่ยวข้อง\n"
            "- ข้อจำกัดเป็นส่วนสุดท้าย ระบุสั้นๆ ไม่ใช่หัวข้อหลัก\n\n"
            f"{INSIGHT_RESPONSE_BLUEPRINT}\n\n"
            f"{MISSING_DATA_POLICY}"
        ),
        "รายงานทางการภาษาไทยสำหรับผู้บริหาร สสจ. พร้อมตารางชื่อคอลัมน์อ่านได้ สูตร LaTeX และข้อเสนอแนะเชิงนโยบาย",
        step="insight", domain=domain_names_en, session_id=session_id,
    )
    insight = _strip_csv_extension_mentions(insight)
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
