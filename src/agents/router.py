"""Router Agent — classifies user questions into health domains d0–d4, dt, obsidian."""
import os
import re

from crewai import Agent, Crew, LLM, Task

from src.domains import Domain, DOMAINS, DOMAIN_LIST_TEXT

# Domains that have CSV files in MinIO (d2–d4 only; d0=general, d1=PostgreSQL)
_CSV_DOMAIN_CODES = {"d2", "d3", "d4"}

# ── ThaiJo keyword detection ──────────────────────────────────────────────────

_THAIJO_KEYWORDS = [
    r"thaijo", r"thai\s*jo",
    r"บทความวิจัย", r"งานวิจัย", r"literature\s*review",
    r"สังเคราะห์งานวิจัย", r"วรรณกรรม",
    r"สร้าง\s*journal", r"สร้าง\s*report\s*วิจัย",
    r"ค้นหาบทความ", r"journal\s*report",
]

_OBSIDIAN_KEYWORDS = [
    r"obsidian", r"คลังความรู้", r"knowledge\s*vault",
    r"เขตสุขภาพ", r"เขต\s*(?:ที่\s*)?10",
    r"อุบลราชธานี", r"ศรีสะเกษ", r"ยโสธร", r"อำนาจเจริญ", r"มุกดาหาร",
    r"สสจ\.?", r"สาธารณสุขจังหวัด",
    r"นโยบาย.*สุขภาพ.*จังหวัด", r"รายงาน.*สสจ",
    r"vault",
]

_ACCIDENT_KEYWORDS = [
    r"อุบัติเหตุ", r"อุบัติเหตุทางถนน", r"จราจร", r"รถชน", r"ชนแล้ว", r"ชนกัน",
    r"เสียชีวิต.*อุบัติเหตุ", r"บาดเจ็บ.*อุบัติเหตุ", r"ผู้บาดเจ็บ", r"ผู้เสียชีวิต",
    r"rti", r"road\s*traffic", r"traffic\s*injur", r"accident",
    r"มอเตอร์ไซค์", r"รถจักรยานยนต์",
]


def _has_thaijo_signal(prompt: str) -> bool:
    """True when the query is asking for a ThaiJo research journal."""
    p = prompt.lower()
    return any(re.search(kw, p) for kw in _THAIJO_KEYWORDS)


def _has_obsidian_signal(prompt: str) -> bool:
    """True when the query is about Obsidian Knowledge Vault / Health Region 10."""
    p = prompt.lower()
    return any(re.search(kw, p) for kw in _OBSIDIAN_KEYWORDS)


def _has_accident_signal(prompt: str) -> bool:
    """True when the query is clearly about road accidents / injuries."""
    p = prompt.lower()
    return any(re.search(kw, p) for kw in _ACCIDENT_KEYWORDS)


# ── Data-analysis intent screening ───────────────────────────────────────────
# Signals that indicate the user wants CSV-based statistical analysis.
# If NONE of these appear, the question is general knowledge → route d0.
_DATA_ANALYSIS_SIGNALS = [
    # ข้อมูล/สถิติ
    r"ข้อมูล", r"สถิติ", r"ตัวเลข", r"ตัวชี้วัด",
    # อัตรา/ร้อยละ
    r"อัตรา", r"ร้อยละ", r"จำนวน.*ราย", r"จำนวน.*คน", r"จำนวน.*ครั้ง",
    # ภูมิศาสตร์ → ต้องการ CSV จังหวัด
    r"จังหวัด", r"อำเภอ", r"ภาค", r"ทั่วประเทศ", r"จาก\s*\d+\s*จังหวัด",
    # เวลา → แนวโน้ม CSV
    r"ปี\s*(?:พ\.ศ\.|25\d\d)", r"แนวโน้ม", r"เพิ่มขึ้น", r"ลดลง", r"ย้อนหลัง",
    # การวิเคราะห์
    r"วิเคราะห์", r"เปรียบเทียบ", r"จัดอันดับ", r"ranking", r"top\s*\d",
    r"สูงสุด\s*\d", r"ต่ำสุด\s*\d",
    # กราฟ/ชาร์ต
    r"กราฟ", r"ชาร์ต", r"chart", r"graph", r"plot",
    # ระบาดวิทยา
    r"ความชุก", r"อุบัติการณ์", r"ระบาดวิทยา", r"prevalence", r"incidence",
    # Red zone / hot spot
    r"red\s*zone", r"hot\s*spot", r"พื้นที่เสี่ยง",
    # รายงาน
    r"รายงาน(?:สรุป|ผล|ข้อมูล)", r"summary.*data", r"data.*report",
]


def _has_data_analysis_signal(prompt: str) -> bool:
    """True when the prompt is asking for CSV-based data/statistics analysis.

    Returns False for general knowledge/advice questions (วิธี, แนะนำ, ปรึกษา, etc.)
    that should route to d0 without touching any CSV domain.
    """
    p = prompt.lower()
    return any(re.search(sig, p) for sig in _DATA_ANALYSIS_SIGNALS)


# ── Step 3: Keyword-based multi-domain detection ──────────────────────────────

_MULTI_KEYWORDS = [
    r"red\s*zone",
    r"วงจร",
    r"หลายกลุ่ม",
    r"ข้ามกลุ่ม",
    r"ทุกช่วงวัย",
    r"หลายมิติ",
    r"ความสัมพันธ์ระหว่าง",
    r"เปรียบเทียบ.{1,20}และ",
    r"วัยเรียน.{1,30}วัยทำงาน",
    r"วัยทำงาน.{1,30}วัยเรียน",
    r"เด็ก.{1,20}ผู้ใหญ่",
    r"อ้วน.{1,20}ncd",
    r"ncd.{1,20}อ้วน",
    r"โรค.{1,20}โภชนาการ",
    r"โภชนาการ.{1,20}โรค",
    r"multi.?domain",
    r"ข้ามสาขา",
    r"ซ้อนทับ",
]

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "d1": ["อุบัติเหตุ", "อุบัติเหตุทางถนน", "รถชน", "จราจร", "rti", "accident", "traffic injury"],
    "d2": ["สุขภาพจิต", "ฆ่าตัวตาย", "ซึมเศร้า", "จิตเวช", "mental"],
    "d3": ["ncd", "เบาหวาน", "ความดัน", "หัวใจ", "หลอดเลือด", "โรคไม่ติดต่อ", "ncds"],
    "d4": ["โภชนาการ", "อ้วน", "bmi", "วัยเรียน", "วัยทำงาน", "ภาวะโภชนาการ", "น้ำหนัก", "nutrition"],
}


def _has_multi_signal(prompt: str) -> bool:
    """Fast keyword check — True means the question almost certainly spans multiple domains."""
    p = prompt.lower()
    return any(re.search(kw, p) for kw in _MULTI_KEYWORDS)


def _keyword_infer_domains(prompt: str) -> list[str]:
    """Infer domain codes from keywords — used as fallback when LLM returns only 1 domain."""
    p = prompt.lower()
    return [code for code, kws in _DOMAIN_KEYWORDS.items() if any(kw in p for kw in kws)]


def _get_llm() -> LLM:
    return LLM(model="gemini/gemini-2.5-flash-lite", api_key=os.getenv("GEMINI_API_KEY"))


def is_accident_question(prompt: str, history_context: str = "") -> bool:
    """True ถ้าคำถามเกี่ยวกับ "อุบัติเหตุ/ความปลอดภัยทางถนน" (d1 → SQL) ไม่ใช่โรค/สุขภาพทั่วไป.

    ใช้ในโหมด stats เป็น "LLM นำ" เพื่อตัดสิน d1 (PostgreSQL) vs CSV (d2/d3/d4)
    ก่อนจะ default ไป CSV เสมอ — keyword (_has_accident_signal) เป็น safety-net:
    ถ้า keyword เจอชัด ๆ ก็ True ทันทีไม่ต้องเรียก LLM (เร็ว+ฟรี); ถ้า keyword miss
    จึงให้ LLM ตัดสิน เพื่อจับคำที่ keyword list ครอบไม่ถึง เช่น
    "คนตายบนถนนเยอะไหม", "สถิติคนเจ็บจากการขับขี่", "ความปลอดภัยทางถนนอุบล".
    """
    # Safety-net: keyword ชัดเจน → ไม่ต้องเสีย latency เรียก LLM
    if _has_accident_signal(prompt):
        return True

    classifier = Agent(
        role="Accident Intent Classifier",
        goal="ตัดสินว่าคำถามเกี่ยวกับอุบัติเหตุ/ความปลอดภัยทางถนน หรือเรื่องโรค/สุขภาพทั่วไป",
        backstory=(
            "คุณเชี่ยวชาญการจำแนกคำถามสุขภาพ โดยเฉพาะการแยกว่าเรื่องไหนเกี่ยวกับ "
            "'อุบัติเหตุทางถนน' (เช่น คนเจ็บ/ตายบนถนน รถชน จราจร การขับขี่ ความปลอดภัยทางถนน "
            "ผู้บาดเจ็บ/เสียชีวิตจากการเดินทาง) กับเรื่อง 'โรค/สุขภาพทั่วไป' "
            "(เช่น เบาหวาน ความดัน สุขภาพจิต โภชนาการ). คุณตอบเพียงคำเดียว ห้ามอธิบายเพิ่ม"
        ),
        llm=_get_llm(),
        verbose=False,
        max_iter=2,
    )
    history_section = f"{history_context}\n\n" if history_context else ""
    task = Task(
        description=(
            f"{history_section}"
            f"คำถาม: {prompt}\n\n"
            "คำถามนี้เกี่ยวกับอะไร ตอบเพียงคำเดียว:\n"
            "- accident → อุบัติเหตุ/ความปลอดภัยทางถนน/รถชน/ผู้บาดเจ็บ-เสียชีวิตจากการเดินทาง\n"
            "- health   → โรค/สุขภาพทั่วไป (เบาหวาน ความดัน สุขภาพจิต โภชนาการ ฯลฯ)"
        ),
        expected_output="คำเดียว: accident หรือ health",
        agent=classifier,
    )
    crew = Crew(agents=[classifier], tasks=[task], verbose=False)
    try:
        result = str(crew.kickoff()).strip().lower()
    except Exception:
        return False  # LLM ล้ม → ปล่อยให้ตกไป CSV ตาม flow เดิม
    return "accident" in result


def route_stats_domains(
    prompt: str, history_context: str = ""
) -> tuple[list[Domain], bool, str]:
    """LLM router สำหรับโหมดสถิติ — อ่าน context การสนทนา + ให้เหตุผล แล้วเลือก domain.

    ต่างจาก route_multi_domain ตรงที่ "ยึดความต่อเนื่องของบทสนทนา" เป็นหลัก:
    ถ้าคำถามล่าสุดเป็น follow-up (ขอแยกย่อย/เจาะจงพื้นที่-ปี-กลุ่ม เช่น "ขอแต่ละอำเภอ",
    "เฉพาะปี 2567") ให้ใช้ domain เดิมที่กำลังคุยอยู่ ไม่ re-classify ใหม่จนหลุดหัวข้อ
    (เดิม router มองคำถามล่าสุดอย่างเดียว → follow-up ที่หัวข้อหายไปถูกจัดผิด domain)

    Returns:
        (domains, is_multi, reasoning) — reasoning = "วิธีคิด" ของ AI เพื่อแสดงให้ผู้ใช้เห็น
    """
    history_section = f"{history_context}\n\n" if history_context else ""
    router = Agent(
        role="Stats Domain Router (context-aware)",
        goal="อ่านบริบทการสนทนาทั้งหมดแล้วเลือก domain ข้อมูลสถิติที่ถูกต้อง พร้อมอธิบายเหตุผล",
        backstory=(
            "คุณเป็น Router ที่เข้าใจ 'ความต่อเนื่องของบทสนทนา' — ไม่ได้ดูแค่คำถามล่าสุด "
            "แต่ดูว่าก่อนหน้านี้กำลังคุยเรื่อง domain ใดอยู่ คำถามตามหลังที่เป็นการขอ "
            "เจาะลึก/แยกย่อย/กรองพื้นที่หรือปี ถือเป็นเรื่องเดิม domain เดิมเสมอ"
        ),
        llm=_get_llm(),
        verbose=False,
        max_iter=3,
    )
    task = Task(
        description=(
            f"{history_section}"
            f"คำถามล่าสุด: {prompt}\n\n"
            "Domain ข้อมูลสถิติ (CSV) ที่มี:\n"
            "- d2 สุขภาพจิต — ฆ่าตัวตาย ซึมเศร้า จิตเวช\n"
            "- d3 โรคไม่ติดต่อ (NCDs) — เบาหวาน ความดัน หัวใจ หลอดเลือด\n"
            "- d4 โภชนาการ — ภาวะอ้วน BMI น้ำหนัก วัยเรียน วัยทำงาน ภาวะโภชนาการ\n\n"
            "กฎการตัดสิน:\n"
            "1. ถ้าคำถามล่าสุดเป็น 'follow-up' (เช่น 'ขอแต่ละอำเภอ', 'เฉพาะปี 2567', "
            "'แยกตามเพศ', 'ขอแค่จังหวัด...') → ใช้ domain เดิมจากประวัติการสนทนา ห้ามเปลี่ยน\n"
            "2. ถ้าเป็นหัวข้อใหม่ → เลือก domain ที่ตรงหัวข้อมากที่สุด (เลือกได้สูงสุด 3)\n\n"
            "ตอบ 2 บรรทัด:\n"
            "REASONING: <อธิบายวิธีคิดสั้น ๆ 1-2 ประโยค ว่าเป็นเรื่องเดิมต่อเนื่องหรือหัวข้อใหม่ "
            "และทำไมจึงเลือก domain นี้>\n"
            "DOMAINS: <รหัส domain คั่นด้วย comma เช่น d4 หรือ d3,d4>"
        ),
        expected_output="REASONING: ... \\nDOMAINS: d4",
        agent=router,
    )
    crew = Crew(agents=[router], tasks=[task], verbose=False)
    try:
        raw = str(crew.kickoff()).strip()
    except Exception:
        raw = ""

    # แยก reasoning + codes
    reasoning = ""
    m_reason = re.search(r"REASONING:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE | re.DOTALL)
    if m_reason:
        reasoning = m_reason.group(1).split("DOMAINS:")[0].strip()
    m_dom = re.search(r"DOMAINS:\s*(.+)", raw, re.IGNORECASE)
    code_src = m_dom.group(1) if m_dom else raw
    codes = list(dict.fromkeys(re.findall(r"\bd[2-4]\b", code_src.lower())))[:3]

    # Fallback: LLM ไม่คืน code → ใช้ keyword inference, สุดท้ายค่อย default d3
    if not codes:
        codes = [c for c in _keyword_infer_domains(prompt) if c in _CSV_DOMAIN_CODES]
    if not codes:
        codes = ["d3"]

    domains = [DOMAINS[c] for c in codes]
    is_multi = len(domains) >= 2
    if not reasoning:
        reasoning = f"เลือก {' + '.join(d.name_th for d in domains)} จากหัวข้อคำถาม"
    return domains, is_multi, reasoning


def route_domain(prompt: str, history_context: str = "") -> Domain:
    """Run the Router Agent and return the matched Domain.

    Fast-path: if ThaiJo keywords detected → return 'dt' immediately.
    """
    # Fast path: Road accidents (high-priority hard rule)
    # ⚠️ ต้องอยู่ "ก่อน" Obsidian เสมอ — ห้ามสลับ! เพราะ _OBSIDIAN_KEYWORDS มีชื่อ
    # จังหวัดเขต 10 เต็ม ๆ อยู่ (เช่น "อุบลราชธานี", "ศรีสะเกษ") ซึ่งคำถามอุบัติเหตุ
    # ก็มักระบุชื่อจังหวัดด้วยเสมอ (โดยเฉพาะหลัง Memory Agent ขยาย "อุบล" → "อุบลราชธานี"
    # ตอน resolve follow-up) ถ้าเช็ค Obsidian ก่อน คำถามอุบัติเหตุที่มีชื่อจังหวัดเต็ม
    # จะถูกชิงไปคลังความรู้ที่ไม่มีข้อมูลอุบัติเหตุเลย ทันที — ตรงกับบั๊กที่เจอจริง
    # (ถามอุบัติเหตุอุบล ตามด้วย "ขอข้อมูลแต่ละอำเภอ" แล้วหลุดไป Obsidian)
    if _has_accident_signal(prompt):
        return DOMAINS["d1"]

    # Fast path: Obsidian Knowledge Vault
    if _has_obsidian_signal(prompt):
        return DOMAINS["obsidian"]

    # Fast path: ThaiJo research journal
    if _has_thaijo_signal(prompt):
        return DOMAINS["dt"]

    # Fast path: no data/statistics signals → general knowledge, skip CSV domains
    if not _has_data_analysis_signal(prompt):
        return DOMAINS["d0"]

    router = Agent(
        role="Router Agent",
        goal="วิเคราะห์คำถามแล้วเลือก domain ที่เหมาะสมที่สุด",
        backstory=(
            "คุณเป็น Router Agent ที่เชี่ยวชาญการจำแนกคำถามสุขภาพว่าเกี่ยวกับ domain ใด "
            "คุณตอบเฉพาะรหัส domain เช่น d1, d2, dt เท่านั้น ห้ามตอบอย่างอื่น "
            "หากคำถามปัจจุบันเป็น follow-up ให้ใช้ domain เดิมจากประวัติการสนทนา"
        ),
        llm=_get_llm(),
        verbose=False,
        max_iter=3,
    )

    history_section = f"{history_context}\n\n" if history_context else ""
    task = Task(
        description=(
            f"{history_section}"
            f"คำถามล่าสุด: {prompt}\n\n"
            f"Domain ที่มี:\n{DOMAIN_LIST_TEXT}\n\n"
            "กฎสำคัญ: ถ้าคำถามมีสัญญาณเรื่องอุบัติเหตุทางถนน/รถชน/RTI/จราจร ให้เลือก d1 ทันที\n"
            "เลือก domain ที่เหมาะสมที่สุด 1 อัน ตอบเฉพาะรหัส เช่น d2 หรือ dt"
        ),
        expected_output="รหัส domain เช่น d2 หรือ dt",
        agent=router,
    )
    crew = Crew(agents=[router], tasks=[task], verbose=False)

    try:
        result = str(crew.kickoff()).strip()
    except Exception:
        result = "d0"

    # Match dt or d0-d4
    m = re.search(r'\b(obsidian|dt|d[0-4])\b', result.lower())
    code = m.group(1) if m else "d0"
    return DOMAINS.get(code, DOMAINS["d0"])


def route_multi_domain(prompt: str, history_context: str = "") -> tuple[list[Domain], bool]:
    """Detect if the question spans multiple CSV domains.

    Returns (domains, is_multi).  is_multi=True when ≥2 CSV domains are needed.
    Uses fast keyword check first; LLM refines the domain codes.
    """
    # Hard rule: accident questions should route to d1 pipeline, not CSV multi-domain
    # ⚠️ ต้องเช็คก่อน Obsidian เสมอ — ดูคำอธิบายเต็มในคอมเมนต์ของ route_domain() ด้านบน
    # (สรุปสั้น ๆ: ชื่อจังหวัดเขต 10 ที่อยู่ใน _OBSIDIAN_KEYWORDS ดันไปตรงกับชื่อจังหวัด
    # ที่คำถามอุบัติเหตุมักระบุด้วยพอดี โดยเฉพาะหลัง Memory Agent ขยายชื่อย่อให้เต็ม)
    if _has_accident_signal(prompt):
        return [DOMAINS["d1"]], False

    # Hard rule: obsidian knowledge vault
    if _has_obsidian_signal(prompt):
        return [DOMAINS["obsidian"]], False

    # Screen: no data/statistics signals → general knowledge, skip all CSV domains
    if not _has_data_analysis_signal(prompt):
        return [DOMAINS["d0"]], False

    force_multi = _has_multi_signal(prompt)

    router = Agent(
        role="Multi-Domain Router Agent",
        goal="วิเคราะห์คำถามและเลือก domain ที่เกี่ยวข้องทั้งหมด (สูงสุด 3 domain)",
        backstory=(
            "คุณเป็น Router Agent ที่เชี่ยวชาญในการจำแนกคำถามสุขภาพที่ซับซ้อน "
            "ซึ่งอาจต้องการข้อมูลจากหลาย domain พร้อมกัน เช่น 'วงจรความอ้วน' ต้องการทั้ง "
            "โภชนาการ (d4) และ NCDs (d3) 'Red Zone พื้นที่เสี่ยง' อาจต้องการ 2-3 domain "
            "คุณตอบเพียง domain codes คั่นด้วย comma ห้ามอธิบายเพิ่ม"
        ),
        llm=_get_llm(),
        verbose=False,
        max_iter=3,
    )

    history_section = f"{history_context}\n\n" if history_context else ""
    multi_hint = (
        "\n⚠️ คำถามนี้มีสัญญาณว่าต้องการข้อมูลจากหลาย domain — ให้เลือก ≥2 domain\n"
        if force_multi else ""
    )
    task = Task(
        description=(
            f"{history_section}"
            f"คำถาม: {prompt}\n"
            f"{multi_hint}\n"
            "กฎสำคัญ: ถ้าคำถามมีคำว่าอุบัติเหตุ/รถชน/RTI ให้ตอบ d1 เท่านั้น\n"
            "Domains ที่มีไฟล์ CSV:\n"
            "- d2: สุขภาพจิต — ฆ่าตัวตาย ซึมเศร้า จิตเวช\n"
            "- d3: โรคไม่ติดต่อ (NCDs) — เบาหวาน ความดัน หัวใจ\n"
            "- d4: โภชนาการ — ภาวะอ้วน BMI วัยเรียน วัยทำงาน\n"
            "- d0: ทั่วไป (ไม่มี CSV)\n"
            "- d1: อุบัติเหตุ (ใช้ฐานข้อมูล ไม่ใช่ CSV)\n\n"
            "เลือก domain codes ที่จำเป็น (สูงสุด 3 domain)\n"
            "ตอบเฉพาะ codes คั่น comma เช่น: d3,d4 หรือ d4 หรือ d2,d3,d4"
        ),
        expected_output="domain codes คั่นด้วย comma เช่น d3,d4",
        agent=router,
    )
    crew = Crew(agents=[router], tasks=[task], verbose=False)

    try:
        result = str(crew.kickoff()).strip().lower()
    except Exception:
        result = "d0"

    codes = list(dict.fromkeys(re.findall(r'\b(d[0-4])\b', result)))[:3]
    if not codes:
        codes = ["d0"]

    csv_codes = [c for c in codes if c in _CSV_DOMAIN_CODES]

    # Guard: LLM claimed multi-domain but no independent multi-domain signal exists —
    # cross-check against keyword evidence and drop any domain with zero support.
    # Prevents e.g. "เบาหวานและความดัน" (purely d3/NCDs) from being inflated into
    # "d2,d3" (+ Mental Health) by a router LLM hallucination, which then cascades
    # into the File Finder loading unrelated files (suicide/self-harm/schizophrenia)
    # and the Insight Analyst fabricating citations to explain the mismatch.
    if len(csv_codes) >= 2 and not force_multi:
        keyword_domains = set(_keyword_infer_domains(prompt))
        if keyword_domains:
            filtered = [c for c in csv_codes if c in keyword_domains]
            if filtered:
                csv_codes = filtered

    # Keyword said multi but LLM returned only 1 — infer missing domains from keywords
    if force_multi and len(csv_codes) < 2:
        inferred = _keyword_infer_domains(prompt)
        for c in inferred:
            if c not in csv_codes:
                csv_codes.append(c)
            if len(csv_codes) >= 3:
                break

    if len(csv_codes) >= 2:
        domains = [DOMAINS[c] for c in csv_codes[:3]]
        return domains, True

    primary = csv_codes[0] if csv_codes else (codes[0] if codes else "d0")
    return [DOMAINS.get(primary, DOMAINS["d0"])], False


def route_with_web_search(prompt: str, history_context: str = "") -> tuple[str, Domain | None]:
    """Router ที่รู้จัก 'tavily' — ให้ LLM ตัดสินใจเองว่าต้องค้น web หรือตอบจากความรู้

    Returns:
        ("tavily", None)         — ต้องค้นข้อมูลจากอินเทอร์เน็ต
        ("d0", DOMAINS["d0"])   — ตอบจากความรู้ทั่วไป
        ("dN", DOMAINS["dN"])   — domain เฉพาะทาง
    """
    # Hard rule: accident queries should go to d1 directly
    # ⚠️ ต้องเช็คก่อน Obsidian เสมอ — ดูคำอธิบายเต็มในคอมเมนต์ของ route_domain() ด้านบน
    if _has_accident_signal(prompt):
        return "d1", DOMAINS["d1"]

    # Hard rule: obsidian knowledge vault
    if _has_obsidian_signal(prompt):
        return "obsidian", DOMAINS["obsidian"]

    router = Agent(
        role="Smart Router Agent",
        goal="วิเคราะห์คำถามแล้วตัดสินใจว่าต้องค้นข้อมูลจากอินเทอร์เน็ตหรือตอบจากความรู้",
        backstory=(
            "คุณเป็น Smart Router ที่เข้าใจว่าคำถามไหนต้องการข้อมูลล่าสุดจากเว็บ "
            "และคำถามไหนตอบได้จากความรู้ทั่วไป "
            "คุณตอบเพียงรหัสเดียว ห้ามอธิบายเพิ่ม"
        ),
        llm=_get_llm(),
        verbose=False,
        max_iter=3,
    )

    history_section = f"{history_context}\n\n" if history_context else ""
    task = Task(
        description=(
            f"{history_section}"
            f"คำถาม: {prompt}\n\n"
            "เลือก 1 ตัวเลือก:\n"
            "- tavily  → คำถามต้องการข้อมูลล่าสุด/ข่าว/เหตุการณ์ปัจจุบัน/ข้อมูลเฉพาะจากเว็บ\n"
            "- d0      → ตอบได้จากความรู้ทั่วไป (คณิตศาสตร์ ความหมาย คำอธิบาย ฯลฯ)\n"
            f"Domain อื่นที่มี:\n{DOMAIN_LIST_TEXT}\n\n"
            "ตอบเพียงรหัสเดียว เช่น tavily หรือ d0 หรือ d2"
        ),
        expected_output="รหัสเดียว เช่น tavily หรือ d0",
        agent=router,
    )
    crew = Crew(agents=[router], tasks=[task], verbose=False)

    try:
        result = str(crew.kickoff()).strip().lower()
    except Exception:
        result = "d0"

    if "tavily" in result:
        return "tavily", None

    m = re.search(r'\b(obsidian|d[0-4])\b', result)
    code = m.group(1) if m else "d0"
    return code, DOMAINS.get(code, DOMAINS["d0"])
