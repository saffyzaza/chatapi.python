"""Router Agent — classifies user questions into health domains d0–d8, dt."""
import os
import re

from crewai import Agent, Crew, LLM, Task

from src.domains import Domain, DOMAINS, DOMAIN_LIST_TEXT

# Domains that have CSV files in MinIO (d2–d8 only; d0=general, d1=PostgreSQL)
_CSV_DOMAIN_CODES = {"d2", "d3", "d4", "d5", "d6", "d7", "d8"}

# ── ThaiJo keyword detection ──────────────────────────────────────────────────

_THAIJO_KEYWORDS = [
    r"thaijo", r"thai\s*jo",
    r"บทความวิจัย", r"งานวิจัย", r"literature\s*review",
    r"สังเคราะห์งานวิจัย", r"วรรณกรรม",
    r"สร้าง\s*journal", r"สร้าง\s*report\s*วิจัย",
    r"ค้นหาบทความ", r"journal\s*report",
]


def _has_thaijo_signal(prompt: str) -> bool:
    """True when the query is asking for a ThaiJo research journal."""
    p = prompt.lower()
    return any(re.search(kw, p) for kw in _THAIJO_KEYWORDS)

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
    "d2": ["สุขภาพจิต", "ฆ่าตัวตาย", "ซึมเศร้า", "จิตเวช", "mental"],
    "d3": ["ncd", "เบาหวาน", "ความดัน", "หัวใจ", "หลอดเลือด", "โรคไม่ติดต่อ", "ncds"],
    "d4": ["โภชนาการ", "อ้วน", "bmi", "วัยเรียน", "วัยทำงาน", "ภาวะโภชนาการ", "น้ำหนัก", "nutrition"],
    "d5": ["ผู้สูงอายุ", "สูงวัย", "elderly", "ผู้สูง"],
    "d6": ["โรคติดต่อ", "ไข้เลือดออก", "มาลาเรีย", "วัณโรค", "ติดเชื้อ"],
    "d7": ["มะเร็ง", "cancer"],
    "d8": ["ประชากร", "population", "ประชาชน"],
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
    return LLM(model="gemini/gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))


def route_domain(prompt: str, history_context: str = "") -> Domain:
    """Run the Router Agent and return the matched Domain.

    Fast-path: if ThaiJo keywords detected → return 'dt' immediately.
    """
    # Fast path: ThaiJo research journal
    if _has_thaijo_signal(prompt):
        return DOMAINS["dt"]

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

    # Match dt or d0-d8
    m = re.search(r'\b(dt|d[0-8])\b', result.lower())
    code = m.group(1) if m else "d0"
    return DOMAINS.get(code, DOMAINS["d0"])


def route_multi_domain(prompt: str, history_context: str = "") -> tuple[list[Domain], bool]:
    """Detect if the question spans multiple CSV domains.

    Returns (domains, is_multi).  is_multi=True when ≥2 CSV domains are needed.
    Uses fast keyword check first; LLM refines the domain codes.
    """
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
            "Domains ที่มีไฟล์ CSV:\n"
            "- d2: สุขภาพจิต — ฆ่าตัวตาย ซึมเศร้า จิตเวช\n"
            "- d3: โรคไม่ติดต่อ (NCDs) — เบาหวาน ความดัน หัวใจ\n"
            "- d4: โภชนาการ — ภาวะอ้วน BMI วัยเรียน วัยทำงาน\n"
            "- d5: ผู้สูงอายุ\n"
            "- d6: โรคติดต่อ — ไข้เลือดออก มาลาเรีย วัณโรค\n"
            "- d7: มะเร็ง\n"
            "- d8: ประชากร\n"
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

    codes = list(dict.fromkeys(re.findall(r'\b(d[0-8])\b', result)))[:3]
    if not codes:
        codes = ["d0"]

    csv_codes = [c for c in codes if c in _CSV_DOMAIN_CODES]

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

    m = re.search(r'\b(d[0-8])\b', result)
    code = m.group(1) if m else "d0"
    return code, DOMAINS[code]
