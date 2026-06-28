"""Obsidian Knowledge Ask Orchestrator — 2-agent pipeline for Knowledge Vault Q&A.

Pipeline:
  ObsidianSearcher (fast LLM) — searches vault, reads notes → raw knowledge
  ObsidianAnswerWriter (pro LLM) — interprets knowledge → structured Thai answer

Entry points:
  run_obsidian_ask(question, province, vault_id) → ObsidianAskResponse
  run_obsidian_ask_with_progress(..., request_id) → ObsidianAskResponse  (SSE)
"""
import json
import logging
import re
import time

from crewai import Agent, Crew, Task, Process, LLM

from src.config import get_settings
from src.agents.agent_defaults import agent_retry_kwargs, kickoff_with_retry
from src.agents.progress import emit_progress
from src.schemas.obsidian import ObsidianAskResponse, ObsidianNoteRef
from src.tools.obsidian import search_obsidian, read_obsidian_note, list_obsidian_notes

logger = logging.getLogger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────────

SEARCHER_PROMPT = """คุณคือ Obsidian Knowledge Searcher ผู้เชี่ยวชาญด้านการค้นหาข้อมูลสุขภาพ
จาก Obsidian Knowledge Vault เขตสุขภาพที่ 10

**เครื่องมือที่ใช้:**
- search_obsidian → ค้นหา notes ด้วยคำสำคัญ, กรองตามจังหวัด/อำเภอ/แท็ก
- read_obsidian_note → อ่านเนื้อหาเต็มของ note (ใช้เมื่อ snippet ไม่พอ)
- list_obsidian_notes → ดูรายการ notes ที่มีในระบบ

**วิธีทำงาน:**
1. วิเคราะห์คำถาม — ระบุคำสำคัญ, จังหวัด, อำเภอ, ประเภทข้อมูล
2. เรียก search_obsidian ด้วยคำสำคัญที่เกี่ยวข้อง (อาจค้นหาหลายรอบ)
3. ถ้า snippet สั้นเกินไป — เรียก read_obsidian_note เพื่ออ่านเต็ม
4. รวบรวมข้อมูลทั้งหมดโดยไม่ตัดทอน ระบุ note_id ที่อ้างอิง

**ข้อสำคัญ:**
- ค้นหาด้วยทั้งคำไทยและชื่อตัวชี้วัด (เช่น "NCD remission", "อัตราฆ่าตัวตาย")
- ถ้าไม่พบผลลัพธ์ ให้ลองค้นด้วยคำกว้างขึ้น หรือใช้ list_obsidian_notes
- ระบุอย่างชัดเจนว่าไม่พบข้อมูลเรื่องใดบ้าง
"""

ANSWER_WRITER_PROMPT = """คุณคือ Health Knowledge Answer Writer ผู้เชี่ยวชาญด้านการสื่อสาร
ข้อมูลสุขภาพเขตสุขภาพที่ 10 สำหรับผู้บริหารสาธารณสุข

รับข้อมูลจาก Knowledge Searcher แล้วเขียนคำตอบภาษาไทย ครอบคลุม:

**รูปแบบคำตอบ:**
1. **สรุปคำตอบ** (2-3 ประโยค ตอบตรงๆ)
2. **ข้อมูลจากคลังความรู้** (ตาราง/รายการ ถ้ามีตัวเลข)
3. **บริบทและการวิเคราะห์** (2-3 ประเด็น)
4. **แหล่งข้อมูล** (ชื่อ note และปีที่อ้างอิง)

**กฎสำคัญ:**
- ใช้เฉพาะข้อมูลที่ Knowledge Searcher ให้มา — ห้ามสร้างตัวเลขขึ้นเอง
- ถ้าข้อมูลไม่พอ ระบุว่า "ไม่พบข้อมูลในคลังความรู้" พร้อมแนะนำคำค้นอื่น
- แปลงปีที่ปรากฏในข้อมูลเป็น พ.ศ. ถ้าเป็น ค.ศ.
- ท้ายคำตอบระบุคำถามติดตาม 2-3 ข้อที่เกี่ยวข้อง
"""


# ── LLM factory ────────────────────────────────────────────────────────────────

def _get_llm(tier: str = "fast") -> LLM:
    s = get_settings()
    if tier == "pro":
        return LLM(
            model=f"gemini/{s.GEMINI_MODEL_PRO}",
            api_key=s.GEMINI_API_KEY,
            temperature=0.2,
            max_tokens=s.REPORT_MAX_TOKENS,
        )
    return LLM(
        model=f"gemini/{s.GEMINI_MODEL}",
        api_key=s.GEMINI_API_KEY,
        temperature=0.1,
        max_tokens=4096,
    )


# ── Agent factories ─────────────────────────────────────────────────────────────

def _create_searcher(llm) -> Agent:
    return Agent(
        role="Obsidian Knowledge Searcher",
        goal="ค้นหาและรวบรวมข้อมูลที่เกี่ยวข้องจาก Obsidian Knowledge Vault อย่างครบถ้วน",
        backstory=(
            "ผู้เชี่ยวชาญด้านการค้นหาข้อมูลสุขภาพระดับอำเภอและจังหวัดในเขตสุขภาพที่ 10 "
            "รู้จักโครงสร้าง Knowledge Vault และรู้ว่าควรค้นหาด้วยคำสำคัญใด "
            "ระบุข้อจำกัดของข้อมูลอย่างตรงไปตรงมา"
        ),
        tools=[search_obsidian, read_obsidian_note, list_obsidian_notes],
        llm=llm,
        verbose=True,
        max_iter=10,
        **agent_retry_kwargs(),
    )


def _create_writer(llm) -> Agent:
    return Agent(
        role="Health Knowledge Answer Writer",
        goal=(
            "เขียนคำตอบภาษาไทยที่ชัดเจน มีโครงสร้าง และระบุแหล่งอ้างอิงจาก Obsidian Vault "
            "สำหรับผู้บริหารสาธารณสุขเขตสุขภาพที่ 10"
        ),
        backstory=(
            "ผู้เชี่ยวชาญด้านการสื่อสารข้อมูลสาธารณสุขระดับเขต "
            "มีประสบการณ์เขียนสรุปข้อมูลสำหรับผู้บริหาร สสจ. และ สสอ. "
            "เขียนเฉพาะสิ่งที่ข้อมูลรองรับ ไม่สร้างข้อมูลขึ้นเอง"
        ),
        llm=llm,
        verbose=True,
        max_iter=5,
        **agent_retry_kwargs(),
    )


# ── Core pipeline ──────────────────────────────────────────────────────────────

def _build_crew(question: str, province: str, vault_id: str):
    llm_fast = _get_llm("fast")
    llm_pro = _get_llm("pro")

    searcher = _create_searcher(llm_fast)
    writer = _create_writer(llm_pro)

    prov_label = province or "ทุกจังหวัดในเขตสุขภาพที่ 10"

    search_task = Task(
        description=(
            SEARCHER_PROMPT + "\n\n"
            f"**คำถาม:** {question}\n"
            f"**จังหวัด:** {prov_label}\n"
            f"**Vault:** {vault_id}\n\n"
            "ค้นหาข้อมูลที่เกี่ยวข้องทั้งหมด ระบุ note_id ของทุก note ที่ใช้"
        ),
        expected_output=(
            "ข้อมูลดิบจาก Knowledge Vault ครบถ้วน พร้อม note_id อ้างอิง "
            "และระบุชัดเจนว่าไม่พบข้อมูลใดบ้าง"
        ),
        agent=searcher,
    )

    answer_task = Task(
        description=(
            ANSWER_WRITER_PROMPT + "\n\n"
            f"**คำถามผู้ใช้:** {question}\n"
            f"**จังหวัด:** {prov_label}\n\n"
            "เขียนคำตอบโดยใช้ข้อมูลจาก Knowledge Searcher เท่านั้น"
        ),
        expected_output=(
            "คำตอบภาษาไทย Markdown ครบ 4 ส่วน (สรุป/ข้อมูล/บริบท/แหล่งอ้างอิง) "
            "และคำถามติดตาม 2-3 ข้อ"
        ),
        agent=writer,
        context=[search_task],
    )

    crew = Crew(
        agents=[searcher, writer],
        tasks=[search_task, answer_task],
        process=Process.sequential,
        verbose=True,
    )
    return crew, search_task, answer_task


# ── Response parsing ───────────────────────────────────────────────────────────

def _extract_note_refs(raw_search: str) -> list[ObsidianNoteRef]:
    """Extract note_id references from the searcher's output."""
    refs = []
    seen = set()
    # Try to parse JSON tool results embedded in the output
    for match in re.finditer(r'"note_id"\s*:\s*"([^"]+)"', raw_search):
        note_id = match.group(1)
        if note_id not in seen:
            seen.add(note_id)
            # Try to extract title near the note_id
            title_match = re.search(
                rf'"note_id"\s*:\s*"{re.escape(note_id)}"[^}}]*?"title"\s*:\s*"([^"]*)"',
                raw_search,
            )
            title = title_match.group(1) if title_match else None

            prov_match = re.search(
                rf'"note_id"\s*:\s*"{re.escape(note_id)}"[^}}]*?"province"\s*:\s*"([^"]*)"',
                raw_search,
            )
            province = prov_match.group(1) if prov_match else None

            dist_match = re.search(
                rf'"note_id"\s*:\s*"{re.escape(note_id)}"[^}}]*?"district"\s*:\s*"([^"]*)"',
                raw_search,
            )
            district = dist_match.group(1) if dist_match else None

            refs.append(ObsidianNoteRef(
                note_id=note_id,
                title=title,
                province=province,
                district=district,
            ))
    return refs[:10]  # cap at 10 references


def _extract_follow_ups(text: str) -> list[str]:
    """Extract numbered follow-up questions from the answer."""
    section = re.search(r"(?:คำถามติดตาม|Follow-up)[:\s]*(.*)", text, re.DOTALL | re.IGNORECASE)
    search_in = section.group(1) if section else text
    matches = re.findall(r"(?:^|\n)\s*\d+\.\s*(.+?)(?=\n|$)", search_in)
    return [m.strip() for m in matches if len(m.strip()) > 5][:3]


def _extract_note_refs_from_db(content: str, question: str, vault_id: str) -> list[ObsidianNoteRef]:
    """Extract note refs by finding note_ids mentioned in content, or fallback to DB search."""
    from src.db.pool import query_db
    refs = []
    seen: set[str] = set()

    # Try JSON note_id pattern first
    for match in re.finditer(r'"note_id"\s*:\s*"([^"]+)"', content):
        note_id = match.group(1)
        if note_id not in seen:
            seen.add(note_id)
            rows = query_db(
                "SELECT note_id, title, province, district FROM obsidian_notes WHERE note_id = %s LIMIT 1",
                (note_id,),
            )
            if rows:
                r = rows[0]
                refs.append(ObsidianNoteRef(
                    note_id=r["note_id"], title=r["title"],
                    province=r["province"], district=r["district"],
                ))

    # Fallback: search DB with question keywords
    if not refs:
        try:
            terms = [t for t in re.split(r'\s+', question) if len(t) >= 3][:5]
            if terms:
                like = "%" + "%".join(terms[:2]) + "%"
                rows = query_db(
                    "SELECT note_id, title, province, district FROM obsidian_notes "
                    "WHERE vault_id = %s AND (content_stripped ILIKE %s OR title ILIKE %s) "
                    "ORDER BY indexed_at DESC LIMIT 3",
                    (vault_id, like, like),
                )
                for r in rows:
                    if r["note_id"] not in seen:
                        seen.add(r["note_id"])
                        refs.append(ObsidianNoteRef(
                            note_id=r["note_id"], title=r["title"],
                            province=r["province"], district=r["district"],
                        ))
        except Exception:
            pass
    return refs[:10]


def _build_response(
    result,
    search_task,
    elapsed: float,
    question: str,
    vault_id: str,
) -> ObsidianAskResponse:
    tasks_output = getattr(result, "tasks_output", [])
    content = str(result)
    raw_search = ""

    if tasks_output:
        content = getattr(tasks_output[-1], "raw", None) or str(tasks_output[-1])
    if len(tasks_output) >= 1:
        raw_search = getattr(tasks_output[0], "raw", None) or str(tasks_output[0])

    all_refs = _extract_note_refs(raw_search) or _extract_note_refs_from_db(raw_search, question, vault_id)

    # กรองเฉพาะ notes ที่ title หรือ note_id ปรากฏในคำตอบจริง
    content_lower = content.lower()
    notes_referenced = [
        r for r in all_refs
        if (r.title and r.title.lower() in content_lower)
        or (r.note_id and r.note_id.replace("::", "/").lower() in content_lower)
    ] or all_refs[:4]  # fallback ถ้า filter ได้ 0 ใช้ 4 อันแรก

    follow_ups = _extract_follow_ups(content)

    return ObsidianAskResponse(
        content=content,
        notes_referenced=notes_referenced,
        follow_ups=follow_ups,
        metadata={
            "pipeline": "obsidian_ask",
            "vault_id": vault_id,
            "elapsed_seconds": round(elapsed, 1),
            "notes_found": len(notes_referenced),
        },
    )


# ── Public entry points ────────────────────────────────────────────────────────

def run_obsidian_ask(
    question: str,
    province: str = "",
    vault_id: str = "health_region_10",
) -> ObsidianAskResponse:
    """Run the 2-agent Obsidian Ask pipeline (synchronous)."""
    start = time.time()
    logger.info("[OBSIDIAN-ASK] question=%s province=%s", question[:80], province or "all")

    crew, search_task, answer_task = _build_crew(question, province, vault_id)
    try:
        result = kickoff_with_retry(crew)
        elapsed = time.time() - start
        logger.info("[OBSIDIAN-ASK] done in %.1fs", elapsed)
        return _build_response(result, search_task, elapsed, question, vault_id)
    except Exception as exc:
        elapsed = time.time() - start
        logger.error("[OBSIDIAN-ASK] failed after %.1fs: %s", elapsed, exc)
        return ObsidianAskResponse(
            content=f"เกิดข้อผิดพลาด: {exc}",
            notes_referenced=[],
            follow_ups=[],
            metadata={"error": str(exc), "pipeline": "obsidian_ask", "elapsed_seconds": round(elapsed, 1)},
        )


def run_obsidian_ask_with_progress(
    question: str,
    province: str = "",
    vault_id: str = "health_region_10",
    request_id: str | None = None,
) -> ObsidianAskResponse:
    """Same as run_obsidian_ask but emits SSE progress events."""
    start = time.time()
    logger.info("[OBSIDIAN-ASK-PROGRESS] question=%s rid=%s", question[:60], request_id)

    emit_progress(request_id, "Obsidian Knowledge Searcher", "running", "กำลังค้นหาข้อมูลจาก Vault...")

    crew, search_task, answer_task = _build_crew(question, province, vault_id)
    try:
        result = kickoff_with_retry(crew)
        elapsed = time.time() - start

        emit_progress(request_id, "Obsidian Knowledge Searcher", "done", "ค้นหาข้อมูลเสร็จ", elapsed)
        emit_progress(request_id, "Health Knowledge Answer Writer", "done", "เขียนคำตอบเสร็จ", elapsed)

        logger.info("[OBSIDIAN-ASK-PROGRESS] done in %.1fs", elapsed)
        return _build_response(result, search_task, elapsed, question, vault_id)
    except Exception as exc:
        elapsed = time.time() - start
        emit_progress(request_id, "Obsidian Knowledge Searcher", "error", str(exc)[:100], elapsed)
        logger.error("[OBSIDIAN-ASK-PROGRESS] failed: %s", exc)
        return ObsidianAskResponse(
            content=f"เกิดข้อผิดพลาด: {exc}",
            notes_referenced=[],
            follow_ups=[],
            metadata={"error": str(exc), "pipeline": "obsidian_ask", "elapsed_seconds": round(elapsed, 1)},
        )
