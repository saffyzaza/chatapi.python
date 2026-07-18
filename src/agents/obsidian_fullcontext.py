"""Obsidian Full Context pipeline.

โหลด note ทั้งหมดจากตาราง obsidian_notes (PostgreSQL) แล้วส่งตรงเข้า Gemini.
ถ้าระบุ province จะโหลดเฉพาะ note ของ province นั้น (~100-200 KB แทนที่ 1.1 MB)
"""
import logging
import os
import re
import time
from pathlib import Path

from src.config import get_settings
from src.db.pool import query_db
from src.agents.progress import emit_progress
from src.agents.text_utils import dedupe_repeated_answer
from src.schemas.obsidian import ObsidianAskResponse, ObsidianNoteRef

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """คุณคือผู้เชี่ยวชาญด้านข้อมูลสุขภาพ เขตสุขภาพที่ 10
(อุบลราชธานี, ศรีสะเกษ, ยโสธร, อำนาจเจริญ, มุกดาหาร)

คุณได้รับเอกสารจาก Obsidian Knowledge Vault ด้านล่าง
ตอบคำถามโดยอ้างอิงจากเอกสารเหล่านั้นเท่านั้น

**รูปแบบคำตอบ (Markdown):**
1. **สรุปคำตอบ** — 2-3 ประโยค ตอบตรงๆ
2. **ข้อมูลจากคลังความรู้** — ตาราง/รายการ ถ้ามีตัวเลข
3. **บริบทและการวิเคราะห์** — 2-3 ประเด็น

ห้ามใส่หัวข้อ "แหล่งข้อมูล" ในคำตอบ — ระบบจะแสดงแหล่งอ้างอิงให้โดยอัตโนมัติ

**กฎ:**
- ใช้เฉพาะข้อมูลจากเอกสาร — ห้ามสร้างตัวเลขขึ้นเอง
- ถ้าหาไม่เจอ ระบุ "ไม่พบข้อมูลในคลังความรู้" พร้อมแนะนำคำค้นอื่น
- แปลง ค.ศ. เป็น พ.ศ. เสมอ
- ท้ายคำตอบระบุคำถามติดตาม 2-3 ข้อ

**ถ้ามี "ประวัติการสนทนาก่อนหน้า" แนบมาด้วย — ตอบต่อแบบบทสนทนาจริง (เหมือน Gemini/ChatGPT):**
- อ่านดูว่าก่อนหน้านี้คุยอะไรไปแล้ว แล้ว "ต่อยอด" จากตรงนั้นอย่างเป็นธรรมชาติ
  ไม่ต้องเริ่มอธิบายซ้ำตั้งแต่ต้นหรือแนะนำตัวซ้ำ — ใช้ฟอร์แมต 4 ส่วนด้านบน
  "เฉพาะ" คำถามแรกของหัวข้อหนึ่ง ๆ ส่วนคำถามต่อเนื่อง (follow-up) ให้ตอบกระชับ
  ตรงประเด็นที่ถามเพิ่ม โดยอ้างอิงสิ่งที่เคยตอบไปก่อนหน้าได้ตามธรรมชาติ เช่น
  "จากข้อมูลที่ให้ไปก่อนหน้านี้เกี่ยวกับ... เมื่อดูเพิ่มเติมในส่วนของ... พบว่า ..."
- ถ้าคำถามต่อเนื่องขอรายละเอียด/มุมมองที่ลึกหรือต่างจากเดิม (เช่น เคยถามภาพรวม
  จังหวัด แล้วถามต่อ "แต่ละอำเภอ" หรือ "เจาะจงปีล่าสุด") ให้ค้นเอกสารและตอบ
  เฉพาะในมุมที่ขอเพิ่มนั้นโดยตรง อย่าตอบภาพรวมซ้ำแบบเดิมอีก
- คำถามตามหลัง (follow-up) มักสั้นและไม่ระบุจังหวัด/หัวข้อซ้ำ — ให้อนุมานบริบท
  จากประวัติการสนทนาเสมอ
"""


# ── DB loader ──────────────────────────────────────────────────────────────────

def _relevance_score(content: str, path: str, keywords: list[str]) -> int:
    """นับจำนวนคีย์เวิร์ดจากคำถามที่ปรากฏใน note (นับใน path ด้วย × น้ำหนัก)"""
    if not keywords:
        return 0
    lc = content.lower()
    lp = path.lower()
    return sum(lc.count(k) + lp.count(k) * 5 for k in keywords)


def _load_vault_context(
    vault_id: str,
    province: str | None,
    question: str = "",
    max_chars: int | None = None,
) -> tuple[str, list[str], dict[str, str]]:
    """โหลด note จาก obsidian_notes (กรองตาม province) แบบ "เลือกที่เกี่ยวข้องที่สุด
    ให้พอดีกับเพดาน context" — กัน Gemini ContextWindowExceededError เมื่อ vault
    โตขึ้นจากการ ingest PDF จนแม้แต่จังหวัดเดียวก็เกิน 1M tokens

    Returns:
        (context_text, relative_file_paths, minio_id_map{rel_path: file_id})
    """
    if max_chars is None:
        max_chars = get_settings().OBSIDIAN_MAX_CONTEXT_CHARS

    rows = query_db(
        "SELECT relative_path, content, file_id FROM obsidian_notes "
        "WHERE vault_id = %s AND province = %s ORDER BY relative_path",
        (vault_id, province),
    ) if province else []

    if province and not rows:
        logger.warning("[fullctx] ไม่พบ note ของ '%s' — โหลดทั้ง vault", province)

    if not rows:
        rows = query_db(
            "SELECT relative_path, content, file_id FROM obsidian_notes "
            "WHERE vault_id = %s ORDER BY relative_path",
            (vault_id,),
        )

    # คีย์เวิร์ดจากคำถาม (ตัดคำสั้น/คำทั่วไปทิ้ง) ใช้จัดอันดับความเกี่ยวข้อง
    keywords = [w.lower() for w in re.split(r"\s+", question) if len(w) >= 3]

    # ให้คะแนนแล้วเรียงจากเกี่ยวข้องมากไปน้อย — โน้ตที่ตรงคำถามจะได้เข้าก่อน
    # ถ้าถึงเพดานตัวอักษรก่อนจะตัดโน้ตที่เหลือ (เกี่ยวข้องน้อยกว่า) ออก
    scored = sorted(
        rows,
        key=lambda r: _relevance_score(r["content"] or "", r["relative_path"], keywords),
        reverse=True,
    )

    parts: list[str] = []
    file_paths: list[str] = []
    minio_id_map: dict[str, str] = {}
    total = 0
    included = 0

    for r in scored:
        content = (r["content"] or "").strip()
        if not content:
            continue
        rel = r["relative_path"]
        block = f"\n\n---\n## FILE: {rel}\n\n{content}"
        if total + len(block) > max_chars and included > 0:
            break  # เต็มเพดานแล้ว (แต่ต้องมีอย่างน้อย 1 โน้ตเสมอ)
        parts.append(block)
        file_paths.append(rel)
        if r.get("file_id"):
            minio_id_map[rel] = r["file_id"]
        total += len(block)
        included += 1

    logger.info(
        "[fullctx] โหลด %d/%d notes (%d chars, cap=%d, vault=%s, province=%s)",
        included, len(rows), total, max_chars, vault_id, province,
    )

    return "\n".join(parts), file_paths, minio_id_map


# ── Gemini call ────────────────────────────────────────────────────────────────

def _call_gemini(system: str, user_message: str, s) -> str:
    """เรียก Gemini Pro ผ่าน litellm (dependency ของ crewai)."""
    import litellm

    os.environ.setdefault("GEMINI_API_KEY", s.GEMINI_API_KEY)
    os.environ.setdefault("GOOGLE_API_KEY", s.GEMINI_API_KEY)

    resp = litellm.completion(
        model=f"gemini/{s.GEMINI_MODEL_PRO}",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        api_key=s.GEMINI_API_KEY,
        max_tokens=s.REPORT_MAX_TOKENS,
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


# ── Follow-up extractor ────────────────────────────────────────────────────────

def _extract_follow_ups(text: str) -> list[str]:
    section = re.search(r"(?:คำถามติดตาม|Follow-up)[:\s]*(.*)", text, re.DOTALL | re.IGNORECASE)
    search_in = section.group(1) if section else text
    matches = re.findall(r"(?:^|\n)\s*\d+\.\s*(.+?)(?=\n|$)", search_in)
    return [m.strip() for m in matches if len(m.strip()) > 5][:3]


# ── Public entry point ─────────────────────────────────────────────────────────

def run_obsidian_ask_fullcontext(
    question: str,
    province: str = "",
    vault_id: str = "health_region_10",
    request_id: str | None = None,
    history_context: str = "",
) -> ObsidianAskResponse:
    """Full context pipeline — โหลด .md ทั้งหมด → Gemini context window โดยตรง.

    history_context: ข้อความสรุปประวัติการสนทนาก่อนหน้า (จาก build_history_context)
    — แนบไปกับคำถามให้ Gemini เห็นบทสนทนาที่ผ่านมา เพื่อให้ตอบคำถามต่อเนื่อง
    (follow-up) ได้อย่างเป็นธรรมชาติแบบ Gemini/ChatGPT แทนที่จะเริ่มนับหนึ่งใหม่
    ทุกครั้งที่ถามต่อ (ดูคอมเมนต์ใน _orchestrate ของ analyze.py ที่
    build_history_context ถูกสร้างขึ้น แล้วส่งต่อมาที่นี่)
    """
    start = time.time()
    s = get_settings()

    emit_progress(request_id, "📂 Context Loader", "running",
                  f"กำลังโหลดเอกสาร{f' จังหวัด{province}' if province else 'ทั้ง vault'}...")

    try:
        context_text, file_paths, minio_id_map = _load_vault_context(
            vault_id, province or None, question=question
        )

        load_elapsed = round(time.time() - start, 1)
        emit_progress(request_id, "📂 Context Loader", "done",
                      f"โหลด {len(file_paths)} ไฟล์ ({load_elapsed}s)", load_elapsed)

        emit_progress(request_id, "🤖 Gemini Answer Writer", "running",
                      "กำลังวิเคราะห์เอกสารและเขียนคำตอบ...")

        prov_label = province or "ทุกจังหวัดในเขตสุขภาพที่ 10"
        # ⚠️ ต่อ "ความจำการสนทนา" เข้า user_message — ไม่งั้นทุกคำถามตามหลัง
        # (follow-up) จะถูกตอบแบบเริ่มนับหนึ่งใหม่ทุกครั้ง ไม่ต่อเนื่องแบบ
        # Gemini/ChatGPT (ใช้รูปแบบเดียวกับ history_section ใน csv_pipeline.py /
        # accident_chat_orchestrator.py — ตรงกับที่ผู้ใช้ขอให้ "ส่ง context history
        # ไปให้ AI ไปด้วย")
        history_section = f"{history_context}\n\n" if history_context else ""
        user_message = (
            f"{history_section}"
            f"**เอกสารจาก Obsidian Knowledge Vault ({prov_label}):**\n"
            f"{context_text}\n\n"
            f"---\n**คำถาม:** {question}\n\n"
            "(ถ้ามี \"ประวัติการสนทนาก่อนหน้า\" แนบมาด้านบน ให้ตอบต่อแบบบทสนทนาจริง "
            "ตามแนวทางในคำสั่งระบบ — ต่อยอดจากที่เคยตอบไปแล้ว ไม่ใช่เริ่มอธิบายใหม่ทั้งหมด)"
        )

        answer = dedupe_repeated_answer(_call_gemini(SYSTEM_PROMPT, user_message, s))

        elapsed = round(time.time() - start, 1)
        emit_progress(request_id, "🤖 Gemini Answer Writer", "done",
                      f"เขียนคำตอบเสร็จ ({elapsed}s)", elapsed)

        note_refs = [
            ObsidianNoteRef(
                note_id=p.replace("/", "::"),
                title=Path(p).stem,
                province=province or None,
                district=None,
                pdf_url=(
                    f"/api/pdf/view/{minio_id_map[p]}"
                    if p in minio_id_map else None
                ),
            )
            for p in file_paths[:15]
        ]
        follow_ups = _extract_follow_ups(answer)

        return ObsidianAskResponse(
            content=answer,
            notes_referenced=note_refs,
            follow_ups=follow_ups,
            metadata={
                "pipeline": "obsidian_fullcontext",
                "vault_id": vault_id,
                "province": province or "all",
                "files_loaded": len(file_paths),
                "elapsed_seconds": elapsed,
            },
        )

    except Exception as exc:
        elapsed = round(time.time() - start, 1)
        emit_progress(request_id, "🤖 Gemini Answer Writer", "error", str(exc)[:120], elapsed)
        logger.exception("[fullctx] ล้มเหลว: %s", exc)
        return ObsidianAskResponse(
            content=f"เกิดข้อผิดพลาด: {exc}",
            notes_referenced=[],
            follow_ups=[],
            metadata={
                "error": str(exc),
                "pipeline": "obsidian_fullcontext",
                "elapsed_seconds": elapsed,
            },
        )
