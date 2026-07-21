"""Obsidian Full Context pipeline.

โหลด note ทั้งหมดจากตาราง obsidian_notes (PostgreSQL) แล้วส่งตรงเข้า Gemini.
ถ้าระบุ province จะโหลดเฉพาะ note ของ province นั้น (~100-200 KB แทนที่ 1.1 MB)
"""
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable

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

**ห้ามเด็ดขาด (anti-leak) — ต้องสรุปใหม่เป็นคำพูดของคุณเองเสมอ:**
- ห้ามคัดลอกเนื้อหาต้นฉบับของเอกสารมาแปะในคำตอบไม่ว่ากรณีใด ได้แก่: บรรทัดที่ขึ้นต้น
  ด้วย "FILE:", บล็อก YAML frontmatter (ข้อความที่คั่นด้วย "---"), wikilink รูปแบบ
  [[...]], หรือเลขหน้า/หัวกระดาษดิบจากต้นฉบับ
- เอกสารที่แนบมาให้เป็น "วัตถุดิบ" สำหรับอ่านทำความเข้าใจเท่านั้น ไม่ใช่สิ่งที่ต้อง
  คัดลอกออกมา — ให้เรียบเรียงประโยคใหม่ด้วยตัวเองเสมอ

**ท้ายคำตอบ (บังคับ, machine-readable — ต้องมีเป๊ะๆ ทุกครั้ง):**
ปิดท้ายคำตอบด้วยบล็อกนี้ (ห้ามมีข้อความอื่นตามหลังบล็อกนี้อีก):
<<<FOLLOWUPS>>>
["คำถามติดตาม 1?", "คำถามติดตาม 2?", "คำถามติดตาม 3?"]
<<<END_FOLLOWUPS>>>
กติกาบล็อกนี้: ต้องเป็น JSON array ของสตริงล้วนๆ 2-3 ข้อ แต่ละข้อเป็นประโยคคำถามสั้นๆ
ที่ลงท้ายด้วยเครื่องหมาย "?" เท่านั้น ห้ามใส่ตัวหนา (**) หรือหัวข้อ ห้ามมีคอมเมนต์อื่นปนในบล็อกนี้

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

# ── Anti-leak guard ──────────────────────────────────────────────────────────
# ป้องกันเนื้อหาดิบของเอกสารต้นฉบับ (raw ingest markers) หลุดเข้าไปในคำตอบที่
# ผู้ใช้เห็นตรงๆ — เคยเจอจริงตอน LLM ตอบคำถามกว้างๆ (เช่น "มีเอกสารอะไรบ้าง")
# แล้วดันคัดลอกบล็อก "## FILE: ..." พร้อม YAML frontmatter และ wikilink ทั้งดุ้น
_FILE_MARKER_RE = re.compile(r"(?m)^\s*#{0,3}\s*FILE:\s*\S+")
_YAML_BLOCK_RE = re.compile(r"(?ms)^---\s*\n.*?\n---\s*(?:\n|$)")
_YAML_FENCE_RE = re.compile(r"```\s*ya?ml.*?```", re.DOTALL | re.IGNORECASE)
_WIKILINK_RE = re.compile(r"\[\[[^\[\]\n]{1,200}\]\]")

# เช็คเฉพาะ "หาง" ของบัฟเฟอร์ที่เพิ่งโตขึ้นระหว่างสตรีม (ไม่ต้องสแกนทั้งก้อนทุกครั้ง)
_LEAK_CHECK_WINDOW = 400

_LEAK_RETRY_SUFFIX = (
    "\n\n⚠️ คำเตือนสำคัญ: คำตอบก่อนหน้าของคุณมีการคัดลอกเนื้อหาต้นฉบับของเอกสาร "
    "(เช่น บรรทัด \"FILE:\", YAML frontmatter ที่คั่นด้วย \"---\", หรือ wikilink "
    "[[...]]) ปนมาโดยตรง ซึ่งห้ามเด็ดขาด กรุณาเขียนคำตอบใหม่ทั้งหมดเป็นคำสรุป "
    "ด้วยคำพูดของคุณเอง ห้ามคัดลอกประโยค/บรรทัดจากเอกสารต้นฉบับมาทั้งดุ้นไม่ว่า"
    "กรณีใดก็ตาม และห้ามลืมปิดท้ายด้วยบล็อก <<<FOLLOWUPS>>> ตามฟอร์แมตที่กำหนด"
)


def _contains_leak(text: str) -> bool:
    """True ถ้าเจอร่องรอยเนื้อหาดิบของเอกสารต้นฉบับหลุดเข้ามาในคำตอบ"""
    if not text:
        return False
    return bool(
        _FILE_MARKER_RE.search(text)
        or _YAML_FENCE_RE.search(text)
        or _YAML_BLOCK_RE.search(text)
        or _WIKILINK_RE.search(text)
    )


def _strip_leaked_blocks(text: str) -> str:
    """ท่าสำรองสุดท้าย (best-effort) — ถ้า retry ด้วยพรอมต์เข้มแล้วยังหลุดอีก
    ให้ตัดบล็อกที่หลุดออกด้วยโค้ดตรงๆ แทนที่จะปล่อยให้ผู้ใช้เห็นเนื้อหาดิบ
    """
    cleaned = _YAML_FENCE_RE.sub("", text)
    cleaned = _YAML_BLOCK_RE.sub("", cleaned)
    cleaned = _FILE_MARKER_RE.sub("", cleaned)
    # wikilink → เก็บแค่ข้อความอ่านง่าย (ตัด [[ ]] และเอาเฉพาะส่วนหลัง | ถ้ามี)
    cleaned = _WIKILINK_RE.sub(
        lambda m: m.group(0).strip("[]").split("|")[-1].strip(), cleaned
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


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

def _call_gemini(
    system: str,
    user_message: str,
    s,
    on_delta: Callable[[str], None] | None = None,
) -> str:
    """เรียก Gemini Pro ผ่าน litellm (dependency ของ crewai).

    on_delta: ถ้าระบุ จะสตรีมคำตอบทีละ token ผ่าน callback นี้แบบเรียลไทม์
    (ลด perceived latency ของคำถามที่ใช้เวลานาน ~50-60s) — มีการ์ดกันเนื้อหาดิบ
    หลุดออกไปสด ๆ ระหว่างสตรีมด้วย: เช็คเฉพาะ "หาง" ของบัฟเฟอร์ที่โตขึ้นทุกครั้ง
    ถ้าเจอร่องรอยเนื้อหาดิบ (FILE:/YAML/wikilink) จะหยุดส่งสดทันที (แต่ยังสะสม
    ข้อความในหน่วยความจำต่อจนจบ เพื่อให้ตัวตรวจสอบระดับบนสุดใน
    run_obsidian_ask_fullcontext ทำ retry/cleanup ได้ตามปกติ — ผู้ใช้จะไม่เห็น
    เนื้อหาดิบเป็นคำตอบสุดท้ายไม่ว่ากรณีใด)
    """
    import litellm

    os.environ.setdefault("GEMINI_API_KEY", s.GEMINI_API_KEY)
    os.environ.setdefault("GOOGLE_API_KEY", s.GEMINI_API_KEY)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

    if on_delta is None:
        resp = litellm.completion(
            model=f"gemini/{s.GEMINI_MODEL_PRO}",
            messages=messages,
            api_key=s.GEMINI_API_KEY,
            max_tokens=s.REPORT_MAX_TOKENS,
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""

    parts: list[str] = []
    forwarding = True
    stream = litellm.completion(
        model=f"gemini/{s.GEMINI_MODEL_PRO}",
        messages=messages,
        api_key=s.GEMINI_API_KEY,
        max_tokens=s.REPORT_MAX_TOKENS,
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        delta = ""
        if chunk.choices and chunk.choices[0].delta:
            delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue
        parts.append(delta)
        if forwarding:
            tail = "".join(parts)[-_LEAK_CHECK_WINDOW:]
            if _contains_leak(tail):
                # หยุดส่งสดตั้งแต่ตอนที่พบร่องรอยแรก — กันไม่ให้ผู้ใช้เห็นเนื้อหา
                # ดิบไหลเข้าจอระหว่างสตรีม ส่วนที่เหลือของคำตอบจะถูกจัดการที่ระดับ
                # guard/retry ของ run_obsidian_ask_fullcontext แทน
                forwarding = False
                logger.warning("[fullctx] พบร่องรอยเนื้อหาดิบระหว่างสตรีม — หยุดส่งสด")
            else:
                on_delta(delta)

    return "".join(parts)


# ── Note-reference title cleanup ────────────────────────────────────────────────
# ไฟล์ PDF ต้นฉบับที่ยาวจะถูกตัดแบ่งเป็นหลาย .md "ส่วน" ตอน ingest (เช่น
# "...-2567-ส่วนที่01", "...-ส่วนที่02", ..., "...-INDEX") — ทุกส่วนของเอกสารเดียวกัน
# จะชี้ minio file_id เดียวกัน ต้องตัดคำต่อท้ายออกเพื่อโชว์เป็นชื่อเอกสารต้นฉบับเดียว
_PART_SUFFIX_RE = re.compile(r"[-_](?:ส่วนที่\s*\d+|part\s*\d+|INDEX)$", re.IGNORECASE)


def _clean_doc_title(stem: str) -> str:
    return _PART_SUFFIX_RE.sub("", stem).strip() or stem


# ── Follow-up extractor (structured, ไม่ใช่ regex เดาจาก markdown headers) ──────

_FOLLOWUP_BLOCK_RE = re.compile(
    r"<<<FOLLOWUPS>>>\s*(.*?)\s*<<<END_FOLLOWUPS>>>", re.DOTALL
)


def _extract_and_strip_followups(text: str) -> tuple[str, list[str]]:
    """ดึง follow_ups จากบล็อก JSON ที่บังคับให้ LLM ปิดท้ายคำตอบด้วยเสมอ แล้วตัด
    บล็อกนั้นออกจาก content ก่อนส่งให้ผู้ใช้เห็น

    เดิม _extract_follow_ups ใช้ regex เดาว่าอะไรคือ "รายการเลขข้อ" ในคำตอบทั้งก้อน
    ซึ่งไปจับเอาหัวข้อ markdown ของคำตอบเอง (เช่น "1. **สรุปคำตอบ**") มาแสดงเป็น
    ปุ่มคำถามแนะนำผิด ๆ — ตอนนี้ใช้ JSON block ที่ระบุตำแหน่งชัดเจนแทน จึงไม่มีทาง
    หยิบข้อความอื่นมาปนได้ และมีการกรองรูปแบบซ้ำอีกชั้นก่อน return
    """
    m = _FOLLOWUP_BLOCK_RE.search(text)
    if not m:
        return text.strip(), []

    content = (text[: m.start()] + text[m.end():]).strip()
    raw = m.group(1).strip()
    follow_ups: list[str] = []
    try:
        items = json.loads(raw)
        if isinstance(items, list):
            for item in items:
                q = str(item).strip()
                # ต้องเป็นประโยคคำถามสั้น ๆ ที่ลงท้ายด้วย "?" เท่านั้น และห้ามมี
                # markdown syntax หลุดมา (กันเคสหัวข้อ **...** ปนเข้ามา)
                if q and q.endswith("?") and 5 < len(q) <= 160 and "**" not in q and "\n" not in q:
                    follow_ups.append(q)
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning("[fullctx] follow_ups block ไม่ใช่ JSON ที่ถูกต้อง: %r", raw[:200])

    return content, follow_ups[:3]


# ── Public entry point ─────────────────────────────────────────────────────────

def run_obsidian_ask_fullcontext(
    question: str,
    province: str = "",
    vault_id: str = "health_region_10",
    request_id: str | None = None,
    history_context: str = "",
    on_delta: Callable[[str], None] | None = None,
) -> ObsidianAskResponse:
    """Full context pipeline — โหลด .md ทั้งหมด → Gemini context window โดยตรง.

    history_context: ข้อความสรุปประวัติการสนทนาก่อนหน้า (จาก build_history_context)
    — แนบไปกับคำถามให้ Gemini เห็นบทสนทนาที่ผ่านมา เพื่อให้ตอบคำถามต่อเนื่อง
    (follow-up) ได้อย่างเป็นธรรมชาติแบบ Gemini/ChatGPT แทนที่จะเริ่มนับหนึ่งใหม่
    ทุกครั้งที่ถามต่อ (ดูคอมเมนต์ใน _orchestrate ของ analyze.py ที่
    build_history_context ถูกสร้างขึ้น แล้วส่งต่อมาที่นี่)

    on_delta: callback รับ token สด ๆ ระหว่างสตรีมคำตอบ (ดู _call_gemini) — ใช้ลด
    perceived latency ของคำถามที่กิน ~50-60s ผู้เรียก (analyze.py) ส่ง callback ที่
    ยิง SSE event "obsidian_chunk" กลับไปอัปเดตแผงสถานะฝั่งหน้าจอแบบเรียลไทม์
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

        raw_answer = _call_gemini(SYSTEM_PROMPT, user_message, s, on_delta=on_delta)
        answer, follow_ups = _extract_and_strip_followups(raw_answer)
        answer = dedupe_repeated_answer(answer)

        # ── Output guard: กันเนื้อหาดิบของเอกสารต้นฉบับหลุดเข้าคำตอบ ─────────
        # (เคยเจอจริง: คำถามกว้าง ๆ อย่าง "มีเอกสารอะไรบ้าง" ทำให้ LLM คัดลอกบล็อก
        # "## FILE: ..." พร้อม YAML frontmatter + wikilink ทั้งดุ้นมาแปะในคำตอบ)
        if _contains_leak(answer):
            logger.warning(
                "[fullctx] ตรวจพบเนื้อหาดิบหลุดในคำตอบ (province=%s) — retry ด้วยพรอมต์ที่เข้มขึ้น",
                province,
            )
            retry_raw = _call_gemini(
                SYSTEM_PROMPT, user_message + _LEAK_RETRY_SUFFIX, s, on_delta=None,
            )
            retry_answer, retry_follow_ups = _extract_and_strip_followups(retry_raw)
            retry_answer = dedupe_repeated_answer(retry_answer)
            if not _contains_leak(retry_answer):
                answer, follow_ups = retry_answer, (retry_follow_ups or follow_ups)
            else:
                logger.error(
                    "[fullctx] เนื้อหาดิบยังหลุดหลัง retry (province=%s) — ตัดออกเองแบบ best-effort",
                    province,
                )
                answer = _strip_leaked_blocks(answer)
                follow_ups = follow_ups or retry_follow_ups

        elapsed = round(time.time() - start, 1)
        emit_progress(request_id, "🤖 Gemini Answer Writer", "done",
                      f"เขียนคำตอบเสร็จ ({elapsed}s)", elapsed)

        # เอกสารต้นฉบับหนึ่งไฟล์ที่ถูกตัดแบ่งเป็นหลาย .md "ส่วน" (ระหว่าง ingest PDF)
        # จะมีหลาย path ใน file_paths แต่ชี้ minio file_id เดียวกัน — dedupe ตรงนี้
        # เพื่อให้อ้างอิงที่โชว์ผู้ใช้เป็น "1 เอกสาร = 1 ลิงก์" ไม่ใช่โผล่ซ้ำเป็น 15-20
        # ป้ายของทุกส่วนย่อย (ส่วนที่ไม่มี PDF ผูกอยู่ ใช้ path ของตัวเองกันซ้ำแทน)
        note_refs: list[ObsidianNoteRef] = []
        seen_dedup_keys: set[str] = set()
        for p in file_paths:
            file_id = minio_id_map.get(p)
            dedup_key = file_id or p
            if dedup_key in seen_dedup_keys:
                continue
            seen_dedup_keys.add(dedup_key)
            note_refs.append(ObsidianNoteRef(
                note_id=p.replace("/", "::"),
                title=_clean_doc_title(Path(p).stem),
                province=province or None,
                district=None,
                pdf_url=f"/api/pdf/view/{file_id}" if file_id else None,
            ))
            if len(note_refs) >= 15:
                break

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
