"""PDF Ingest Router — แปลง PDF → Markdown chunks → บันทึกใน PostgreSQL (obsidian_notes).
จัดระเบียบตามโครงสร้าง เขต10 / จังหวัด / อำเภอ ของสาธารณสุขเขต 10.
หมายเหตุ (028): MD chunks ถูกเก็บใน database โดยตรง (ไม่ใช่ filesystem)
"""
import io
import json
import logging
import os
import re
import time
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator
from concurrent.futures import ThreadPoolExecutor, as_completed

import pdfplumber
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types

from src.config import get_settings
from src.tools.minio import _get_client, _bucket
from src.db.pool import execute_db, query_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pdf", tags=["PDF Ingest"])

# ── In-memory job store ───────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}


def _pdf_bucket() -> str:
    return get_settings().PDF_BUCKET


def _ensure_pdf_bucket() -> None:
    """Create the PDF bucket if it doesn't exist."""
    client = _get_client()
    bucket = _pdf_bucket()
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            logger.info(f"Created PDF bucket: {bucket}")
    except Exception as e:
        logger.warning(f"Could not ensure PDF bucket: {e}")



# ── Zone 10 Geographic Structure ─────────────────────────────────────────────
ZONE10_PROVINCES: dict[str, list[str]] = {
    "ส่วนกลาง": [],
    "อุบลราชธานี": [
        "เมืองอุบลราชธานี", "ศรีเมืองใหม่", "โขงเจียม", "เขื่องใน", "เขมราฐ",
        "เดชอุดม", "นาจะหลวย", "น้ำยืน", "บุณฑริก", "ตระการพืชผล",
        "กุดข้าวปุ้น", "ม่วงสามสิบ", "วารินชำราบ", "พิบูลมังสาหาร", "ตาลสุม",
        "โพธิ์ไทร", "สำโรง", "ดอนมดแดง", "สิรินธร", "ทุ่งศรีอุดม",
        "นาเยีย", "นาตาล", "เหล่าเสือโก้ก", "สว่างวีระวงศ์", "น้ำขุ่น",
    ],
    "ศรีสะเกษ": [
        "เมืองศรีสะเกษ", "ยางชุมน้อย", "กันทรารมย์", "กันทรลักษ์", "ขุขันธ์",
        "ไพรบึง", "ปรางค์กู่", "ขุนหาญ", "ราษีไศล", "อุทุมพรพิสัย",
        "บึงบูรพ์", "ห้วยทับทัน", "โนนคูณ", "ศรีรัตนะ", "น้ำเกลี้ยง",
        "วังหิน", "ภูสิงห์", "เมืองจันทร์", "เบญจลักษ์", "พยุห์",
        "โพธิ์ศรีสุวรรณ", "ศิลาลาด",
    ],
    "ยโสธร": [
        "เมืองยโสธร", "ทรายมูล", "กุดชุม", "คำเขื่อนแก้ว", "ป่าติ้ว",
        "มหาชนะชัย", "ค้อวัง", "เลิงนกทา", "ไทยเจริญ",
    ],
    "อำนาจเจริญ": [
        "เมืองอำนาจเจริญ", "ชานุมาน", "ปทุมราชวงศา", "พนา",
        "เสนางคนิคม", "หัวตะพาน", "ลืออำนาจ",
    ],
    "มุกดาหาร": [
        "เมืองมุกดาหาร", "นิคมคำสร้อย", "ดอนตาล", "ดงหลวง",
        "คำชะอี", "หว้านใหญ่", "หนองสูง",
    ],
}

# Alias map for fuzzy matching (common abbreviations / alternative spellings)
_PROVINCE_ALIASES: dict[str, str] = {
    "อุบล": "อุบลราชธานี",
    "อุบลฯ": "อุบลราชธานี",
    "ศรีสะเกษ": "ศรีสะเกษ",
    "ศรีสะเกษ": "ศรีสะเกษ",
    "ยโสธร": "ยโสธร",
    "อำนาจ": "อำนาจเจริญ",
    "มุกดา": "มุกดาหาร",
    "มุกดาหาร": "มุกดาหาร",
}


# ── Gemini client ─────────────────────────────────────────────────────────────
def _get_gemini_client():
    s = get_settings()
    return genai.Client(api_key=s.GEMINI_API_KEY)


def _gemini_model() -> str:
    return get_settings().GEMINI_MODEL_PRO  # gemini-2.5-pro (markdown conversion)


def _gemini_model_fast() -> str:
    return get_settings().GEMINI_MODEL      # gemini-2.0-flash (filename / location)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitize_filename(name: str) -> str:
    """ลบอักขระพิเศษที่ไม่เหมาะกับชื่อไฟล์ แต่รักษา Thai/Unicode"""
    name = re.sub(r'[\\/:*?"<>|]', '-', name)
    name = name.strip('. ')
    return name[:120] if len(name) > 120 else name


# ── DB helpers ────────────────────────────────────────────────────────────────

VAULT_ID = "health_region_10"


def _upsert_note(
    note_id: str,
    relative_path: str,
    title: str,
    province: str | None,
    district: str | None,
    note_type: str,
    tags: list[str],
    source_file: str,
    content: str,
    file_id: str | None = None,
    chunk_index: int = 0,
    is_index: bool = False,
    year: int | None = None,
) -> None:
    """UPSERT หนึ่ง note เข้า obsidian_notes."""
    import re as _re
    # strip YAML frontmatter เพื่อเก็บใน content_stripped
    stripped = _re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, count=1, flags=_re.DOTALL).strip()
    execute_db(
        """
        INSERT INTO obsidian_notes
            (note_id, vault_id, relative_path, title, province, district,
             note_type, tags, source_file, year, content, content_stripped,
             file_id, chunk_index, is_index, updated_at, indexed_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),NOW())
        ON CONFLICT (note_id) DO UPDATE SET
            title            = EXCLUDED.title,
            province         = EXCLUDED.province,
            district         = EXCLUDED.district,
            note_type        = EXCLUDED.note_type,
            tags             = EXCLUDED.tags,
            source_file      = EXCLUDED.source_file,
            year             = EXCLUDED.year,
            content          = EXCLUDED.content,
            content_stripped = EXCLUDED.content_stripped,
            file_id          = EXCLUDED.file_id,
            chunk_index      = EXCLUDED.chunk_index,
            is_index         = EXCLUDED.is_index,
            updated_at       = NOW()
        """,
        (
            note_id, VAULT_ID, relative_path, title, province, district,
            note_type, tags, source_file, year, content, stripped,
            file_id, chunk_index, is_index,
        ),
    )


def _extract_pages(file_bytes: bytes) -> list[str]:
    """Extract text from each page of PDF."""
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return pages


def _chunk_pages(pages: list[str], chunk_size: int) -> list[list[str]]:
    """Split pages into chunks of chunk_size."""
    chunks = []
    for i in range(0, len(pages), chunk_size):
        chunks.append(pages[i: i + chunk_size])
    return chunks


# ── AI: Detect province & district from content ───────────────────────────────

def _build_zone10_list_text() -> str:
    """สร้าง text แสดงรายชื่อจังหวัด/อำเภอทั้งหมดให้ AI ใช้อ้างอิง."""
    lines = ["สาธารณสุขเขต 10 ประกอบด้วย 5 จังหวัด:"]
    for province, districts in ZONE10_PROVINCES.items():
        lines.append(f"\n**{province}**: {', '.join(districts)}")
    return "\n".join(lines)


def _ai_detect_location(client, uploaded_file, sample_text: str, original_pdf_name: str) -> dict:
    """
    ให้ Gemini Pro วิเคราะห์เนื้อหา PDF แล้วระบุจังหวัดและอำเภอในเขต 10.

    Returns:
        {
            "province": "อุบลราชธานี" | None,
            "district": "เมืองอุบลราชธานี" | None,
            "confidence": "high" | "medium" | "low",
            "reason": "เหตุผล..."
        }
    """
    zone_info = _build_zone10_list_text()

    prompt = f"""คุณเป็น AI ผู้เชี่ยวชาญด้านสาธารณสุขเขต 10 ของประเทศไทย มีหน้าที่ระบุว่าเอกสาร PDF นี้เป็นของ "จังหวัดใด" ในเขต 10

{zone_info}

═══════════════════════════════════════
กฎการวิเคราะห์ (ห้ามละเลยข้อใดข้อหนึ่ง)
═══════════════════════════════════════
1. อ่านเนื้อหา PDF ที่แนบมาทุกหน้าก่อนตอบ
2. ค้นหาเบาะแสต่อไปนี้ แล้วให้ confidence ตามนี้:

   ★ confidence="high" — เมื่อพบสิ่งเหล่านี้อย่างน้อย 1 อย่าง:
     • ชื่อจังหวัดปรากฏในเอกสาร (แม้แค่ครั้งเดียว)
     • ชื่ออำเภอใดๆ ของจังหวัดนั้นปรากฏในเอกสาร
     • ชื่อโรงพยาบาล / สสจ. / หน่วยงานสาธารณสุขของจังหวัดนั้น
     • ที่อยู่หรือสถานที่ที่ระบุจังหวัดนั้น

   ★ confidence="medium" — เมื่อมีหลักฐานทางอ้อม:
     • ข้อมูลสถิติของพื้นที่ที่น่าจะเป็นจังหวัดนั้น แต่ไม่ได้ระบุชื่อตรงๆ
     • บริบทของเอกสารที่ชี้ไปยังจังหวัดนั้นโดยรวม

   ★ confidence="low" — เฉพาะเมื่อไม่พบหลักฐานใดๆ เลย

3. ห้ามเดาโดยไม่มีหลักฐาน
4. หากเอกสารเปรียบเทียบทุกจังหวัดหรือเป็นระดับประเทศ → ระบุ "ส่วนกลาง"
5. หากไม่พบหลักฐานของจังหวัดใดเลย → ระบุ "ส่วนกลาง" ห้ามเดาเป็นอุบลราชธานี

═══════════════════════════════════════
ตัวอย่างการวิเคราะห์ที่ถูกต้อง
═══════════════════════════════════════
✅ พบคำว่า "ศรีสะเกษ" หรือ "ขุขันธ์" → province: "ศรีสะเกษ", confidence: "high"
✅ พบ "โรงพยาบาลเลิงนกทา" → province: "ยโสธร", district: "เลิงนกทา", confidence: "high"
✅ พบ "สสจ.มุกดาหาร" → province: "มุกดาหาร", confidence: "high"
✅ พบ "อุบลราชธานี" แค่ 1 ครั้ง → confidence: "high" (พบหลักฐาน)
✅ ข้อมูลเขต 10 ทุกจังหวัด → province: "ส่วนกลาง", confidence: "high"
❌ ห้าม: ระบุ "อุบลราชธานี" โดยไม่มีหลักฐานใดๆ

ตอบในรูปแบบ JSON เท่านั้น:
{{"province":"ชื่อจังหวัด หรือ ส่วนกลาง","district":"ชื่ออำเภอ หรือ null","confidence":"high หรือ medium หรือ low","reason":"อธิบายหลักฐานที่พบ"}}"""

    try:
        response = client.models.generate_content(
            model=_gemini_model(),  # Pro: ความแม่นยำสูงสุดสำหรับ location
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=400,
                temperature=0.02,  # ลด temperature ให้ deterministic มากขึ้น
            ),
        )
        raw = response.text.strip()
        logger.info(f"AI location raw response: {raw}")
        # Extract JSON — รองรับ nested objects และ markdown code blocks
        json_match = re.search(r'\{.*?\}', raw.replace('\n', ' '), re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            province = data.get("province", "").strip()
            district = (data.get("district") or "").strip() or None

            # Normalize confidence value
            conf_raw = str(data.get("confidence", "low")).strip().lower()
            if conf_raw in ("high", "สูง", "high confidence"):
                confidence = "high"
            elif conf_raw in ("medium", "กลาง", "ปานกลาง", "medium confidence"):
                confidence = "medium"
            else:
                confidence = "low"

            # "ส่วนกลาง" เป็น valid province
            if province == "ส่วนกลาง":
                return {
                    "province": "ส่วนกลาง",
                    "district": None,
                    "confidence": confidence,
                    "reason": data.get("reason", ""),
                }

            # Validate province is in Zone 10
            if province and province not in ZONE10_PROVINCES:
                province = _PROVINCE_ALIASES.get(province, None)

            # Validate district belongs to province
            if province and district and district not in ZONE10_PROVINCES.get(province, []):
                district = None  # district not valid for this province

            if province:
                return {
                    "province": province,
                    "district": district,
                    "confidence": confidence,
                    "reason": data.get("reason", ""),
                }
    except Exception as e:
        logger.warning(f"AI location detection failed: {e}")

    # Fallback: keyword search in text only (ignoring filename to prevent misclassification)
    combined = sample_text[:3000].lower()
    best_province = None
    best_district = None
    best_count = 0

    for province, districts in ZONE10_PROVINCES.items():
        if province == "ส่วนกลาง":
            continue
        count = combined.count(province.lower()) + combined.count(province[:4].lower())
        if count > best_count:
            best_count = count
            best_province = province
            best_district = None
            for district in districts:
                if district.lower() in combined:
                    best_district = district
                    break

    if best_province and best_count >= 1:  # แค่พบ 1 ครั้งก็ถือว่า medium
        conf = "high" if best_count >= 3 else "medium"
        return {"province": best_province, "district": best_district, "confidence": conf, "reason": f"keyword match (พบ {best_count} ครั้ง)"}

    return {"province": "ส่วนกลาง", "district": None, "confidence": "medium", "reason": "ไม่พบความเชื่อมโยงกับจังหวัดใดในเขต 10"}



def _resolve_rel_path(
    province: str | None,
    district: str | None,
    base_filename: str,
) -> str:
    """
    คำนวณ relative path (ใช้เป็น note_id prefix) ตามโครงสร้างเขต 10.

    - มีจังหวัดเท่านั้น → เขต10/{province}/{base_filename}
    - มีจังหวัด+อำเภอ  → เขต10/{province}/{district}/{base_filename}
    - ไม่มีจังหวัด     → {base_filename}
    """
    safe_name = _sanitize_filename(base_filename)
    if province and district:
        return f"เขต10/{province}/{district}/{safe_name}"
    if province:
        return f"เขต10/{province}/{safe_name}"
    return safe_name


def _update_province_index_db(
    province: str,
    district: str | None,
    base_filename: str,
    index_note_id: str,
    total_pages: int,
):
    """อัปเดต INDEX note ของจังหวัด/อำเภอใน DB เพิ่ม link ไปยังเอกสารใหม่."""
    safe_name = _sanitize_filename(base_filename)
    date_str = datetime.now().strftime('%d/%m/%Y')

    if district:
        doc_link = f"{district}/{safe_name}/{safe_name}-INDEX"
    else:
        doc_link = f"{safe_name}/{safe_name}-INDEX"

    entry = f"\n- [[{doc_link}|📄 {base_filename}]] ({total_pages} หน้า · {date_str})"

    try:
        # อัปเดต INDEX note ของจังหวัด
        prov_note_id = f"{VAULT_ID}::เขต10/{province}/INDEX"
        rows = query_db(
            "SELECT content FROM obsidian_notes WHERE note_id = %s",
            (prov_note_id,),
        )
        if rows:
            old_content = rows[0]["content"] or ""
            new_content = old_content + entry
            execute_db(
                "UPDATE obsidian_notes SET content = %s, content_stripped = %s, updated_at = NOW() WHERE note_id = %s",
                (new_content, new_content, prov_note_id),
            )

        # อัปเดต INDEX note ของอำเภอ (ถ้ามี)
        if district:
            dist_note_id = f"{VAULT_ID}::เขต10/{province}/{district}/INDEX"
            rows = query_db(
                "SELECT content FROM obsidian_notes WHERE note_id = %s",
                (dist_note_id,),
            )
            if not rows:
                # สร้าง INDEX note ของอำเภอใหม่
                dist_content = (
                    f"# {district} — {province}\n\n"
                    f"> **เขตสุขภาพที่ 10** · [[{province}/INDEX|← {province}]]\n\n"
                    f"## เอกสาร\n"
                )
                _upsert_note(
                    note_id=dist_note_id,
                    relative_path=f"เขต10/{province}/{district}/INDEX.md",
                    title=f"INDEX — {district}",
                    province=province,
                    district=district,
                    note_type="MOC",
                    tags=["index", province, district],
                    source_file="",
                    content=dist_content,
                    is_index=True,
                )
                rows = [{"content": dist_content}]
            old_content = rows[0]["content"] or ""
            dist_entry = f"\n- [[{safe_name}/{safe_name}-INDEX|📄 {base_filename}]] ({total_pages} หน้า · {date_str})"
            new_content = old_content + dist_entry
            execute_db(
                "UPDATE obsidian_notes SET content = %s, content_stripped = %s, updated_at = NOW() WHERE note_id = %s",
                (new_content, new_content, dist_note_id),
            )
    except Exception as e:
        logger.warning(f"Failed to update province index in DB: {e}")


# ── AI: Generate filename from content ───────────────────────────────────────

def _ai_generate_filename(client, uploaded_file, original_pdf_name: str) -> str:
    """ให้ Gemini วิเคราะห์เนื้อหาและสร้างชื่อโฟลเดอร์ภาษาไทยที่ระบุได้ชัดเจน"""
    prompt = f"""วิเคราะห์ไฟล์ PDF นี้ทั้งหมด แล้วตั้งชื่อโฟลเดอร์สำหรับเอกสารนี้ โดยใช้รูปแบบ:
[prefix]ชื่อเรื่อง-สถานที่-ปี

**prefix** (เลือกอันเดียว):
- R_ = รายงาน / สรุปผลการดำเนินงาน / ผลการประเมิน
- M_ = งานวิจัย / วิทยานิพนธ์ / จะแนะการศึกษา
- F_ = ฟอร์มสำรวจ / สถิติ / ข้อมูลระดับเขต
- P_ = แผนงาน / โครงการ / นโยบาย
- A_ = คู่มือ / แนวทางปฏิบัติ / มาตรฐาน
- I_ = บันทึกการประชุม / รายงานการตรวจราชการ

**ชื่อเรื่อง** (บังคับ):
- ต้องระบุเนื้อหาสำคัญสูงสุดให้ชัดเจน เช่น: พฤติกรรมสุขภาพ-ผู้สูงอายุ, โรคทางเดินอาหาร, อะฟลาทอกสาย
- หามใช้คำกำกวม เช่น: การดำเนินงาน, สาธารณสุข

**สถานที่** (ใส่ถ้าระบุไว้ในเอกสาร): ชื่อจังหวัดหรืออำเภอ เช่น: มุกดาหาร, อุบลราชธานี

**ปี** (ใส่ถ้าพบ): ปี พ.ศ. สั้น เช่น: 2565, 2567

ตัวอย่างชื่อที่ดี:
- R_รายงานตรวจราชการ-มุกดาหาร-2565
- M_พฤติกรรมสุขภาพ-ผู้สูงอายุ-อุบล-2564
- F_สถิติอะฟลาทอกสาย-ศรีสะเกษ
- P_แผนปฏิบัติการ-เขต10-2566

ชื่อไฟล์ต้นฉบับ: {original_pdf_name}
กฎ: ใช้ขีดกลาง (-) แทนช่องว่าง ห้ามไอ้อักษรพิเศษ ความยาว 20-100 ตัวอักษร ตอบเฉพาะชื่อเท่านั้น"""

    try:
        response = client.models.generate_content(
            model=_gemini_model_fast(),
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=120,
                temperature=0.2,
            ),
        )
        raw_name = response.text.strip()
        logger.info(f"AI filename raw response: {raw_name}")
        name = raw_name.split('\n')[0].strip()
        name = re.sub(r'[\\/:*?"<>|]', '-', name)
        name = name.strip('- ')
        return name if name else _sanitize_filename(original_pdf_name.replace('.pdf', ''))
    except Exception as e:
        logger.warning(f"AI filename generation failed: {e}")
        base = original_pdf_name.replace('.pdf', '').replace('.PDF', '')
        return _sanitize_filename(base)



# ── AI: Convert text chunk to Markdown ───────────────────────────────────────

def _ai_convert_to_markdown(
    client,
    uploaded_file,
    chunk_index: int,
    total_chunks: int,
    base_filename: str,
    page_start: int,
    page_end: int,
    prev_link: str | None,
    next_link: str | None,
    province: str | None,
    district: str | None,
    location_confidence: str,
) -> str:
    """ให้ Gemini 3 Pro แปลงข้อความเป็น Markdown สวยงาม พร้อม frontmatter และ wikilinks."""

    nav_links = []
    if prev_link:
        nav_links.append(f"← [[{prev_link}|ส่วนก่อนหน้า]]")
    if next_link:
        nav_links.append(f"[[{next_link}|ส่วนถัดไป]] →")
    nav_str = "  |  ".join(nav_links) if nav_links else ""

    location_tag = ""
    location_breadcrumb = ""
    if province:
        location_tag = f", {province}"
        if district:
            location_tag += f", {district}"
            location_breadcrumb = f"[[เขต10/INDEX|เขต 10]] > [[เขต10/{province}/INDEX|{province}]] > [[เขต10/{province}/{district}/INDEX|{district}]]"
        else:
            location_breadcrumb = f"[[เขต10/INDEX|เขต 10]] > [[เขต10/{province}/INDEX|{province}]]"

    tags_hint = f'pdf-ingest{location_tag}'

    prompt = f"""คุณเป็น AI ผู้เชี่ยวชาญด้านการจัดการความรู้ (Knowledge Management) สาธารณสุขเขต 10
ดึงเนื้อหาจากไฟล์ PDF ที่แนบมา เฉพาะหน้าที่ {page_start} ถึงหน้าที่ {page_end} เท่านั้น แล้วแปลงให้เป็น Markdown ที่สวยงาม อ่านง่าย และเป็นระเบียบ

**ข้อกำหนด:**
1. เริ่มด้วย YAML frontmatter ดังนี้ (ห้ามเปลี่ยนรูปแบบ):
```yaml
---
title: "{base_filename} (ส่วนที่ {chunk_index}/{total_chunks})"
source_pages: "หน้า {page_start}–{page_end}"
part: {chunk_index}/{total_chunks}
province: "{province or 'ไม่ระบุ'}"
district: "{district or 'ไม่ระบุ'}"
location_confidence: "{location_confidence}"
tags: [{tags_hint}]
created: {datetime.now().strftime('%Y-%m-%d')}
---
```

2. หลัง frontmatter ใส่ breadcrumb location และ navigation bar:
```
{location_breadcrumb}

{nav_str if nav_str else '*ไม่มีลิ้งนำทาง*'}
```
จากนั้นใส่เส้นคั่น `---`

3. แปลงเนื้อหาเป็น Markdown:
   - ใช้ # ## ### สำหรับหัวข้อ
   - ใช้ bullet list และ numbered list ตามความเหมาะสม
   - สร้าง table หากมีข้อมูลตาราง
   - ใช้ **ตัวหนา** สำหรับคำสำคัญ
   - รักษาความหมายเดิมไว้ทุกประการ
   - **สำคัญมาก:** ข้อความต้นฉบับอาจมีสระหรือวรรณยุกต์ภาษาไทยตกหล่น (เช่น 'มุ งเน น' หรือ 'ด าน' หรือมีอักขระสี่เหลี่ยม) ให้คุณช่วยแก้ไขคำให้ถูกต้องตามความหมายและบริบทของภาษาไทยโดยอัตโนมัติ (เช่น แก้เป็น 'มุ่งเน้น', 'ด้าน')
   - ถ้าเนื้อหาเป็นภาษาไทย ให้ใช้ภาษาไทย

4. ท้ายไฟล์ใส่ navigation ซ้ำ และ `---`

กรุณาดึงและแปลงเนื้อหาเฉพาะจากไฟล์ PDF หน้าที่ {page_start} ถึงหน้าที่ {page_end} เท่านั้น (ห้ามดึงหน้าอื่นมาปนเด็ดขาด)"""

    try:
        response = client.models.generate_content(
            model=_gemini_model(),
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=8192,
                temperature=0.2,
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"AI markdown conversion failed for chunk {chunk_index}: {e}")
        return f"""---
title: "{base_filename} (ส่วนที่ {chunk_index}/{total_chunks})"
source_pages: "หน้า {page_start}–{page_end}"
part: {chunk_index}/{total_chunks}
province: "{province or 'ไม่ระบุ'}"
district: "{district or 'ไม่ระบุ'}"
tags: [pdf-ingest]
created: {datetime.now().strftime('%Y-%m-%d')}
---

{location_breadcrumb}

{nav_str}

---

(เนื้อหา PDF ไม่สามารถแปลงเป็น Markdown ได้สำเร็จ)

---

{nav_str}
"""


# ── Core ingest function ──────────────────────────────────────────────────────

def _do_ingest(
    job_id: str,
    file_id: str,
    original_name: str,
    override_province: str | None = None,
    override_district: str | None = None,
    override_folder_name: str | None = None,
) -> None:
    """รัน ingest ใน background thread."""
    job = _jobs[job_id]
    job["status"] = "running"
    job["progress"] = []

    def log(msg: str):
        logger.info(f"[{job_id}] {msg}")
        job["progress"].append({"time": time.time(), "msg": msg})

    try:
        s = get_settings()
        chunk_size = s.PDF_INGEST_PAGES_PER_CHUNK

        # 1. ดึง PDF จาก MinIO
        log(f"📥 กำลังดึงไฟล์ {original_name} จาก MinIO...")
        client_minio = _get_client()
        resp = client_minio.get_object(_pdf_bucket(), file_id)
        file_bytes = resp.read()
        log(f"✅ ดึงไฟล์สำเร็จ ({len(file_bytes):,} bytes)")

        # 2. Extract pages
        log("📄 กำลังอ่านหน้า PDF...")
        pages = _extract_pages(file_bytes)
        total_pages = len(pages)
        log(f"✅ อ่าน PDF สำเร็จ: {total_pages} หน้า")

        if total_pages == 0:
            job["status"] = "error"
            job["error"] = "ไม่สามารถอ่านข้อความจาก PDF ได้ (อาจเป็น scanned PDF)"
            return

        # 3. Chunk pages
        chunks = _chunk_pages(pages, chunk_size)
        total_chunks = len(chunks)
        log(f"✂️ แบ่งเป็น {total_chunks} ส่วน (ส่วนละ {chunk_size} หน้า)")

        # 4. Gemini client
        gemini_client = _get_gemini_client()

        # 5. อัปโหลด PDF ให้ Gemini (Native PDF Understanding)
        log("📤 กำลังอัปโหลด PDF ให้ AI...")
        tmp_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        uploaded_file = None
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                uploaded_file = gemini_client.files.upload(file=tmp_path, config={'mime_type': 'application/pdf'})
                break
            except Exception as upload_err:
                if attempt == max_attempts:
                    raise upload_err
                wait_time = attempt * 2
                log(f"⚠️ อัปโหลดล้มเหลวชั่วคราว ({upload_err}) กำลังลองใหม่ใน {wait_time} วินาที... (ครั้งที่ {attempt}/{max_attempts})")
                time.sleep(wait_time)
        log("✅ อัปโหลดไฟล์ให้ AI สำเร็จ (หลีกเลี่ยงปัญหาฟอนต์ภาษาไทยของ PDF)")

        # ไฟล์ใหญ่ (เช่น PDF ~100MB) ต้องรอ Gemini ประมวลผลไฟล์ก่อน (state: PROCESSING → ACTIVE)
        # ไม่งั้นเรียก generate_content ทันทีจะได้ 400 INVALID_ARGUMENT ทุก chunk พร้อมกัน
        log("⏳ กำลังรอ AI ประมวลผลไฟล์...")
        wait_elapsed = 0
        poll_interval = 3
        max_wait = 300
        while uploaded_file.state.name == "PROCESSING":
            if wait_elapsed >= max_wait:
                raise RuntimeError(f"AI ประมวลผลไฟล์ไม่เสร็จภายใน {max_wait} วินาที")
            time.sleep(poll_interval)
            wait_elapsed += poll_interval
            uploaded_file = gemini_client.files.get(name=uploaded_file.name)
        if uploaded_file.state.name == "FAILED":
            raise RuntimeError(f"AI ประมวลผลไฟล์ล้มเหลว: {uploaded_file.error}")
        log(f"✅ ไฟล์พร้อมใช้งาน (state: {uploaded_file.state.name})")

        sample_text = "\n\n".join(pages[:min(5, total_pages)])

        run_ai_name = not override_folder_name
        run_ai_loc = not override_province or override_province == "auto"

        base_filename = override_folder_name or ""
        province = override_province if (override_province and override_province != "auto") else None
        district = override_district if (override_district and override_district != "auto") else None
        confidence = "manual"
        reason = "กำหนดโดยผู้ใช้"

        if run_ai_name or run_ai_loc:
            log("🤖 AI กำลังวิเคราะห์ข้อมูลเพิ่มเติม...")
            with ThreadPoolExecutor(max_workers=2) as meta_pool:
                fut_name = None
                fut_loc = None
                if run_ai_name:
                    fut_name = meta_pool.submit(_ai_generate_filename, gemini_client, uploaded_file, original_name)
                if run_ai_loc:
                    fut_loc  = meta_pool.submit(_ai_detect_location,  gemini_client, uploaded_file, sample_text, original_name)
                
                if fut_name:
                    base_filename = fut_name.result()
                if fut_loc:
                    location = fut_loc.result()
                    province = location["province"]
                    district = location["district"]
                    confidence = location["confidence"]
                    reason = location["reason"]

        log(f"📝 ชื่อเอกสาร: {base_filename}")
        if province == "ส่วนกลาง":
            district = None
        if district == "none" or district == "ไม่มี" or not district:
            district = None

        if not province or province not in ZONE10_PROVINCES:
            province = "ส่วนกลาง"
            district = None

        loc_str = f"{province}"
        if district:
            loc_str += f" / {district}"
        log(f"📍 ตำแหน่ง: {loc_str} (ความแม่นยำ: {confidence})")
        log(f"   เหตุผล: {reason}")

        # 6. Resolve relative path prefix (ใช้แทน filesystem path)
        rel_path = _resolve_rel_path(province, district, base_filename)
        log(f"📁 โครงสร้าง DB path: {rel_path}/")

        # Pre-compute all chunk names
        safe_name = _sanitize_filename(base_filename)
        chunk_filenames = []
        for i in range(total_chunks):
            if total_chunks == 1:
                chunk_filenames.append(safe_name)
            else:
                chunk_filenames.append(f"{safe_name}-ส่วนที่{i+1:02d}")

        # 7. Convert each chunk to MD (parallel)
        created_files: list[str] = []
        md_results: dict[int, str] = {}

        def _convert_chunk(i: int) -> tuple[int, str]:
            chunk_num = i + 1
            page_start = i * chunk_size + 1
            page_end = min(page_start + chunk_size - 1, total_pages)
            prev_link = chunk_filenames[i - 1] if i > 0 else None
            next_link = chunk_filenames[i + 1] if i < total_chunks - 1 else None
            content = _ai_convert_to_markdown(
                client=gemini_client,
                uploaded_file=uploaded_file,
                chunk_index=chunk_num,
                total_chunks=total_chunks,
                base_filename=base_filename,
                page_start=page_start,
                page_end=page_end,
                prev_link=prev_link,
                next_link=next_link,
                province=province,
                district=district,
                location_confidence=confidence,
            )
            return i, content

        max_parallel = min(s.PDF_INGEST_MAX_PARALLEL, total_chunks)
        log(f"⚡ แปลง {total_chunks} ส่วนพร้อมกัน (สูงสุด {max_parallel} threads)...")

        with ThreadPoolExecutor(max_workers=max_parallel) as chunk_pool:
            futures = {chunk_pool.submit(_convert_chunk, i): i for i in range(total_chunks)}
            for fut in as_completed(futures):
                idx, content = fut.result()
                chunk_num = idx + 1
                page_start = idx * chunk_size + 1
                page_end = min(page_start + chunk_size - 1, total_pages)
                log(f"✅ แปลงส่วนที่ {chunk_num}/{total_chunks} สำเร็จ (หน้า {page_start}–{page_end})")
                md_results[idx] = content

        # ── บันทึก chunks ลง Database (แทน filesystem) ────────────────────────
        current_year = datetime.now().year
        for i in range(total_chunks):
            chunk_name = chunk_filenames[i]
            note_id = f"{VAULT_ID}::{rel_path}/{chunk_name}"
            chunk_num = i + 1
            page_start = i * chunk_size + 1
            page_end = min(page_start + chunk_size - 1, total_pages)
            _upsert_note(
                note_id=note_id,
                relative_path=f"{rel_path}/{chunk_name}.md",
                title=f"{base_filename} (ส่วนที่ {chunk_num}/{total_chunks})",
                province=province,
                district=district,
                note_type="report",
                tags=["pdf-ingest", province or "ส่วนกลาง"],
                source_file=original_name,
                content=md_results[i],
                file_id=file_id,
                chunk_index=i + 1,
                is_index=False,
                year=current_year,
            )
            created_files.append(f"{rel_path}/{chunk_name}.md")
            log(f"💾 บันทึก DB: {rel_path}/{chunk_name}.md")

        # 7.5 Register PDF ใน obsidian_pdf_assets (link กลับ MinIO)
        log("📎 ลงทะเบียน PDF ต้นฉบับใน obsidian_pdf_assets...")
        # สร้าง index_note_id ก่อนเพื่อใช้เป็น FK
        index_note_id = f"{VAULT_ID}::{rel_path}/{safe_name}-INDEX"
        try:
            s_cfg = get_settings()
            scheme = "https" if s_cfg.MINIO_USE_SSL else "http"
            minio_url = f"{scheme}://{s_cfg.minio_endpoint_url}/{s_cfg.PDF_BUCKET}/{file_id}"
            execute_db(
                """
                INSERT INTO obsidian_pdf_assets
                    (province, note_id, filename, minio_path, minio_url, file_size, content_type)
                VALUES (%s, %s, %s, %s, %s, %s, 'application/pdf')
                ON CONFLICT (minio_path) DO UPDATE SET
                    filename  = EXCLUDED.filename,
                    note_id   = EXCLUDED.note_id,
                    minio_url = EXCLUDED.minio_url,
                    file_size = EXCLUDED.file_size
                """,
                (province or "ส่วนกลาง", index_note_id, original_name, file_id, minio_url, len(file_bytes)),
            )
            log(f"✅ ลงทะเบียน PDF ใน obsidian_pdf_assets เรียบร้อย")
        except Exception as pdf_reg_err:
            logger.warning(f"Failed to register PDF in obsidian_pdf_assets: {pdf_reg_err}")
            log(f"⚠️ ไม่สามารถลงทะเบียน PDF ได้: {pdf_reg_err}")

        # 8. Create INDEX note ใน DB
        log("📚 กำลังสร้าง Index note ใน DB...")
        breadcrumb = ""
        if province:
            breadcrumb = f"\n\n[[เขต10/INDEX|เขต 10]] > [[เขต10/{province}/INDEX|{province}]]"
            if district:
                breadcrumb += f" > [[เขต10/{province}/{district}/INDEX|{district}]]"
            breadcrumb += f" > 📁 {safe_name}"

        index_lines = [
            f"# {base_filename} — ดัชนีเนื้อหา\n",
            f"> **แหล่งที่มา:** `{original_name}` | **MinIO ID:** `{file_id}`\n",
            f"> **จำนวนหน้า:** {total_pages} หน้า | **แบ่งเป็น:** {total_chunks} ส่วน\n",
            f"> **จังหวัด:** {province or 'ไม่ระบุ'} | **อำเภอ:** {district or 'ไม่ระบุ'}\n",
            f"> **วันที่ ingest:** {datetime.now().strftime('%d %B %Y %H:%M')}\n",
            breadcrumb,
            "\n---\n\n## 📎 ไฟล์ PDF ต้นฉบับ (MinIO)\n",
            f"- file_id: `{file_id}` (ดูผ่าน /pdf/view/{file_id})\n",
            "\n## 📑 รายการส่วน\n",
        ]
        for i, fname in enumerate(chunk_filenames):
            page_start = i * chunk_size + 1
            page_end = min(page_start + chunk_size - 1, total_pages)
            index_lines.append(f"- [[{fname}|ส่วนที่ {i+1}: หน้า {page_start}–{page_end}]]")

        index_content = "\n".join(index_lines)
        _upsert_note(
            note_id=index_note_id,
            relative_path=f"{rel_path}/{safe_name}-INDEX.md",
            title=f"{base_filename} — ดัชนี",
            province=province,
            district=district,
            note_type="MOC",
            tags=["pdf-ingest", "index", province or "ส่วนกลาง"],
            source_file=original_name,
            content=index_content,
            file_id=file_id,
            chunk_index=0,
            is_index=True,
            year=current_year,
        )
        index_filename = f"{safe_name}-INDEX.md"
        log(f"✅ สร้าง Index note: {rel_path}/{index_filename}")

        # 9. อัปเดต INDEX note ของจังหวัด/อำเภอใน DB
        if province:
            _update_province_index_db(province, district, base_filename, index_note_id, total_pages)
            log(f"🔗 อัปเดต Index จังหวัด {province} ใน DB แล้ว")

        # 9.5 Update metadata in MinIO to persist ingest status
        try:
            from minio.commonconfig import CopySource
            import urllib.parse
            bucket = _pdf_bucket()
            stat = client_minio.stat_object(bucket, file_id)
            meta = {k.lower(): v for k, v in (stat.metadata or {}).items()}
            
            custom_meta = {}
            for k, v in meta.items():
                if k.startswith('x-amz-meta-'):
                    key = k[len('x-amz-meta-'):]
                    custom_meta[key] = v
            
            custom_meta['ingested'] = 'true'
            custom_meta['province'] = urllib.parse.quote(province or "")
            custom_meta['district'] = urllib.parse.quote(district or "")
            custom_meta['savedat'] = urllib.parse.quote(rel_path or "")
            custom_meta['confidence'] = confidence or "low"
            # Ensure Content-Type is preserved if present
            content_type = stat.metadata.get('Content-Type') or stat.metadata.get('content-type') or 'application/pdf'
            custom_meta['Content-Type'] = content_type
            
            client_minio.copy_object(
                bucket,
                file_id,
                CopySource(bucket, file_id),
                metadata=custom_meta,
                metadata_directive='REPLACE'
            )
            log(f"💾 อัปเดตสถานะ Ingested ลงใน MinIO Metadata เรียบร้อยแล้ว")
        except Exception as meta_err:
            logger.warning(f"Failed to update MinIO metadata: {meta_err}", exc_info=True)
            log(f"⚠️ ไม่สามารถอัปเดตสถานะลงใน MinIO Metadata ได้: {meta_err}")

        job["status"] = "completed"
        job["result"] = {
            "base_filename": base_filename,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "created_files": created_files,
            "index_file": f"{rel_path}/{index_filename}",
            "vault_path": f"[database] {rel_path}",
            "province": province,
            "district": district,
            "location_confidence": confidence,
            "saved_at": rel_path,
        }
        log(f"🎉 Ingest สำเร็จ! บันทึก {len(created_files)} ไฟล์ MD ที่ {rel_path}")

    except Exception as e:
        logger.error(f"Error in background ingest: {e}", exc_info=True)
        job["status"] = "error"
        job["error"] = str(e)
        log(f"❌ ล้มเหลว: {e}")
    finally:
        if uploaded_file:
            try:
                gemini_client.files.delete(name=uploaded_file.name)
                log("🧹 ลบไฟล์ PDF ชั่วคราวออกจาก Google Cloud")
            except Exception as e:
                logger.warning(f"Failed to delete file from Gemini: {e}")
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass


# ── Routes ────────────────────────────────────────────────────────────────────

import urllib.parse as _urllib_parse
from fastapi import UploadFile, File

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """รับไฟล์ PDF และบันทึกลงใน MinIO pdf-library bucket."""
    if not (file.filename or "").lower().endswith(".pdf") and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    _ensure_pdf_bucket()
    client_minio = _get_client()
    bucket = _pdf_bucket()

    original_name = file.filename or "document.pdf"

    # Check duplicate name in MinIO
    try:
        objects = list(client_minio.list_objects(bucket, recursive=True))
        for obj in objects:
            if obj.object_name.endswith("__apa.json") or obj.object_name.endswith("__path.json"):
                continue
            try:
                stat = client_minio.stat_object(bucket, obj.object_name)
                meta = {k.lower(): v for k, v in (stat.metadata or {}).items()}
                orig_name = _urllib_parse.unquote(meta.get("x-amz-meta-name", ""))
                if orig_name == original_name:
                    raise HTTPException(
                        status_code=400,
                        detail=f"ไฟล์ '{original_name}' เคยอัปโหลดในระบบแล้ว"
                    )
            except HTTPException:
                raise
            except Exception:
                pass
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Error checking duplicate file in MinIO: {e}")

    # Generate unique file ID
    import random
    for _ in range(100):
        file_id = str(random.randint(100000, 999999))
        try:
            client_minio.stat_object(bucket, file_id)
        except Exception:
            break  # ID is available

    file_bytes = await file.read()
    file_size = len(file_bytes)
    original_name = file.filename or "document.pdf"
    uploaded_at = int(time.time() * 1000)

    import io as _io
    stream = _io.BytesIO(file_bytes)
    meta = {
        "x-amz-meta-name": _urllib_parse.quote(original_name[:150]),
        "x-amz-meta-extension": "pdf",
        "x-amz-meta-size": str(file_size),
        "x-amz-meta-uploadedat": str(uploaded_at),
        "Content-Type": "application/pdf",
    }
    client_minio.put_object(bucket, file_id, stream, file_size, metadata=meta)
    logger.info(f"Uploaded PDF to {bucket}/{file_id}: {original_name}")

    return {
        "id": file_id,
        "name": original_name,
        "size": file_size,
        "uploadedAt": uploaded_at,
        "ingested": False,
    }


@router.post("/ingest/{file_id}")
async def ingest_pdf(
    file_id: str,
    background_tasks: BackgroundTasks,
    original_name: str = "document.pdf",
    province: str = None,
    district: str = None,
    folder_name: str = None,
):
    """เริ่ม ingest PDF จาก MinIO → Markdown → Obsidian vault (จัดเก็บตามจังหวัด/อำเภอ เขต 10)."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "file_id": file_id,
        "original_name": original_name,
        "status": "queued",
        "progress": [],
        "created_at": time.time(),
    }
    background_tasks.add_task(_do_ingest, job_id, file_id, original_name, province, district, folder_name)
    return {"job_id": job_id, "status": "queued"}


@router.get("/ingest/status/{job_id}")
async def get_ingest_status(job_id: str):
    """ดู status และ progress ของ ingest job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


async def _stream_job_progress(job_id: str) -> AsyncIterator[str]:
    """SSE stream สำหรับ real-time progress."""
    last_count = 0
    max_wait = 300  # 5 minutes timeout
    start = time.time()

    while time.time() - start < max_wait:
        if job_id not in _jobs:
            yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
            return

        job = _jobs[job_id]
        progress = job.get("progress", [])

        if len(progress) > last_count:
            for msg_obj in progress[last_count:]:
                yield f"data: {json.dumps({'msg': msg_obj['msg'], 'status': job['status']})}\n\n"
            last_count = len(progress)

        if job["status"] in ("completed", "error"):
            final_data = {
                "status": job["status"],
                "done": True,
                "result": job.get("result"),
                "error": job.get("error"),
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            return

        import asyncio
        await asyncio.sleep(1)

    yield f"data: {json.dumps({'error': 'Timeout', 'done': True})}\n\n"


@router.get("/ingest/stream/{job_id}")
async def stream_ingest_progress(job_id: str):
    """SSE endpoint สำหรับ real-time progress streaming."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return StreamingResponse(
        _stream_job_progress(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/view/{file_id}")
async def view_pdf(file_id: str):
    """Stream PDF จาก MinIO pdf-library bucket."""
    from fastapi.responses import Response
    client_minio = _get_client()
    bucket = _pdf_bucket()
    try:
        resp = client_minio.get_object(bucket, file_id)
        data = resp.read()
        return Response(
            content=data,
            media_type="application/pdf",
            headers={"Content-Disposition": "inline"},
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"File not found: {exc}")


@router.get("/files")
async def list_pdf_files():
    """List PDF files ทั้งหมดใน MinIO bucket พร้อม cross-check ข้อมูล ingest จาก DB."""
    import urllib.parse

    client_minio = _get_client()
    bucket = _pdf_bucket()

    # ดึงข้อมูล ingest ทั้งหมดจาก DB ครั้งเดียว (key = file_id)
    db_all = query_db(
        """
        SELECT file_id,
               source_file,
               COUNT(*) AS chunks,
               MIN(province)  AS province,
               MIN(district)  AS district,
               MIN(relative_path) AS sample_path
        FROM obsidian_notes
        WHERE file_id IS NOT NULL OR source_file IS NOT NULL
        GROUP BY file_id, source_file
        """,
        ()
    )
    # Map 1: by file_id (exact MinIO object name) — highest priority
    db_by_fileid: dict = {}
    # Map 2: by source_file name (original PDF name) — fallback for legacy data
    db_by_sourcefile: dict = {}
    for r in db_all:
        fid = str(r["file_id"]) if r["file_id"] else None
        sfn = str(r["source_file"]) if r["source_file"] else None
        if fid:
            prev = db_by_fileid.get(fid)
            if not prev or int(r["chunks"]) > int(prev["chunks"]):
                db_by_fileid[fid] = r
        if sfn:
            prev = db_by_sourcefile.get(sfn)
            if not prev or int(r["chunks"]) > int(prev["chunks"]):
                db_by_sourcefile[sfn] = r

    try:
        objects = list(client_minio.list_objects(bucket, recursive=True))
        files = []
        for obj in objects:
            name = obj.object_name
            if name.endswith("__apa.json") or name.endswith("__path.json") or name.endswith(".json"):
                continue
            try:
                stat = client_minio.stat_object(bucket, name)
                meta = {k.lower(): v for k, v in (stat.metadata or {}).items()}
                ext = meta.get("x-amz-meta-extension", "").lower()
                orig_name = urllib.parse.unquote(meta.get("x-amz-meta-name", name))

                if ext and ext not in ("pdf", ""):
                    continue
                if not orig_name.lower().endswith(".pdf") and not name.lower().endswith(".pdf") and ext != "pdf":
                    continue

                # --- ข้อมูลจาก DB (source of truth) ---
                # Priority: match by file_id > match by source_file name > none
                db_row = db_by_fileid.get(str(name))
                if not db_row:
                    db_row = db_by_sourcefile.get(orig_name)
                db_ingested = db_row is not None and int(db_row.get("chunks") or 0) > 0
                db_province = db_row["province"] if db_row else None
                db_district = db_row["district"] if db_row else None
                db_chunks = int(db_row["chunks"]) if db_row else 0
                db_saved_at = "/".join(db_row["sample_path"].split("/")[:-1]) if db_row and db_row.get("sample_path") else None

                # --- ข้อมูลจาก MinIO metadata (fallback) ---
                meta_ingested = meta.get("x-amz-meta-ingested") == "true"
                meta_province = urllib.parse.unquote(meta.get("x-amz-meta-province", "")) or None
                meta_district = urllib.parse.unquote(meta.get("x-amz-meta-district", "")) or None
                meta_savedat = urllib.parse.unquote(meta.get("x-amz-meta-savedat", "")) or None
                meta_confidence = meta.get("x-amz-meta-confidence", "") or None

                # --- in-memory job (running/queued override) ---
                in_memory_job = next(
                    (j for j in _jobs.values() if j.get("file_id") == name),
                    None
                )

                if in_memory_job:
                    job_status = in_memory_job.get("status")
                    job_result = in_memory_job.get("result") or {}
                    ingested = db_ingested or (job_status == "completed") or meta_ingested
                    province = job_result.get("province") or db_province or meta_province
                    district = job_result.get("district") or db_district or meta_district
                    saved_at = job_result.get("saved_at") or db_saved_at or meta_savedat
                    location_confidence = job_result.get("location_confidence") or meta_confidence
                    ingest_status = job_status
                    ingest_job_id = in_memory_job.get("job_id")
                else:
                    ingested = db_ingested or meta_ingested
                    province = db_province or meta_province
                    district = db_district or meta_district
                    saved_at = db_saved_at or meta_savedat
                    location_confidence = meta_confidence
                    ingest_status = "completed" if ingested else None
                    ingest_job_id = None

                files.append({
                    "id": name,
                    "name": orig_name,
                    "size": obj.size,
                    "uploaded_at": meta.get("x-amz-meta-uploadedat", str(int(obj.last_modified.timestamp() * 1000)) if obj.last_modified else ""),
                    "ingested": ingested,
                    "ingestJobId": ingest_job_id,
                    "ingestStatus": ingest_status,
                    "province": province,
                    "district": district,
                    "saved_at": saved_at,
                    "location_confidence": location_confidence,
                    "notes_count": db_chunks,   # จำนวน chunks ใน DB
                })
            except Exception:
                pass
        files.sort(key=lambda f: f.get("uploaded_at", ""), reverse=True)
        return {"files": files}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/files/{file_id:path}")
async def delete_pdf_file(file_id: str):
    """ลบไฟล์ PDF จาก MinIO bucket."""
    client_minio = _get_client()
    bucket = _pdf_bucket()
    try:
        client_minio.remove_object(bucket, file_id)
        # ลบ metadata files ที่เกี่ยวข้อง (ถ้ามี)
        for suffix in ("__apa.json", "__path.json"):
            try:
                client_minio.remove_object(bucket, file_id + suffix)
            except Exception:
                pass
        return {"deleted": True, "file_id": file_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/vault/files")
async def list_vault_files():
    """List Markdown notes จาก database (obsidian_notes) แบบ tree structure.

    หมายเหตุ (028): เปลี่ยนจาก filesystem scan เป็น DB query
    """
    rows = query_db(
        """
        SELECT note_id, relative_path, title, province, district, note_type,
               is_index, chunk_index, file_id,
               length(content) AS size,
               EXTRACT(EPOCH FROM updated_at)::bigint AS modified_at
        FROM obsidian_notes
        WHERE vault_id = %s
        ORDER BY province NULLS LAST, relative_path
        """,
        (VAULT_ID,),
    )

    # ── Build tree grouped by province/district ────────────────────────────────
    from collections import defaultdict

    # group notes by province
    by_province: dict[str, list] = defaultdict(list)
    root_notes = []
    for r in rows:
        prov = r.get("province")
        if prov and r["relative_path"].startswith("เขต10/"):
            by_province[prov].append(r)
        else:
            root_notes.append(r)

    def _note_to_file(r: dict) -> dict:
        rel = r["relative_path"]  # e.g. เขต10/อุบล/docname/chunk.md
        name = rel.split("/")[-1]  # filename
        return {
            "type": "file",
            "name": name,
            "path": rel.replace(".md", ""),   # use path without .md as note_id suffix
            "size": r.get("size") or 0,
            "modified_at": r.get("modified_at") or 0,
            "file_id": r.get("file_id"),
            "is_index": r.get("is_index") or False,
        }

    zone10_tree = []
    for province in ZONE10_PROVINCES:
        prov_notes = by_province.get(province, [])
        if not prov_notes:
            continue

        # group by district
        by_district: dict[str | None, list] = defaultdict(list)
        for r in prov_notes:
            by_district[r.get("district")].append(r)

        prov_children = []
        for dist, dist_notes in by_district.items():
            # group by document folder (base_name = path segment after province/district)
            by_doc: dict[str, list] = defaultdict(list)
            for r in dist_notes:
                parts = r["relative_path"].replace(".md", "").split("/")
                # parts: [เขต10, province, (district?), docfolder, filename]
                doc_folder = parts[-2] if len(parts) >= 2 else "root"
                by_doc[doc_folder].append(r)

            if dist:
                dist_children = []
                for doc_folder, doc_notes in sorted(by_doc.items()):
                    folder_files = [_note_to_file(r) for r in doc_notes]
                    # Try get file_id from index note
                    idx_note = next((r for r in doc_notes if r.get("is_index")), None)
                    dist_children.append({
                        "type": "folder",
                        "name": doc_folder,
                        "path": f"เขต10/{province}/{dist}/{doc_folder}",
                        "children": folder_files,
                        "file_id": idx_note["file_id"] if idx_note else None,
                    })
                prov_children.append({
                    "type": "folder",
                    "name": dist,
                    "path": f"เขต10/{province}/{dist}",
                    "children": dist_children,
                })
            else:
                for doc_folder, doc_notes in sorted(by_doc.items()):
                    folder_files = [_note_to_file(r) for r in doc_notes]
                    idx_note = next((r for r in doc_notes if r.get("is_index")), None)
                    prov_children.append({
                        "type": "folder",
                        "name": doc_folder,
                        "path": f"เขต10/{province}/{doc_folder}",
                        "children": folder_files,
                        "file_id": idx_note["file_id"] if idx_note else None,
                    })

        zone10_tree.append({
            "type": "folder",
            "name": province,
            "path": f"เขต10/{province}",
            "children": prov_children,
        })

    root_files = [_note_to_file(r) for r in root_notes]

    return {
        "zone10": zone10_tree,
        "root_files": root_files,
        "provinces": list(ZONE10_PROVINCES.keys()),
        "vault_path": "[database]",
    }


@router.get("/vault/db-stats")
async def get_vault_db_stats():
    """
    แสดงสถิติข้อมูล Vault ที่เก็บใน database — ใช้สำหรับตรวจสอบว่าข้อมูลเข้า DB จริง.
    """
    # summary
    summary = query_db("""
        SELECT
            COUNT(*)                                    AS total_notes,
            COUNT(CASE WHEN is_index THEN 1 END)        AS index_notes,
            COUNT(CASE WHEN NOT is_index THEN 1 END)    AS chunk_notes,
            COUNT(DISTINCT province)                    AS provinces,
            COUNT(DISTINCT source_file)                 AS source_files,
            COUNT(file_id)                              AS with_pdf_link,
            MAX(updated_at)                             AS last_updated
        FROM obsidian_notes
        WHERE vault_id = %s
    """, (VAULT_ID,))

    # per province breakdown
    by_province = query_db("""
        SELECT
            COALESCE(province, '(ไม่ระบุ)')    AS province,
            COUNT(*)                            AS notes,
            COUNT(CASE WHEN is_index THEN 1 END) AS index_notes,
            COUNT(DISTINCT district)            AS districts,
            COUNT(file_id)                      AS with_pdf,
            COUNT(DISTINCT source_file)         AS documents
        FROM obsidian_notes
        WHERE vault_id = %s
        GROUP BY province
        ORDER BY province NULLS LAST
    """, (VAULT_ID,))

    # recent 10 notes
    recent = query_db("""
        SELECT
            note_id,
            relative_path,
            title,
            province,
            district,
            is_index,
            chunk_index,
            file_id,
            TO_CHAR(updated_at, 'YYYY-MM-DD HH24:MI') AS updated
        FROM obsidian_notes
        WHERE vault_id = %s
        ORDER BY updated_at DESC
        LIMIT 10
    """, (VAULT_ID,))

    s = summary[0] if summary else {}
    return {
        "vault_id": VAULT_ID,
        "summary": {
            "total_notes": s.get("total_notes", 0),
            "index_notes": s.get("index_notes", 0),
            "chunk_notes": s.get("chunk_notes", 0),
            "provinces": s.get("provinces", 0),
            "source_files": s.get("source_files", 0),
            "with_pdf_link": s.get("with_pdf_link", 0),
            "last_updated": str(s.get("last_updated", "")) if s.get("last_updated") else None,
        },
        "by_province": by_province,
        "recent_notes": recent,
    }


@router.get("/zone10")
async def get_zone10_structure():
    """ดูโครงสร้าง 5 จังหวัดเขต 10 พร้อมรายชื่ออำเภอ."""
    return {
        "zone": "สาธารณสุขเขต 10",
        "provinces": {
            province: {
                "name": province,
                "districts": districts,
                "district_count": len(districts),
            }
            for province, districts in ZONE10_PROVINCES.items()
        },
        "total_provinces": len(ZONE10_PROVINCES),
        "total_districts": sum(len(d) for d in ZONE10_PROVINCES.values()),
    }


# ── Vault CRUD endpoints (DB-based since migration 028) ───────────────────────

from fastapi import Body


def _note_id_from_path(path: str) -> str:
    """แปลง relative path (จาก frontend) เป็น note_id ใน DB."""
    # path อาจมี .md หรือไม่ก็ได้ — normalize โดยไม่มี .md
    clean = path.replace(".md", "")
    return f"{VAULT_ID}::{clean}"


@router.get("/vault/file")
async def read_vault_file(path: str):
    """อ่านเนื้อหา Markdown note จาก database."""
    note_id = _note_id_from_path(path)
    rows = query_db(
        "SELECT note_id, relative_path, title, content, is_index, "
        "length(content) AS size, EXTRACT(EPOCH FROM updated_at)::bigint AS modified_at "
        "FROM obsidian_notes WHERE note_id = %s",
        (note_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"Note not found: {path}")
    r = rows[0]
    name = r["relative_path"].split("/")[-1]
    return {
        "path": path,
        "name": name,
        "content": r["content"] or "",
        "size": r.get("size") or 0,
        "modified_at": r.get("modified_at") or 0,
    }


@router.put("/vault/file")
async def write_vault_file(
    path: str,
    body: dict = Body(...),
):
    """อัปเดตหรือสร้าง Markdown note ใน database."""
    import re as _re
    content = body.get("content", "")
    note_id = _note_id_from_path(path)
    stripped = _re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, count=1, flags=_re.DOTALL).strip()
    name = path.replace(".md", "").split("/")[-1]
    relative_path = path if path.endswith(".md") else path + ".md"

    rows = query_db("SELECT note_id FROM obsidian_notes WHERE note_id = %s", (note_id,))
    if rows:
        execute_db(
            "UPDATE obsidian_notes SET content = %s, content_stripped = %s, updated_at = NOW() WHERE note_id = %s",
            (content, stripped, note_id),
        )
    else:
        # สร้าง note ใหม่
        _upsert_note(
            note_id=note_id,
            relative_path=relative_path,
            title=name,
            province=None,
            district=None,
            note_type="report",
            tags=[],
            source_file="",
            content=content,
        )
    return {"path": path, "size": len(content.encode()), "modified_at": int(time.time())}


@router.delete("/vault/file")
async def delete_vault_file(path: str):
    """ลบ note หรือ prefix ทั้งหมดออกจาก database."""
    note_id = _note_id_from_path(path)
    # ลบ exact match
    rows = query_db("SELECT note_id FROM obsidian_notes WHERE note_id = %s", (note_id,))
    if rows:
        execute_db("DELETE FROM obsidian_notes WHERE note_id = %s", (note_id,))
        return {"deleted": True, "path": path, "count": 1}
    # ลบทั้ง prefix (folder delete)
    prefix = note_id + "::"
    # note_id format: VAULT_ID::rel/path — ลบที่มี rel/path ขึ้นต้นด้วย clean path
    clean = path.replace(".md", "")
    execute_db(
        "DELETE FROM obsidian_notes WHERE vault_id = %s AND relative_path LIKE %s",
        (VAULT_ID, f"{clean}%"),
    )
    return {"deleted": True, "path": path}


@router.post("/vault/rename")
async def rename_vault_file(body: dict = Body(...)):
    """เปลี่ยนชื่อ note ใน database (อัปเดต note_id และ relative_path)."""
    old_path = body.get("old_path", "")
    new_path = body.get("new_path", "")
    if not old_path or not new_path:
        raise HTTPException(status_code=400, detail="old_path and new_path required")
    old_id = _note_id_from_path(old_path)
    new_id = _note_id_from_path(new_path)
    new_rel = new_path if new_path.endswith(".md") else new_path + ".md"
    rows = query_db("SELECT note_id FROM obsidian_notes WHERE note_id = %s", (old_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="Source not found")
    execute_db(
        "UPDATE obsidian_notes SET note_id = %s, relative_path = %s, updated_at = NOW() WHERE note_id = %s",
        (new_id, new_rel, old_id),
    )
    return {"renamed": True, "old_path": old_path, "new_path": new_path}


@router.get("/vault/folders")
async def list_vault_folders():
    """List folder paths ที่มีใน obsidian_notes (สำหรับ dropdown)."""
    rows = query_db(
        "SELECT DISTINCT relative_path FROM obsidian_notes WHERE vault_id = %s ORDER BY relative_path",
        (VAULT_ID,),
    )
    folders: set[str] = set()
    for r in rows:
        parts = r["relative_path"].replace(".md", "").split("/")
        for i in range(1, len(parts)):
            folders.add("/".join(parts[:i]))
    return {"folders": sorted(folders)}


# ── PDF View Endpoints ────────────────────────────────────────────────────────

@router.get("/view/obsidian/{asset_id}")
async def view_obsidian_pdf(asset_id: int):
    """Stream PDF จาก MinIO โดยค้นหา minio_path จาก obsidian_pdf_assets.

    ใช้ asset_id (PK ของ obsidian_pdf_assets) เพื่อ serve PDF ต้นฉบับแบบ inline.
    """
    from fastapi.responses import Response
    rows = query_db(
        "SELECT minio_path, filename, content_type FROM obsidian_pdf_assets WHERE id = %s",
        (asset_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_id}")
    r = rows[0]
    minio_path = r["minio_path"]
    content_type = r.get("content_type") or "application/pdf"
    filename = r.get("filename") or "document.pdf"

    # ลอง pdf-library bucket ก่อน (file_id เก็บ minio_path โดยตรง)
    client_minio = _get_client()
    s_cfg = get_settings()
    for bucket in (s_cfg.PDF_BUCKET, "obsidian-pdfs", s_cfg.MINIO_BUCKET):
        try:
            resp = client_minio.get_object(bucket, minio_path)
            data = resp.read()
            return Response(
                content=data,
                media_type=content_type,
                headers={"Content-Disposition": f'inline; filename="{filename}"'},
            )
        except Exception:
            continue
    raise HTTPException(status_code=404, detail=f"PDF not found in MinIO: {minio_path}")


@router.get("/view/obsidian/by-note/{note_id:path}")
async def view_obsidian_pdf_by_note(note_id: str):
    """Stream PDF ต้นฉบับของ note ที่กำหนด (ค้นจาก obsidian_pdf_assets.note_id)."""
    from fastapi.responses import Response
    full_note_id = f"{VAULT_ID}::{note_id}" if not note_id.startswith(VAULT_ID) else note_id
    rows = query_db(
        "SELECT id, minio_path, filename, content_type FROM obsidian_pdf_assets WHERE note_id = %s",
        (full_note_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"No PDF linked to note: {note_id}")
    r = rows[0]
    minio_path = r["minio_path"]
    content_type = r.get("content_type") or "application/pdf"
    filename = r.get("filename") or "document.pdf"

    client_minio = _get_client()
    s_cfg = get_settings()
    for bucket in (s_cfg.PDF_BUCKET, "obsidian-pdfs", s_cfg.MINIO_BUCKET):
        try:
            resp = client_minio.get_object(bucket, minio_path)
            data = resp.read()
            return Response(
                content=data,
                media_type=content_type,
                headers={"Content-Disposition": f'inline; filename="{filename}"'},
            )
        except Exception:
            continue
    raise HTTPException(status_code=404, detail=f"PDF not found in MinIO: {minio_path}")


# -- Filesystem to Database Migration -----------------------------------------

@router.post("/vault/migrate-from-filesystem")
async def migrate_vault_from_filesystem():
    """
    (Legacy) ระบบย้ายมาใช้ database แล้ว — ไม่มี filesystem vault อีกต่อไป.
    ข้อมูลทั้งหมดอยู่ใน obsidian_notes table แล้ว.
    """
    rows = query_db(
        "SELECT COUNT(*) AS total FROM obsidian_notes WHERE vault_id = %s",
        (VAULT_ID,)
    )
    total = rows[0]["total"] if rows else 0
    return {
        "message": f"Vault is already database-native. {total} notes in DB. No filesystem migration needed.",
        "vault_path": "[database]",
        "imported": 0,
        "errors": [],
        "files": [],
    }
