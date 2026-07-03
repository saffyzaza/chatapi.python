"""PDF Ingest Router — แปลง PDF → Markdown chunks → บันทึกใน Obsidian vault (musya knowlads).
จัดระเบียบตามโครงสร้าง เขต10 / จังหวัด / อำเภอ ของสาธารณสุขเขต 10.
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
    """ลบอักขระพิเศษที่ไม่เหมาะกับชื่อไฟล์ แต่รักษา Thai/Unicode

    ตัดความยาวตามจำนวน **byte ของ UTF-8** ไม่ใช่จำนวนตัวอักษร — ภาษาไทย
    เข้ารหัสเป็น 3 byte/ตัวอักษร ชื่อไฟล์ 120 ตัวอักษรอาจยาวถึง 360 byte
    ซึ่งเกินขีดจำกัด 255 byte/path-component ของ ext4 ทำให้เกิด
    OSError: [Errno 36] File name too long ตอน mkdir/เขียนไฟล์
    """
    name = re.sub(r'[\\/:*?"<>|]', '-', name)
    name = name.strip('. ')
    max_bytes = 150  # เผื่อที่ให้ suffix ที่ต่อท้ายภายหลัง เช่น -ส่วนที่99, -INDEX.md, .pdf
    encoded = name.encode('utf-8')
    if len(encoded) <= max_bytes:
        return name
    # ตัดที่ระดับ byte แล้ว decode กลับแบบไม่ทำลาย multi-byte character ที่ถูกตัดครึ่ง
    return encoded[:max_bytes].decode('utf-8', errors='ignore')


def _get_vault_path() -> Path:
    s = get_settings()
    vault = Path(s.OBSIDIAN_VAULT_PATH)
    vault.mkdir(parents=True, exist_ok=True)
    return vault


def _get_zone10_base() -> Path:
    """เส้นทาง เขต10/ ภายใน vault."""
    zone = _get_vault_path() / "เขต10"
    zone.mkdir(parents=True, exist_ok=True)
    return zone


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



def _resolve_save_path(
    vault_path: Path,
    province: str | None,
    district: str | None,
    base_filename: str,
) -> tuple[Path, str]:
    """
    คำนวณ path ที่จะบันทึกไฟล์ และ relative path สำหรับแสดงผล.
    ทุก PDF จะถูกเก็บในโฟลเดอร์ชื่อเดียวกับเอกสารเสมอ เพื่อไม่ให้ไฟล์ MD กระจัดกระจาย

    - มีจังหวัดเท่านั้น → เขต10/{province}/{base_filename}/
    - มีจังหวัด+อำเภอ  → เขต10/{province}/{district}/{base_filename}/
    - (บังคับให้มีจังหวัดเสมอ หากไม่มีจะใช้ อุบลราชธานี เป็นค่าเริ่มต้น)

    Returns:
        (absolute_path, display_relative_path)
    """
    zone_base = vault_path / "เขต10"
    safe_name = _sanitize_filename(base_filename)

    if not district:
        # ระบุเฉพาะจังหวัด → โฟลเดอร์เอกสารใต้จังหวัด
        save_dir = zone_base / province / safe_name
        rel = f"เขต10/{province}/{safe_name}"
    else:
        # ระบุทั้งจังหวัดและอำเภอ
        save_dir = zone_base / province / district / safe_name
        rel = f"เขต10/{province}/{district}/{safe_name}"

    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir, rel


def _update_province_index(
    vault_path: Path,
    province: str,
    district: str | None,
    base_filename: str,
    index_filename: str,
    total_pages: int,
):
    """อัปเดต INDEX.md ของจังหวัด/อำเภอ เพิ่ม link ไปยังโฟลเดอร์เอกสารใหม่."""
    zone_base = vault_path / "เขต10"
    safe_name = _sanitize_filename(base_filename)
    date_str = datetime.now().strftime('%d/%m/%Y')

    # wikilink ชี้ไปที่ INDEX ภายในโฟลเดอร์ของเอกสาร
    if district:
        doc_index_link = f"{district}/{safe_name}/{index_filename.replace('.md', '')}"
    else:
        doc_index_link = f"{safe_name}/{index_filename.replace('.md', '')}"

    entry = f"\n- [[{doc_index_link}|📄 {base_filename}]] ({total_pages} หน้า · {date_str})"

    try:
        # อัปเดต INDEX.md ของจังหวัด
        prov_index = zone_base / province / "INDEX.md"
        if prov_index.exists():
            content = prov_index.read_text(encoding="utf-8")
            if "[[เขต10/INDEX" in content:
                content = content.replace("[[เขต10/INDEX", f"{entry}\n\n[[เขต10/INDEX")
            else:
                content += entry
            prov_index.write_text(content, encoding="utf-8")

        # อัปเดต INDEX.md ของอำเภอ (ถ้ามี)
        if district:
            dist_index = zone_base / province / district / "INDEX.md"
            if not dist_index.exists():
                dist_index.write_text(
                    f"# {district} — {province}\n\n"
                    f"> **เขตสุขภาพที่ 10** · [[{province}/INDEX|← {province}]]\n\n"
                    f"## เอกสาร\n",
                    encoding="utf-8",
                )
            content = dist_index.read_text(encoding="utf-8")
            dist_entry = f"\n- [[{safe_name}/{index_filename.replace('.md', '')}|📄 {base_filename}]] ({total_pages} หน้า · {date_str})"
            content += dist_entry
            dist_index.write_text(content, encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to update province index: {e}")


# ── AI: Generate filename from content ───────────────────────────────────────

def _ai_generate_filename(client, uploaded_file, original_pdf_name: str) -> str:
    """ให้ Gemini 3 Pro วิเคราะห์เนื้อหาและสร้างชื่อไฟล์ภาษาไทย."""
    prompt = f"""วิเคราะห์ไฟล์ PDF ที่แนบมานี้ทั้งหมด แล้วสรุป 'ชื่อเอกสาร' ภาษาไทยที่สื่อความหมายชัดเจนและครอบคลุมเนื้อหาหลักของเอกสารนี้มากที่สุด

กฎการตั้งชื่อ:
- ต้องเป็นชื่อที่อ่านแล้วเข้าใจทันทีว่าเป็นเอกสารเกี่ยวกับอะไร
- ห้ามใช้คำเดี่ยวๆ ที่ไม่มีความหมาย เช่น "การ", "รายงาน", "สรุป"
- หากเอกสารมีชื่อเรื่องหรือหัวข้อหลักอยู่แล้ว ให้นำมาใช้ได้เลย
- ห้ามมีนามสกุลไฟล์ (เช่น .pdf)
- ใช้ขีดกลาง (-) แทนช่องว่าง
- ความยาว 10 ถึง 80 ตัวอักษร
- ตอบกลับมา **เฉพาะชื่อไฟล์เท่านั้น** ห้ามมีคำอธิบายเพิ่มเติมใดๆ ทั้งสิ้น"""

    try:
        response = client.models.generate_content(
            model=_gemini_model_fast(),  # Flash: เร็วกว่า Pro ~3x สำหรับ task นี้
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=100,
                temperature=0.3,
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

        sample_text = "\n\n".join(pages[:min(5, total_pages)])

        run_ai_name = not override_folder_name
        run_ai_loc = not override_province or override_province == "auto"

        base_filename = _sanitize_filename(override_folder_name) if override_folder_name else ""
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
                    # ตัดความยาวทันทีที่ได้ชื่อจาก AI — ทุกจุดที่ใช้ base_filename ต่อ
                    # (ชื่อไฟล์ chunk, PDF, index) จะได้ค่าที่ปลอดภัยแล้วเหมือนกันหมด
                    base_filename = _sanitize_filename(fut_name.result())
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

        # 6. Resolve save directory — สร้างโฟลเดอร์ชื่อเอกสารเสมอ
        vault_path = _get_vault_path()
        save_dir, rel_path = _resolve_save_path(vault_path, province, district, base_filename)
        log(f"📁 สร้างโฟลเดอร์: {rel_path}/")

        # Pre-compute all chunk filenames
        chunk_filenames = []
        for i in range(total_chunks):
            if total_chunks == 1:
                chunk_filenames.append(base_filename)
            else:
                chunk_filenames.append(f"{base_filename}-ส่วนที่{i+1:02d}")

        # 7. Convert each chunk to MD (parallel)
        created_files: list[str] = []
        md_results: dict[int, str] = {}

        # ── สร้าง args สำหรับแต่ละ chunk ────────────────────────────────────
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

        # ── บันทึกตามลำดับ ────────────────────────────────────────────────────
        for i in range(total_chunks):
            md_filename = chunk_filenames[i] + ".md"
            md_path = save_dir / md_filename
            md_path.write_text(md_results[i], encoding="utf-8")
            created_files.append(f"{rel_path}/{md_filename}")
            log(f"💾 บันทึก: {rel_path}/{md_filename}")

        # 7.5 Save original PDF to Vault
        pdf_filename = f"{base_filename}.pdf"
        pdf_path = save_dir / pdf_filename
        pdf_path.write_bytes(file_bytes)
        created_files.append(f"{rel_path}/{pdf_filename}")
        log(f"💾 บันทึก PDF ต้นฉบับ: {rel_path}/{pdf_filename}")

        # 8. Create index file
        log("📚 กำลังสร้างไฟล์ Index...")
        safe_name = _sanitize_filename(base_filename)
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
            "\n---\n\n## 📎 ไฟล์แนบ\n",
            f"- [[{pdf_filename}]]\n",
            "\n## 📑 รายการส่วน\n",
        ]
        for i, fname in enumerate(chunk_filenames):
            page_start = i * chunk_size + 1
            page_end = min(page_start + chunk_size - 1, total_pages)
            index_lines.append(f"- [[{fname}|ส่วนที่ {i+1}: หน้า {page_start}–{page_end}]]")

        index_filename = f"{base_filename}-INDEX.md"
        index_path = save_dir / index_filename
        index_path.write_text("\n".join(index_lines), encoding="utf-8")
        log(f"✅ สร้าง Index: {rel_path}/{index_filename}")

        # 9. Update province/district INDEX.md
        if province:
            _update_province_index(vault_path, province, district, base_filename, index_filename, total_pages)
            log(f"🔗 อัปเดต Index จังหวัด {province} แล้ว")

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
            "vault_path": str(save_dir),
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
    """List PDF files ทั้งหมดใน MinIO bucket."""
    import urllib.parse

    client_minio = _get_client()
    bucket = _pdf_bucket()

    try:
        objects = list(client_minio.list_objects(bucket, recursive=True))
        files = []
        for obj in objects:
            name = obj.object_name
            if name.endswith("__apa.json") or name.endswith("__path.json"):
                continue
            try:
                stat = client_minio.stat_object(bucket, name)
                meta = {k.lower(): v for k, v in (stat.metadata or {}).items()}
                ext = meta.get("x-amz-meta-extension", "").lower()
                orig_name = urllib.parse.unquote(meta.get("x-amz-meta-name", name))

                if ext == "pdf" or orig_name.lower().endswith(".pdf"):
                    # Check in-memory jobs (first priority for running/queued, fallback for completed)
                    in_memory_job = next(
                        (j for j in _jobs.values() if j.get("file_id") == name),
                        None
                    )
                    
                    # Read metadata persisted in MinIO
                    meta_ingested = meta.get("x-amz-meta-ingested") == "true"
                    meta_province = urllib.parse.unquote(meta.get("x-amz-meta-province", "")) or None
                    meta_district = urllib.parse.unquote(meta.get("x-amz-meta-district", "")) or None
                    meta_savedat = urllib.parse.unquote(meta.get("x-amz-meta-savedat", "")) or None
                    meta_confidence = meta.get("x-amz-meta-confidence", "") or None
                    
                    if in_memory_job:
                        job_status = in_memory_job.get("status")
                        job_result = in_memory_job.get("result") or {}
                        
                        ingested = (job_status == "completed") or meta_ingested
                        province = job_result.get("province") or meta_province
                        district = job_result.get("district") or meta_district
                        saved_at = job_result.get("saved_at") or meta_savedat
                        location_confidence = job_result.get("location_confidence") or meta_confidence
                        ingest_status = job_status
                        ingest_job_id = in_memory_job.get("job_id")
                    else:
                        ingested = meta_ingested
                        province = meta_province
                        district = meta_district
                        saved_at = meta_savedat
                        location_confidence = meta_confidence
                        ingest_status = "completed" if meta_ingested else None
                        ingest_job_id = None

                    files.append({
                        "id": name,
                        "name": orig_name,
                        "size": obj.size,
                        "uploaded_at": meta.get("x-amz-meta-uploadedat", ""),
                        "ingested": ingested,
                        "ingestJobId": ingest_job_id,
                        "ingestStatus": ingest_status,
                        "province": province,
                        "district": district,
                        "saved_at": saved_at,
                        "location_confidence": location_confidence,
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
    """List Markdown files ใน Obsidian vault แบบ tree structure."""
    vault_path = _get_vault_path()
    zone_base = vault_path / "เขต10"

    def scan_dir(path: Path, relative: str = "") -> list[dict]:
        items = []
        try:
            for entry in sorted(path.iterdir()):
                rel = f"{relative}/{entry.name}" if relative else entry.name
                if entry.is_dir():
                    children = scan_dir(entry, rel)
                    items.append({
                        "type": "folder",
                        "name": entry.name,
                        "path": rel,
                        "children": children,
                    })
                elif entry.suffix == ".md":
                    stat = entry.stat()
                    items.append({
                        "type": "file",
                        "name": entry.name,
                        "path": rel,
                        "size": stat.st_size,
                        "modified_at": stat.st_mtime,
                    })
        except Exception:
            pass
        return items

    # Build zone 10 tree
    zone10_tree = []
    if zone_base.exists():
        for province in ZONE10_PROVINCES:
            prov_path = zone_base / province
            if prov_path.exists():
                zone10_tree.append({
                    "type": "folder",
                    "name": province,
                    "path": f"เขต10/{province}",
                    "children": scan_dir(prov_path, f"เขต10/{province}"),
                })

    # Root level items (folders/files not in zone10)
    root_files = []
    for entry in sorted(vault_path.iterdir()):
        if entry.name == "เขต10":
            continue  # skip zone10 — handled separately
        if entry.is_dir():
            # โฟลเดอร์เอกสารที่ไม่ระบุจังหวัด
            root_files.append({
                "type": "folder",
                "name": entry.name,
                "path": entry.name,
                "children": scan_dir(entry, entry.name),
            })
        elif entry.is_file() and entry.suffix == ".md":
            stat = entry.stat()
            root_files.append({
                "type": "file",
                "name": entry.name,
                "path": entry.name,
                "size": stat.st_size,
                "modified_at": stat.st_mtime,
            })

    return {
        "zone10": zone10_tree,
        "root_files": root_files,
        "provinces": list(ZONE10_PROVINCES.keys()),
        "vault_path": str(vault_path),
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


# ── Vault CRUD endpoints ──────────────────────────────────────────────────────

from fastapi import Body

def _safe_vault_path(vault_path: Path, rel: str) -> Path:
    """ป้องกัน path traversal — ตรวจสอบว่า path อยู่ใน vault เสมอ."""
    resolved = (vault_path / rel).resolve()
    if not str(resolved).startswith(str(vault_path.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    return resolved


@router.get("/vault/file")
async def read_vault_file(path: str):
    """อ่านเนื้อหาของไฟล์ Markdown ใน vault."""
    vault_path = _get_vault_path()
    target = _safe_vault_path(vault_path, path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    content = target.read_text(encoding="utf-8")
    stat = target.stat()
    return {
        "path": path,
        "name": target.name,
        "content": content,
        "size": stat.st_size,
        "modified_at": stat.st_mtime,
    }


@router.put("/vault/file")
async def write_vault_file(
    path: str,
    body: dict = Body(...),
):
    """สร้างหรืออัปเดตเนื้อหาไฟล์ Markdown ใน vault."""
    content = body.get("content", "")
    vault_path = _get_vault_path()
    target = _safe_vault_path(vault_path, path)
    # สร้างโฟลเดอร์ parent ถ้ายังไม่มี
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    stat = target.stat()
    return {"path": path, "size": stat.st_size, "modified_at": stat.st_mtime}


@router.delete("/vault/file")
async def delete_vault_file(path: str):
    """ลบไฟล์หรือโฟลเดอร์ออกจาก vault."""
    import shutil
    vault_path = _get_vault_path()
    target = _safe_vault_path(vault_path, path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Not found")
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return {"deleted": True, "path": path}


@router.post("/vault/rename")
async def rename_vault_file(body: dict = Body(...)):
    """เปลี่ยนชื่อ หรือย้ายไฟล์/โฟลเดอร์ภายใน vault."""
    old_path = body.get("old_path", "")
    new_path = body.get("new_path", "")
    if not old_path or not new_path:
        raise HTTPException(status_code=400, detail="old_path and new_path required")
    vault_path = _get_vault_path()
    src = _safe_vault_path(vault_path, old_path)
    dst = _safe_vault_path(vault_path, new_path)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source not found")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    return {"renamed": True, "old_path": old_path, "new_path": new_path}


@router.get("/vault/folders")
async def list_vault_folders():
    """List โฟลเดอร์ทั้งหมดใน vault (สำหรับ dropdown เลือก target)."""
    vault_path = _get_vault_path()
    folders: list[str] = []

    def collect(path: Path, rel: str = ""):
        for entry in sorted(path.iterdir()):
            if entry.is_dir():
                r = f"{rel}/{entry.name}" if rel else entry.name
                folders.append(r)
                collect(entry, r)

    try:
        collect(vault_path)
    except Exception:
        pass
    return {"folders": folders}
