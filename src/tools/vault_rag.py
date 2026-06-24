"""Obsidian Vault RAG — อ่าน MD files จาก vault ตามจังหวัด เขต 10 เพื่อใช้เป็น context เสริม."""
import re
from pathlib import Path

from src.config import get_settings

# ─── Zone 10 province aliases ───────────────────────────────────────────────
_ZONE10_PROVINCES = [
    "อุบลราชธานี",
    "ศรีสะเกษ",
    "ยโสธร",
    "อำนาจเจริญ",
    "มุกดาหาร",
]

_PROVINCE_ALIASES: dict[str, str] = {
    # อุบล
    "อุบลราชธานี": "อุบลราชธานี",
    "อุบล": "อุบลราชธานี",
    "อุบลฯ": "อุบลราชธานี",
    # ศรีสะเกษ
    "ศรีสะเกษ": "ศรีสะเกษ",
    "ศรีสะเกษ": "ศรีสะเกษ",
    # ยโสธร
    "ยโสธร": "ยโสธร",
    # อำนาจเจริญ
    "อำนาจเจริญ": "อำนาจเจริญ",
    "อำนาจ": "อำนาจเจริญ",
    # มุกดาหาร
    "มุกดาหาร": "มุกดาหาร",
    "มุกดา": "มุกดาหาร",
}


def detect_province_from_prompt(prompt: str) -> str | None:
    """ตรวจจับชื่อจังหวัดในเขต 10 จากคำถาม คืนชื่อจังหวัดเต็ม หรือ None."""
    text = prompt.strip()
    # ตรงตัวก่อน (longest match)
    for alias in sorted(_PROVINCE_ALIASES.keys(), key=len, reverse=True):
        if alias in text:
            return _PROVINCE_ALIASES[alias]
    return None


def _get_vault_zone10_path() -> Path:
    s = get_settings()
    return Path(s.OBSIDIAN_VAULT_PATH) / "เขต10"


def _collect_md_files(province_dir: Path) -> list[Path]:
    """รวบรวม MD files ทั้งหมดในโฟลเดอร์จังหวัด เรียงตาม modified_at ล่าสุดก่อน."""
    if not province_dir.exists():
        return []
    files = list(province_dir.rglob("*.md"))
    # เรียงตาม mtime ล่าสุดก่อน
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _strip_frontmatter(content: str) -> str:
    """ลบ YAML frontmatter ออก เก็บเฉพาะเนื้อหา."""
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            return content[end + 3:].lstrip()
    return content


def _clean_wikilinks(content: str) -> str:
    """แปลง [[link|label]] → label และลบ nav links ที่ซ้ำซ้อน."""
    # [[path|label]] → label
    content = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', content)
    # [[path]] → path (basename)
    content = re.sub(r'\[\[([^\]]+)\]\]', lambda m: m.group(1).split('/')[-1], content)
    return content


def read_vault_context(province: str, max_chars: int = 8000) -> str:
    """อ่าน MD files จากโฟลเดอร์จังหวัดใน vault และรวมเป็น context string.

    Args:
        province: ชื่อจังหวัดเต็ม เช่น 'อุบลราชธานี'
        max_chars: จำกัดขนาด context สูงสุด (default 8,000 chars)

    Returns:
        String รวมเนื้อหา MD files สำหรับใช้เป็น context
        หรือ empty string ถ้าไม่มีไฟล์
    """
    zone10 = _get_vault_zone10_path()
    province_dir = zone10 / province

    if not province_dir.exists():
        return ""

    files = _collect_md_files(province_dir)
    if not files:
        return ""

    parts: list[str] = []
    total_chars = 0

    # INDEX files ก่อน (สรุปภาพรวม)
    index_files = [f for f in files if "INDEX" in f.name.upper()]
    chunk_files = [f for f in files if "INDEX" not in f.name.upper()]
    ordered = index_files + chunk_files

    for fpath in ordered:
        if total_chars >= max_chars:
            break
        try:
            raw = fpath.read_text(encoding="utf-8")
            cleaned = _strip_frontmatter(raw)
            cleaned = _clean_wikilinks(cleaned)
            cleaned = cleaned.strip()
            if not cleaned:
                continue

            # ตัดให้พอดีกับ budget ที่เหลือ
            remaining = max_chars - total_chars
            if len(cleaned) > remaining:
                cleaned = cleaned[:remaining] + "…"

            # relative path สำหรับ label
            try:
                rel = fpath.relative_to(zone10.parent)
            except ValueError:
                rel = fpath.name

            parts.append(f"### 📄 {rel}\n{cleaned}")
            total_chars += len(cleaned)
        except Exception:
            continue

    if not parts:
        return ""

    header = (
        f"## 📚 เอกสารจาก Obsidian Vault — {province} (เขต 10)\n"
        f"*(ข้อมูลจากไฟล์ที่ Ingest แล้ว {len(parts)}/{len(files)} ไฟล์)*\n\n"
    )
    return header + "\n\n".join(parts)


def get_vault_summary(province: str) -> dict:
    """สรุปจำนวนไฟล์และขนาดใน vault ของจังหวัด (ไม่อ่านเนื้อหา)."""
    zone10 = _get_vault_zone10_path()
    province_dir = zone10 / province
    if not province_dir.exists():
        return {"province": province, "exists": False, "file_count": 0, "total_bytes": 0}

    files = _collect_md_files(province_dir)
    total_bytes = sum(f.stat().st_size for f in files)
    return {
        "province": province,
        "exists": True,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": [str(f.name) for f in files[:10]],  # preview 10 files
    }
