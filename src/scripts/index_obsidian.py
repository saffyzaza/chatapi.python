"""Index Obsidian vault — walks .md files and UPSERTs into obsidian_notes table."""
import logging
import re
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (meta_dict, body_without_frontmatter)."""
    meta: dict = {}
    body = content
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            fm_block = content[3:end].strip()
            body = content[end + 4:].lstrip()
            for line in fm_block.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    v = v.strip().strip('"').strip("'")
                    if v.startswith("[") and v.endswith("]"):
                        meta[k.strip()] = [x.strip().strip('"') for x in v[1:-1].split(",") if x.strip()]
                    else:
                        meta[k.strip()] = v
    return meta, body


def _infer_province(path: Path, vault_root: Path) -> str | None:
    """Infer province from the top-level folder under vault root."""
    try:
        rel = path.relative_to(vault_root)
        parts = rel.parts
        if len(parts) >= 2:
            return parts[0]
    except ValueError:
        pass
    return None


def _infer_district(path: Path, vault_root: Path) -> str | None:
    """Infer district from folder name starting with อ."""
    try:
        rel = path.relative_to(vault_root)
        for part in rel.parts[1:-1]:
            if part.startswith("อ."):
                return part[2:]
    except ValueError:
        pass
    return None


def _extract_year(filename: str, meta: dict) -> int | None:
    """Extract Buddhist year from filename or frontmatter."""
    if "year" in meta:
        try:
            return int(meta["year"])
        except (ValueError, TypeError):
            pass
    m = re.search(r'(256[0-9]|257[0-9])', filename)
    if m:
        return int(m.group(1)) - 543
    return None


def _infer_note_type(meta: dict, filename: str) -> str:
    if "type" in meta:
        return str(meta["type"]).lower()
    low = filename.lower()
    if "moc" in low or "000_" in low:
        return "MOC"
    if "report" in low or "รายงาน" in filename:
        return "report"
    if "policy" in low or "นโยบาย" in filename:
        return "policy"
    if "research" in low or "วิจัย" in filename:
        return "research"
    return "district"


def index_vault(vault_id: str = "health_region_10") -> dict:
    """Walk vault folder and UPSERT all .md files into obsidian_notes."""
    try:
        from src.config import get_settings
        from src.db.pool import execute_db, query_db
    except ImportError:
        from config import get_settings
        from db.pool import execute_db, query_db

    s = get_settings()
    vault_root = Path(s.OBSIDIAN_VAULT_PATH) / s.OBSIDIAN_DEFAULT_VAULT
    if not vault_root.exists():
        vault_root = Path(s.OBSIDIAN_VAULT_PATH)

    if not vault_root.exists():
        raise FileNotFoundError(f"Vault path not found: {vault_root}")

    logger.info("[index] Scanning vault: %s", vault_root)

    md_files = [p for p in vault_root.rglob("*.md") if ".obsidian" not in p.parts]
    logger.info("[index] Found %d .md files", len(md_files))

    inserted = 0
    updated = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    for md_path in md_files:
        try:
            content = md_path.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                skipped += 1
                continue

            meta, body = _parse_frontmatter(content)
            rel_path = str(md_path.relative_to(vault_root)).replace("\\", "/")
            note_id = f"{vault_id}::{rel_path}"
            title = meta.get("title") or md_path.stem
            province = meta.get("province") or _infer_province(md_path, vault_root)
            district = meta.get("district") or _infer_district(md_path, vault_root)
            note_type = _infer_note_type(meta, md_path.name)
            tags_raw = meta.get("tags", [])
            tags = tags_raw if isinstance(tags_raw, list) else [tags_raw] if tags_raw else []
            year = _extract_year(md_path.name, meta)
            source_file = meta.get("source_file") or meta.get("source")

            existing = query_db(
                "SELECT note_id FROM obsidian_notes WHERE note_id = %s", (note_id,)
            )
            is_new = len(existing) == 0

            execute_db(
                """
                INSERT INTO obsidian_notes
                    (note_id, vault_id, relative_path, title, province, district,
                     note_type, tags, source_file, year, content, content_stripped, indexed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (note_id) DO UPDATE SET
                    title           = EXCLUDED.title,
                    province        = EXCLUDED.province,
                    district        = EXCLUDED.district,
                    note_type       = EXCLUDED.note_type,
                    tags            = EXCLUDED.tags,
                    source_file     = EXCLUDED.source_file,
                    year            = EXCLUDED.year,
                    content         = EXCLUDED.content,
                    content_stripped = EXCLUDED.content_stripped,
                    indexed_at      = EXCLUDED.indexed_at
                """,
                (
                    note_id, vault_id, rel_path, title, province, district,
                    note_type, tags, source_file, year,
                    content, body, datetime.utcnow(),
                ),
            )
            if is_new:
                inserted += 1
            else:
                updated += 1

        except Exception as e:
            logger.error("[index] Error on %s: %s", md_path, e)
            errors += 1

    total_indexed = inserted + updated
    execute_db(
        "UPDATE obsidian_vaults SET note_count = %s, indexed_at = %s, vault_path = %s WHERE vault_id = %s",
        (total_indexed, datetime.utcnow(), str(vault_root), vault_id),
    )

    elapsed = round(time.time() - start_time, 2)
    logger.info("[index] Done: inserted=%d updated=%d skipped=%d errors=%d elapsed=%.2fs",
                inserted, updated, skipped, errors, elapsed)
    return {
        "vault_id": vault_id,
        "vault_path": str(vault_root),
        "inserted": inserted,
        "updated": updated,
        "unchanged": skipped,
        "errors": errors,
        "total_files": len(md_files),
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    vid = sys.argv[1] if len(sys.argv) > 1 else "health_region_10"
    result = index_vault(vid)
    print(result)
