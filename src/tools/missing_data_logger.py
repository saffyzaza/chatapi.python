"""Missing-Data Logger — บันทึก "คำถามที่ระบบไม่มีข้อมูลตอบ" ไว้ให้ admin.

จุดประสงค์: เมื่อ pipeline ตอบ "ไม่พบข้อมูล" (relevance gate ไม่ผ่าน / ไม่พบไฟล์ /
ถามจังหวัดนอกเขต) ให้บันทึกคำถามไว้ → admin ใช้เป็น "รายการคำขอข้อมูล" เพื่อตัดสินใจ
ว่าควรเพิ่มชุดข้อมูลอะไรเข้าระบบ (ปิด loop ของข้อความ "กรุณาแจ้ง admin เพิ่มข้อมูล")

Log format: one .txt file per day  →  missing_data_logs/missing_data_YYYY-MM-DD.txt
รูปแบบ + การ parse/aggregate เลียนแบบ error_logger.py เพื่อให้ดูแลรักษาแบบเดียวกัน
"""
import re
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

# Root อยู่ข้าง ๆ main.py เช่นเดียวกับ error_logs/
MISSING_DATA_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "missing_data_logs"

_DIVIDER = "=" * 72


# ── Write ──────────────────────────────────────────────────────────────────────

def log_missing_data(
    prompt: str,
    domain: str = "",
    reason: str = "",
    session_id: str = "",
) -> None:
    """บันทึกคำถามที่ไม่มีข้อมูลตอบ 1 รายการลงไฟล์ log ของวันนี้.

    reason: สาเหตุ เช่น "irrelevant_file" / "no_file_found" / "out_of_zone10"
    เรียกจาก thread ใดก็ได้ (เปิดไฟล์แบบ append)
    """
    MISSING_DATA_LOG_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_file = MISSING_DATA_LOG_DIR / f"missing_data_{today}.txt"

    entry_id  = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    block = (
        f"\n{_DIVIDER}\n"
        f"ENTRY ID  : {entry_id}\n"
        f"TIMESTAMP : {timestamp}\n"
        f"SESSION   : {session_id or '-'}\n"
        f"DOMAIN    : {domain or '-'}\n"
        f"REASON    : {reason or '-'}\n"
        f"QUESTION  : {(prompt or '')[:500]}\n"
        f"{_DIVIDER}\n"
    )

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(block)
    except Exception:
        pass  # logging ต้องไม่ทำให้ pipeline ล้ม


# ── Read / Parse ───────────────────────────────────────────────────────────────

def _parse_txt_file(path: Path) -> list[dict]:
    entries: list[dict] = []
    text = path.read_text(encoding="utf-8")
    for block in re.split(r"={60,}", text):
        block = block.strip()
        if not block:
            continue
        entry: dict = {}
        for line in block.splitlines():
            if " : " in line:
                key, _, value = line.partition(" : ")
                mapping = {
                    "entry id":  "id",
                    "timestamp": "timestamp",
                    "session":   "session_id",
                    "domain":    "domain",
                    "reason":    "reason",
                    "question":  "question",
                }
                field = key.strip().lower()
                if field in mapping:
                    entry[mapping[field]] = value.strip()
        if entry.get("id"):
            entries.append(entry)
    return entries


def read_all_missing_data(days: int = 30) -> list[dict]:
    """คืนรายการคำขอข้อมูลที่ไม่มีตอบ จาก N วันล่าสุด (ใหม่สุดก่อน)."""
    if not MISSING_DATA_LOG_DIR.exists():
        return []
    entries: list[dict] = []
    for log_file in sorted(MISSING_DATA_LOG_DIR.glob("missing_data_*.txt"), reverse=True)[:days]:
        try:
            entries.extend(_parse_txt_file(log_file))
        except Exception:
            pass
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return entries


def aggregate_missing_data(entries: list[dict]) -> dict:
    """สรุปคำขอข้อมูล → นับตาม domain/reason + คำถามที่พบบ่อย."""
    by_domain:   Counter = Counter()
    by_reason:   Counter = Counter()
    by_question: Counter = Counter()
    for e in entries:
        if e.get("domain") and e["domain"] != "-":
            by_domain[e["domain"]] += 1
        if e.get("reason") and e["reason"] != "-":
            by_reason[e["reason"]] += 1
        if e.get("question"):
            by_question[e["question"]] += 1
    return {
        "total":        len(entries),
        "by_domain":    dict(by_domain.most_common()),
        "by_reason":    dict(by_reason.most_common()),
        "top_requests": dict(by_question.most_common(20)),
        "latest":       entries[:10],
    }
