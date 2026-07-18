"""Shared post-processing for LLM-generated answers.

Gemini (โดยเฉพาะตอนโดน context ยาว/ใกล้ max_tokens) บางครั้งเข้าสู่ภาวะ
"repetition loop" คือเขียนคำตอบทั้งชุดซ้ำอีกรอบ (บางทีก็ paraphrase ไม่เหมือน
เดิมเป๊ะ) หรือวนซ้ำเฉพาะบล็อกท้าย ๆ เช่น "คำถามติดตาม" หลายรอบ — ทำให้ output
ที่ผู้ใช้เห็น "เบิ้ล" กันหลายอัน dedupe ตรงนี้ตัดส่วนที่วนซ้ำทิ้งแบบระมัดระวัง
(เก็บคำตอบรอบแรกที่สมบูรณ์ไว้เสมอ ไม่ตัดเนื้อหาที่ถูกต้อง)
"""
import re

# หัวข้อ "คำถามติดตาม" ที่เป็น "หัวข้อจริง" (ขึ้นต้นบรรทัด, มี #/​** นำหน้าได้)
# — ไม่ใช่คำว่า "คำถามติดตาม" ที่โผล่กลางประโยค (in-prose)
_FOLLOWUP_HEADER_RE = re.compile(
    r"(?m)^[ \t]*(?:#{1,4}[ \t]*)?(?:\*\*)?[ \t]*(?:คำถามติดตาม|Follow-?up)"
)
_NUM_ITEM_RE = re.compile(r"^[ \t]*\d+[\.\)]\s")
_HEADER_RE = re.compile(r"^\**\s*([^\n*]{4,40})")


def dedupe_repeated_answer(text: str) -> str:
    """ตัดคำตอบที่ถูกเขียนซ้ำจาก repetition loop ของ LLM.

    หลักการหลัก: บล็อก "คำถามติดตาม" คือ "ส่วนท้ายสุด" ของคำตอบเสมอ (ตาม
    SYSTEM_PROMPT) — ดังนั้นเก็บเนื้อหาถึง "บล็อกคำถามติดตามอันแรก" ให้ครบ แล้ว
    ตัดทุกอย่างหลังจากนั้นทิ้ง (เพราะเป็นการวนซ้ำ/เขียนคำตอบใหม่ทั้งชุด)

    ครอบคลุมทุกรูปแบบที่เจอจริง:
      • คำตอบทั้งชุด (structured 4 ส่วน) วนซ้ำ 3-5 รอบ แต่ละรอบจบด้วย
        "**คำถามติดตาม**"  → ตัดหลังบล็อกแรก
      • คำตอบแบบสนทนา (paraphrase) วนซ้ำ 2 รอบ, บล็อกคำถามติดตามซ้ำ → ตัดหลังบล็อกแรก
      • เฉพาะบล็อกคำถามติดตามวนซ้ำ 5 รอบ (ตัวคำตอบไม่ซ้ำ) → ตัดหลังบล็อกแรก

    ปลอดภัยกับคำตอบปกติ (บล็อกคำถามติดตามอันเดียว, ไม่มีอะไรต่อท้าย → no-op)
    และมี guard `len < 200` กันไม่ให้ไปยุ่งกับคำตอบสั้น ๆ
    """
    if not text or len(text) < 200:
        return text

    # ── Rule หลัก: ตัดหลัง "บล็อกคำถามติดตามอันแรก" ──────────────────────────
    hdrs = list(_FOLLOWUP_HEADER_RE.finditer(text))
    if hdrs:
        h = hdrs[0]
        lines = text[h.start():].split("\n")
        kept = [lines[0]]           # บรรทัดหัวข้อ "คำถามติดตาม"
        i = 1
        # ข้ามบรรทัดว่างหลังหัวข้อ (บางฟอร์แมตเว้นบรรทัด)
        while i < len(lines) and lines[i].strip() == "":
            kept.append(lines[i]); i += 1
        # เก็บรายการคำถามที่เป็นเลขข้อ (1. 2. 3. ...) ติดกัน
        n_items = 0
        while i < len(lines) and _NUM_ITEM_RE.match(lines[i]):
            kept.append(lines[i]); i += 1; n_items += 1
        # ตัด "restart" ที่ถูก glue ท้ายข้อสุดท้าย (เช่น "...ครับ?สวัสดีครับ ...")
        # ให้เหลือถึงเครื่องหมาย "?" ตัวแรกของข้อนั้น
        if n_items and "?" in kept[-1]:
            kept[-1] = kept[-1][: kept[-1].index("?") + 1]

        # ใช้กฎนี้เฉพาะเมื่อจับรายการคำถามได้จริง (กันเผลอตัดคำถามทิ้งถ้าฟอร์แมตแปลก)
        if n_items:
            candidate = (text[: h.start()] + "\n".join(kept)).rstrip()
            if len(candidate) < len(text.rstrip()):
                return candidate
            return text
        # ไม่มีรายการเลขข้อ → ตกไปใช้ Rule '---' ด้านล่างแทน

    # ── Rule สำรอง: คำตอบซ้ำคั่นด้วย '---' และ segment หลังขึ้นหัวข้อเดียวกับแรก ──
    dash = list(re.finditer(r"\n-{3,}\n", text))
    if dash:
        fh = _HEADER_RE.match(text[: dash[0].start()].strip())
        if fh:
            key = fh.group(1).strip()
            for m in dash:
                sh = _HEADER_RE.match(text[m.end():].lstrip())
                if sh and sh.group(1).strip() == key:
                    return text[: m.start()].rstrip()
    return text
