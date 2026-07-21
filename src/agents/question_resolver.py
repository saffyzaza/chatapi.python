"""Memory Agent — แปลง follow-up question ให้ครบถ้วนโดยใช้บริบทการสนทนา"""
import os
import litellm
from src.tools.error_logger import log_agent_error

_SYSTEM = (
    "คุณเป็นผู้ช่วยที่เข้าใจบริบทการสนทนาภาษาไทย "
    "และสามารถปรับคำถาม follow-up ที่ไม่ครบถ้วนให้ชัดเจนสมบูรณ์"
)

_PROMPT = """บริบทการสนทนาก่อนหน้า (เรียงจากเก่าไปใหม่ — ข้อความ "ล่าสุด" คือประเด็นที่
กำลังคุยอยู่ตอนนี้ และควรใช้เป็นหลักในการตีความคำถามใหม่ ไม่ใช่ประเด็นแรกสุดของบทสนทนา):
{history}

คำถามใหม่: "{prompt}"

ภารกิจ:
ถ้าคำถามใหม่อ้างถึงสิ่งที่คุยไปก่อน โดยไม่ระบุชัดเจน เช่น:
- "ของจังหวัด" โดยไม่ระบุชื่อจังหวัด
- "โรคนั้น" / "ข้อมูลเดิม" / "ปีเดิม" / "ที่กล่าวถึง"
- "ขอทุกอำเภอ" โดยไม่ระบุจังหวัด
- "ขอเพิ่มเติม" / "แล้วเมื่อเทียบกับ..." / "แบ่งรายอำเภอได้ไหม"
- ใช้คำสรรพนาม เช่น "นั้น" "ที่ว่า" "ดังกล่าว"

ให้เขียนคำถามใหม่ให้ครบถ้วนสมบูรณ์ โดยใส่ข้อมูลที่ขาดไปจากบริบทก่อนหน้า

**กรณีพิเศษที่ต้องระวัง — ประโยค "ยืนยัน/เจาะจงขอบเขต" (scope-confirmation):**
บางครั้งคำถามใหม่ไม่ใช่คำถามจริง ๆ แต่เป็นการ "ย้ำ/ชี้แจงขอบเขต" ของสิ่งที่กำลังคุย
อยู่ ณ ตอนนั้น เช่น "ผมถามถึงจังหวัด X", "หมายถึง X นะ", "เจาะจงที่ X", "คือ X อ่ะ"
— ประโยคแบบนี้ไม่ได้แปลว่าผู้ใช้อยากย้อนกลับไปคำถามแรกสุดของบทสนทนาใหม่ทั้งหมด
แต่หมายถึง "เอาหัวข้อ/คำถามล่าสุดที่เพิ่งคุยกันอยู่ (ข้อความท้ายสุดของประวัติ) มา
เติมเงื่อนไข X เข้าไปให้ครบ" เท่านั้น เช่น:
- ประวัติ: ถามภาพรวมเอกสารจังหวัด X → ถามต่อเรื่อง "อุบัติเหตุจราจร" (ไม่ระบุจังหวัด)
  → คำถามใหม่: "ผมถามถึงจังหวัด X" → ต้องตีความเป็น "อุบัติเหตุจราจร จังหวัด X"
  (ผูกกับหัวข้อ "อุบัติเหตุจราจร" ที่เพิ่งถามล่าสุด ไม่ใช่กลับไปถามภาพรวมเอกสารอีก)
ให้ยึดหลัก: หัวข้อ/คำถามเนื้อหาที่ "เจาะจงที่สุด" และอยู่ "ล่าสุด" ในประวัติ คือฐานที่
ต้องนำมาเติมคำชี้แจงขอบเขตใหม่เข้าไป ไม่ใช่หัวข้อแรกสุดหรือหัวข้อภาพรวมของบทสนทนา

ถ้าคำถามชัดเจนสมบูรณ์อยู่แล้ว ตอบว่า: UNCHANGED

ตอบเฉพาะคำถามที่ปรับปรุงแล้ว หรือ UNCHANGED เท่านั้น (ห้ามอธิบายเพิ่ม ห้ามใส่คำนำ):"""


def resolve_question(prompt: str, history_context: str, gemini_key: str) -> tuple[str, bool]:
    """Resolve a follow-up question using conversation history.

    Returns:
        (resolved_prompt, was_changed)
    """
    if not history_context or not gemini_key:
        return prompt, False
    try:
        resp = litellm.completion(
            model="gemini/gemini-2.5-flash-lite",
            api_key=gemini_key,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": _PROMPT.format(
                    history=history_context, prompt=prompt
                )},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        result = (resp.choices[0].message.content or "").strip()
        # Remove surrounding quotes if model added them
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        if result.upper() == "UNCHANGED" or not result:
            return prompt, False
        # Sanity: reject if output is absurdly long (model hallucinating)
        if len(result) > max(len(prompt) * 6, 400):
            return prompt, False
        return result, True
    except Exception as exc:
        log_agent_error(
            str(exc), agent_name="Memory Agent",
            step="memory", domain="", prompt=prompt[:120],
        )
        return prompt, False
