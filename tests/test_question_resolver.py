"""Regression tests for src/agents/question_resolver.py (Memory Agent).

ล็อกไว้ว่าพรอมต์ต้องมีคำแนะนำเรื่อง "ประโยคยืนยัน/เจาะจงขอบเขต" (เช่น
"ผมถามถึง X", "หมายถึง X นะ") อยู่เสมอ — เกิดจากบั๊กที่เจอจริงตอนทดสอบ: ผู้ใช้
พิมพ์ "ผมถามถึงจังหวัดอุบล" หลังจากเพิ่งคุยเรื่อง "อุบัติเหตุจราจร" ค้างอยู่ แต่
Memory Agent กลับขยายคำถามใหม่กว้างขึ้นย้อนกลับไปหาหัวข้อแรกสุดของบทสนทนาแทนที่
จะเกาะประเด็นล่าสุด (อุบัติเหตุ) ไว้

หมายเหตุ: ไม่ได้เทสต์ผลลัพธ์จริงจาก Gemini (ต้องยิง API จริง) — เทสต์นี้ล็อกแค่ว่า
"คำสั่งพรอมต์ที่ป้อนให้โมเดล" มีตัวอย่าง/กติกาที่ถูกต้องอยู่ครบ กัน regression ที่
อาจเกิดจากใครมาแก้พรอมต์แล้วเผลอลบคำแนะนำนี้ทิ้งในอนาคต
"""
from src.agents import question_resolver as qr


class TestPromptTemplate:
    def test_includes_scope_confirmation_guidance(self):
        rendered = qr._PROMPT.format(history="ประวัติทดสอบ", prompt="ผมถามถึงจังหวัดอุบล")
        assert "ยืนยัน/เจาะจงขอบเขต" in rendered
        assert "ผมถามถึง" in rendered

    def test_instructs_to_anchor_on_latest_specific_topic_not_conversation_start(self):
        rendered = qr._PROMPT.format(history="ประวัติทดสอบ", prompt="ผมถามถึงจังหวัดอุบล")
        assert "ล่าสุด" in rendered
        assert "ไม่ใช่หัวข้อแรกสุด" in rendered or "ไม่ใช่กลับไปถาม" in rendered

    def test_prompt_renders_without_crashing_for_various_inputs(self):
        for prompt in ["ผมถามถึงจังหวัดอุบล", "หมายถึงจังหวัดศรีสะเกษนะ", "3. อุบัติเหตุจราจร"]:
            rendered = qr._PROMPT.format(history="ประวัติทดสอบ", prompt=prompt)
            assert prompt in rendered


class TestResolveQuestionGuards:
    def test_short_circuits_without_history(self):
        resolved, changed = qr.resolve_question("คำถามอะไรก็ได้", "", "some-key")
        assert resolved == "คำถามอะไรก็ได้"
        assert changed is False

    def test_short_circuits_without_gemini_key(self):
        resolved, changed = qr.resolve_question("คำถามอะไรก็ได้", "ประวัติเก่า", "")
        assert resolved == "คำถามอะไรก็ได้"
        assert changed is False
