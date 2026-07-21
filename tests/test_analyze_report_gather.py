"""Regression tests for report-gather citation assembly in src/routers/analyze.py

บั๊กที่แก้: ตอน "สร้างรายงาน" (report-gather) รวม 5 แหล่งข้อมูล (Obsidian/สถิติ/
ThaiJo/PubMed/Tavily) เป็นรายงานเดียว — เดิม `report_source_parts` (วัตถุดิบที่ส่งให้
Report Generator ใช้เขียนส่วน "เอกสารอ้างอิง" ของรายงานฉบับจริง) มีแค่ ThaiJo/PubMed/
Tavily ที่พก URL ไปด้วย ส่วนเอกสารจากคลังความรู้ (Obsidian, ซึ่งตอนนี้มี pdf_url ต่อ
เอกสารแล้วหลัง dedupe — ดู test_obsidian_fullcontext.py) ไม่เคยถูกแนบ URL ไปกับ
รายงานที่สร้างขึ้นเลย ทั้งที่หน้าแชทปกติโชว์ลิงก์ PDF ให้กดได้แล้ว
"""
import types

from src.routers.analyze import _obsidian_notes_to_articles_text
from src.config import get_settings


def _note(title, province=None, pdf_url=None, note_id="x"):
    return types.SimpleNamespace(title=title, province=province, pdf_url=pdf_url, note_id=note_id)


class TestObsidianNotesToArticlesText:
    def test_empty_notes_returns_empty_string(self):
        assert _obsidian_notes_to_articles_text([]) == ""
        assert _obsidian_notes_to_articles_text(None) == ""

    def test_includes_pdf_url_when_present(self):
        notes = [_note("R_รายงานตรวจราชการ-อุบลราชธานี-2566", "อุบลราชธานี", "/api/pdf/view/815316")]
        text = _obsidian_notes_to_articles_text(notes)
        assert "R_รายงานตรวจราชการ-อุบลราชธานี-2566" in text
        assert "อุบลราชธานี" in text
        assert "/api/pdf/view/815316" in text

    def test_pdf_url_is_absolute_not_relative(self):
        """ล็อกบั๊กที่เจอจริง: ส่ง path สัมพัทธ์ตรง ๆ เข้าไปในข้อความให้ Report
        Generator (LLM) อ่าน แล้ว LLM ไม่ยอมทำเป็นลิงก์ <a href> ให้ (ต่างจาก
        ThaiJo/PubMed ที่ใช้ URL เต็มรูปแบบ https://... แล้วถูกทำเป็นลิงก์ถูกต้อง)
        — ต้องต่อ PUBLIC_APP_URL นำหน้าให้เป็น absolute URL เสมอ
        """
        notes = [_note("เอกสาร A", "อุบลราชธานี", "/api/pdf/view/815316")]
        text = _obsidian_notes_to_articles_text(notes)
        base = get_settings().PUBLIC_APP_URL.rstrip("/")
        assert f"URL:       {base}/api/pdf/view/815316" in text
        assert "URL:       /api/pdf/view/815316" not in text  # ต้องไม่ใช่ path สัมพัทธ์ล้วน ๆ

    def test_falls_back_to_dash_when_no_pdf_url(self):
        notes = [_note("หมายเหตุอื่นๆ", None, None)]
        text = _obsidian_notes_to_articles_text(notes)
        assert "หมายเหตุอื่นๆ" in text
        assert "URL:       -" in text

    def test_formats_multiple_notes_as_separate_numbered_blocks(self):
        notes = [
            _note("เอกสาร A", "อุบลราชธานี", "/api/pdf/view/1"),
            _note("เอกสาร B", "ศรีสะเกษ", "/api/pdf/view/2"),
        ]
        text = _obsidian_notes_to_articles_text(notes)
        assert "--- เอกสารคลังความรู้ที่ 1 ---" in text
        assert "--- เอกสารคลังความรู้ที่ 2 ---" in text
        assert "/api/pdf/view/1" in text
        assert "/api/pdf/view/2" in text
