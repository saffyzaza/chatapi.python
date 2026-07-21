"""Regression tests for _linkify_bare_urls in src/agents/thaijo_agent.py

บั๊กที่แก้: พรอมต์สั่งให้ LLM ห่อ URL ในส่วน "เอกสารอ้างอิง" ด้วย <a href> เสมอ แต่
เป็นพฤติกรรม non-deterministic ของ LLM — บางรอบเขียนแค่ "URL: https://..." เป็น
plain text เฉยๆ ทำให้ผู้ใช้กดลิงก์ไม่ได้ทั้งที่ URL ถูกต้อง (ยืนยันจากรายงานจริงที่
สร้างแล้ว URL ทุกรายการในลิสต์ "เอกสารอ้างอิง" ไม่ถูกทำเป็นลิงก์เลยแม้แต่อันเดียว
ทั้งที่ก่อนหน้านี้เคยเห็น LLM ทำให้ถูกในบางรอบ) — จึงต้อง post-process บังคับ
ห่อ URL ที่หลุดมาเป็น <a> เสมอ ไม่พึ่งพรอมต์อย่างเดียว
"""
from src.agents.thaijo_agent import _linkify_bare_urls


class TestLinkifyBareUrls:
    def test_wraps_bare_url_in_anchor_tag(self):
        html = '<li>[R1] สรุปผลการดำเนินงาน. URL: http://localhost:3000/api/pdf/view/912883</li>'
        result = _linkify_bare_urls(html)
        assert '<a href="http://localhost:3000/api/pdf/view/912883"' in result
        assert 'target="_blank"' in result

    def test_does_not_double_wrap_existing_anchor(self):
        html = '<li><a href="https://he04.tci-thaijo.org/x">https://he04.tci-thaijo.org/x</a></li>'
        result = _linkify_bare_urls(html)
        assert result.count("<a ") == 1
        assert "<a href=\"https://he04.tci-thaijo.org/x\"><a" not in result

    def test_leaves_href_attribute_values_untouched(self):
        html = '<a href="https://example.com/page">คลิกที่นี่</a>'
        result = _linkify_bare_urls(html)
        assert result == html

    def test_strips_trailing_punctuation_from_wrapped_url(self):
        html = 'อ้างอิงจาก https://example.com/report. ต่อด้วยประโยคถัดไป'
        result = _linkify_bare_urls(html)
        assert '>https://example.com/report</a>. ต่อด้วย' in result

    def test_wraps_multiple_bare_urls_in_reference_list(self):
        html = (
            '<ol class="ref-list">'
            '<li>[1] เอกสาร A. URL: http://localhost:3000/api/pdf/view/1</li>'
            '<li>[2] เอกสาร B. URL: https://he04.tci-thaijo.org/index.php/x/article/view/2</li>'
            '</ol>'
        )
        result = _linkify_bare_urls(html)
        assert '<a href="http://localhost:3000/api/pdf/view/1"' in result
        assert '<a href="https://he04.tci-thaijo.org/index.php/x/article/view/2"' in result

    def test_empty_and_no_url_input_unchanged(self):
        assert _linkify_bare_urls("") == ""
        assert _linkify_bare_urls("<p>ไม่มี URL ในข้อความนี้</p>") == "<p>ไม่มี URL ในข้อความนี้</p>"

    def test_strips_trailing_closing_bracket_citation_style(self):
        """บั๊กจริงที่เจอ: รูปแบบ [URL: http://...] วงเล็บเหลี่ยมปิดท้ายถูกกลืน
        เข้าไปเป็นส่วนหนึ่งของ URL/ลิงก์ไปด้วย ทำให้ลิงก์พาไปหน้าที่ไม่มีจริง"""
        html = '1. สำนักงานสาธารณสุขจังหวัดยโสธร. [URL: http://localhost:3000/api/pdf/view/912883]'
        result = _linkify_bare_urls(html)
        assert '<a href="http://localhost:3000/api/pdf/view/912883" target="_blank" rel="noopener noreferrer">http://localhost:3000/api/pdf/view/912883</a>]' in result
        assert 'view/912883]"' not in result
