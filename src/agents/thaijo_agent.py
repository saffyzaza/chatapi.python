"""ThaiJo Research Agent — CrewAI + Gemini pipeline.

Flow:
  [Step 1] ThaiJo Fetcher     — httpx GET → JSON articles
  [Step 2] Report Planner     — Gemini: วางแผนโครงสร้างรายงาน
  [Step 3] Report Generator   — Gemini: สร้าง structured JSON report

SSE events → queue:
  {"type": "agent_start",  "step": "fetcher",   "agentName": "ThaiJo Fetcher"}
  {"type": "agent_done",   "step": "fetcher",   "result": "พบ X บทความ", "articleCount": X}
  {"type": "agent_start",  "step": "planner",   "agentName": "Report Planner"}
  {"type": "agent_done",   "step": "planner",   "result": "...plan..."}
  {"type": "agent_start",    "step": "generator",  "agentName": "Report Generator"}
  {"type": "generator_chunk","html": "...chunk...",}  ← streamed while generating
  {"type": "agent_done",    "step": "generator",  "result": "สร้าง HTML report สำเร็จ (N chars)"}
  {"type": "final",         "reportHtml": "...", "reportTitle": "...", "articleCount": N, "agentSteps": [...]}
"""
import asyncio
import json
import os
import re
from typing import Any

import httpx
import litellm
from bs4 import BeautifulSoup
from crewai import Agent, Crew, LLM, Task
from openai import OpenAI

from src.config import get_settings
from src.agents.csv_pipeline import _run_agent, _is_agent_error
from src.tools.error_logger import log_agent_error
from src.agents.thaijo_prompts import build_prompt, doc_type_label, JOURNAL_CSS

# ── Journal HTML CSS (mirrors journalHtmlStyles.ts) ──────────────────────────

_JOURNAL_CSS = """
  @import url('https://fonts.googleapis.com/css2?family=Sarabun:ital,wght@0,400;0,600;0,700;1,400&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Sarabun','TH Sarabun New',sans-serif; font-size: 14px; line-height: 1.65; color: #111; background: #c8c8c8; padding: 20px 0; }
  .page {
    width: 794px;       /* A4 = 210mm @ 96dpi */
    min-height: 1123px; /* A4 = 297mm @ 96dpi */
    margin: 0 auto 24px auto;
    background: #fff;
    padding: 72px 72px 80px 72px; /* ~1 inch margins */
    box-shadow: 0 3px 24px rgba(0,0,0,0.22);
    position: relative;
    page-break-after: always;
  }
  @media print {
    body { background: #fff; padding: 0; }
    .page { box-shadow: none; margin: 0; }
  }
  .page-num { position: absolute; bottom: 28px; right: 72px; font-size: 11px; color: #777; }
  .tag-research { font-size: 13px; font-weight: 600; color: #c0392b; font-style: italic; margin-bottom: 6px; }
  .article-title { font-size: 15px; font-weight: 700; text-align: center; margin: 12px 0 8px; line-height: 1.5; color: #111; }
  .authors { text-align: center; font-size: 13px; font-style: italic; margin-bottom: 4px; color: #222; }
  .affiliations { text-align: center; font-size: 12px; color: #333; line-height: 1.6; margin-bottom: 4px; }
  .corresponding { text-align: center; font-size: 12px; color: #333; margin-bottom: 14px; }
  .divider { border: none; border-top: 1px solid #ccc; margin: 16px 0; }
  .section-heading { font-size: 14px; font-weight: 700; margin: 18px 0 8px; color: #111; }
  .body-para { font-size: 13px; line-height: 1.8; margin-bottom: 10px; text-indent: 2em; text-align: justify; }
  .keywords { font-size: 12px; margin: 10px 0 8px; }
  .keywords span.kw-label { font-weight: 700; }
  .abstract-box { background: #f8f9fa; border-left: 3px solid #1a3c6e; padding: 12px 16px; margin: 10px 0; }
  .abstract-label { font-size: 12px; font-weight: 700; color: #1a3c6e; margin-bottom: 4px; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 10px 0; }
  .data-table { width: 100%; border-collapse: collapse; font-size: 12px; margin: 12px 0; }
  .data-table caption { font-size: 12px; font-weight: 700; margin-bottom: 4px; text-align: left; color: #1a3c6e; }
  .data-table th { background: #1a3c6e; color: #fff; padding: 6px 8px; text-align: left; font-weight: 600; }
  .data-table td { border: 1px solid #ccc; padding: 5px 8px; }
  .data-table tr:nth-child(even) td { background: #f5f8fc; }
  .ref-list { list-style: decimal; padding-left: 1.6em; font-size: 12px; line-height: 1.7; color: #111; }
  .ref-list li { margin-bottom: 8px; text-align: justify; word-break: break-all; }
  .ref-list a { color: #1a55a0; text-decoration: underline; word-break: break-all; }
  .source-badge { display: inline-block; background: #e8f5ee; color: #1a6b3c; border: 1px solid #aad5b8; border-radius: 12px; font-size: 11px; padding: 2px 8px; margin-top: 6px; }
  sup.cite { font-size: 10px; color: #1a55a0; font-weight: 700; vertical-align: super; line-height: 0; }
  .ref-list li::marker { color: #1a3c6e; font-weight: 700; }
  .chart-box { margin: 16px 0; background: #f8faff; border: 1px solid #dce8f8; border-radius: 6px; padding: 12px 16px; }
  .chart-title { font-size: 12px; font-weight: 700; color: #1a3c6e; margin-bottom: 8px; }
  .chart-note { font-size: 11px; color: #666; margin-top: 6px; text-align: center; }
"""

# ── Generator format rules (adapted from promptplan.js) ───────────────────────

_FORMAT_RULES = """กฎบังคับการเขียน HTML:
1. ย่อหน้าต้องมี 3-5 ประโยคต่อเนื่องเป็นก้อนเดียวใน <p class="body-para"> ห้ามตัดบรรทัดกลางย่อหน้า
2. หัวข้อ section ต้องเป็นชื่อที่มีความหมาย ห้ามใช้ 'ส่วนที่ 1' หรือ 'หน้า 1'
3. ห้ามใช้ markdown ห้ามมี ```html block — ตอบ HTML ล้วนเท่านั้น
4. ตารางใช้ <table class="data-table"> เฉพาะเมื่อมีข้อมูลเชิงตัวเลขจริง ห้ามสร้างตารางเปล่า
5. อ้างอิงทุกแหล่งใน <ol class="ref-list"> ท้ายบทความ พร้อม URL
6. CRITICAL: ทุกครั้งที่กล่าวถึงข้อมูล ผล หรือข้อความจากบทความใด ต้องใส่เลขอ้างอิงแบบ inline ด้วย <sup class="cite">[n]</sup> ทันทีหลังข้อความนั้น เช่น ...พบว่าร้อยละ 78.6<sup class="cite">[1]</sup> ของผู้ป่วย...
   — เลข n ตรงกับลำดับใน ref-list ท้ายเล่ม ทุก paragraph ต้องมีการอ้างอิงอย่างน้อย 1 แห่ง
7. CHART: สร้างกราฟด้วย Chart.js สำหรับข้อมูลเชิงตัวเลขทุกชุด โดยใช้รูปแบบนี้:
   <div class="chart-box">
     <p class="chart-title">ชื่อกราฟ</p>
     <canvas id="chartN" style="max-height:280px"></canvas>
     <p class="chart-note">หมายเหตุ: ...</p>
   </div>
   <script>
   new Chart(document.getElementById('chartN'), {{
     type: 'bar',  /* หรือ 'line', 'pie', 'doughnut' ตามความเหมาะสม */
     data: {{ labels: [...], datasets: [{{ label: '...', data: [...], backgroundColor: [...] }}] }},
     options: {{ responsive: true, plugins: {{ legend: {{ position: 'bottom' }}, title: {{ display: false }} }} }}
   }});
   </script>
   — ใช้ id ไม่ซ้ำกัน (chart1, chart2, chart3, ...) — สร้างกราฟอย่างน้อย 2 กราฟในรายงาน"""


def _build_generator_prompt(query: str, plan: str, articles_text: str, article_count: int) -> str:
    return f"""คุณคือนักวิชาการอาวุโสด้านสาธารณสุขไทย เขียน Journal Report ฉบับสมบูรณ์ระดับวารสาร TCI อย่างน้อย 10 หน้า A4

หัวข้อ: {query}
แผนรายงาน:
{plan}

บทความที่ค้นพบ ({article_count} บทความ):
{articles_text}

{_FORMAT_RULES}
6. แต่ละหน้าต้องมีเนื้อหาหนาแน่น เต็มหน้า — ห้ามมีหน้าว่าง
7. ย่อหน้าต้องละเอียด 4-6 ประโยค พร้อมข้อมูลเชิงลึก ตัวเลข และการวิเคราะห์

โครงสร้าง (CRITICAL — ต้องสร้าง 10 <div class="page"> แยกกัน ขั้นต่ำ):

หน้า 1 — หน้าปก:
  <span class="tag-research">Review article</span>
  <h2 class="article-title"> ชื่อเต็มภาษาไทย
  <p class="authors"> ชื่อผู้แต่งทุกคน
  <p class="affiliations"> สังกัดและที่อยู่
  <hr class="divider">
  <div class="abstract-box"> บทคัดย่อภาษาไทย (5-6 ประโยค ครอบคลุมวัตถุประสงค์ วิธีการ ผล และข้อเสนอแนะ)
  <div class="abstract-box"> Abstract ภาษาอังกฤษ (ครบถ้วนเช่นเดียวกัน)
  <div class="keywords"> คำสำคัญ (5-7 คำ)
  <span class="page-num">1</span>

หน้า 2 — บทนำ (ส่วนที่ 1):
  <h3 class="section-heading">บทนำ</h3>
  4-5 <p class="body-para"> ครอบคลุม: ความสำคัญของปัญหา, สถานการณ์ระดับโลก, สถานการณ์ในประเทศไทย
  <span class="page-num">2</span>

หน้า 3 — บทนำ (ส่วนที่ 2) + วัตถุประสงค์:
  4-5 <p class="body-para"> ครอบคลุม: ปัจจัยที่เกี่ยวข้อง, งานวิจัยก่อนหน้า, ช่องว่างของความรู้
  <h3 class="section-heading">วัตถุประสงค์การวิจัย</h3>
  1-2 <p class="body-para">
  <span class="page-num">3</span>

หน้า 4 — วิธีการศึกษา:
  <h3 class="section-heading">วิธีการศึกษา</h3>
  <h3 class="section-heading">รูปแบบการวิจัย</h3> + 2 <p class="body-para">
  <h3 class="section-heading">แหล่งข้อมูลและเกณฑ์การคัดเลือก</h3> + 2 <p class="body-para">
  <h3 class="section-heading">การวิเคราะห์ข้อมูล</h3> + 2 <p class="body-para">
  <span class="page-num">4</span>

หน้า 5 — ผลการศึกษา (ส่วนที่ 1):
  <h3 class="section-heading">ผลการศึกษา</h3>
  <h3 class="section-heading">ลักษณะทั่วไปของบทความที่ทบทวน</h3> + 3 <p class="body-para">
  <table class="data-table"> ตารางสรุปบทความ (ชื่อผู้แต่ง, ปี, วัตถุประสงค์, ผล)
  <span class="page-num">5</span>

หน้า 6 — ผลการศึกษา (ส่วนที่ 2):
  <h3 class="section-heading">ผลการวิเคราะห์ข้อมูล</h3> + 3-4 <p class="body-para"> พร้อมตัวเลขสถิติ
  <table class="data-table"> ตารางข้อมูลเชิงปริมาณที่สำคัญ
  <span class="page-num">6</span>

หน้า 7 — ผลการศึกษา (ส่วนที่ 3) + ปัจจัยที่เกี่ยวข้อง:
  <h3 class="section-heading">ปัจจัยที่มีอิทธิพล</h3> + 4-5 <p class="body-para">
  <span class="page-num">7</span>

หน้า 8 — อภิปรายผล:
  <h3 class="section-heading">อภิปรายผล</h3>
  4-5 <p class="body-para"> เปรียบเทียบกับงานวิจัยอื่น อธิบายกลไก และสาเหตุของความแตกต่าง
  <span class="page-num">8</span>

หน้า 9 — สรุปและข้อเสนอแนะ:
  <h3 class="section-heading">สรุปผล</h3> + 2-3 <p class="body-para">
  <h3 class="section-heading">ข้อเสนอแนะเชิงปฏิบัติ</h3> + 2 <p class="body-para">
  <h3 class="section-heading">ข้อเสนอแนะเชิงนโยบาย</h3> + 2 <p class="body-para">
  <h3 class="section-heading">ข้อจำกัดของการศึกษา</h3> + 1 <p class="body-para">
  <span class="page-num">9</span>

หน้า 10 — เอกสารอ้างอิง:
  <h3 class="section-heading">เอกสารอ้างอิง</h3>
  <ol class="ref-list"> รายการอ้างอิงทุกบทความพร้อม URL
  <span class="page-num">10</span>

CSS ที่ต้องใส่ใน <style>:
{_JOURNAL_CSS}

ใน <head> ต้องมี Chart.js CDN:
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>

ตำแหน่งกราฟที่แนะนำ:
- หน้า 5: กราฟแท่ง (bar) แสดงสรุปผลลัพธ์หลักจากบทความที่ทบทวน
- หน้า 6: กราฟเส้น (line) หรือกราฟวงกลม (doughnut) แสดงข้อมูลสถิติ
- หน้า 7: กราฟแท่งแนวนอน (bar horizontal) แสดงปัจจัยที่มีอิทธิพล

ตอบเป็น HTML เท่านั้น เริ่มด้วย <!DOCTYPE html> ห้ามมี text หรือ ``` ก่อนหรือหลัง HTML:"""


def _fallback_html(query: str, article_count: int, articles: list[dict]) -> str:
    refs = "".join(
        f'<li>{a.get("reference", "-")}</li>'
        for a in articles if a.get("reference")
    )
    return f"""<!DOCTYPE html><html lang="th"><head><meta charset="UTF-8">
<style>{_JOURNAL_CSS}</style></head><body>
<div class="page">
  <span class="tag-research">Review article</span>
  <h2 class="article-title">{query}</h2>
  <h3 class="section-heading">สรุป</h3>
  <p class="body-para">พบ {article_count} บทความที่เกี่ยวข้องกับ {query}</p>
  <h3 class="section-heading">เอกสารอ้างอิง</h3>
  <ol class="ref-list">{refs}</ol>
</div></body></html>"""


# ── Mock / Demo articles ───────────────────────────────────────────────────────

_MOCK_ARTICLES: list[dict] = [
    {
        "pdf_url": "https://he01.tci-thaijo.org/index.php/jnat-ned/article/view/200911/140466",
        "summary": (
            "สรุปบทความวิชาการ: การเสริมพลังอำนาจผู้ป่วยโรคไม่ติดต่อเรื้อรัง\n\n"
            "บทความนี้ศึกษาการเสริมพลังอำนาจผู้ป่วยโรคไม่ติดต่อเรื้อรัง (NCDs) "
            "โดยเฉพาะผู้ป่วยโรคความดันโลหิตสูงและเบาหวาน ในการดูแลตนเองและมารับการรักษาต่อเนื่อง "
            "พบว่าร้อยละของผู้ป่วยที่ขึ้นทะเบียนและมารับการรักษาต่อเนื่องสูงถึง 82.4% "
            "เมื่อมีโปรแกรมเสริมพลังอำนาจ เทียบกับ 63.1% ในกลุ่มควบคุม\n\n"
            "ปัจจัยที่ส่งผลต่อการมารับการรักษาต่อเนื่อง ได้แก่ ความรู้เกี่ยวกับโรค "
            "การสนับสนุนจากครอบครัว และความสะดวกในการเข้าถึงบริการสาธารณสุข "
            "ข้อเสนอแนะ: ควรพัฒนาระบบติดตามผู้ป่วยเชิงรุกและส่งเสริมการดูแลตนเองที่บ้าน"
        ),
        "reference": (
            "แดนสีแก้ว ส, แสนโสม ด, รวยสูงเนิน ว, เมธากาญจนศักดิ์ น. "
            "การเสริมพลังอำนาจผู้ป่วยโรคไม่ติดต่อเรื้อรัง ในการลดการบริโภคอาหารที่มีโซเดียมสูง "
            "เพื่อป้องกันโรคไตเรื้อรัง. J Nurs Ther Care. 2019;37(2):238-46. "
            "available at: https://he01.tci-thaijo.org/index.php/jnat-ned/article/view/200911"
        ),
    },
    {
        "pdf_url": "https://he01.tci-thaijo.org/index.php/jnat-ned/article/view/150620/110338",
        "summary": (
            "สรุปบทความวิชาการ: ปัจจัยที่มีอิทธิพลต่อความมั่นคงด้านสุขภาพผู้ป่วยโรคเรื้อรังไม่ติดต่อ\n\n"
            "การศึกษาในจังหวัดพะเยา พบว่าผู้ป่วยโรคความดันโลหิตสูงที่ขึ้นทะเบียนในระบบ "
            "มารับการรักษาต่อเนื่องเฉลี่ยร้อยละ 78.6 โดยปัจจัยที่มีอิทธิพลสูงสุด ได้แก่ "
            "การรับรู้ความรุนแรงของโรค (OR=3.21) และการได้รับการสนับสนุนจากเจ้าหน้าที่สาธารณสุข (OR=2.87)\n\n"
            "ผู้ป่วยที่ไม่มารับการรักษาต่อเนื่องส่วนใหญ่มีเหตุผล ได้แก่ "
            "ไม่มีอาการ (45.2%) ระยะทางไกล (23.8%) และภาระงาน (18.6%) "
            "แนวทางแก้ไข: ระบบการนัดหมายอัตโนมัติและการดูแลที่บ้านโดยอาสาสมัครสาธารณสุข"
        ),
        "reference": (
            "แสงศรีจันทร์ ศ, โอภาสนันท์ ป, เกศหอม ม. "
            "ปัจจัยที่มีอิทธิพลต่อความมั่นคงด้านสุขภาพของผู้ป่วยกลุ่มโรคเรื้อรังไม่ติดต่อ "
            "ในจังหวัดพะเยา. J Nurs Ther Care. 2018;36(3):117-26. "
            "available at: https://he01.tci-thaijo.org/index.php/jnat-ned/article/view/150620"
        ),
    },
]

_DEMO_PROMPT = "ร้อยละผู้ป่วยความดันโลหิตสูงที่ขึ้นทะเบียนและมารับการรักษาต่อเนื่อง"

_DEMO_ROUTER_REASONING = (
    "คำถามนี้เกี่ยวกับข้อมูลทางคลินิกและการดูแลผู้ป่วยโรคความดันโลหิตสูง "
    "ซึ่งเป็นหัวข้อที่มีงานวิจัยใน ThaiJo จำนวนมาก "
    "จะค้นหาบทความที่เกี่ยวข้องจากฐานข้อมูล ThaiJo แล้วสังเคราะห์เป็น Journal Report อัตโนมัติ"
)


# ── LLM ───────────────────────────────────────────────────────────────────────

def _get_llm() -> LLM:
    return LLM(model="gemini/gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))


# ── Step 0: Keyword Extractor ─────────────────────────────────────────────

_KEYWORD_SYSTEM = (
    "You are an expert at searching Thai academic journals in the TCI ThaiJo database. "
    "The TCI ThaiJo full-text index is in English only — Thai search terms return zero results. "
    "Always translate topic and location to English medical/academic terminology."
)

_KEYWORD_PROMPT_TMPL = """From this Thai research question: "{prompt}"

Extract the best English search keywords for TCI ThaiJo.

Reply with JSON only — no markdown, no ``` :
{{
    "term": "2-4 English words, medical/academic terms only",
    "page": 1,
    "size": 5,
    "strict": true,
    "title": false,
    "author": false,
    "abstract": true,
    "reasoning": "why this term was chosen"
}}

Rules:
- term MUST be in English — the API indexes English content only
- Translate Thai province names: อุบล→Ubon, เชียงใหม่→Chiang Mai, ขอนแก่น→Khon Kaen, etc.
- Translate Thai medical terms: ความดันโลหิตสูง→hypertension, เบาหวาน→diabetes, ซึมเศร้า→depression, etc.
- Keep term short (2-4 words), focus on the core concept
- author: true only if the prompt names a specific researcher
- size: 3-8 depending on topic specificity

Examples:
  "โรคความดันโลหิตสูง จังหวัดอุบล" → term: "hypertension Ubon"
  "โรคซึมเศร้าในผู้ป่วยเบาหวาน"    → term: "depression diabetes patients"
  "นโยบายสาธารณสุข ความดันโลหิตสูง" → term: "hypertension public health policy\""""


def _extract_search_payload(prompt: str, gemini_key: str) -> dict:
    """Use Gemini flash to extract English TCI ThaiJo search keyword from Thai prompt."""
    default = {
        "term": prompt, "page": 1, "size": 5,
        "strict": True, "title": False, "author": False, "abstract": True,
        "reasoning": "ใช้ prompt โดยตรง (fallback)",
    }
    if not gemini_key:
        return default
    try:
        resp = litellm.completion(
            model="gemini/gemini-2.0-flash",
            api_key=gemini_key,
            messages=[
                {"role": "system", "content": _KEYWORD_SYSTEM},
                {"role": "user",   "content": _KEYWORD_PROMPT_TMPL.format(prompt=prompt)},
            ],
            temperature=0.1,
        )
        text = resp.choices[0].message.content or ""
        data = _extract_json(text)
        if data and "term" in data:
            data.setdefault("page", 1)
            data.setdefault("size", 5)
            data.setdefault("strict", True)
            data.setdefault("title", False)
            data.setdefault("author", False)
            data.setdefault("abstract", True)
            data["size"] = max(1, min(int(data["size"]), 10))
            return data
    except Exception as exc:
        log_agent_error(str(exc), agent_name="Keyword Extractor",
                        step="keyword", domain="thaijo", prompt=prompt)
    return default


# ── Step 1: ThaiJo Fetcher ─────────────────────────────────────────────────

_THAIJO_API_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; thaijo-api/1.0)",
    "Accept": "application/json,text/plain,*/*",
}
_THAIJO_HTML_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; thaijo-api/1.0)",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
}


def _summarize_pdf(pdf_url: str, term: str) -> str:
    """Summarize a PDF URL via OpenAI GPT-4.1, with Redis cache."""
    from src.tools.thaijo_cache import get_cached_summary, save_cached_summary

    cached = get_cached_summary(pdf_url)
    if cached:
        return cached

    s = get_settings()
    if not s.OPENAI_API_KEY:
        return f"[ไม่มี OPENAI_API_KEY — ข้ามการสรุป PDF]"

    try:
        client = OpenAI(api_key=s.OPENAI_API_KEY)
        prompt = (
            f"คุณเป็นนักวิจัยด้าน{term}\n\n"
            f"มีบทความวิชาการที่ลิงก์: {pdf_url}\n\n"
            "สรุปเนื้อหาบทความเป็นภาษาไทย 3-5 ย่อหน้า สำนวนเชิงวิชาการ\n"
            "โครงสร้าง:\n"
            "1. บทนำและความสำคัญ\n"
            "2. แนวคิดและผลการศึกษา\n"
            "3. ข้อค้นพบสำคัญ\n"
            "4. ข้อเสนอแนะและสรุป\n\n"
            "หากรายละเอียดเชิงตัวเลขไม่ครบ ให้สรุปเชิงแนวคิด"
        )
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": f"นักวิจัยด้าน{term}"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        summary = response.choices[0].message.content or ""
        save_cached_summary(pdf_url, summary)
        return summary
    except Exception as exc:
        log_agent_error(str(exc), agent_name="ThaiJo Fetcher",
                        step="summarize", domain="thaijo", prompt=pdf_url)
        return ""


def _translate_to_thai(abstract_en: str, title: str, openai_key: str) -> str:
    """Translate English abstract to Thai using OpenAI."""
    from src.tools.thaijo_cache import get_cached_summary, save_cached_summary
    cache_key = f"translate:{abstract_en[:80]}"
    cached = get_cached_summary(cache_key)
    if cached:
        return cached
    try:
        client = OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "แปลบทคัดย่อวิชาการเป็นภาษาไทย สำนวนเชิงวิชาการ"},
                {"role": "user", "content": f"หัวข้อ: {title}\n\nบทคัดย่อ:\n{abstract_en}\n\nแปลเป็นภาษาไทย:"},
            ],
            temperature=0.2,
            max_tokens=1000,
        )
        result = response.choices[0].message.content or ""
        save_cached_summary(cache_key, result)
        return result
    except Exception:
        return ""


def _is_thai(text: str) -> bool:
    """Check if text contains Thai characters."""
    return any("฀" <= c <= "๿" for c in (text or ""))


def _build_citation(result: dict) -> str:
    """Build APA-style citation from TCI ThaiJo API result."""
    authors = result.get("authors", [])
    names = []
    for a in authors[:6]:
        fn = a.get("full_name", {})
        name = fn.get("th_TH") or fn.get("en_US") or ""
        if name:
            names.append(name)
    author_str = ", ".join(names) if names else "ผู้แต่งไม่ระบุ"

    pub_date = result.get("datePublished") or result.get("issueDatePublished") or ""
    year = pub_date[:4] if pub_date else ""

    title = (result.get("title", {}).get("th_TH") or
             result.get("title", {}).get("en_US") or "")

    journal_url = result.get("thaijoUrl", "")
    journal = journal_url.split("/index.php/")[-1] if "/index.php/" in journal_url else journal_url

    pages = result.get("pages", "")
    url = result.get("articleUrl", "")

    parts = [author_str]
    if year:
        parts.append(f"({year})")
    if title:
        parts.append(f"{title}.")
    if journal:
        parts.append(journal)
    if pages:
        parts.append(f"หน้า {pages}.")
    if url:
        parts.append(url)

    return " ".join(parts)


def fetch_thaijo_articles(payload: dict) -> list[dict]:
    """POST ThaiJo search → use Thai abstract/title from API response directly.

    No HTML scraping needed — API response already contains th_TH content.
    Falls back to OpenAI translation only when abstract is non-Thai.
    payload keys: term, page, size, strict, title, author, abstract
    Returns list of {pdf_url, summary, reference}.
    """
    s = get_settings()
    search_url = f"{s.THAIJO_API_URL}/articles/search/"
    term = payload.get("term", "")
    size = payload.get("size", s.THAIJO_MAX_RESULTS)

    search_body = {
        "term":     term,
        "page":     payload.get("page", 1),
        "size":     size,
        "strict":   payload.get("strict", True),
        "title":    payload.get("title", False),
        "author":   payload.get("author", False),
        "abstract": payload.get("abstract", True),
    }

    try:
        resp = httpx.post(
            search_url,
            json=search_body,
            headers=_THAIJO_API_HEADERS,
            timeout=30,
            follow_redirects=True,
        )
        resp.raise_for_status()
        raw_results = resp.json().get("result", [])
    except Exception as exc:
        log_agent_error(str(exc), agent_name="ThaiJo Fetcher",
                        step="fetcher", domain="thaijo", prompt=term)
        return []

    articles = []
    for result in raw_results[:size]:
        article_url = result.get("articleUrl", "")
        if not article_url:
            continue

        # ── Prefer Thai content, fall back to English ──────────────────────
        title_th = (result.get("title", {}).get("th_TH") or
                    result.get("title", {}).get("en_US") or "")
        abstract_th = (result.get("abstract_clean", {}).get("th_TH") or
                       result.get("abstract_clean", {}).get("en_US") or "")

        if not abstract_th:
            continue

        # If abstract is not Thai → translate via OpenAI
        if not _is_thai(abstract_th) and s.OPENAI_API_KEY:
            translated = _translate_to_thai(abstract_th, title_th, s.OPENAI_API_KEY)
            if translated:
                abstract_th = translated

        summary = f"**{title_th}**\n\n{abstract_th}"
        citation = _build_citation(result)

        articles.append({
            "pdf_url":   article_url,
            "summary":   summary,
            "reference": citation,
        })

    return articles


def _articles_to_text(articles: list[dict]) -> str:
    if not articles:
        return "[ไม่มีบทความจาก API — ใช้ความรู้ทั่วไป]"
    lines = []
    for i, a in enumerate(articles, 1):
        lines.append(
            f"--- บทความที่ {i} ---\n"
            f"Reference: {a.get('reference', '-')}\n"
            f"Summary:   {a.get('summary',   '-')}\n"
            f"URL:       {a.get('pdf_url',   '-')}"
        )
    return "\n\n".join(lines)


# ── JSON Schema description for generator prompt ───────────────────────────

_JSON_SCHEMA_DESC = """
{
  "title":        "ชื่อรายงานภาษาไทย",
  "subtitle":     "English title",
  "journal_name": "ชื่อวารสาร",
  "volume_info":  "Vol. X(X): pp–pp, YYYY",
  "authors":      ["ชื่อผู้แต่ง 1", "ชื่อผู้แต่ง 2"],
  "affiliations": ["สังกัด 1"],
  "corresponding":"email หรือชื่อผู้ติดต่อ",
  "abstract_th":  "บทคัดย่อภาษาไทย 3-5 ประโยค",
  "abstract_en":  "Abstract 2-3 sentences",
  "keywords_th":  ["คำ1", "คำ2", "คำ3"],
  "keywords_en":  ["word1", "word2"],
  "introduction": ["ย่อหน้าบทนำ 1", "ย่อหน้าบทนำ 2"],
  "methods":      ["ย่อหน้าวิธีการ 1"],
  "results": [
    {"heading": "หัวข้อผลลัพธ์ 1", "paragraphs": ["ย่อหน้า 1", "ย่อหน้า 2"]},
    {"heading": "หัวข้อผลลัพธ์ 2", "paragraphs": ["ย่อหน้า 1"]}
  ],
  "table_head":     ["คอลัมน์ 1", "คอลัมน์ 2", "คอลัมน์ 3"],
  "table_rows":     [["แถว1ค1", "แถว1ค2", "แถว1ค3"]],
  "discussion":     ["ย่อหน้าอภิปราย 1", "ย่อหน้าอภิปราย 2"],
  "recommendation": ["ข้อเสนอแนะ 1", "ข้อเสนอแนะ 2"],
  "fig1_cap":       "คำอธิบายภาพ 1",
  "fig2_cap":       "คำอธิบายภาพ 2",
  "references":     ["Reference 1 APA", "Reference 2 APA"],
  "source_count":   3
}
"""


def _extract_json(text: str) -> dict | None:
    """Extract first JSON object from LLM text response."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m2 = re.search(r"\{.*\}", text, re.DOTALL)
    if m2:
        try:
            return json.loads(m2.group())
        except Exception:
            pass
    return None


def _fallback_report(query: str, article_count: int, articles: list[dict]) -> dict:
    return {
        "title":        f"รายงานการวิจัย: {query}",
        "subtitle":     f"Research Report: {query}",
        "journal_name": "ThaiJo Research Summary",
        "volume_info":  "",
        "authors":      [],
        "affiliations": [],
        "corresponding":"",
        "abstract_th":  (
            f"รายงานนี้สรุปข้อมูลเกี่ยวกับ {query} "
            f"จากการค้นหาในฐานข้อมูล ThaiJo พบ {article_count} บทความ"
        ),
        "abstract_en":  f"This report summarizes research on {query} from ThaiJo.",
        "keywords_th":  query.split()[:5],
        "keywords_en":  [],
        "introduction": [f"การวิจัยเกี่ยวกับ {query} มีความสำคัญต่อสาธารณสุข"],
        "methods":      ["รวบรวมข้อมูลจากฐานข้อมูล ThaiJo"],
        "results":      [{"heading": "ผลการค้นหา",
                          "paragraphs": [f"พบ {article_count} บทความที่เกี่ยวข้อง"]}],
        "table_head":   [],
        "table_rows":   [],
        "discussion":   [f"จากการสังเคราะห์ {article_count} บทความ พบประเด็นสำคัญ"],
        "recommendation":["ควรศึกษาเพิ่มเติมในประเด็นที่เกี่ยวข้อง"],
        "fig1_cap":     "",
        "fig2_cap":     "",
        "references":   [a.get("reference", "") for a in articles if a.get("reference")],
        "source_count": article_count,
    }


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_thaijo_pipeline(
    prompt: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    session_id: str = "",
    use_mock: bool = False,          # ← True = ใช้ mock articles แทนการ GET จริง
    doc_type: str = "policy",        # ← policy | plan | workplan
) -> None:
    """Stream ThaiJo research pipeline via SSE queue (Gemini).

    Args:
        use_mock: ถ้า True ให้ข้ามการ GET ThaiJo API และใช้ _MOCK_ARTICLES แทน
    """
    llm = _get_llm()

    def put(ev: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(queue.put(ev), loop)

    agent_steps: list[dict] = []

    # ── [Demo only] Simulate Router + Reasoning steps ─────────────────────
    if use_mock:
        # Simulate Router Agent step
        put({"type": "agent_start", "step": "router", "agentName": "Router Agent"})
        router_result = "domain: dt — วิจัย ThaiJo (ThaiJo Research)"
        put({
            "type": "agent_done",
            "step": "router",
            "agentName": "Router Agent",
            "result": router_result,
            "domain": {"code": "dt", "nameTh": "วิจัย ThaiJo", "nameEn": "ThaiJo Research"},
        })
        agent_steps.append({"step": "router", "agentName": "Router Agent",
                            "result": router_result})

        # Simulate Reasoning Narrator step
        put({"type": "agent_start", "step": "reasoning", "agentName": "Reasoning Narrator"})
        put({"type": "agent_done", "step": "reasoning", "agentName": "Reasoning Narrator",
             "result": _DEMO_ROUTER_REASONING})
        agent_steps.append({"step": "reasoning", "agentName": "Reasoning Narrator",
                            "result": _DEMO_ROUTER_REASONING})

    # ── STEP 0: Keyword Extractor ──────────────────────────────────────────
    if use_mock:
        search_payload = {
            "term": _DEMO_PROMPT, "page": 1, "size": 3,
            "strict": True, "title": True, "author": False, "abstract": True,
            "reasoning": "Demo mode — ใช้ prompt สำเร็จรูป",
        }
    else:
        put({"type": "agent_start", "step": "keyword", "agentName": "Keyword Extractor"})
        search_payload = _extract_search_payload(prompt, os.getenv("GEMINI_API_KEY", ""))
        put({
            "type":          "agent_done",
            "step":          "keyword",
            "agentName":     "Keyword Extractor",
            "result":        f"keyword: \"{search_payload['term']}\"",
            "searchPayload": {k: v for k, v in search_payload.items() if k != "reasoning"},
            "reasoning":     search_payload.get("reasoning", ""),
        })
        agent_steps.append({"step": "keyword", "agentName": "Keyword Extractor",
                            "result": f"keyword: \"{search_payload['term']}\""})

    # ── STEP 1: ThaiJo Fetcher ─────────────────────────────────────────────
    put({"type": "agent_start", "step": "fetcher", "agentName": "ThaiJo Fetcher"})

    if use_mock:
        articles = _MOCK_ARTICLES
        fetcher_result = f"[Demo] ใช้ข้อมูลตัวอย่าง {len(_MOCK_ARTICLES)} บทความ ('{_DEMO_PROMPT}')"
    else:
        articles = fetch_thaijo_articles(search_payload)
        fetcher_result = (
            f"พบ {len(articles)} บทความสำหรับ '{search_payload['term']}'"
            if articles else
            f"ไม่พบบทความจาก ThaiJo API — สร้างรายงานจากความรู้ทั่วไป"
        )

    article_count = len(articles)
    articles_text = _articles_to_text(articles)

    # Use original Thai prompt as report title; English term was for API search only
    query_for_plan = _DEMO_PROMPT if use_mock else prompt

    put({"type": "agent_done", "step": "fetcher", "agentName": "ThaiJo Fetcher",
         "result": fetcher_result, "articleCount": article_count,
         "isDemo": use_mock})
    agent_steps.append({"step": "fetcher", "agentName": "ThaiJo Fetcher",
                        "result": fetcher_result})

    # ── STEP 2: Report Planner ─────────────────────────────────────────────
    put({"type": "agent_start", "step": "planner", "agentName": "Report Planner"})

    planner = Agent(
        role="ThaiJo Report Planner",
        goal="วางแผนโครงสร้างรายงานวิชาการจากบทความที่ค้นพบ อย่างกระชับ",
        backstory=(
            "คุณเป็น editor วารสารวิชาการที่เชี่ยวชาญการสังเคราะห์งานวิจัยสาธารณสุขไทย "
            "คุณวางแผนโครงสร้างรายงานที่ครบถ้วน ระบุ theme และ section สำคัญ"
        ),
        llm=llm, verbose=False, max_iter=3,
    )

    plan = _run_agent(
        planner,
        (
            f"หัวข้อ: {query_for_plan}\n\n"
            f"บทความที่พบ ({article_count} บทความ):\n{articles_text}\n\n"
            "วางแผนโครงสร้างรายงาน (ไม่เกิน 10 บรรทัด):\n"
            "1. Theme หลัก 3-5 ประเด็น\n"
            "2. Section ที่ควรมี\n"
            "3. ข้อมูลสำหรับตาราง\n"
            "4. ประเด็น discussion สำคัญ"
        ),
        "แผนโครงสร้างรายงานสั้นๆ ไม่เกิน 10 บรรทัด",
        step="planner", session_id=session_id,
    )

    put({"type": "agent_done", "step": "planner", "agentName": "Report Planner",
         "result": plan, "docType": doc_type, "docTypeLabel": doc_type_label(doc_type)})
    agent_steps.append({"step": "planner", "agentName": "Report Planner", "result": plan})

    # ── STEP 3: Report Generator (streaming HTML) ────────────────────────────
    put({"type": "agent_start", "step": "generator", "agentName": "Report Generator"})

    gen_prompt = build_prompt(doc_type, query_for_plan, plan, articles_text, article_count)
    html_parts: list[str] = []
    stream_error = ""

    try:
        stream = litellm.completion(
            model="gemini/gemini-2.5-pro",
            api_key=os.getenv("GEMINI_API_KEY"),
            messages=[{"role": "user", "content": gen_prompt}],
            stream=True,
            temperature=0.3,
        )
        for chunk in stream:
            delta = ""
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta.content or ""
            if delta:
                html_parts.append(delta)
                put({"type": "generator_chunk", "html": delta})
    except Exception as exc:
        stream_error = str(exc)
        log_agent_error(stream_error, agent_name="Report Generator",
                        step="generator", domain="thaijo", prompt=query_for_plan)

    full_html = "".join(html_parts).strip()

    # Strip markdown fences if model wrapped output
    if full_html.startswith("```"):
        full_html = re.sub(r"^```[a-z]*\n?", "", full_html)
        full_html = re.sub(r"\n?```$", "", full_html).strip()

    if not full_html or "<html" not in full_html:
        full_html = _fallback_html(query_for_plan, article_count, articles)
        put({"type": "generator_chunk", "html": full_html})

    done_msg = f"สร้าง HTML report สำเร็จ ({len(full_html)} ตัวอักษร)"
    put({"type": "agent_done", "step": "generator", "agentName": "Report Generator",
         "result": done_msg})
    agent_steps.append({"step": "generator", "agentName": "Report Generator",
                        "result": done_msg})

    # ── FINAL EVENT ────────────────────────────────────────────────────────
    put({
        "type":         "final",
        "message":      f"สร้าง journal report จาก {article_count} บทความสำเร็จ",
        "reportHtml":   full_html,
        "reportTitle":  query_for_plan,
        "articleCount": article_count,
        "agentSteps":   agent_steps,
    })
