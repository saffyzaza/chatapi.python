"""ThaiJo Prompt Components — 3 document types for report generation.

doc_type:
  "policy"   — การขอนโยบาย (Policy Brief)
  "plan"     — การเขียนแผน (Strategic Plan Writing)
  "workplan" — การวางแผนงาน (Operational Work Plan)
"""
from __future__ import annotations

# ── Shared Journal CSS ─────────────────────────────────────────────────────────

JOURNAL_CSS = """
  @import url('https://fonts.googleapis.com/css2?family=Sarabun:ital,wght@0,400;0,600;0,700;1,400&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Sarabun','TH Sarabun New',sans-serif; font-size: 14px; line-height: 1.65; color: #111; background: #c8c8c8; padding: 20px 0; }
  .page {
    width: 794px;
    min-height: 1123px;
    margin: 0 auto 24px auto;
    background: #fff;
    padding: 72px 72px 80px 72px;
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
  .divider { border: none; border-top: 1px solid #ccc; margin: 16px 0; }
  .section-heading { font-size: 14px; font-weight: 700; margin: 18px 0 8px; color: #111; }
  .sub-heading { font-size: 13px; font-weight: 600; margin: 12px 0 6px; color: #1a3c6e; }
  .body-para { font-size: 13px; line-height: 1.8; margin-bottom: 10px; text-indent: 2em; text-align: justify; }
  .keywords { font-size: 12px; margin: 10px 0 8px; }
  .keywords span.kw-label { font-weight: 700; }
  .abstract-box { background: #f8f9fa; border-left: 3px solid #1a3c6e; padding: 12px 16px; margin: 10px 0; }
  .abstract-label { font-size: 12px; font-weight: 700; color: #1a3c6e; margin-bottom: 4px; }
  .exec-box { background: #eef5ee; border-left: 4px solid #2e9e5b; padding: 14px 18px; margin: 12px 0; border-radius: 4px; }
  .exec-label { font-size: 12px; font-weight: 700; color: #1a6b3c; margin-bottom: 6px; }
  .highlight-box { background: #fff8e1; border-left: 4px solid #f59e0b; padding: 12px 16px; margin: 12px 0; border-radius: 4px; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 10px 0; }
  .three-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin: 10px 0; }
  .data-table { width: 100%; border-collapse: collapse; font-size: 12px; margin: 12px 0; }
  .data-table caption { font-size: 12px; font-weight: 700; margin-bottom: 4px; text-align: left; color: #1a3c6e; }
  .data-table th { background: #1a3c6e; color: #fff; padding: 6px 8px; text-align: left; font-weight: 600; }
  .data-table td { border: 1px solid #ccc; padding: 5px 8px; }
  .data-table tr:nth-child(even) td { background: #f5f8fc; }
  .ref-list { list-style: decimal; padding-left: 1.6em; font-size: 12px; line-height: 1.7; color: #111; }
  .ref-list li { margin-bottom: 8px; text-align: justify; word-break: break-all; }
  .ref-list li::marker { color: #1a3c6e; font-weight: 700; }
  .ref-list a { color: #1a55a0; text-decoration: underline; word-break: break-all; }
  sup.cite { font-size: 10px; color: #1a55a0; font-weight: 700; vertical-align: super; line-height: 0; }
  .chart-box { margin: 16px 0; background: #f8faff; border: 1px solid #dce8f8; border-radius: 6px; padding: 12px 16px; }
  .chart-title { font-size: 12px; font-weight: 700; color: #1a3c6e; margin-bottom: 8px; }
  .chart-note { font-size: 11px; color: #666; margin-top: 6px; text-align: center; }
  .kpi-card { background: #f0f7ff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 10px 14px; margin: 6px 0; }
  .kpi-label { font-size: 11px; color: #1a3c6e; font-weight: 600; }
  .kpi-value { font-size: 18px; font-weight: 700; color: #1a55a0; }
  .timeline-row { display: flex; gap: 8px; align-items: flex-start; margin: 6px 0; }
  .timeline-dot { width: 10px; height: 10px; border-radius: 50%; background: #1a3c6e; flex-shrink: 0; margin-top: 4px; }
  .badge { display: inline-block; border-radius: 12px; font-size: 11px; padding: 2px 10px; font-weight: 600; }
  .badge-green { background: #d1fae5; color: #065f46; }
  .badge-blue  { background: #dbeafe; color: #1e40af; }
  .badge-orange{ background: #fef3c7; color: #92400e; }
"""

# ── Shared chart rule (for f-string) ──────────────────────────────────────────

_CHART_RULE = """7. CHART: สร้างกราฟด้วย Chart.js สำหรับข้อมูลเชิงตัวเลขทุกชุด:
   <div class="chart-box">
     <p class="chart-title">ชื่อกราฟ</p>
     <canvas id="chartN" style="max-height:280px"></canvas>
     <p class="chart-note">ที่มา: ...</p>
   </div>
   <script>
   new Chart(document.getElementById('chartN'), {{
     type: 'bar',
     data: {{ labels: [...], datasets: [{{ label: '...', data: [...], backgroundColor: [...] }}] }},
     options: {{ responsive: true, plugins: {{ legend: {{ position: 'bottom' }} }} }}
   }});
   </script>
   — id ไม่ซ้ำ (chart1, chart2, ...) — อย่างน้อย 2 กราฟต่อรายงาน"""

_BASE_RULES = """กฎบังคับการเขียน HTML:
1. ย่อหน้าต้องมี 3-5 ประโยคต่อเนื่องใน <p class="body-para"> ห้ามตัดกลางย่อหน้า
2. หัวข้อ section ต้องมีความหมาย ห้ามใช้ 'ส่วนที่ 1' หรือ 'หน้า 1'
3. ห้ามใช้ markdown ห้ามมี ```html — ตอบ HTML ล้วนเท่านั้น
4. ตารางใช้ <table class="data-table"> เฉพาะเมื่อมีตัวเลขจริง ห้ามสร้างตารางเปล่า
5. อ้างอิงทุกแหล่งใน <ol class="ref-list"> ท้ายเล่ม พร้อม URL
6. INLINE CITE: ทุกครั้งที่กล่าวถึงข้อมูล/ผลจากบทความ ให้ใส่ <sup class="cite">[n]</sup> ทันที เช่น ...ร้อยละ 78.6<sup class="cite">[1]</sup>...
{chart_rule}
8. แต่ละหน้าต้องเนื้อหาหนาแน่น เต็มหน้า ห้ามมีหน้าว่าง
9. ย่อหน้าละ 4-6 ประโยค พร้อมข้อมูลเชิงลึกและตัวเลข""".format(chart_rule=_CHART_RULE)

_HEAD_SCRIPT = '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>'


# ── 1. Policy Brief — การขอนโยบาย ────────────────────────────────────────────

def build_policy_prompt(query: str, plan: str, articles_text: str, article_count: int) -> str:
    return f"""คุณคือนักวิชาการด้านนโยบายสาธารณสุขไทย เขียน Policy Brief ฉบับสมบูรณ์ระดับ WHO/สธ. อย่างน้อย 10 หน้า A4

หัวข้อ: {query}
แผนรายงาน:
{plan}

บทความที่ค้นพบ ({article_count} บทความ):
{articles_text}

{_BASE_RULES}

โครงสร้าง Policy Brief (CRITICAL — ต้องสร้าง 10 <div class="page"> แยกกัน):

หน้า 1 — ปกและบทสรุปผู้บริหาร:
  <span class="tag-research">Policy Brief</span>
  <h2 class="article-title"> ชื่อ Policy Brief ภาษาไทย
  <p class="authors"> ผู้จัดทำ / หน่วยงาน
  <hr class="divider">
  <div class="exec-box"> <p class="exec-label">บทสรุปผู้บริหาร</p> + 4-5 ประโยคสรุปสถานการณ์ ปัญหา ข้อเสนอ
  <div class="keywords"> คำสำคัญ
  <span class="page-num">1</span>

หน้า 2 — ความเป็นมาและขนาดของปัญหา:
  <h3 class="section-heading">ความเป็นมาและความสำคัญของปัญหา</h3>
  4-5 <p class="body-para"> + ตัวเลขระบาดวิทยาและผลกระทบ
  <span class="page-num">2</span>

หน้า 3 — สถานการณ์และบริบทในประเทศไทย:
  <h3 class="section-heading">สถานการณ์ปัจจุบันในประเทศไทย</h3>
  4-5 <p class="body-para"> + chart (bar/line) แสดงแนวโน้ม
  <span class="page-num">3</span>

หน้า 4-5 — หลักฐานจากงานวิจัย:
  <h3 class="section-heading">ผลการทบทวนหลักฐานเชิงประจักษ์</h3>
  ตาราง + 4-5 <p class="body-para"> + กราฟเปรียบเทียบผล
  <span class="page-num">4/5</span>

หน้า 6 — ปัจจัยและอุปสรรค:
  <h3 class="section-heading">ปัจจัยที่มีผลต่อปัญหา</h3>
  chart (horizontal bar) + 3-4 <p class="body-para">
  <span class="page-num">6</span>

หน้า 7 — ทางเลือกนโยบาย (Policy Options):
  <h3 class="section-heading">ทางเลือกนโยบาย</h3>
  3 ทางเลือกในตาราง (ทางเลือก, ข้อดี, ข้อจำกัด, ความเป็นไปได้) + อธิบาย
  <span class="page-num">7</span>

หน้า 8 — ข้อเสนอแนะเชิงนโยบาย:
  <h3 class="section-heading">ข้อเสนอแนะเชิงนโยบาย</h3>
  ระยะสั้น/กลาง/ยาว + หน่วยงานรับผิดชอบ
  <span class="page-num">8</span>

หน้า 9 — แผนดำเนินงานเบื้องต้นและตัวชี้วัด:
  <h3 class="section-heading">แผนการดำเนินงานและตัวชี้วัด</h3>
  ตาราง KPI + 2-3 <p class="body-para">
  <span class="page-num">9</span>

หน้า 10 — เอกสารอ้างอิง:
  <h3 class="section-heading">เอกสารอ้างอิง</h3>
  <ol class="ref-list"> ทุกบทความพร้อม URL
  <span class="page-num">10</span>

CSS ใส่ใน <style>:
{JOURNAL_CSS}

ใน <head>:
{_HEAD_SCRIPT}

ตอบเป็น HTML เท่านั้น เริ่มด้วย <!DOCTYPE html> ห้ามมี text หรือ ``` ก่อนหรือหลัง HTML:"""


# ── 2. Strategic Plan — การเขียนแผน ──────────────────────────────────────────

def build_plan_prompt(query: str, plan: str, articles_text: str, article_count: int) -> str:
    return f"""คุณคือนักวางแผนยุทธศาสตร์สาธารณสุขไทย เขียนแผนยุทธศาสตร์ฉบับสมบูรณ์ระดับกระทรวง/กรม อย่างน้อย 10 หน้า A4

หัวข้อ: {query}
แผนรายงาน:
{plan}

บทความที่ค้นพบ ({article_count} บทความ):
{articles_text}

{_BASE_RULES}

โครงสร้าง Strategic Plan (CRITICAL — ต้องสร้าง 10 <div class="page"> แยกกัน):

หน้า 1 — ปกและบทนำ:
  <span class="tag-research">แผนยุทธศาสตร์</span>
  <h2 class="article-title"> ชื่อแผนยุทธศาสตร์
  <p class="authors"> หน่วยงานที่รับผิดชอบ / ระยะเวลา
  <hr class="divider">
  <div class="exec-box"> บทสรุปแผนยุทธศาสตร์ (วิสัยทัศน์ พันธกิจ เป้าหมายหลัก)
  <span class="page-num">1</span>

หน้า 2 — บริบทและสถานการณ์:
  <h3 class="section-heading">บริบทและสถานการณ์ปัจจุบัน</h3>
  SWOT หรือ PESTEL + 4-5 <p class="body-para"> + chart
  <span class="page-num">2</span>

หน้า 3 — วิสัยทัศน์ พันธกิจ และเป้าประสงค์:
  <h3 class="section-heading">วิสัยทัศน์และพันธกิจ</h3>
  <div class="highlight-box"> วิสัยทัศน์
  <h3 class="section-heading">เป้าประสงค์เชิงยุทธศาสตร์</h3>
  kpi-card x3 + 2-3 <p class="body-para">
  <span class="page-num">3</span>

หน้า 4 — ยุทธศาสตร์และกลยุทธ์:
  <h3 class="section-heading">ยุทธศาสตร์หลัก</h3>
  3-4 ยุทธศาสตร์ แต่ละอันมีกลยุทธ์ย่อย + ตาราง
  <span class="page-num">4</span>

หน้า 5-6 — แผนปฏิบัติการ (Action Plan):
  <h3 class="section-heading">แผนปฏิบัติการ</h3>
  ตาราง (กิจกรรม, ผู้รับผิดชอบ, ระยะเวลา, งบประมาณ, ตัวชี้วัด) + chart แสดง timeline
  <span class="page-num">5/6</span>

หน้า 7 — งบประมาณและทรัพยากร:
  <h3 class="section-heading">แผนงบประมาณ</h3>
  ตาราง breakdown งบ + chart (doughnut) สัดส่วนงบ
  <span class="page-num">7</span>

หน้า 8 — การติดตามและประเมินผล:
  <h3 class="section-heading">ระบบการติดตามและประเมินผล</h3>
  ตาราง KPI (ตัวชี้วัด, ค่าเป้าหมาย, ความถี่, ผู้รับผิดชอบ) + 2-3 <p class="body-para">
  <span class="page-num">8</span>

หน้า 9 — ปัจจัยความสำเร็จและความเสี่ยง:
  <h3 class="section-heading">ปัจจัยความสำเร็จ</h3> + 2 <p class="body-para">
  <h3 class="section-heading">การบริหารความเสี่ยง</h3> ตาราง (ความเสี่ยง, โอกาส, ผลกระทบ, มาตรการ)
  <span class="page-num">9</span>

หน้า 10 — เอกสารอ้างอิง:
  <h3 class="section-heading">เอกสารอ้างอิง</h3>
  <ol class="ref-list"> ทุกบทความพร้อม URL
  <span class="page-num">10</span>

CSS ใส่ใน <style>:
{JOURNAL_CSS}

ใน <head>:
{_HEAD_SCRIPT}

ตอบเป็น HTML เท่านั้น เริ่มด้วย <!DOCTYPE html> ห้ามมี text หรือ ``` ก่อนหรือหลัง HTML:"""


# ── 3. Work Plan — การวางแผนงาน ──────────────────────────────────────────────

def build_workplan_prompt(query: str, plan: str, articles_text: str, article_count: int) -> str:
    return f"""คุณคือผู้จัดการโครงการสาธารณสุขไทย เขียนแผนงาน/โครงการฉบับสมบูรณ์ระดับสสจ./รพ. อย่างน้อย 10 หน้า A4

หัวข้อ: {query}
แผนรายงาน:
{plan}

บทความที่ค้นพบ ({article_count} บทความ):
{articles_text}

{_BASE_RULES}

โครงสร้าง Work Plan (CRITICAL — ต้องสร้าง 10 <div class="page"> แยกกัน):

หน้า 1 — ชื่อโครงการและภาพรวม:
  <span class="tag-research">แผนงาน/โครงการ</span>
  <h2 class="article-title"> ชื่อโครงการ
  <p class="authors"> หน่วยงาน / ผู้รับผิดชอบ
  <hr class="divider">
  <div class="exec-box"> ภาพรวมโครงการ (วัตถุประสงค์หลัก กลุ่มเป้าหมาย ระยะเวลา งบประมาณรวม)
  <div class="three-col"> 3 kpi-card แสดงตัวชี้วัดหลัก
  <span class="page-num">1</span>

หน้า 2 — หลักการและเหตุผล:
  <h3 class="section-heading">หลักการและเหตุผล</h3>
  4-5 <p class="body-para"> + chart แสดงสถานการณ์ปัจจุบัน
  <span class="page-num">2</span>

หน้า 3 — วัตถุประสงค์และกลุ่มเป้าหมาย:
  <h3 class="section-heading">วัตถุประสงค์</h3> วัตถุประสงค์ทั่วไปและเฉพาะ
  <h3 class="section-heading">กลุ่มเป้าหมายและพื้นที่ดำเนินการ</h3>
  ตาราง (กลุ่ม, จำนวน, พื้นที่) + 2 <p class="body-para">
  <span class="page-num">3</span>

หน้า 4 — วิธีดำเนินการ (ส่วนที่ 1):
  <h3 class="section-heading">วิธีดำเนินการ</h3>
  <h3 class="sub-heading">ระยะที่ 1: การเตรียมการ</h3> กิจกรรมและขั้นตอน
  <h3 class="sub-heading">ระยะที่ 2: การดำเนินการ</h3> กิจกรรมและขั้นตอน
  timeline-row + chart (bar) แสดง Gantt
  <span class="page-num">4</span>

หน้า 5 — วิธีดำเนินการ (ส่วนที่ 2) + ระยะเวลา:
  <h3 class="sub-heading">ระยะที่ 3: การติดตามและประเมิน</h3>
  ตาราง Gantt Chart (กิจกรรม vs เดือน ม.ค.–ธ.ค.)
  <span class="page-num">5</span>

หน้า 6 — ผู้รับผิดชอบและหน่วยงาน:
  <h3 class="section-heading">โครงสร้างการบริหารโครงการ</h3>
  ตาราง RACI (กิจกรรม, ผู้รับผิดชอบ, ผู้สนับสนุน, ผู้ตรวจสอบ) + 2 <p class="body-para">
  <span class="page-num">6</span>

หน้า 7 — งบประมาณ:
  <h3 class="section-heading">งบประมาณรายละเอียด</h3>
  ตาราง (หมวด, รายการ, จำนวน, ราคาต่อหน่วย, รวม) + chart (doughnut) สัดส่วนงบ
  <span class="page-num">7</span>

หน้า 8 — ตัวชี้วัดและการติดตาม:
  <h3 class="section-heading">ตัวชี้วัดความสำเร็จ (KPIs)</h3>
  ตาราง (ตัวชี้วัด, นิยาม, ค่าเป้าหมาย, วิธีวัด, ความถี่, ผู้รับผิดชอบ) + 2 <p class="body-para">
  <span class="page-num">8</span>

หน้า 9 — ความเสี่ยงและแผนรองรับ:
  <h3 class="section-heading">การบริหารความเสี่ยง</h3>
  ตาราง Risk Matrix (ความเสี่ยง, โอกาสเกิด, ผลกระทบ, ระดับ, มาตรการ) + badge สีระดับความเสี่ยง
  <h3 class="section-heading">ปัจจัยความสำเร็จ</h3> + 2 <p class="body-para">
  <span class="page-num">9</span>

หน้า 10 — เอกสารอ้างอิง:
  <h3 class="section-heading">เอกสารอ้างอิง</h3>
  <ol class="ref-list"> ทุกบทความพร้อม URL
  <span class="page-num">10</span>

CSS ใส่ใน <style>:
{JOURNAL_CSS}

ใน <head>:
{_HEAD_SCRIPT}

ตอบเป็น HTML เท่านั้น เริ่มด้วย <!DOCTYPE html> ห้ามมี text หรือ ``` ก่อนหรือหลัง HTML:"""


# ── Custom prompt (user-selected topics from wizard) ─────────────────────────

def build_custom_prompt(
    query: str,
    plan: str,
    articles_text: str,
    article_count: int,
    topic_plan: str,
    doc_type: str,
) -> str:
    """สร้าง prompt สำหรับ Gemini โดยใช้หัวข้อที่ผู้ใช้เลือกผ่าน wizard เป็นโครงสร้างหลัก"""
    _DOC_LABEL = {"policy": "Policy Brief", "plan": "แผนยุทธศาสตร์", "workplan": "แผนปฏิบัติงาน"}
    _DOC_ROLE  = {
        "policy":   "นักวิชาการด้านนโยบายสาธารณสุขไทย",
        "plan":     "นักวางแผนยุทธศาสตร์สาธารณสุขไทย",
        "workplan": "ผู้จัดการโครงการสาธารณสุขไทย",
    }
    _TAG_LABEL = {"policy": "Policy Brief", "plan": "แผนยุทธศาสตร์", "workplan": "แผนงาน/โครงการ"}
    doc_label  = _DOC_LABEL.get(doc_type, "รายงาน")
    doc_role   = _DOC_ROLE.get(doc_type, "ผู้เชี่ยวชาญ")
    tag_label  = _TAG_LABEL.get(doc_type, "รายงาน")

    # Parse "- Title" / "- Title: Note" bullets
    topics: list[dict] = []
    for raw in topic_plan.strip().splitlines():
        line = raw.strip().lstrip("-").strip()
        if not line:
            continue
        if ":" in line:
            title, note = line.split(":", 1)
            topics.append({"title": title.strip(), "note": note.strip()})
        else:
            topics.append({"title": line, "note": ""})

    if not topics:
        return ""  # caller falls back to standard prompt

    # Build per-section page instructions
    section_lines: list[str] = []
    for i, t in enumerate(topics):
        note_hint = f"\n   คำแนะนำจากผู้ใช้: {t['note']}" if t["note"] else ""
        section_lines.append(
            f"หน้า {i + 2} — {t['title']}:{note_hint}\n"
            f"  <h3 class=\"section-heading\">{t['title']}</h3>\n"
            "  เขียนเนื้อหาละเอียด 4-6 ย่อหน้า แต่ละย่อหน้า 4-6 ประโยค "
            "พร้อมข้อมูลเชิงตัวเลข/งานวิจัยอ้างอิง และ <sup class=\"cite\">[n]</sup>\n"
            "  เพิ่มตารางหรือกราฟ Chart.js ถ้ามีข้อมูลเชิงปริมาณ\n"
            f"  <span class=\"page-num\">{i + 2}</span>"
        )

    sections_text  = "\n\n".join(section_lines)
    total_pages    = len(topics) + 2  # cover + sections + references

    return f"""คุณคือ{doc_role} เขียน{doc_label}ฉบับสมบูรณ์ระดับมืออาชีพ อย่างน้อย {total_pages} หน้า A4

หัวข้อ: {query}
ประเภทเอกสาร: {doc_label}

หัวข้อที่ผู้ใช้กำหนด (ต้องปฏิบัติตามอย่างเคร่งครัด):
{topic_plan}

แนวทางเพิ่มเติมจาก AI Planner:
{plan}

บทความที่ค้นพบ ({article_count} บทความ):
{articles_text}

{_BASE_RULES}

โครงสร้างที่ MUST ใช้ — สร้าง {total_pages} <div class="page"> แยกกัน:

หน้า 1 — ปกและบทสรุปผู้บริหาร:
  <span class="tag-research">{tag_label}</span>
  <h2 class="article-title">ชื่อ{doc_label}ภาษาไทย — {query}</h2>
  <p class="authors">ผู้จัดทำ / หน่วยงาน</p>
  <hr class="divider">
  <div class="exec-box"><p class="exec-label">บทสรุปผู้บริหาร</p>สรุปประเด็นสำคัญที่ครอบคลุมทุกหัวข้อที่เลือก 5-6 ประโยค</div>
  <span class="page-num">1</span>

{sections_text}

หน้า {total_pages} — เอกสารอ้างอิง:
  <h3 class="section-heading">เอกสารอ้างอิง</h3>
  <ol class="ref-list">อ้างอิงทุกบทความพร้อม URL</ol>
  <span class="page-num">{total_pages}</span>

CSS ใส่ใน <style>:
{JOURNAL_CSS}

ใน <head>:
{_HEAD_SCRIPT}

ตอบเป็น HTML เท่านั้น เริ่มด้วย <!DOCTYPE html> ห้ามมี text หรือ ``` ก่อนหรือหลัง HTML:"""


# ── Factory ───────────────────────────────────────────────────────────────────

DOC_TYPES = {
    "policy":   ("การขอนโยบาย", build_policy_prompt),
    "plan":     ("การเขียนแผน", build_plan_prompt),
    "workplan": ("การวางแผนงาน", build_workplan_prompt),
}

DEFAULT_DOC_TYPE = "policy"


def build_prompt(
    doc_type: str,
    query: str,
    plan: str,
    articles_text: str,
    article_count: int,
    topic_plan: str = "",
) -> str:
    # When the user picked specific topics via wizard, use the custom prompt
    if topic_plan:
        custom = build_custom_prompt(query, plan, articles_text, article_count, topic_plan, doc_type)
        if custom:
            return custom
    # Otherwise fall back to the fixed document-type template
    _, builder = DOC_TYPES.get(doc_type, DOC_TYPES[DEFAULT_DOC_TYPE])
    return builder(query, plan, articles_text, article_count)


def doc_type_label(doc_type: str) -> str:
    label, _ = DOC_TYPES.get(doc_type, DOC_TYPES[DEFAULT_DOC_TYPE])
    return label
