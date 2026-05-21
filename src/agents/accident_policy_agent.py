"""Zone 10 accident policy agents — เขตสุขภาพที่ 10.

Three-agent sequential pipeline:
  1. Zone10SqlFetcher   (fast LLM) — runs all 7 zone10_accident tools
  2. Zone10PolicyAnalyst (pro LLM) — interprets via RTI / Haddon Matrix framework
  3. Zone10ReportWriter  (pro LLM) — writes สสส./สสจ./ศปถ. policy report
"""
from crewai import Agent
from src.agents.agent_defaults import agent_retry_kwargs

from src.tools.zone10_accident import (
    get_zone10_top_roads,
    get_zone10_time_bands,
    get_zone10_motorcycle_severity,
    get_zone10_car_serious_injuries,
    get_zone10_environment_risk,
    get_zone10_yearly_kpi,
    get_zone10_monthly_risk,
)

ZONE10_TOOLS = [
    get_zone10_top_roads,
    get_zone10_time_bands,
    get_zone10_motorcycle_severity,
    get_zone10_car_serious_injuries,
    get_zone10_environment_risk,
    get_zone10_yearly_kpi,
    get_zone10_monthly_risk,
]

# ── Prompts ───────────────────────────────────────────────────────────────────

SQL_FETCHER_PROMPT = """คุณคือ Zone 10 SQL Data Fetcher ผู้เชี่ยวชาญด้านการดึงข้อมูลอุบัติเหตุ
จากฐานข้อมูลสำหรับเขตสุขภาพที่ 10 (อุบลราชธานี, ศรีสะเกษ, ยโสธร, อำนาจเจริญ, มุกดาหาร)

ให้เรียกใช้เครื่องมือทั้ง 7 ตัวต่อไปนี้สำหรับจังหวัดที่ระบุ แล้วรวบรวมผลลัพธ์:

1. get_zone10_top_roads       → Q1: ถนนเสี่ยงสูงสุด Top 10
2. get_zone10_time_bands      → Q2: การกระจายตามช่วงเวลา EMS
3. get_zone10_motorcycle_severity → Q3: อุบัติเหตุจักรยานยนต์แยกตามความรุนแรง
4. get_zone10_car_serious_injuries → Q4: อุบัติเหตุรถยนต์/บาดเจ็บสาหัส
5. get_zone10_environment_risk → Q5: สภาพแสง/ถนน กับความรุนแรง
6. get_zone10_yearly_kpi      → Q6: แนวโน้ม YoY เสียชีวิต/สาหัส
7. get_zone10_monthly_risk    → Q7: รูปแบบความเสี่ยงรายเดือน/เทศกาล

**Output:** รวมผลลัพธ์ทุก Q ไม่ตัดทอน ไม่สรุปเอง
"""

POLICY_ANALYST_PROMPT = """คุณคือ Zone 10 RTI Policy Analyst ผู้เชี่ยวชาญด้านนโยบายอุบัติเหตุทางถนน

วิเคราะห์ข้อมูลที่ SQL Fetcher ดึงมา ด้วยกรอบ Haddon Matrix และ 4 หมวดนโยบาย

## กรอบ Haddon Matrix

|          | ก่อนเกิดเหตุ | ขณะเกิดเหตุ | หลังเกิดเหตุ |
|----------|-------------|------------|------------|
| คน       | Q3/Q4 พฤติกรรมเสี่ยง | Q3/Q4 ความรุนแรง | EMS |
| รถ/ถนน   | Q1 จุดเสี่ยง | Q5 สภาพแวดล้อม | Q2 เวลา EMS |
| ระบบ     | Q6 แนวโน้ม KPI | Q7 เทศกาล | Q6 เป้าหมาย |

## 4 หมวดนโยบาย (Output JSON):

```json
{
  "hotspot": {
    "top_roads": [...],
    "ems_time_bands": [...],
    "key_findings": "...",
    "recommendations": [...]
  },
  "human_behavior": {
    "motorcycle_findings": "...",
    "car_findings": "...",
    "data_limitations": "...",
    "recommendations": [...]
  },
  "environment": {
    "risk_conditions": [...],
    "key_findings": "...",
    "recommendations": [...]
  },
  "kpi": {
    "yearly_trend": [...],
    "festival_risk": [...],
    "kpi_status": "ผ่าน/ไม่ผ่าน",
    "recommendations": [...]
  },
  "haddon_matrix": {
    "pre_event": "...",
    "event": "...",
    "post_event": "..."
  }
}
```
"""

REPORT_WRITER_PROMPT = """คุณคือ Zone 10 Policy Report Writer ผู้เขียนรายงานนโยบาย
สำหรับ สสส./สสจ./ศปถ. ในรูปแบบรายงานตรวจราชการกระทรวงสาธารณสุข

## โครงสร้างรายงาน (Markdown)

### ส่วนที่ 1 — สถานการณ์อุบัติเหตุทางถนน เขตสุขภาพที่ 10
### ส่วนที่ 2 — จุดเสี่ยง (Black Spot) และช่วงเวลาอันตราย
### ส่วนที่ 3 — ปัจจัยพฤติกรรมและสภาพแวดล้อม
### ส่วนที่ 4 — ผลการดำเนินงานตามตัวชี้วัด (KPI)
### ส่วนที่ 5 — ข้อเสนอแนะเชิงนโยบาย (3 ระยะ)
### ข้อจำกัดของข้อมูล

**กฎ:**
- ใช้ภาษาทางการ
- ตัวเลขทุกตัวต้องมาจากข้อมูลที่ Analyst ให้มา
- ปี ค.ศ. → พ.ศ. ทุกครั้ง (CE + 543)
"""


# ── Factory functions ─────────────────────────────────────────────────────────

def create_zone10_sql_fetcher(llm) -> Agent:
    return Agent(
        role="Zone 10 SQL Data Fetcher",
        goal="ดึงข้อมูลอุบัติเหตุทางถนนในเขตสุขภาพที่ 10 โดยเรียกเครื่องมือทั้ง 7 ตัว",
        backstory=(
            "ผู้เชี่ยวชาญด้านฐานข้อมูลอุบัติเหตุ ดึงข้อมูลดิบครบถ้วน "
            "รายงานตามที่ข้อมูลในฐานข้อมูลระบุ ไม่ตีความหรือสรุปเอง"
        ),
        tools=ZONE10_TOOLS,
        llm=llm,
        verbose=True,
        max_iter=10,
        **agent_retry_kwargs(),
    )


def create_zone10_policy_analyst(llm) -> Agent:
    return Agent(
        role="Zone 10 RTI Policy Analyst",
        goal=(
            "วิเคราะห์ข้อมูลอุบัติเหตุเขตสุขภาพที่ 10 ด้วยกรอบ Haddon Matrix "
            "จัดกลุ่มเป็น 4 หมวดนโยบาย และผลิต JSON structured analysis"
        ),
        backstory=(
            "ผู้เชี่ยวชาญด้านนโยบายความปลอดภัยทางถนนระดับเขตสุขภาพ "
            "มีประสบการณ์วิเคราะห์ข้อมูลอุบัติเหตุสำหรับรายงานตรวจราชการ "
            "แยก 'ข้อเท็จจริงจากข้อมูล' กับ 'การตีความ' อย่างชัดเจน"
        ),
        llm=llm,
        verbose=True,
        max_iter=8,
        **agent_retry_kwargs(),
    )


def create_zone10_report_writer(llm) -> Agent:
    return Agent(
        role="Zone 10 RTI Policy Report Writer",
        goal=(
            "เขียนรายงานนโยบายอุบัติเหตุทางถนน เขตสุขภาพที่ 10 "
            "ในรูปแบบรายงานตรวจราชการ ครบถ้วนตาม 5 ส่วนหลัก"
        ),
        backstory=(
            "ผู้เชี่ยวชาญด้านการเขียนรายงานราชการสาธารณสุข "
            "เขียนภาษาทางการที่ชัดเจน ใช้เฉพาะข้อมูลที่ Analyst ให้มา"
        ),
        llm=llm,
        verbose=True,
        max_iter=6,
        **agent_retry_kwargs(),
    )
