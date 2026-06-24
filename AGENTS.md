# 🤖 แคตตาล็อก AI Agents — `chatapi.python/src/agents/`

> โฟลเดอร์นี้คือหัวใจของระบบ AI ทั้งหมด มีไฟล์ ~22 ไฟล์ (รวม 1 ไฟล์ `.js` ที่ดูแปลกตา)
> ทุก agent สร้างผ่าน **CrewAI** (Agent/Crew/Task) และ/หรือเรียก LLM โดยตรงผ่าน **litellm** (สำหรับ streaming)
> โมเดลหลักคือ **Gemini** (`gemini-2.5-flash-lite` สำหรับงานเร็ว, `gemini-2.5-pro` สำหรับงานที่ต้องคิดเยอะ)
>
> ดูว่า router ไหนเรียก agent ตัวไหนได้ที่ [`STRUCTURE.md` หัวข้อ 5](./STRUCTURE.md#5-routers--endpoint-ทั้งหมด-srcrouters)

---

## ภาพรวม — จัดกลุ่ม agent ทั้งหมด

| กลุ่ม | ไฟล์ | บทบาท |
|---|---|---|
| **A. วิเคราะห์อุบัติเหตุ (SQL-based)** | `accident_chat_orchestrator`, `accident_policy_agent`, `accident_policy_orchestrator`, `analyst_accident` | ตอบคำถาม/สร้างรายงานนโยบายอุบัติเหตุเขต 10 จากฐานข้อมูล Postgres โดยตรง |
| **B. วิเคราะห์ CSV (multi-step pipelines)** | `csv_pipeline`, `multi_csv_pipeline`, `compare_agent`, `report_agent`, `database_agent`, `workplan_agent` | อ่าน/วิเคราะห์ไฟล์ CSV ใน MinIO ด้วย pandas ผ่าน agent หลายตัวต่อกันเป็นสาย (pipeline) |
| **C. ความรู้ & งานวิจัย** | `thaijo_agent`, `thaijo_prompts`, `obsidian_fullcontext`, `obsidian_agent`, `obsidian_progress` | สังเคราะห์งานวิจัยจาก ThaiJO และค้นคำตอบจากคลังความรู้ Obsidian |
| **D. การจัดเส้นทาง & ส่วนกลาง** | `router`, `question_resolver`, `progress`, `tavily_pipeline` | ตัดสินใจว่าจะส่งคำถามไป pipeline ไหน, แก้คำถามที่พูดสั้น/ไม่ครบ, ติดตามความคืบหน้าแบบ real-time, ค้นเว็บ |
| **E. โครงสร้างพื้นฐานร่วม** | `agent_defaults`, `prompt_profile`, `error_monitor_agent` | ตั้งค่า CrewAI กลาง + รับมือ rate limit, แม่แบบ prompt ตามโดเมน, สรุป error log ด้วย LLM |
| **ไฟล์แปลก** | `promptplan.js` | constant `PROMPT_PLAN` เขียนเป็น JS แต่อยู่ในโฟลเดอร์ Python ล้วน (ดูข้อสังเกตท้ายเอกสาร) |

---

## กลุ่ม A — วิเคราะห์อุบัติเหตุ (SQL-based, สำหรับโดเมน `d1`)

### `accident_chat_orchestrator.py`
แชทถาม-ตอบเชิงนโยบายอุบัติเหตุแบบสนทนา — จุดเริ่มที่ router `accident_chat.py` เรียกใช้
- **ฟังก์ชันหลัก**: `run_accident_chat(question, province, district, year_start, year_end)` → `AccidentChatResponse`, และ `run_accident_chat_with_progress(...)` (เวอร์ชันสตรีม SSE)
- **Agent ภายใน 2 ตัว**: `AccidentSQLAgent` (Gemini flash — เรียก SQL tools ดึงข้อมูลดิบ) → `AccidentAnswerAgent` (Gemini pro — ตีความข้อมูล เขียนคำตอบเชิงนโยบายเป็นภาษาไทย)
- **ใช้ tools**: `accident_chat_sql.py` (15+ เครื่องมือเฉพาะทาง)
- **ข้อจำกัดข้อมูล**: ตาราง `fact_accident_person` ว่างเปล่า, `road_name` ส่วนใหญ่เป็น NULL

### `accident_policy_agent.py`
นิยาม agent (ไม่ใช่ orchestrator) สำหรับ pipeline สร้าง Policy Brief เขต 10
- `create_zone10_sql_fetcher()` — ดึงข้อมูลตอบ 7 คำถามนโยบาย (Q1-Q7) ผ่าน 7 tools ใน `zone10_accident.py`
- `create_zone10_policy_analyst()` — วิเคราะห์ข้อมูล แปลงเป็น JSON sections
- `create_zone10_report_writer()` — เขียน Policy Brief ฉบับเต็ม (HTML/Markdown)

### `accident_policy_orchestrator.py`
Entry point ของ pipeline สร้าง Policy Brief — เรียก agent จาก `accident_policy_agent.py` มาต่อกันเป็น crew
- **ฟังก์ชันหลัก**: `run_zone10_analysis(provinces, questions, year_range)` → `dict` (มี `policy_brief`, `sections`, `charts`, `metadata`)
- **ลำดับการทำงาน (sequential crew)**: SQL Fetcher → Policy Analyst → Report Writer
- **เวลาที่ใช้**: ประมาณ 2-5 นาที (ช้ากว่า accident_chat เพราะลึกกว่า)

### `analyst_accident.py`
agent เสริมสำหรับวิเคราะห์ข้อมูลอุบัติเหตุเชิงลึก (ไฟล์ขนาดเล็ก ~3.5 KB) — ดูเหมือนเป็นตัวช่วยที่ orchestrator ตัวอื่นเรียกใช้สำหรับ metric เฉพาะทาง

> 🔎 **สังเกต**: `accident_chat_orchestrator` กับ `accident_policy_orchestrator` มีโครงสร้างคล้ายกันมาก (ดึง SQL → วิเคราะห์ → เขียนคำตอบ) ต่างกันที่ความลึก/ความเร็ว — อาจสกัด logic ร่วม (เช่น SQL fetcher pattern, retry, progress emit) ออกมาเป็น base class/helper ที่ใช้ร่วมกันได้

---

## กลุ่ม B — วิเคราะห์ CSV (สำหรับโดเมน `d0`, `d2`-`d4`)

ทุก pipeline ในกลุ่มนี้ใช้แนวคิดเดียวกัน: **หาไฟล์ CSV ใน MinIO → อ่าน schema → ให้ LLM เขียนโค้ด pandas → รันโค้ดจริงในแซนด์บ็อกซ์ → สรุปผลเป็นภาษาไทย** (อธิบายดีไซน์เต็ม ๆ ใน [`AI_CSV_Data_Analyst_System.md`](./AI_CSV_Data_Analyst_System.md))

### `csv_pipeline.py` — pipeline พื้นฐาน (วิเคราะห์โดเมนเดียว) — ไฟล์ใหญ่สุดอันดับ 2 (45 KB)
- **ฟังก์ชันหลัก**: `run_pipeline(prompt, queue, loop, domain, ...)` → สตรีม SSE event ทีละขั้น
- **6 agents เรียงต่อกัน**: File Finder → Schema Analyst → Prompt Profiler → Python Code Generator → Python Executor → Insight Analyst
- **ใช้ tools**: `minio.py` (list/read/execute), `error_logger.py`
- รองรับโดเมน `d0`, `d2`, `d3`, `d4`
- มีการ retry เมื่อได้ response ว่าง และบันทึก error ลง `error_logs/`

### `multi_csv_pipeline.py` — pipeline ข้ามโดเมน — **ไฟล์ใหญ่สุด (48 KB) และแก้ล่าสุด (7 มิ.ย.)**
- **ฟังก์ชันหลัก**: `run_multi_pipeline(prompt, queue, loop, domains, ...)` → SSE events

- **🔄 จุดที่เพิ่งปรับปรุงใหม่ (อัปเดตจากผู้ใช้): วิธีหาไฟล์ข้อมูลเปลี่ยนจาก "หา CSV โดยตรง" → "หาโฟลเดอร์ก่อน แล้วค่อยดึง CSV จากโฟลเดอร์นั้น"**

  > 🎯 **หลักการออกแบบสำคัญที่สุดของจุดนี้ — "ให้ AI เป็นคนเลือกโฟลเดอร์"**: การ**ตัดสินใจ**ว่าควรใช้ข้อมูลจากหมวด/ตัวชี้วัดไหน เป็นหน้าที่ของ **AI agent ล้วน ๆ** (ใช้ความเข้าใจความหมาย/บริบทของคำถามเทียบกับชื่อโฟลเดอร์ — งานที่ AI ถนัดกว่า) ส่วน **"โค้ด" ทำหน้าที่แค่ช่วงท้ายที่ deterministic ล้วน ๆ** คือแปลง "ชื่อโฟลเดอร์ที่ AI เลือกแล้ว" ให้กลายเป็น file ID จริงผ่าน path index (งานที่โค้ดถนัดกว่า แม่นยำ ไม่มีสิทธิ์เพี้ยน) → คือการ **แบ่งงานตามจุดแข็งของแต่ละฝ่าย**: AI ตัดสินใจ "ความหมาย", โค้ดจัดการ "การค้นหา/แปลงค่าที่ต้องแม่นยำ 100%"

  อ่านจากคอมเมนต์ในโค้ด (บรรทัด 398-405) สรุปเหตุผล + กลไกใหม่ได้ดังนี้:
  - **แบบเดิม**: agent เห็นรายชื่อไฟล์แบบแบน ๆ (`[ID:xxx] ชื่อไฟล์.csv` ที่มักถูกตัดสั้น) แล้วเดาจาก keyword ในชื่อไฟล์ — มองไม่เห็น "หมวด/ตัวชี้วัด" ที่แท้จริง ทำให้เลือกผิดโดเมนได้ง่าย (เช่น เลือกไฟล์สุขภาพจิตไปตอบคำถามเรื่องเบาหวาน/ความดัน) — ปัญหาคือ **ให้ AI ทำงานที่โค้ดควรทำ (จำ/พิมพ์ ID) และไม่ได้ให้ข้อมูลที่ AI ต้องการ (ชื่อตัวชี้วัดเต็ม ๆ) สำหรับตัดสินใจ**
  - **แบบใหม่ (ปัจจุบัน)** — agent ชื่อ **"Multi-Domain Folder Navigator Agent"** ทำงาน 3 ขั้น:
    1. ระบบสร้าง **"folder tree"** จาก path metadata จริงของไฟล์ (ฟังก์ชัน `list_csv_tree_impl`) แสดงให้ agent ดูก่อน (ไม่ใช่ list ไฟล์แบบแบน) — **นี่คือสิ่งที่ทำให้ AI "เลือกโฟลเดอร์" ได้แม่นขึ้น เพราะเห็นชื่อตัวชี้วัดเต็ม ๆ ไม่ถูกตัดทอน**
    2. agent แตกคำถามเป็นหัวข้อย่อย แล้ว**เลือก "ชื่อโฟลเดอร์ตัวชี้วัด"** ที่ตรงความหมายที่สุดด้วยตัวเอง — ใช้ชื่อโฟลเดอร์ระดับลึกสุด (เพราะเป็น "ชื่อตัวชี้วัดเต็ม ๆ" ไม่ถูกตัดทอนแบบชื่อไฟล์ เช่น "ร้อยละของผู้ป่วยเบาหวานชนิดที่ 2 ที่เข้าสู่โรคเบาหวานระยะสงบ (DM remission)") เลือกได้ไม่เกิน `MAX_FILES` (5 โฟลเดอร์) — **ห้ามตอบเป็น `[ID:...]` หรือชื่อไฟล์ ตอบเฉพาะ "ชื่อโฟลเดอร์" เท่านั้น (ตัดสินใจแค่ "อะไร" ปล่อยให้โค้ดจัดการ "อย่างไร")**
    3. **โค้ด** (ไม่ใช่ AI) resolve "ชื่อโฟลเดอร์ที่ agent เลือก" → file ID จริงแบบ deterministic เองผ่าน `_resolve_folders_to_files()` + `path_index` (agent ไม่ต้องเดา/พิมพ์ ID เอง → ตัด error จากการจำ/พิมพ์ ID ผิดไปด้วยในตัว เพราะขั้นตอนนี้ไม่ใช่จุดแข็งของ LLM)
    4. มี **fallback ป้องกันพัง**: ถ้า resolve ได้ไฟล์น้อยกว่า 2 ไฟล์ (เช่น folder navigation ล้มเหลว หรือไม่มี path metadata) จะถอยกลับไปใช้วิธีเดิม คือ keyword scoring แบบแบน (`_keyword_select` + `resolve_file_id`)

- **ส่วนเสริมอื่นที่เพิ่มจาก `csv_pipeline` ธรรมดา**:
  - **Geographic Key Detector** (`_detect_geo_keys`) — หาคอลัมน์ที่ใช้ join ข้ามไฟล์ได้อัตโนมัติ (เช่น จังหวัด, อำเภอ)
  - **Domain Coverage Validator** (`_enforce_domain_coverage`) — เช็คว่าทุกโดเมนที่เลือกมีไฟล์ข้อมูลอย่างน้อย 1 ไฟล์
  - รายงานความคืบหน้าเป็นรายไฟล์ (ไม่ใช่เป็นชุด)
- **agent chain**: Multi-Domain Folder Navigator (เลือก "ชื่อโฟลเดอร์" → โค้ด resolve เป็นไฟล์ให้เอง) → Domain Coverage Validator → Schema Analyzer (ต่อไฟล์) → Geo Key Detector → Code Generator → Executor → Cross-Domain Insight
- **ผลลัพธ์**: วิเคราะห์ข้ามมุมมองหลายโดเมน หารูปแบบ/ความสัมพันธ์ร่วม

### `compare_agent.py` — เปรียบเทียบ 2 ชุดข้อมูล (23 KB)
- **ฟังก์ชันหลัก**: `run_compare_pipeline(prompt, queue, loop, session_id)` → SSE events
- **6 ขั้นตอน**: File Finder A, File Finder B → Schema Analyst (อ่านทั้งคู่) → Code Generator (สร้างโค้ด join/diff/สถิติ) → Python Executor → Insight Analyst
- **ผลลัพธ์**: รายงาน Markdown เปรียบเทียบ (ผล join, สหสัมพันธ์, การทดสอบทางสถิติ)

### `report_agent.py` — สร้างรายงานสรุปครบวงจร (18.6 KB)
- **ฟังก์ชันหลัก**: `run_report_pipeline(prompt, queue, loop, session_id)` → SSE events
- **5 ขั้นตอน**: File Finder → Schema Analyst → Data Analyst → Python Executor → Report Writer
- **ผลลัพธ์**: รายงาน Markdown ภาษาไทยแบบเต็ม (บทสรุป ข้อเสนอแนะ กราฟ)

### `database_agent.py` — วิเคราะห์ไฟล์ที่ผู้ใช้แนบมาเอง (24 KB)
- **ฟังก์ชันหลัก**: `run_database_pipeline(prompt, queue, loop, session_id, attached_files)` → SSE events
- **4 ขั้นตอน**: Schema Analyst → Code Generator → Python Executor → Insight Analyst (ข้าม File Finder เพราะผู้ใช้แนบไฟล์มาเอง)
- **อินพุต**: file ID จาก MinIO (หรือ temp file local)

### `workplan_agent.py` — สร้างแผนงานล้วน ๆ จาก LLM (16 KB, ไม่ใช้ CSV)
- **ฟังก์ชันหลัก**: `run_workplan_pipeline(prompt, queue, loop, session_id, doc_type)` → สตรีม HTML
- **2 ขั้นตอน**: Plan Analyzer (CrewAI) → Plan Writer (litellm สตรีม)
- **ผลลัพธ์**: เอกสาร HTML รูปแบบราชการ ~15 หน้า (ปก วัตถุประสงค์ กิจกรรม ตัวชี้วัด งบประมาณ บทเรียนที่ได้)
- ใช้ `promptplan.js` (constant `PROMPT_PLAN`) เป็นแนวทาง prompt

> 🔎 **สังเกต**: `compare_agent` กับ `report_agent` มีขั้นตอน "File Finder + Schema Analyst" ซ้ำกัน — สามารถแยกออกมาเป็น shared component/helper เพื่อลดโค้ดซ้ำได้

---

## กลุ่ม C — ความรู้ & งานวิจัย (สำหรับโดเมน `dt`, `obsidian`)

### `thaijo_agent.py` — สังเคราะห์บทความวิจัยเป็นรายงาน — ไฟล์ใหญ่ที่สุดในกลุ่ม agent (64 KB)
- **ฟังก์ชันหลัก**:
  - `run_thaijo_pipeline(prompt, queue, loop, session_id, use_mock)` — ค้นบทความสด ๆ จาก ThaiJO API แล้วสร้างรายงาน
  - `run_thaijo_report_pipeline(query, articles_text, ...)` — สร้างรายงานจากบทความที่ดึงมาแล้ว (pre-fetched)
  - `run_topic_planner(query, articles_text, doc_type)` — แนะนำหัวข้อ/โครงรายงาน
- **ขั้นตอน**: ThaiJo Fetcher (httpx เรียก TCI-ThaiJO API, default 5 ผลลัพธ์) → Report Planner (Gemini วางโครง) → Report Generator (litellm สตรีม HTML)
- **ผลลัพธ์**: journal HTML แบบ A4 (ฟอนต์ Sarabun, มี page break, ข้อมูลผู้เขียน คำสำคัญ บทคัดย่อ วิธีการ ผลลัพธ์ เอกสารอ้างอิง)
- **ใช้ tools**: `thaijo_cache.py` (cache สรุป PDF บน Redis)

### `thaijo_prompts.py` — แม่แบบ prompt ของ ThaiJO (28 KB)
- เก็บ `_JOURNAL_CSS` (สไตล์ A4/ฟอนต์ Sarabun), แม่แบบบทความ, กฎการจัดรูปแบบ
- ใช้ร่วมกันโดย `thaijo_agent.py` และ `workplan_agent.py`

### `obsidian_fullcontext.py` — pipeline ถาม-ตอบคลังความรู้แบบเต็มบริบท (9.4 KB, แก้ล่าสุด 6 มิ.ย.)
- **ฟังก์ชันหลัก**: `run_obsidian_ask_fullcontext(question, province, vault_id, request_id)` → `ObsidianAskResponse`
- **แนวทาง**: โหลดไฟล์ `.md` ทั้งหมดจาก vault (หรือกรองตามจังหวัด) ใส่ใน context เดียวแล้วส่งให้ Gemini ตอบ — **ไม่ผ่านการทำดัชนี DB** ใช้ filesystem ตรง ๆ (~100-200 KB ต่อจังหวัด, ~1.1 MB ทั้ง vault)
- **ผลลัพธ์**: `ObsidianAskResponse(content, notes_referenced[], follow_ups[])`
- **เรียกใช้โดย**: router `obsidian.py` (`/ask`) และ `analyze.py` (mode `obsidian`)

### `obsidian_agent.py` — pipeline Obsidian อีกแบบ (17.6 KB)
- มีอยู่คู่กับ `obsidian_fullcontext.py` — จากชื่อและขนาดไฟล์ คาดว่าเป็น**แนวทางที่เก่ากว่า**หรือแบบที่อิงดัชนี DB (ตรงข้ามกับ full-context ที่อ่าน filesystem ตรง)

### `obsidian_progress.py` — ติดตามความคืบหน้าของงาน Obsidian (2.2 KB)
- ตัวช่วยคู่กับ `obsidian_fullcontext.py` สำหรับรายงานสถานะระหว่างค้นหา/ตอบคำถาม

> 🔎 **สังเกต**: `obsidian_agent.py` กับ `obsidian_fullcontext.py` ทำงานคล้ายกัน (ตอบคำถามจากคลังความรู้) — ควรตรวจสอบว่าตัวไหนเป็น production path จริง (จากที่อ่านโค้ด router ดูเหมือน `obsidian_fullcontext` คือตัวที่ใช้งานอยู่ตอนนี้) แล้วพิจารณาลบ/รวมตัวที่ไม่ได้ใช้แล้ว

---

## กลุ่ม D — การจัดเส้นทาง & ส่วนกลาง (cross-cutting)

### `router.py` — ตัดสินใจว่าคำถามควรไป pipeline ไหน (18 KB, แก้ล่าสุด 7 มิ.ย.)
- **ฟังก์ชันหลัก**:
  - `route_domain(prompt, history_context)` → `(domain, is_multi)`
  - `route_multi_domain(prompt, history_context)` → `(domains[], is_multi)`
  - `route_with_web_search(prompt, history_context)` → `(decision, domain)`
  - `_has_accident_signal(prompt)` → `bool` (ดักคำถามอุบัติเหตุให้ไป SQL fast-path แทน CSV)
- **ตรวจจับคำสำคัญ**: ThaiJO ("thaijo", "บทความวิจัย"...), Obsidian ("obsidian", "คลังความรู้", "เขตสุขภาพที่ 10", ชื่อจังหวัด), อุบัติเหตุ ("อุบัติเหตุ", "ถนน", "RTI")
- เป็นตัวตัดสินว่าจะใช้ pipeline ไหนใน `analyze.py`

### `question_resolver.py` — Memory Agent แก้คำถามที่พูดสั้น/ไม่ครบ (3.5 KB)
- **ฟังก์ชันหลัก**: `resolve_question(prompt, history_context, gemini_key)` → `(resolved_prompt, was_changed)`
- **ตัวอย่าง**: ผู้ใช้ถาม "ของจังหวัดนั้นล่ะ" → ขยายเป็น "อุบัติเหตุของจังหวัดอุบลราชธานี..." (อิงจากบทสนทนาก่อนหน้า)
- เรียก Gemini flash-lite ผ่าน litellm พร้อม history context, คืนค่า `"UNCHANGED"` ถ้าคำถามชัดเจนอยู่แล้ว

### `progress.py` — โครงสร้างติดตามความคืบหน้าของ agent แบบ real-time (2.6 KB)
- `create_progress_queue(request_id)` → Queue, `emit_progress(request_id, agent_name, agent_icon, status, message, elapsed_seconds)`, `remove_progress_queue(request_id)`
- Data class `AgentProgress` (request_id, agent_name, agent_icon, status, message, elapsed_seconds, order)
- ทุก pipeline แบบสตรีมเรียกใช้ตัวนี้เพื่อส่ง event ผ่าน SSE ไปแสดงที่ frontend (เช่น `AgentPipelinePanel.tsx`)

### `tavily_pipeline.py` — ค้นหาเว็บแล้วสรุปคำตอบ (8.8 KB)
- **ฟังก์ชันหลัก**: `run_tavily_pipeline(prompt, queue, loop, session_id)` → SSE events
- **agent**: `TavilySearchAgent` (เรียก tool `tavily_search`) → `TavilyAnswerWriter` (สังเคราะห์คำตอบภาษาไทยจากผลค้นเว็บ)

---

## กลุ่ม E — โครงสร้างพื้นฐานร่วม (ไม่ใช่ agent โดยตรง แต่ทุก agent ต้องพึ่งพา)

### `agent_defaults.py` — ตั้งค่า CrewAI กลาง + รับมือ rate limit (3.2 KB)
- `_patch_gemini_429_backoff()` — **monkey-patch** `Agent` ของ CrewAI ให้หน่วงเวลาเมื่อเจอ 429 RESOURCE_EXHAUSTED
- `agent_retry_kwargs()` → `dict` (ค่า `max_retry_limit` จาก settings)
- `kickoff_with_retry(crew, max_attempts)` → ผลลัพธ์ (รัน crew พร้อม retry อัตโนมัติเมื่อโดน 429)
- **สำคัญ**: ไฟล์นี้คือกลไกที่ทำให้ทั้งระบบ "ทนทาน" ต่อ rate limit ของ Gemini — แทบทุก orchestrator import จากที่นี่

### `prompt_profile.py` — แม่แบบ prompt ตามโดเมน (8.8 KB)
- ค่าคงที่: `ANALYST_CORE_POLICY`, `CODE_GENERATOR_CORE_POLICY`, `INSIGHT_RESPONSE_BLUEPRINT`, `MISSING_DATA_POLICY`, ฟังก์ชัน `join_prompt()`
- ใช้โดย `csv_pipeline.py` เพื่อฉีดความเชี่ยวชาญเฉพาะโดเมนเข้าไปใน prompt ของแต่ละ agent

### `error_monitor_agent.py` — agent สรุป error log (4.1 KB)
- **ฟังก์ชันหลัก**: `run_error_monitor(days)` → `dict(aggregate, report, entries)`
- ใช้ LLM สรุปรูปแบบของ error ที่เกิดขึ้นในช่วงเวลาที่กำหนด ให้คนอ่านเข้าใจง่าย (ใช้ใน endpoint `/api/errors/summary`)

---

## ไฟล์ที่ดูแปลกตา — `promptplan.js`

- **เป็น JavaScript ไฟล์เดียว** ในโฟลเดอร์ที่เหลือเป็น Python ทั้งหมด (60 KB — ใหญ่เป็นอันดับ 2 ของทั้งโฟลเดอร์!)
- เก็บ constant ชื่อ `PROMPT_PLAN` ซึ่งเป็นแนวทาง/กฎการสร้างแผนงาน 15 หน้า ที่ดูเหมือนตั้งใจให้ `workplan_agent.py` ใช้
- **ข้อเสนอแนะ**: ย้ายเนื้อหาไปไว้ใน `.py` (เช่น `src/agents/prompt_profile.py` หรือไฟล์ใหม่ `src/agents/workplan_prompts.py`) แล้วลบไฟล์ `.js` นี้ทิ้ง — เพื่อไม่ให้สับสนว่าทำไมมีไฟล์ JS อยู่ใน backend Python

---

## สรุปขนาดไฟล์ (เรียงจากใหญ่ไปเล็ก — ดูว่าไฟล์ไหน "หนัก" ที่สุด)

| ไฟล์ | ขนาด | กลุ่ม |
|---|---|---|
| `promptplan.js` | 60 KB | (แปลก — ดูด้านบน) |
| `thaijo_agent.py` | 64 KB | C |
| `multi_csv_pipeline.py` | 48 KB | B |
| `csv_pipeline.py` | 45 KB | B |
| `thaijo_prompts.py` | 28 KB | C |
| `database_agent.py` | 24 KB | B |
| `compare_agent.py` | 24 KB | B |
| `accident_chat_orchestrator.py` | 15 KB | A |
| `report_agent.py` | 18.6 KB | B |
| `obsidian_agent.py` | 17.6 KB | C |
| `workplan_agent.py` | 16 KB | B |
| `accident_policy_orchestrator.py` | 9.4 KB | A |
| `obsidian_fullcontext.py` | 9.4 KB | C |
| `accident_policy_agent.py` | 8.8 KB | A |
| `tavily_pipeline.py` | 8.8 KB | D |
| `prompt_profile.py` | 8.8 KB | E |
| `analyst_accident.py` | 3.5 KB | A |
| `question_resolver.py` | 3.5 KB | D |
| `agent_defaults.py` | 3.2 KB | E |
| `progress.py` | 2.6 KB | D |
| `obsidian_progress.py` | 2.2 KB | C |
| `error_monitor_agent.py` | 4.1 KB | E |

> 💡 ไฟล์ที่ "หนัก" ที่สุด (`thaijo_agent.py`, `multi_csv_pipeline.py`, `csv_pipeline.py`, `thaijo_prompts.py`) ล้วนผสมทั้ง logic การรัน pipeline และ prompt template/HTML/CSS ขนาดใหญ่ไว้ในไฟล์เดียว — ถ้าจะ refactor ในอนาคต จุดแรกที่น่าแยกคือ "ย้าย prompt/template/CSS ออกจากไฟล์ logic" (เริ่มจาก `thaijo_prompts.py` ที่แยกไว้แล้วเป็นตัวอย่างที่ดี)

---

## ไฟล์ที่แก้ไขล่าสุด (เรียงจากใหม่ไปเก่า — เผื่ออยากรู้ว่ากำลังโฟกัสส่วนไหนอยู่)

1. `multi_csv_pipeline.py` — 7 มิ.ย. 17:48 (ใหม่สุด — กำลังพัฒนา multi-domain pipeline อยู่)
2. `router.py` — 7 มิ.ย. 17:42
3. `obsidian_fullcontext.py` — 6 มิ.ย. 17:34
4. `database_agent.py` — 2 มิ.ย. 17:52
5. `thaijo_agent.py` — 2 มิ.ย. 15:54
6. `csv_pipeline.py`, `error_monitor_agent.py`, `question_resolver.py`, `tavily_pipeline.py` — 2 มิ.ย. 15:15
7. ไฟล์ที่เหลือ — แก้ล่าสุดช่วง 24-25 พ.ค. (ดูเหมือนเป็นชุดที่เขียนตอนตั้งโครงสร้างโปรเจกต์ครั้งแรก)

> 📌 จากรอยแก้ไข ดูเหมือนงานล่าสุดที่ทำค้างอยู่คือ **`multi_csv_pipeline.py`** (วิเคราะห์ข้ามโดเมน) และ **`router.py`** — น่าจะเป็นจุดที่น่ากลับไปดูต่อก่อนเป็นอันดับแรก

---

*สร้างจากการอ่านซอร์สโค้ดจริง 2026-06-08 — หากแก้โค้ดภายหลัง ให้ตรวจสอบความถูกต้องกับโค้ดจริงเสมอ โดยเฉพาะเรื่อง "ตัวไหน primary/ตัวไหนซ้ำซ้อน" ซึ่งเป็นข้อสังเกตจากภายนอก ไม่ใช่ข้อเท็จจริงที่ยืนยันแล้ว*
