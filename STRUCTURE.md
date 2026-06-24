# 🔧 โครงสร้างโปรเจกต์ — `chatapi.python` (Health Analytics API)

> Backend แบบ FastAPI ที่เป็น "สมอง" ของระบบ AI ทั้งหมด — รวม agent/orchestrator/tools, ต่อ LLM (Gemini/OpenAI ผ่าน CrewAI/litellm),
> รัน pipeline วิเคราะห์ข้อมูลหลายแบบ แล้วสตรีมคำตอบกลับเป็น Server-Sent Events (SSE)
>
> ดูภาพรวมทั้งระบบที่ [`../STRUCTURE.md`](../STRUCTURE.md) และแคตตาล็อก agent แบบละเอียดที่ [`AGENTS.md`](./AGENTS.md)

---

## 1. ภาพรวม

- **ชื่อแอป**: "Health Analytics API" v2.0 — `CSV Data Analyst + Zone 10 Accident Policy Agent`
- **เฟรมเวิร์ก**: FastAPI + Uvicorn, entry point คือ [`main.py`](./main.py)
- **AI orchestration**: CrewAI (multi-agent crews), litellm (เรียก LLM แบบ unified streaming), Gemini เป็นโมเดลหลัก (`gemini-2.5-flash-lite` / `gemini-2.5-pro`), OpenAI เป็นออปชัน
- **โดเมนหลัก**: "เขตสุขภาพที่ 10" — อุบลราชธานี ศรีสะเกษ ยโสธร อำนาจเจริญ มุกดาหาร (ดู [`src/domains.py`](./src/domains.py))
- **Data stores**: PostgreSQL + pgvector (DB `musyadata`), MinIO (เก็บไฟล์/CSV), Redis (ประวัติแชท + cache)
- **เปิดใช้งาน**: `http://localhost:8000` — Swagger ที่ `/docs`, หน้ารวม UI ทดสอบที่ `/ui`

## 2. ผังโฟลเดอร์

```
chatapi.python/
├── main.py                      ← entry point: สร้าง FastAPI app, register routers, mount /static, หน้า /ui
├── requirements.txt / Dockerfile / docker-compose.yml
├── .env / .env.example          ← ตัวแปรแวดล้อม (API keys, DB, MinIO, Redis, Obsidian)
├── AI_CSV_Data_Analyst_System.md← เอกสารออกแบบ pipeline วิเคราะห์ CSV (6 ขั้นตอน) — เอกสารอ้างอิง ไม่ใช่โค้ด
├── check_obsidian.py            ← สคริปต์ตรวจสุขภาพระบบ Obsidian (เช็ค DB tables + filesystem)
├── check_paths.py               ← สคริปต์ตรวจ/validate path ของไฟล์ (diagnostic)
├── db_check_out.txt             ← ผลลัพธ์ที่ได้จากการรันสคริปต์ตรวจสอบ (log/ผลลัพธ์ที่ dump ไว้)
├── data/, database/             ← ข้อมูล + SQL schema/migration ของ Postgres
├── musya_20260517_2118.dump     ← Postgres dump สำรอง (pg_dump)
└── src/
    ├── config.py                ← Settings (Pydantic) โหลดจาก .env, LRU cache
    ├── domains.py               ← นิยามโดเมนสุขภาพ d0–d4, dt, obsidian
    ├── history.py               ← ประวัติแชทบน Redis (multi-turn, follow-up)
    ├── db/pool.py               ← PostgreSQL connection pool (psycopg2 ThreadedConnectionPool)
    ├── agents/                  ← 🤖 22 ไฟล์ AI agent/orchestrator (ดู AGENTS.md)
    ├── routers/                 ← 8 ไฟล์ FastAPI router (กำหนด endpoint)
    ├── tools/                   ← 8 ไฟล์ CrewAI @tool (ความสามารถที่ agent เรียกใช้ได้)
    ├── schemas/                 ← Pydantic request/response models
    ├── scripts/                 ← สคริปต์ index Obsidian vault + sync PDF เข้า MinIO
    ├── static/                  ← หน้า HTML ทดสอบ/เดโม (mount เป็น /static)
    ├── obsidian_knowledge/      ← 🌿 ข้อมูลจริงของคลังความรู้ Obsidian (ไฟล์ .md)
    ├── obsidian_knowledge_rename/ ← (ดูข้อสังเกตด้านล่าง — โฟลเดอร์ที่อาจเหลือจากการจัดระเบียบ)
    └── obsidian_knowledge.7z    ← ไฟล์บีบอัดสำรองของ vault (ซ้ำกับ ../obsidian.7z ที่ root workspace)
```

---

## 3. Entry point — `main.py`

ไฟล์นี้ทำ 4 อย่าง:
1. โหลด `.env`, ตั้งค่า logging, สร้าง FastAPI app (`title="Health Analytics API"`, version 2.0.0) พร้อม CORS (origin จาก `settings.cors_origin_list`)
2. **Register routers ทั้งหมด 8 ตัว** (ดูหัวข้อ 5):
   `analyze`, `accident_chat`, `accident_policy`, `error_log`, `thaijo`, `tools_router`, `obsidian`, `db_explorer`
3. Mount `/static` → เสิร์ฟไฟล์ HTML ใน `src/static/`
4. สร้างหน้า **`/ui`** — index แบบการ์ดที่ลิงก์ไปหน้า UI ทดสอบทั้ง 11 หน้า (รายชื่อใน `_UI_PAGES`)

รันด้วย `uvicorn main:app --host ... --port 8000 --reload`

---

## 4. โครงสร้างพื้นฐาน — config / db / history / domains

### [`src/config.py`](./src/config.py) — Settings
- `Settings` (Pydantic BaseSettings) โหลดค่าใน `.env`, แคชด้วย `get_settings()` (lru_cache)
- กลุ่มค่าที่ตั้ง: PostgreSQL (`DB_HOST/PORT/USER/PASSWORD/NAME=musyadata`), Gemini/OpenAI API keys + เลือกโมเดล (`gemini-2.5-flash-lite`, `gemini-2.5-pro`), Redis URL, MinIO (`endpoint/bucket/credentials`), Obsidian (`vault path`, `default vault`, `search threshold`), CORS origins, host/port/log level
- มีค่า retry สำหรับ Gemini 429 (`GEMINI_RETRY_LIMIT=3`, `GEMINI_RETRY_DELAY=60s`) — ใช้คู่กับ `agent_defaults.py`

### [`src/db/pool.py`](./src/db/pool.py) — PostgreSQL connection pool
- ใช้ `psycopg2.pool.ThreadedConnectionPool` (min 2 / max 20 connections)
- `get_sync_pool()` — สร้าง/คืน pool แบบ global singleton
- `get_db_connection()` — context manager สำหรับยืม/คืน connection
- `query_db(sql, params)` → `list[dict]` (สำหรับ SELECT)
- `execute_db(sql, params)` → จำนวนแถวที่กระทบ (สำหรับ INSERT/UPDATE/DELETE)

### [`src/history.py`](./src/history.py) — ประวัติแชทบน Redis
- เก็บบทสนทนาแบบ persistent (ใช้ร่วมกันได้หลาย worker), จำกัด 6 เทิร์นล่าสุด, TTL 24 ชม.
- `get_history(session_id)`, `append_history(session_id, role, text)`, `build_history_context(history)` (จัดรูปแบบเป็นข้อความไทยให้ LLM ใช้)
- ใช้โดย pipeline แทบทุกตัวเพื่อ "เข้าใจคำถามต่อเนื่อง" (follow-up) — คู่กับ `question_resolver.py`

### [`src/domains.py`](./src/domains.py) — นิยามโดเมนสุขภาพ
นิยาม 7 โดเมนที่ระบบรู้จัก (รหัส, ชื่อไทย/อังกฤษ, prefix โฟลเดอร์ CSV, คำอธิบายความเชี่ยวชาญ):

| รหัส | โดเมน | หมายเหตุ |
|---|---|---|
| `d0` | ทั่วไป (General Advisor) | ผู้เชี่ยวชาญข้ามโดเมน |
| `d1` | อุบัติเหตุทางถนน (Road Accidents) | ⚠️ ใช้ PostgreSQL โดยตรง ไม่ใช่ CSV pipeline |
| `d2` | สุขภาพจิต (Mental Health) | ฆ่าตัวตาย ซึมเศร้า บริการจิตเวช |
| `d3` | โรคไม่ติดต่อ — NCDs | เบาหวาน ความดัน หัวใจ หลอดเลือดสมอง |
| `d4` | โภชนาการ (Nutrition) | ภาวะทุพโภชนาการ โรคอ้วน ความมั่นคงทางอาหาร |
| `dt` | งานวิจัย ThaiJO | สังเคราะห์วรรณกรรม |
| `obsidian` | คลังความรู้ (Knowledge Vault) | เอกสารนโยบาย/งานวิจัยเขตสุขภาพที่ 10 |

---

## 5. Routers — endpoint ทั้งหมด (`src/routers/`)

| ไฟล์ | Prefix | Endpoint หลัก | หน้าที่ / เรียก agent ไหน |
|---|---|---|---|
| [`analyze.py`](./src/routers/analyze.py) | `/api`, `/api/chat` | `POST /api/analyze`, `POST /api/chat` (alias), `GET /health` | **จุดเข้าหลักของระบบแชท** — orchestrator รวม: Memory agent → Router agent → เลือก pipeline (CSV เดี่ยว/หลายโดเมน, SQL อุบัติเหตุ, ThaiJO, Obsidian, Tavily) ตาม `mode` (`normal`/`stats`/`obsidian`/`tavily`/`thaijo`/`report-gather`/`multi`) สตรีมผลลัพธ์แบบ SSE, จำกัด concurrency ด้วย semaphore (สูงสุด 5 pipeline/worker) |
| [`accident_chat.py`](./src/routers/accident_chat.py) | `/api/accident-chat` | `POST /ask`, `POST /ask/stream`, `POST /quick`, `GET /provinces`, `GET /districts`, `GET /sample-questions` | แชทถาม-ตอบเชิงนโยบายอุบัติเหตุ — เรียก `accident_chat_orchestrator.run_accident_chat()` (2-agent: SQL agent → คนเขียนคำตอบ); `/quick` ดึงข้อมูลดิบเร็ว ๆ ผ่าน 15+ tools โดยไม่ผ่าน LLM |
| [`accident_policy.py`](./src/routers/accident_policy.py) | `/api/accident-policy` | `GET /zone10/data`, `POST /zone10` | สร้าง Policy Brief เขต 10 — `GET /zone10/data` ดึง SQL ดิบ 7 คำถาม (เร็ว <5s); `POST /zone10` รัน 3-agent crew เต็มรูปแบบ (SQL Fetcher → Policy Analyst → Report Writer ใช้เวลา 2-5 นาที) |
| [`obsidian.py`](./src/routers/obsidian.py) | `/api/obsidian` | `POST /search`, `GET /notes`, `GET /notes/{id}`, `POST /ask` (+stream), `GET /status`, `GET /vaults`, `POST /index`, `POST /pdfs/sync`, `GET /pdfs*` | จัดการคลังความรู้ Obsidian — ค้นหาแบบ pg_trgm (`/search`, ไม่ใช้ LLM), ถาม-ตอบเต็มบริบท (`/ask` → `obsidian_fullcontext.py`), ดัชนี vault ใหม่ (`/index`), ซิงก์ PDF เข้า MinIO (`/pdfs/sync`) |
| [`thaijo.py`](./src/routers/thaijo.py) | `/api/thaijo` | `POST /api/thaijo`, `POST /api/thaijo/report`, `POST /api/thaijo/topics`, `POST /api/thaijo/demo` | ค้นงานวิจัยจาก ThaiJO API แล้วสังเคราะห์เป็นรายงาน/journal HTML — เรียก `thaijo_agent.py` (Fetcher → Planner → Generator) |
| [`tools_router.py`](./src/routers/tools_router.py) | `/api` | `POST /api/compare`, `POST /api/report`, `POST /api/workplan`, `POST /api/database` | เครื่องมือวิเคราะห์ CSV เพิ่มเติม — เปรียบเทียบ 2 ชุดข้อมูล (`compare_agent`), รายงานสรุป (`report_agent`), แผนงานล้วน ๆ จาก LLM (`workplan_agent`), วิเคราะห์ไฟล์แนบของผู้ใช้ (`database_agent`) |
| [`db_explorer.py`](./src/routers/db_explorer.py) | `/api/db` | `GET /tables`, `GET /tables/{name}/columns`, `GET /tables/{name}/rows` | สำรวจฐานข้อมูลแบบอ่านอย่างเดียว (read-only) — list ตาราง/คอลัมน์/แถวแบบแบ่งหน้า มี validation ชื่อตาราง/คอลัมน์ + ตัด vector field ที่ยาวเกิน |
| [`error_log.py`](./src/routers/error_log.py) | `/api/errors` | `GET /api/errors`, `GET /api/errors/stats`, `GET /api/errors/summary`, `DELETE /api/errors` | ดู/วิเคราะห์ error log ของ agent — `/summary` ใช้ LLM สรุปเป็นภาษาไทย, ไฟล์ log เก็บเป็น text รายวัน (`error_logs/agent_errors_YYYY-MM-DD.txt`) |

> 💡 endpoint ที่ frontend เรียกผ่าน proxy (`app/api/python/[prefix]/...`) ได้แก่ `accident-chat`, `accident-policy`, `db`, `obsidian` เท่านั้น — ส่วน `analyze`/`chat`/`thaijo`/`tools` frontend เรียกผ่าน `app/api/chat` ที่ส่งต่อ (proxy) เองอีกที

---

## 6. Tools — ความสามารถที่ agent เรียกใช้ (`src/tools/`)

Tools คือฟังก์ชันที่ห่อด้วย CrewAI `@tool` decorator ให้ agent เรียกใช้ได้ระหว่างทำงาน:

| ไฟล์ | ความสามารถ | ใช้โดย agent ไหน |
|---|---|---|
| [`zone10_accident.py`](./src/tools/zone10_accident.py) | SQL tools เฉพาะ Policy Brief เขต 10 — 7 ฟังก์ชัน (`_query_top_roads`, `_query_Q2..Q7_*`) ดึงข้อมูลจากตาราง `mart_province_road`, `dim_geography` | `accident_policy_agent` (Zone10 SQL Fetcher) |
| [`accident_chat_sql.py`](./src/tools/accident_chat_sql.py) | SQL tools สำหรับแชทอุบัติเหตุแบบสนทนา — **15+ tools** แบ่ง 4 กลุ่ม: Hotspot/วิศวกรรมถนน (Q1-5), พฤติกรรม/แคมเปญ (Q6-10, ข้อมูล person ว่างเปล่า), ช่วงเวลา/ฤดูกาล (Q11-15), ติดตาม KPI (Q16-20) — คืนค่าเป็นตาราง Markdown พร้อม footnote ข้อจำกัดข้อมูล | `accident_chat_orchestrator` (AccidentSQLAgent) |
| [`minio.py`](./src/tools/minio.py) | จัดการไฟล์ใน MinIO + รันโค้ด Python — `list_csv_files`, `read_csv_schema`, `execute_python_code` (รันใน subprocess แยก, มี pandas/numpy/scipy), `resolve_file_id`/`fallback_find_file` (fuzzy match ชื่อไฟล์ที่ผู้ใช้พิมพ์), `read_file_bytes/text/extension` | pipeline วิเคราะห์ CSV เกือบทั้งหมด (`csv_pipeline`, `multi_csv_pipeline`, `compare_agent`, `report_agent`, `database_agent`) |
| [`obsidian.py`](./src/tools/obsidian.py) | ค้นคลังความรู้ Obsidian แบบ 2 ระดับ — **Tier 1** (เร็ว/แม่น): ค้นแบบ chunk + pg_trgm similarity + ขยายผลตาม wikilink graph (ต้องมีตาราง `obsidian_note_chunks` จาก migration 026); **Tier 2** (fallback): ค้นทั้งโน้ตด้วย `LIKE` — auto-เลือก tier ที่ใช้ได้ | `routers/obsidian.py`, `obsidian_fullcontext.py` |
| [`tavily_search.py`](./src/tools/tavily_search.py) | ค้นเว็บผ่าน Tavily API — `tavily_search(query, max_results=2)` คืนคำตอบ + URL + สรุปผลลัพธ์ (ต้องมี `TAVILY_API_KEY`) | `tavily_pipeline.py` |
| [`weather_tool.py`](./src/tools/weather_tool.py) | เช็คสภาพอากาศผ่าน Open-Meteo (ฟรี ไม่ต้อง API key) — geocode สถานที่ → ดึงพยากรณ์ → แปล WMO weather code เป็นภาษาไทย | เปิดให้ agent เรียกได้ (ยังไม่เห็นจุดเรียกใช้ชัดเจนในโค้ดที่อ่าน) |
| [`error_logger.py`](./src/tools/error_logger.py) | บันทึก/อ่าน/สรุป error ของ agent — `log_agent_error`, `classify_error` (จัดประเภท: auth/quota/timeout/empty_response/tool_error), `read_all_errors`, `aggregate_errors`, `clear_all_logs` | `csv_pipeline`, `error_monitor_agent`, router `error_log` |
| [`thaijo_cache.py`](./src/tools/thaijo_cache.py) | Cache สรุป PDF ของ ThaiJO บน Redis — `get_cached_summary`/`set_cached_summary` (key = `thaijo_pdf:{sha256(url)}`) | `thaijo_agent.py` (เร่งความเร็วเมื่อเจอ PDF ซ้ำ) |

---

## 7. Schemas — Pydantic models (`src/schemas/`)

| ไฟล์ | โมเดลหลัก |
|---|---|
| [`accident_chat.py`](./src/schemas/accident_chat.py) | `AccidentChatRequest`, `AccidentChatQuickRequest`, `AccidentChatResponse` |
| [`accident_policy.py`](./src/schemas/accident_policy.py) | `AccidentPolicyRequest`, `AccidentPolicyResponse`, `Zone10DataResponse` |
| [`analyze.py`](./src/schemas/analyze.py) | `HistoryMessage`, `AnalyzeRequest` (`sessionId`, `prompt`, `history[]`, `mode`, `tools[]`) |
| [`obsidian.py`](./src/schemas/obsidian.py) | `ObsidianSearchRequest/Response`, `ObsidianAskRequest/Response`, `ObsidianNoteRef`, `ObsidianVaultInfo`, `ObsidianStatusResponse`, `ObsidianPdfAsset`, `ObsidianNotePdfPair`, `ObsidianIndexRequest/Result`, `ObsidianPdfSyncResult` ฯลฯ |
| [`thaijo.py`](./src/schemas/thaijo.py) | `ThaiJoRequest`, `ThaiJoGenerateRequest`, `ThaiJoTopicsRequest` |
| [`tools.py`](./src/schemas/tools.py) | `CompareRequest`, `ReportRequest`, `WorkplanRequest`, `DatabaseRequest` |

---

## 8. หน้า UI ทดสอบ (`src/static/`, mount เป็น `/static`)

`main.py` กำหนดรายการไว้ใน `_UI_PAGES` แล้วสร้างหน้า index ที่ `/ui`:

| ไฟล์ | ชื่อ | ใช้ทดสอบอะไร |
|---|---|---|
| `unified_test_ui.html` | 🤖 Unified Agent Test | แชทรวมทุก pipeline + streaming |
| `policy_brief_ui.html` | 📋 Policy Brief | สร้าง Policy Brief อัตโนมัติ |
| `accident_chat_ui.html` | 🚗 Accident Chat | วิเคราะห์อุบัติเหตุ Zone 10 |
| `accident_policy_ui.html` | 📊 Accident Policy | ข้อมูลนโยบายอุบัติเหตุ |
| `document_agent_test_ui.html` | 📄 Document Agent Test | ทดสอบ Document Agent |
| `document_upload_ui.html` | 📁 Document Upload | อัปโหลด + APA Citation |
| `citation_test_ui.html` | 🔖 Citation Test | ระบบ Citation |
| `db_explorer_ui.html` | 🗄️ DB Explorer | สำรวจตาราง DB |
| `thaijo_research_ui.html` | 🔬 ThaiJO Research | ค้นงานวิจัย ThaiJO |
| `test_ui.html` | 🧪 Test UI | ทดสอบ Tools & Pipeline ทั่วไป |
| `obsidian_knowledge_ui.html` | 🌿 Obsidian Knowledge Vault | คลังความรู้สุขภาพเขต 10 |

เปิดดูได้ที่ `http://localhost:8000/ui` (รวมลิงก์) หรือ `http://localhost:8000/docs` (Swagger)

---

## 9. คลังความรู้ Obsidian (`src/obsidian_knowledge/`)

โฟลเดอร์ที่เก็บไฟล์ `.md` จริงของคลังความรู้ (sync ผ่าน Docker volume แบบ read-only ให้ container backend):

```
obsidian_knowledge/
├── 000_Health_Region_10_MOC.md   ← Master index / Map of Content
├── อุบลราชธานี/
├── ศรีสะเกษ/
├── ยโสธร/
├── อำนาจเจริญ/
└── มุกดาหาร/
    └── สรุปผลรายไฟล์/
        ├── งานวิจัย/
        ├── ตรวจราชการ/
        └── (โฟลเดอร์โดเมน D1_Road, D2_Mental, D3_NCD, D4_Nutrition ฯลฯ)
```

- เนื้อหา: MOC notes, สรุปผลรายจังหวัด/อำเภอ, เอกสารนโยบาย, บทสรุปงานวิจัย, รายงานตรวจราชการ
- ดัชนีเข้า Postgres ผ่านตาราง: `obsidian_vaults`, `obsidian_notes`, `obsidian_note_chunks` (full-text search), `obsidian_note_links` (wikilink graph)
- การ index/sync ทำผ่านสคริปต์ใน `src/scripts/`:
  - [`index_obsidian.py`](./src/scripts/index_obsidian.py) — เดินไฟล์ในระบบไฟล์ แล้ว upsert เข้า DB
  - [`sync_obsidian_pdfs.py`](./src/scripts/sync_obsidian_pdfs.py) — อัปโหลด PDF ดิบเข้า MinIO และบันทึกลง DB
- มี archive สำรอง: `src/obsidian_knowledge.7z` และโฟลเดอร์ `src/obsidian_knowledge_rename/` (ดูข้อสังเกต)

---

## 10. ไฟล์ระดับบนสุดอื่น ๆ

| ไฟล์ | หน้าที่ |
|---|---|
| [`AI_CSV_Data_Analyst_System.md`](./AI_CSV_Data_Analyst_System.md) | เอกสารออกแบบ pipeline วิเคราะห์ CSV 6 ขั้นตอน (File Finder → Schema Analyst → Code Generator → Executor → Insight Analyst) — เอกสารอ้างอิง/ดีไซน์ ไม่ใช่โค้ดที่รัน |
| [`check_obsidian.py`](./check_obsidian.py) | สคริปต์วินิจฉัย — เชื่อมต่อ Postgres เช็คตาราง `obsidian_*`, list vault, นับจำนวนโน้ต, ตรวจว่าโฟลเดอร์ vault มีอยู่จริง |
| [`check_paths.py`](./check_paths.py) | สคริปต์ตรวจสอบ/validate path ของไฟล์ (diagnostic ลักษณะคล้าย check_obsidian) |
| [`db_check_out.txt`](./db_check_out.txt) | ผลลัพธ์ที่ dump ไว้จากการรันสคริปต์ตรวจสอบข้างต้น |
| `musya_20260517_2118.dump` | ไฟล์สำรองฐานข้อมูล (pg_dump) ลงวันที่ 2026-05-17 |

---

## 11. การไหลของ request ตัวอย่าง — วิเคราะห์ CSV ผ่าน `/api/analyze`

```
ผู้ใช้ถามคำถาม → POST /api/analyze (จาก frontend ผ่าน /api/chat)
   │
   ▼
1. Memory Agent (question_resolver.resolve_question)
   - เช็คว่าเป็นคำถามต่อเนื่องไหม ขยายคำถามที่พูดสั้น ๆ ("ของจังหวัดนั้น" → ชื่อจังหวัดเต็ม)
   - ใช้ history จาก Redis (history.py)
   ▼
2. Router Agent (router.route_multi_domain)
   - จัดประเภทโดเมน: d0 / d1 (อุบัติเหตุ → ไป SQL) / d2-d4 (CSV) / dt (ThaiJO) / obsidian
   - ตัดสินใจ: เดี่ยวหรือหลายโดเมน, ใช้ CSV/SQL/อื่น ๆ
   ▼
3. เลือก pipeline ตามผลลัพธ์ข้อ 2 + mode ที่ผู้ใช้ระบุ:
   ├─ d1 (อุบัติเหตุ)      → accident_chat_orchestrator.run_accident_chat()  (2 agents + accident_chat_sql tools)
   ├─ d2-d4 เดี่ยว         → csv_pipeline.run_pipeline()        (6 agents: หา CSV → วิเคราะห์ schema → เขียนโค้ด → รันโค้ด → สรุปผล)
   ├─ d2-d4 หลายโดเมน      → multi_csv_pipeline.run_multi_pipeline() (หาไฟล์โดยดู "folder tree" แล้วเลือกชื่อโฟลเดอร์ตัวชี้วัดก่อน ค่อย resolve เป็นไฟล์ CSV จริง — แม่นกว่าการเดาจากชื่อไฟล์ตรง ๆ — บวก geo-key detector + domain validator; รายละเอียดดู AGENTS.md)
   ├─ dt (ThaiJO)          → thaijo_agent.run_thaijo_pipeline()  (Fetcher → Planner → Generator, สตรีม HTML)
   └─ obsidian             → obsidian_fullcontext.run_obsidian_ask_fullcontext() (โหลด .md ทั้งหมด → ส่งให้ Gemini)
   ▼
4. ระหว่างรัน — ส่ง progress event แบบ SSE ผ่าน progress.py
   { type: "agent_start"/"agent_done"/"final", agentName, message, agentSteps[] ... }
   ▼
5. บันทึกประวัติ — history.append_history(session_id, "assistant", คำตอบ) ลง Redis
   ▼
ส่ง StreamingResponse (text/event-stream) กลับไปยัง frontend → แสดงผลแบบ real-time
```

---

## 12. ข้อจำกัดของข้อมูลที่ต้องรู้ (มีผลต่อคำตอบของ AI)

1. **ข้อมูลอุบัติเหตุ**: ตาราง `fact_accident_person` **ว่างเปล่า** (ไม่มีข้อมูลหมวกกันน็อค/เข็มขัด/อายุ/เพศ), คอลัมน์ `road_name` ใน `mart_province_road` **ส่วนใหญ่เป็น NULL** — ปีในฐานข้อมูลเป็น **ค.ศ. (2021-2026)**, ถ้าผู้ใช้ถามเป็น พ.ศ. agent ต้องแปลง (+543)
2. **Obsidian vault**: ใช้ hybrid search (Tier 1 chunk-based, Tier 2 LIKE fallback); โหลดทั้ง vault มีขนาด ~1.1 MB (ทั้งหมด) หรือ ~100-200 KB (กรองตามจังหวัด)
3. **Agent errors**: 429 RESOURCE_EXHAUSTED จัดการด้วย exponential backoff ใน `agent_defaults.py`; empty response จะ retry อัตโนมัติ; error อื่น ๆ ถูกบันทึกลง `error_logs/`
4. **การหาไฟล์ CSV**: ชื่อไฟล์ใน MinIO เป็นรหัสตัวเลข ส่วน path จริงเก็บใน metadata (`x-amz-meta-path`) — ใช้ fuzzy matching จับคำอธิบายของผู้ใช้ อาจพลาดถ้าคำอธิบายกำกวม

---

## 13. ตัวแปรแวดล้อมที่ต้องตั้งค่า (`.env`)

```
# LLM
GEMINI_API_KEY=...
OPENAI_API_KEY=...
TAVILY_API_KEY=...                 # ออปชัน — สำหรับ web search

# Redis
REDIS_URL=redis://localhost:6379/0

# MinIO
MINIO_ENDPOINT=minio (หรือ localhost ตอน dev เดี่ยว)
MINIO_PORT=9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_USE_SSL=false
MINIO_BUCKET=fileapa

# PostgreSQL
DB_HOST=postgres (หรือ localhost)
DB_PORT=5432
DB_NAME=musyadata (compose ของ frontend) หรือ musyadatabase (standalone compose)
DB_USER=postgres
DB_PASSWORD=1234

# Server
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# Obsidian
OBSIDIAN_VAULT_PATH=src/obsidian_knowledge
OBSIDIAN_DEFAULT_VAULT=health_region_10
OBSIDIAN_ENABLED=true

# Tracing (CrewAI)
CREWAI_TRACING_ENABLED=false
OTEL_SDK_DISABLED=true
```

---

## 14. ข้อสังเกต / สิ่งที่อาจอยากปรับปรุง

1. **`src/agents/promptplan.js`** — ไฟล์ JS อยู่ในโฟลเดอร์ Python agents (เก็บ `PROMPT_PLAN` constant สำหรับ Workplan Agent) — ควรย้ายไป `src/prompts/` หรือแปลงเป็น `.py` และ inline เข้า `workplan_agent.py`
2. **agent ที่ทับซ้อนกัน** — `obsidian_agent.py` vs `obsidian_fullcontext.py` (ไม่ชัดว่าตัวไหนเป็น primary), `accident_chat_orchestrator.py` vs `accident_policy_orchestrator.py` (โครงสร้างคล้ายกันมาก), `compare_agent.py` vs `report_agent.py` (มี File Finder + Schema Analyst ซ้ำกัน — อาจแยกเป็น component ที่ใช้ร่วมกันได้) — ดูรายละเอียดใน [AGENTS.md](./AGENTS.md)
3. **`error_logs/`** — ไฟล์ log สะสมรายวันโดยไม่มี retention policy — ควรพิจารณาตั้งระยะเวลาลบอัตโนมัติ (เช่น เก็บ 30 วัน)
4. **`progress.py`** — ผูกรายชื่อ agent ของแต่ละ pipeline ไว้ตรง ๆ (เช่น `ACCIDENT_CHAT_PIPELINE_AGENTS`) — เพิ่ม pipeline ใหม่ต้อง hardcode ทุกครั้ง อาจเปลี่ยนเป็น registry pattern
5. **โดเมนถูก hardcode ซ้ำ** — รหัสโดเมน (d0-d4, dt, obsidian) ปรากฏซ้ำใน `router.py`, `analyze.py`, `config.py` — ควร import จาก `domains.py` แหล่งเดียว
6. **`obsidian_knowledge.7z`**, **`obsidian_knowledge_rename/`** — ดูเหมือนไฟล์ที่ค้างจากการจัดระเบียบ vault ก่อนหน้า อาจลบได้ถ้าไม่ใช้แล้ว (ตรวจสอบก่อน)
7. **หน้า static UI 11 หน้า** — แยกไฟล์ HTML แยกกันหมด อาจรวม/ทำเป็น template เดียวได้ เพื่อลดงานดูแล

---

*สร้างจากการอ่านซอร์สโค้ดจริง 2026-06-08 — หากแก้โค้ดภายหลัง ให้ตรวจสอบความถูกต้องกับโค้ดจริงเสมอ*
