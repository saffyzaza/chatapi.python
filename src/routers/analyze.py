"""Analyze router — SSE streaming pipeline for health domain Q&A."""
import asyncio
import json
import os
import re
import threading
import time
from typing import Any

# จำกัด 5 AI pipelines พร้อมกันต่อ worker (4 workers = 20 concurrent รวม)
_AI_SEMAPHORE = threading.BoundedSemaphore(5)

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.agents.router import route_domain, route_with_web_search, route_multi_domain, _has_accident_signal, is_accident_question
from src.agents.csv_pipeline import run_pipeline
from src.agents.multi_csv_pipeline import run_multi_pipeline
from src.agents.thaijo_agent import run_thaijo_pipeline
from src.history import get_history, append_history, build_history_context
from src.schemas.analyze import AnalyzeRequest
from src.tools.vault_rag import detect_province_from_prompt, read_vault_context, get_vault_summary
from src.domains import DOMAINS as _DOMAINS


router = APIRouter(tags=["analyze"])


_URL_RE = re.compile(r"https?://[^\s)\]\"'<>]+")


def _tavily_raw_to_articles_text(raw_data: str) -> str:
    """แปลงผลดิบจาก Tavily Search Agent เป็นบล็อกแยกต่อ URL แบบเดียวกับ ThaiJo/PubMed
    articles_text ("--- ... ที่ N ---") เพื่อให้ Report Generator อ้างอิงแยกทีละแหล่ง
    แทนที่จะเห็น Tavily เป็น "แหล่งเดียว" แล้วอ้างอิงรวมเป็น 1 reference เท่านั้น
    (เช่น "Tavily Research (2567)...") ทั้งที่ Tavily หาเจอ 10 แหล่งจริง

    ⚠️ ตั้งใจใช้แค่ URL-extraction แบบกว้าง ๆ (ไม่ผูกกับโครงสร้างข้อความเฉพาะ) เพราะ
    Search Agent (LLM) มักไม่ pass-through ผล tool ตรง ๆ เป็น task output ของตัวเอง —
    บางครั้ง paraphrase ใหม่เป็น prose รูปแบบอื่น (เช่น "ชื่อ (URL: ...): สรุป") แบบไม่
    แน่นอนในแต่ละครั้ง ลอง parse โครงสร้าง "N. Title\\n URL:\\n สรุป:" ตรง ๆ มาก่อนแล้ว
    พบว่าพลาดบ่อยเพราะ Agent เปลี่ยนรูปแบบ — ดึงเฉพาะ URL (ทนทานต่อรูปแบบข้อความรอบข้าง
    ทุกแบบ) มาสร้างบล็อกแยกต่อแหล่งแทน ไม่ต้องพึ่งการดึง title/summary แม่นยำ เพราะ
    เนื้อหา/บริบทเต็มยังอยู่ใน Tavily Answer Writer narrative ที่แนบไปพร้อมกันอยู่แล้ว
    (ดู tavily_result_holder['msg'] ตรงจุดที่เรียกใช้ฟังก์ชันนี้)
    """
    urls = list(dict.fromkeys(_URL_RE.findall(raw_data or "")))  # unique, keep order
    if not urls:
        return ""
    lines = [f"--- แหล่งข้อมูลเว็บที่ {i} ---\nURL: {url}" for i, url in enumerate(urls, 1)]
    return "\n\n".join(lines)


def _orchestrate(
    prompt: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    session_id: str = "",
    client_history: list[dict[str, Any]] | None = None,
    mode: str = "normal",
) -> None:
    """Full pipeline entry point — runs in a background thread."""
    def put(ev: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(queue.put(ev), loop)

    def finish() -> None:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    try:
        # Merge history
        raw_history = client_history or get_history(session_id)
        if raw_history and raw_history[-1].get("role") == "user":
            raw_history = raw_history[:-1]
        history_context = build_history_context(raw_history)
        history_section = f"{history_context}\n\n" if history_context else ""

        if session_id:
            append_history(session_id, "user", prompt)

        # ── Memory Agent: แปลง follow-up question ให้ครบถ้วน ─────────────────
        if history_context:
            from src.agents.question_resolver import resolve_question
            put({"type": "agent_start", "step": "memory", "agentName": "Memory Agent"})
            resolved, was_changed = resolve_question(
                prompt, history_context, os.getenv("GEMINI_API_KEY", "")
            )
            if was_changed:
                put({
                    "type": "agent_done", "step": "memory", "agentName": "Memory Agent",
                    "result": f"ปรับคำถาม: {resolved}",
                })
                prompt = resolved  # ← downstream agents ทั้งหมดใช้ resolved prompt
            else:
                put({
                    "type": "agent_done", "step": "memory", "agentName": "Memory Agent",
                    "result": "คำถามชัดเจน ไม่ต้องปรับ",
                })

        # ── Vault RAG: ดึงเอกสาร Obsidian ตามจังหวัดที่พบในคำถาม ───────────────
        vault_ctx = ""
        reasoning = ""  # default — stats path จะ override ด้วย narrator agent
        vault_province = detect_province_from_prompt(prompt)
        if vault_province:
            summary = get_vault_summary(vault_province)
            if summary.get("file_count", 0) > 0:
                put({
                    "type": "agent_start",
                    "step": "vault_rag",
                    "agentName": "Vault RAG",
                })
                vault_ctx = read_vault_context(vault_province, max_chars=8000)
                put({
                    "type": "agent_done",
                    "step": "vault_rag",
                    "agentName": "Vault RAG",
                    "result": (
                        f"📚 โหลดเอกสาร {vault_province} จาก Obsidian vault "
                        f"({summary['file_count']} ไฟล์ · {len(vault_ctx):,} chars)"
                    ),
                    "province": vault_province,
                    "file_count": summary["file_count"],
                })

        # ── Stats mode: อุบัติเหตุใช้ PostgreSQL (SQL) ตรง ๆ, d2-d4 ใช้ CSV pipeline ──
        # (เดิมโค้ดนี้ถูกวางไว้ผิดที่ใต้ mode == "tavily" ทำให้ปุ่ม "สถิติ" ไม่เคยเรียก
        # Accident SQL Agent เลย และปุ่ม "ค้นหาทั่วไป" ก็ไปเรียก CSV pipeline แทนที่จะ
        # ค้นเว็บจริง — ดู mode == "tavily" ด้านล่างสำหรับ web search ที่ถูกต้อง)

        if mode == "stats":
            put({"type": "agent_start", "step": "router", "agentName": "Router Agent"})

            # ── Accident routing: d1 uses PostgreSQL not CSV ──────────────────
            # LLM นำเสมอ — is_accident_question() เช็ค keyword ก่อน (เร็ว) แล้วถ้า
            # miss จึงให้ LLM ตัดสิน เพื่อจับคำถามอุบัติเหตุที่ keyword list ครอบไม่ถึง
            # (เดิมพึ่ง keyword ล้วน → คำถามอุบัติเหตุที่ไม่มีคำตรง ๆ หลุดไป CSV/NCD)
            if is_accident_question(prompt, history_context):
                import concurrent.futures
                import re as _re
                from src.agents.accident_chat_orchestrator import run_accident_chat
                from src.tools.accident_chat_sql import (
                    detect_zone10_provinces,
                    detect_out_of_zone10_provinces,
                    ZONE10_PROVINCES as _Z10,
                )

                # ── Out-of-zone guard: ถามจังหวัดนอกเขตสุขภาพที่ 10 → แจ้งตรง ๆ ──────
                # ระบบมีข้อมูลอุบัติเหตุเฉพาะ 5 จังหวัดเขต 10 — ถ้าผู้ใช้ถามขอนแก่น/
                # อุดรธานี ฯลฯ ต้องแจ้งว่าไม่มีข้อมูล ไม่ใช่เงียบ ๆ คืนข้อมูลเขต 10 แทน
                _out = detect_out_of_zone10_provinces(prompt)
                _inz = detect_zone10_provinces(prompt)
                if _out and not _inz:
                    from src.tools.missing_data_logger import log_missing_data
                    log_missing_data(prompt, domain="d1", reason="out_of_zone10", session_id=session_id)
                    _prov_list = "\n".join(f"  • {p}" for p in _Z10)
                    warn = (
                        f"## ไม่พบข้อมูล\n\n"
                        f"ระบบไม่มีข้อมูลอุบัติเหตุทางถนนของจังหวัด {', '.join(_out)}\n\n"
                        f"ฐานข้อมูลครอบคลุมเฉพาะ **เขตสุขภาพที่ 10** ซึ่งมี 5 จังหวัด:\n"
                        f"{_prov_list}\n\n"
                        f"หากต้องการข้อมูล กรุณาระบุจังหวัดในเขตสุขภาพที่ 10 "
                        f"หรือแจ้งผู้ดูแลระบบ (admin) เพื่อเพิ่มข้อมูลจังหวัดที่ต้องการ"
                    )
                    put({
                        "type": "agent_done", "step": "router", "agentName": "Router Agent",
                        "result": "อุบัติเหตุทางถนน (SQL) — จังหวัดนอกเขต 10",
                        "domain": {"code": "d1", "nameTh": "อุบัติเหตุทางถนน", "nameEn": "Road Accidents"},
                    })
                    if session_id:
                        append_history(session_id, "assistant", warn)
                    put({"type": "result", "content": warn,
                         "domain": {"code": "d1", "nameTh": "อุบัติเหตุทางถนน", "nameEn": "Road Accidents"}})
                    return

                # ── ดึง "จังหวัด" ที่ระบุในคำถาม → เจาะข้อมูลตรงจังหวัด ────────────
                # ถ้าระบุจังหวัดเขต 10 เดียว ส่งต่อให้ pipeline filter ตรงจังหวัดนั้น
                # (ไม่งั้นจะค้นทั้ง 5 จังหวัดเขต 10 ทั้งที่ผู้ใช้ถามแค่จังหวัดเดียว)
                _provs = detect_zone10_provinces(prompt)
                _province = _provs[0] if len(_provs) == 1 else ""

                # ── ดึง "ปี พ.ศ." (25xx) แล้วแปลงเป็น ค.ศ. — DB เก็บปีเป็น ค.ศ. ──────
                # (ปีในฐานข้อมูล = ค.ศ. 2021-2026; พ.ศ. = ค.ศ. + 543) ถ้าผู้ใช้ระบุปี
                # ให้ scope ตรงปีนั้น ไม่งั้นใช้ช่วงเต็ม 2021-2026 ตามเดิม
                _be_years = [int(y) for y in _re.findall(r"25\d\d", prompt)]
                _ce_years = [y - 543 for y in _be_years if 2021 <= y - 543 <= 2026]
                _y_start, _y_end = (min(_ce_years), max(_ce_years)) if _ce_years else (2021, 2026)

                put({
                    "type": "agent_done", "step": "router", "agentName": "Router Agent",
                    "result": "อุบัติเหตุทางถนน (SQL)",
                    "domain": {"code": "d1", "nameTh": "อุบัติเหตุทางถนน", "nameEn": "Road Accidents"},
                })
                _scope_label = f"{_province or 'เขตสุขภาพที่ 10'} พ.ศ. {_y_start + 543}-{_y_end + 543}"
                put({"type": "agent_start", "step": "accident_sql", "agentName": "Accident SQL Agent"})
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    # ⚠️ ส่ง history_context ต่อให้ pipeline อุบัติเหตุด้วยเสมอ — ไม่งั้น
                    # คำถามต่อเนื่อง (follow-up) เช่น "ขอข้อมูลแต่ละอำเภอ" จะถูกตอบแบบ
                    # เริ่มนับหนึ่งใหม่ ไม่รู้ว่าเทิร์นก่อนหน้าให้ข้อมูลอะไรไปแล้ว
                    # ทำให้คุยต่อเนื่องไม่เป็นธรรมชาติ (ผู้ใช้อยากให้เหมือนแชท Gemini)
                    acc_result = ex.submit(
                        run_accident_chat, prompt, _province, "", _y_start, _y_end, history_context
                    ).result()
                put({"type": "agent_done", "step": "accident_sql", "agentName": "Accident SQL Agent",
                     "result": f"ดึงข้อมูลอุบัติเหตุทางถนน {_scope_label} สำเร็จ"})
                if session_id:
                    append_history(session_id, "assistant", acc_result.answer)
                put({"type": "result", "content": acc_result.answer,
                     "domain": {"code": "d1", "nameTh": "อุบัติเหตุทางถนน", "nameEn": "Road Accidents"}})
                return

            # ── LLM router (context-aware) — อ่านบริบทการสนทนา + ให้เหตุผล ─────────
            # ใช้ LLM ดู context ทั้งบทสนทนาแล้วตัดสิน domain เอง โดยยึดความต่อเนื่อง:
            # follow-up (ขอแยกย่อย/เจาะจงพื้นที่-ปี) → domain เดิม ไม่ re-classify จนหลุด
            # หัวข้อ (แทน keyword-correction แบบเดิมที่กันได้เฉพาะเคสมี keyword ตรง ๆ)
            from src.agents.router import route_stats_domains
            csv_domains, is_multi, route_reasoning = route_stats_domains(prompt, history_context)
            csv_domains = [d for d in csv_domains if d.code in ("d2", "d3", "d4")] or [_DOMAINS["d3"]]
            is_multi = len(csv_domains) >= 2
            domain_names_th = " + ".join(d.name_th for d in csv_domains)
            put({
                "type": "agent_done",
                "step": "router",
                "agentName": "Router Agent",
                "result": f"สถิติ: {domain_names_th}",
                "reasoning": route_reasoning,
                "domain": {
                    "code": "multi" if is_multi else csv_domains[0].code,
                    "nameTh": domain_names_th,
                    "nameEn": " + ".join(d.name_en for d in csv_domains),
                },
            })
            if is_multi:
                run_multi_pipeline(
                    prompt=prompt, queue=queue, loop=loop, domains=csv_domains,
                    history_context=history_context, history_section=history_section,
                    session_id=session_id,
                )
            else:
                run_pipeline(
                    prompt=prompt, queue=queue, loop=loop, domain=csv_domains[0],
                    history_context=history_context, history_section=history_section,
                    session_id=session_id,
                )
            return

        # ── Tavily mode: ค้นหาข้อมูลจากอินเทอร์เน็ตด้วย Tavily Search จริง ──────────
        if mode == "tavily":
            from src.agents.tavily_pipeline import run_tavily_pipeline
            run_tavily_pipeline(
                prompt=prompt, queue=queue, loop=loop,
                session_id=session_id, history_section=history_section,
                reasoning=reasoning,
            )
            return

        # ── Obsidian mode: forced Knowledge Vault routing ─────────────────────
        if mode == "obsidian":
            put({"type": "agent_start", "step": "obsidian_search", "agentName": "Obsidian Knowledge Searcher"})
            from src.agents.obsidian_fullcontext import run_obsidian_ask_fullcontext
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                obs_result = ex.submit(
                    run_obsidian_ask_fullcontext,
                    prompt,
                    # ⚠️ ต้องตรวจจับจังหวัดจาก prompt เอง — run_obsidian_ask_fullcontext
                    # ไม่ได้ "infer จากคำถามเอง" ตามที่คอมเมนต์เดิมเข้าใจผิด ถ้าส่ง ""
                    # จะโหลดทั้ง vault (~1.1MB+) ทุกครั้ง เคยทำให้ Gemini context window
                    # เกิน 1,048,576 token จนพังทั้ง request (ContextWindowExceededError)
                    detect_province_from_prompt(prompt) or "",
                    "health_region_10",
                    history_context=history_context,
                ).result()
            put({"type": "agent_done", "step": "obsidian_search", "agentName": "Obsidian Knowledge Searcher",
                 "result": f"พบ {len(obs_result.notes_referenced)} notes"})
            if session_id:
                append_history(session_id, "assistant", obs_result.content)
            put({"type": "result", "content": obs_result.content,
                 "notesReferenced": [n.model_dump() for n in obs_result.notes_referenced],
                 "followUps": obs_result.follow_ups,
                 "domain": {"code": "obsidian", "nameTh": "คลังความรู้สุขภาพ เขต 10", "nameEn": "Obsidian Knowledge Vault"}})
            return

        # ── Report-Gather mode: รัน thaijo + obsidian + stats แล้วรวมผลสำหรับ wizard ──
        if mode == "report-gather":
            import concurrent.futures as _cf
            from src.agents.thaijo_agent import (
                _extract_search_payload,
                fetch_thaijo_articles,
                _articles_to_text,
            )
            from src.agents.obsidian_fullcontext import run_obsidian_ask_fullcontext
            from src.domains import DOMAINS as _DOMAINS

            api_key = os.getenv("GEMINI_API_KEY", "")
            report_title = prompt

            # ── Guard: ถามจังหวัดนอกเขตสุขภาพที่ 10 → แจ้งเตือนตรง ๆ ไม่แอบแทนข้อมูล ──
            # ระบบนี้มีข้อมูลเฉพาะ 5 จังหวัดเขต 10 (อุบลฯ ศรีสะเกษ ยโสธร อำนาจเจริญ มุกดาหาร)
            # ทั้งใน SQL accident และคลังความรู้ Obsidian — ถ้าผู้ใช้ถามกาฬสินธุ์/ขอนแก่น ฯลฯ
            # เดิมระบบจะคืนข้อมูลทั้งเขต 10 มาแทนเงียบ ๆ (เพราะ extract province คืน "")
            from src.tools.accident_chat_sql import (
                detect_out_of_zone10_provinces,
                detect_zone10_provinces,
                ZONE10_PROVINCES as _Z10,
            )
            _out = detect_out_of_zone10_provinces(prompt)
            _inz = detect_zone10_provinces(prompt)
            if _out and not _inz:
                _out_label = ", ".join(_out)
                _prov_list = "\n".join(f"  • {p}" for p in _Z10)
                warn = (
                    f"⚠️ ไม่มีข้อมูลจังหวัด {_out_label} ในระบบ\n\n"
                    f"ระบบนี้ครอบคลุมเฉพาะ **เขตสุขภาพที่ 10** ซึ่งมี 5 จังหวัด:\n"
                    f"{_prov_list}\n\n"
                    f"ทั้งข้อมูลสถิติอุบัติเหตุ (SQL) และคลังความรู้สุขภาพ (Obsidian) "
                    f"มีเฉพาะ 5 จังหวัดข้างต้น จึงไม่สามารถสร้างรายงานของจังหวัด "
                    f"{_out_label} ได้\n\n"
                    f"หากต้องการ ลองถามใหม่โดยระบุจังหวัดในเขตสุขภาพที่ 10 ครับ"
                )
                put({"type": "text_stream_start", "articleCount": 0})
                for i in range(0, len(warn), 200):
                    put({"type": "text_chunk", "text": warn[i:i + 200]})
                put({
                    "type": "final",
                    "message": warn,
                    "textResult": warn,
                    "articlesText": "",
                    "reportTitle": report_title,
                    "articleCount": 0,
                    "agentSteps": [],
                })
                return

            # ── ยิง 3 agent พร้อมกัน (parallel) ──────────────────────────────

            # ── ThaiJo worker — wrapper queue forwards all steps real-time ──────
            thaijo_result: dict = {}

            class _ThaijoQ:
                _FORWARD = {"agent_start", "agent_done", "crew_plan"}
                async def put(self, ev: Any) -> None:  # type: ignore[override]
                    if not isinstance(ev, dict):
                        return
                    ev_type = ev.get("type", "")
                    if ev_type == "final":
                        thaijo_result["articles_text"] = ev.get("articlesText", "")
                        thaijo_result["article_count"] = ev.get("articleCount", 0)
                        thaijo_result["term"]          = ev.get("reportTitle", prompt)
                        thaijo_result["full_text"]     = ev.get("textResult", "")
                    elif ev_type in self._FORWARD:
                        await queue.put(ev)

            def _worker_thaijo() -> None:
                run_thaijo_pipeline(prompt=prompt, queue=_ThaijoQ(), loop=loop)

            # ── PubMed worker — wrapper queue forwards all steps real-time ──────
            pubmed_result: dict = {}

            class _PubmedQ:
                _FORWARD = {"agent_start", "agent_done"}
                async def put(self, ev: Any) -> None:  # type: ignore[override]
                    if not isinstance(ev, dict):
                        return
                    ev_type = ev.get("type", "")
                    if ev_type == "final":
                        pubmed_result["articles_text"] = ev.get("articlesText", "")
                        pubmed_result["article_count"] = ev.get("articleCount", 0)
                        pubmed_result["full_text"]     = ev.get("textResult", "")
                    elif ev_type in self._FORWARD:
                        await queue.put(ev)

            def _worker_pubmed() -> None:
                from src.agents.pubmed_agent import run_pubmed_pipeline
                run_pubmed_pipeline(prompt=prompt, queue=_PubmedQ(), loop=loop)

            # ── Obsidian worker — 2 agent steps แสดง real-time ──────────────────
            obsidian_result: dict = {}
            def _worker_obsidian() -> None:
                put({"type": "agent_start", "step": "obsidian_search",
                     "agentName": "Obsidian Knowledge Searcher"})
                put({"type": "agent_start", "step": "obsidian_answer",
                     "agentName": "Health Knowledge Answer Writer"})
                # ⚠️ เหตุผลเดียวกับ obsidian mode ด้านบน — ต้องตรวจจับจังหวัดเอง
                # ไม่งั้นโหลดทั้ง vault แล้วเสี่ยงชน Gemini context window limit
                obs = run_obsidian_ask_fullcontext(prompt, detect_province_from_prompt(prompt) or "", "health_region_10")
                note_titles = ", ".join(n.title for n in obs.notes_referenced[:3]) if obs.notes_referenced else "ไม่พบ notes"
                put({"type": "agent_done", "step": "obsidian_search",
                     "agentName": "Obsidian Knowledge Searcher",
                     "result": "ค้นหาข้อมูลจาก Vault สำเร็จ",
                     "reasoning": f"ค้นหาใน vault health_region_10 ด้วยคำถาม: \"{prompt[:100]}\" — พบ notes ที่เกี่ยวข้อง: {note_titles}"})
                put({"type": "agent_done", "step": "obsidian_answer",
                     "agentName": "Health Knowledge Answer Writer",
                     "result": f"พบ {len(obs.notes_referenced)} notes ในคลังความรู้",
                     "reasoning": obs.content[:400] if obs.content else "ไม่พบข้อมูลในคลังความรู้"})
                obsidian_result["content"] = obs.content

            # ── Stats worker — forward events real-time ผ่าน wrapper queue ────
            stats_final_holder: dict = {}

            class _StatsQ:
                _FORWARD = {"agent_start", "agent_done", "crew_plan"}
                async def put(self, ev: Any) -> None:  # type: ignore[override]
                    if not isinstance(ev, dict):
                        return
                    ev_type = ev.get("type", "")
                    if ev_type == "final":
                        stats_final_holder["msg"] = ev.get("message", "")
                    elif ev_type in self._FORWARD:
                        await queue.put(ev)

            # ── Tavily worker — ค้นหาข้อมูลจากอินเทอร์เน็ตเพิ่มเติม ─────────────────
            tavily_result_holder: dict = {}

            class _TavilyQ:
                _FORWARD = {"agent_start", "agent_done"}
                async def put(self, ev: Any) -> None:  # type: ignore[override]
                    if not isinstance(ev, dict):
                        return
                    ev_type = ev.get("type", "")
                    if ev_type == "final":
                        tavily_result_holder["msg"] = ev.get("message", "")
                    elif ev_type == "agent_done" and ev.get("step") == "search":
                        # ⚠️ เก็บผลดิบ (title/url/summary แยกรายการ) จาก Search Agent ไว้ด้วย —
                        # ใช้จัดรูปแบบเป็น per-source block ให้ report generator อ้างอิงแยกทีละ
                        # แหล่งได้ (เหมือน ThaiJo/PubMed) แทนที่จะเห็นแค่ narrative ก้อนเดียวจาก
                        # Answer Writer ที่มักถูกอ้างอิงรวมเป็น 1 reference เท่านั้น — ดูคอมเมนต์
                        # ที่ report_source_parts ด้านล่าง
                        tavily_result_holder["raw_data"] = ev.get("result", "")
                        await queue.put(ev)
                    elif ev_type in self._FORWARD:
                        await queue.put(ev)

            def _worker_tavily() -> None:
                from src.agents.tavily_pipeline import run_tavily_pipeline
                run_tavily_pipeline(
                    prompt=prompt, queue=_TavilyQ(), loop=loop,
                    session_id="", history_section=history_section,
                )

            def _extract_province_from_prompt(text: str) -> str:
                """ดึงชื่อจังหวัดเขต 10 จาก prompt — คืน '' ถ้าไม่พบ"""
                mapping = {
                    "อุบล": "อุบลราชธานี", "อุบลราชธานี": "อุบลราชธานี",
                    "ศรีสะเกษ": "ศรีสะเกษ",
                    "ยโสธร": "ยโสธร",
                    "อำนาจเจริญ": "อำนาจเจริญ",
                    "มุกดาหาร": "มุกดาหาร",
                }
                for kw, full in mapping.items():
                    if kw in text:
                        return full
                return ""  # ทุกจังหวัดเขต 10

            def _worker_stats() -> None:
                put({"type": "agent_start", "step": "stats_gather", "agentName": "Stats Analyst"})

                # ⚠️ อุบัติเหตุทางถนน (d1) เก็บใน PostgreSQL ไม่ใช่ CSV/MinIO (ดูคอมเมนต์
                # _CSV_DOMAIN_CODES ใน router.py: "d1=PostgreSQL") — ต้อง redirect ไปยัง
                # accident pipeline เหมือนที่ mode == "stats" ทำไว้แล้ว (ดูบรรทัด ~109)
                # ไม่อย่างนั้น run_multi_pipeline จะถูกบังคับให้ค้นเฉพาะใน d2/d3/d4
                # (ไม่มีข้อมูลอุบัติเหตุอยู่เลยสักไฟล์) แล้วสุ่มเลือก CSV ผิด domain มาแทน
                # (เช่น สุขภาพจิต/โภชนาการ) — ตรงกับปัญหาที่ผู้ใช้รายงานว่าถามอุบัติเหตุ
                # แล้วระบบไม่ไปค้นหา domain อุบัติเหตุเลย
                #
                # ⚠️ ใช้ is_accident_question() ไม่ใช่ _has_accident_signal() เฉย ๆ —
                # ตัวหลังเช็คแค่ keyword ตรง ๆ (พลาดได้ถ้าพิมพ์ผิด/ใช้คำพ้องที่ list ไม่มี
                # เช่น "อุบัเหตุ" สะกดผิด หรือ "ความปลอดภัยทางถนน" ที่ไม่มีคำว่าอุบัติเหตุ)
                # is_accident_question() เช็ค keyword ก่อน (เร็ว) แล้วถ้า keyword ไม่เจอ
                # จึงให้ LLM ช่วยตัดสินใจอีกที — เหมือนที่ mode == "stats" ทำอยู่แล้ว
                if is_accident_question(prompt, history_context):
                    # ── เรียก SQL โดยตรง ไม่ผ่าน CrewAI/LLM ────────────────
                    # (LLM ล้มเหลวด้วย "None or empty" เมื่อ tool output รวมใหญ่เกิน)
                    from src.tools.accident_chat_sql import (
                        _query_kpi_trend,
                        _query_province_executive_summary,
                        _query_hotspot_roads,
                    )
                    province = _extract_province_from_prompt(prompt)
                    parts = []
                    try:
                        parts.append(_query_kpi_trend(province, 2021, 2025))
                    except Exception:
                        pass
                    try:
                        parts.append(_query_province_executive_summary(province, 2024))
                    except Exception:
                        pass
                    try:
                        parts.append(_query_hotspot_roads(province, 5, 2021, 2025))
                    except Exception:
                        pass
                    if parts:
                        stats_final_holder["msg"] = "\n\n".join(parts)
                        put({"type": "agent_done", "step": "stats_gather", "agentName": "Stats Analyst",
                             "result": "ดึงข้อมูลสถิติอุบัติเหตุทางถนนสำเร็จ (SQL โดยตรง)"})
                    else:
                        put({"type": "agent_done", "step": "stats_gather", "agentName": "Stats Analyst",
                             "result": "ไม่สามารถดึงข้อมูลสถิติได้ในขณะนี้"})
                    return

                csv_domains = [_DOMAINS["d2"], _DOMAINS["d3"], _DOMAINS["d4"]]
                run_multi_pipeline(
                    prompt=prompt, queue=_StatsQ(), loop=loop,
                    domains=csv_domains, history_context=history_context,
                    history_section=history_section, session_id="",
                )

            # ── รัน 5 worker พร้อมกัน (parallel) แล้วรอให้ครบทุกตัว ──────────────
            # ⚠️ เดิมโค้ดจบแค่ตรงนี้ (return ทันที) — worker ทั้ง 4 ตัวข้างบนถูก
            # define ไว้แต่ไม่เคยถูกเรียกเลย ทำให้ "สร้างรายงาน" ไม่ทำอะไรเลย
            # ปิด stream ทันทีแบบไม่มี event ใด ๆ ส่งกลับ — ฝั่ง frontend เห็นเป็น
            # "กำลังประมวลผล" ค้างตลอดไป (ไม่มี final/result/error ให้ resolve)
            put({"type": "text_stream_start", "articleCount": 0})
            with _cf.ThreadPoolExecutor(max_workers=5) as ex:
                # ⚠️ ทยอยเริ่มแต่ละ worker ห่างกัน ~1.5s แทนที่จะยิง LLM ทั้ง 5 ตัว
                # พร้อมกันเป๊ะในวินาทีเดียว — ลดโอกาสชน 429 quota-per-minute
                # (input token) ตอนมีหลาย request วิ่งพร้อมกันอยู่แล้ว โดยแทบไม่
                # กระทบเวลารวม เพราะยังรันซ้อนกัน (concurrent) อยู่เหมือนเดิม
                futures = {}
                for worker, name in (
                    (_worker_obsidian, "obsidian"),
                    (_worker_stats, "stats"),
                    (_worker_thaijo, "thaijo"),
                    (_worker_tavily, "tavily"),
                    (_worker_pubmed, "pubmed"),
                ):
                    futures[ex.submit(worker)] = name
                    time.sleep(1.5)
                for fut in _cf.as_completed(futures):
                    name = futures[fut]
                    try:
                        fut.result()
                    except Exception as exc:
                        put({"type": "agent_done", "step": name, "agentName": name, "result": f"ผิดพลาด: {exc}"})

            # ── รวมผลลัพธ์จากทั้ง 5 แหล่งเป็นรายงานเดียว ──────────────────────────
            sections = []
            if obsidian_result.get("content"):
                sections.append(f"## คลังความรู้สุขภาพ เขต 10\n\n{obsidian_result['content']}")
            if stats_final_holder.get("msg"):
                sections.append(f"## สถิติสาธารณสุข\n\n{stats_final_holder['msg']}")
            if thaijo_result.get("full_text"):
                sections.append(f"## งานวิจัยที่เกี่ยวข้อง (ThaiJo)\n\n{thaijo_result['full_text']}")
            if pubmed_result.get("full_text"):
                sections.append(f"## งานวิจัยทางการแพทย์ (PubMed)\n\n{pubmed_result['full_text']}")
            if tavily_result_holder.get("msg"):
                sections.append(f"## ข้อมูลจากอินเทอร์เน็ต\n\n{tavily_result_holder['msg']}")

            combined_text = "\n\n---\n\n".join(sections) if sections else (
                "ไม่พบข้อมูลที่เกี่ยวข้องจากแหล่งข้อมูลใดเลย (สถิติ/คลังความรู้/งานวิจัย/PubMed/เว็บ) "
                "ลองระบุคำถามให้เจาะจงมากขึ้น เช่น ระบุจังหวัดหรือหัวข้อสุขภาพที่ต้องการ"
            )

            for i in range(0, len(combined_text), 400):
                put({"type": "text_chunk", "text": combined_text[i:i + 400]})

            if session_id:
                append_history(session_id, "assistant", combined_text)

            # ── รวม "วัตถุดิบ" สำหรับ report generator ────────────────────────────
            # ⚠️ เดิม articlesText = เฉพาะบทความ ThaiJo → report generator มองไม่เห็น
            # เนื้อหา+แหล่งอ้างอิงจาก Tavily Research เลย ทำให้ "อ้างอิงของ research
            # ไม่ถูกนำไปทำอ้างอิงในรายงาน" — ต้องแนบเนื้อหา Tavily (ซึ่งมี "แหล่งอ้างอิง"
            # ต่อท้ายอยู่แล้ว) เข้าไปด้วย เพื่อให้แหล่งอ้างอิงไหลต่อเข้ารายงานฉบับจริง
            report_source_parts = []
            if thaijo_result.get("articles_text"):
                report_source_parts.append(thaijo_result["articles_text"])
            if pubmed_result.get("articles_text"):
                report_source_parts.append(
                    "--- บทความวิจัยทางการแพทย์จาก PubMed ---\n"
                    + pubmed_result["articles_text"]
                )
            if tavily_result_holder.get("msg"):
                # ⚠️ ใช้บล็อกแยกต่อแหล่ง (จาก raw_data) แทน narrative ก้อนเดียว — ดูเหตุผล
                # เต็มใน _tavily_raw_to_articles_text ด้านบน fallback เป็น narrative เดิม
                # ถ้า parse ไม่ได้ (เช่น raw_data ว่าง/รูปแบบเปลี่ยน) กันไม่ให้ข้อมูลหาย
                tavily_articles_text = _tavily_raw_to_articles_text(tavily_result_holder.get("raw_data", ""))
                report_source_parts.append(
                    "--- ข้อมูลค้นคว้าเพิ่มเติมจากอินเทอร์เน็ต (Tavily Research) ---\n"
                    + (tavily_articles_text or tavily_result_holder["msg"])
                )
            combined_articles_text = "\n\n".join(report_source_parts)

            put({
                "type": "final",
                "message": combined_text,
                "textResult": combined_text,
                "articlesText": combined_articles_text,
                "articleCount": thaijo_result.get("article_count", 0) + pubmed_result.get("article_count", 0),
                "reportTitle": report_title,
                "agentSteps": [],
            })
            return

        # ── Normal mode: multi-domain aware routing ───────────────────────────

        # STEP 0: Router (detects single vs multi-domain)
        put({"type": "agent_start", "step": "router", "agentName": "Router Agent"})
        domains, is_multi = route_multi_domain(prompt, history_context)
        domain = domains[0]
        domain_names_th = " + ".join(d.name_th for d in domains)
        domain_names_en = " + ".join(d.name_en for d in domains)
        put({
            "type": "agent_done",
            "step": "router",
            "agentName": "Router Agent",
            "result": f"{'Multi-Domain' if is_multi else 'Domain'}: {domain_names_th}",
            "domain": {
                "code": "multi" if is_multi else domain.code,
                "nameTh": domain_names_th,
                "nameEn": domain_names_en,
            },
        })

        # ── Accident domain in normal mode → redirect to Obsidian ──────────
        # ผู้ใช้ไม่ได้เลือก stats tool → ไม่ควรใช้ Accident SQL Agent
        # ให้ตอบจาก Obsidian Knowledge Vault แทน (มีข้อมูลนโยบายอุบัติเหตุ)
        if domain.code == "d1":
            from src.agents.router import DOMAINS as _ROUTER_DOMAINS
            domain = domains[0] = _ROUTER_DOMAINS.get("obsidian", domain)

        # ── Obsidian Knowledge Vault pipeline ────────────────────────────────
        if domain.code == "obsidian":
            put({"type": "agent_start", "step": "obsidian_search", "agentName": "Obsidian Knowledge Searcher"})
            from src.agents.obsidian_fullcontext import run_obsidian_ask_fullcontext
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                obs_result = ex.submit(
                    run_obsidian_ask_fullcontext,
                    prompt,
                    # ⚠️ เหตุผลเดียวกับ mode == "obsidian" ด้านบน — ตรวจจับจังหวัดเอง
                    # ไม่งั้นโหลดทั้ง vault แล้วเสี่ยงชน Gemini context window limit
                    detect_province_from_prompt(prompt) or "",
                    "health_region_10",
                    history_context=history_context,
                ).result()
            put({"type": "agent_done", "step": "obsidian_search", "agentName": "Obsidian Knowledge Searcher",
                 "result": f"พบ {len(obs_result.notes_referenced)} notes"})
            if session_id:
                append_history(session_id, "assistant", obs_result.content)
            put({"type": "result", "content": obs_result.content,
                 "notesReferenced": [n.model_dump() for n in obs_result.notes_referenced],
                 "followUps": obs_result.follow_ups,
                 "domain": {"code": "obsidian", "nameTh": "คลังความรู้สุขภาพ เขต 10", "nameEn": "Obsidian Knowledge Vault"}})
            return

        # ── ThaiJo Research pipeline ──────────────────────────────────────────
        if domain.code == "dt" or mode == "thaijo":
            run_thaijo_pipeline(prompt=prompt, queue=queue, loop=loop, session_id=session_id,
                                history_context=history_context)
            return

        # mode=multi forces multi-domain pipeline regardless of router decision
        if mode == "multi":
            is_multi = True

        # STEP 2+: Multi-domain or single-domain pipeline
        if is_multi:
            run_multi_pipeline(
                prompt=prompt,
                queue=queue,
                loop=loop,
                domains=domains,
                history_context=history_context,
                history_section=history_section,
                session_id=session_id,
                reasoning=reasoning,
                vault_ctx=vault_ctx,
            )
        else:
            run_pipeline(
                prompt=prompt,
                queue=queue,
                loop=loop,
                domain=domain,
                history_context=history_context,
                history_section=history_section,
                session_id=session_id,
                reasoning=reasoning,
                vault_ctx=vault_ctx,
            )

    except Exception as exc:
        put({"type": "error", "message": str(exc)})
    finally:
        finish()
        _AI_SEMAPHORE.release()


async def _handle_analyze(request: AnalyzeRequest) -> StreamingResponse:
    if not _AI_SEMAPHORE.acquire(blocking=False):
        async def busy_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'ระบบกำลังประมวลผลเต็มความสามารถ กรุณารอสักครู่แล้วลองใหม่'}, ensure_ascii=False)}\n\n"
        return StreamingResponse(busy_stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    client_history = (
        [{"role": m.role, "text": m.text} for m in request.history]
        if request.history else None
    )

    thread = threading.Thread(
        target=_orchestrate,
        args=(request.prompt, queue, loop),
        kwargs={"session_id": request.sessionId, "client_history": client_history, "mode": request.mode},
        daemon=True,
    )
    thread.start()

    async def stream():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    return await _handle_analyze(request)


@router.post("/api/chat")
async def chat(request: AnalyzeRequest):
    return await _handle_analyze(request)
