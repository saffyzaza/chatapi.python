"""Analyze router — SSE streaming pipeline for health domain Q&A."""
import asyncio
import json
import os
import threading
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

router = APIRouter(tags=["analyze"])


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

        # ── Stats mode: force multi-domain CSV routing, AI selects domains ────────
        if mode == "stats":
            from src.domains import DOMAINS as _DOMAINS
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

        # ── Obsidian mode: forced Knowledge Vault routing ─────────────────────
        if mode == "obsidian":
            put({"type": "agent_start", "step": "obsidian_search", "agentName": "Obsidian Knowledge Searcher"})
            from src.agents.obsidian_fullcontext import run_obsidian_ask_fullcontext
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                obs_result = ex.submit(
                    run_obsidian_ask_fullcontext,
                    prompt,
                    "",
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

            # ── Obsidian worker — 2 agent steps แสดง real-time ──────────────────
            obsidian_result: dict = {}
            def _worker_obsidian() -> None:
                put({"type": "agent_start", "step": "obsidian_search",
                     "agentName": "Obsidian Knowledge Searcher"})
                put({"type": "agent_start", "step": "obsidian_answer",
                     "agentName": "Health Knowledge Answer Writer"})
                obs = run_obsidian_ask_fullcontext(prompt, "", "health_region_10")
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
                # accident pipeline เหมือนที่ mode == "stats" ทำไว้แล้ว (ดูบรรทัด ~75)
                # ไม่อย่างนั้น run_multi_pipeline จะถูกบังคับให้ค้นเฉพาะใน d2/d3/d4
                # (ไม่มีข้อมูลอุบัติเหตุอยู่เลยสักไฟล์) แล้วสุ่มเลือก CSV ผิด domain มาแทน
                # (เช่น สุขภาพจิต/โภชนาการ) — ตรงกับปัญหาที่ผู้ใช้รายงานว่าถามอุบัติเหตุ
                # แล้วระบบไม่ไปค้นหา domain อุบัติเหตุเลย
                if _has_accident_signal(prompt):
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
                put({"type": "agent_done", "step": "stats_gather", "agentName": "Stats Analyst",
                     "result": "วิเคราะห์สถิติสำเร็จ"})

            # ── ขั้นที่ 1: Stats (accident_chat_sql) เสร็จก่อน ──────────────────────────
            with _cf.ThreadPoolExecutor(max_workers=1) as pool:
                fs = pool.submit(_worker_stats)
                _cf.wait([fs], timeout=300)

            # ── ขั้นที่ 2: Thaijo + Obsidian + Tavily รันพร้อมกัน ────────────────────────
            with _cf.ThreadPoolExecutor(max_workers=3) as pool:
                ft = pool.submit(_worker_thaijo)
                fo = pool.submit(_worker_obsidian)
                fv = pool.submit(_worker_tavily)
                _cf.wait([ft, fo, fv], timeout=180)

            stats_final_msg  = stats_final_holder.get("msg", "")
            tavily_final_msg = tavily_result_holder.get("msg", "")

            # ── รวมผลลัพธ์ ────────────────────────────────────────────
            thaijo_raw       = thaijo_result.get("articles_text", "")
            thaijo_full_text = thaijo_result.get("full_text", "")
            total_articles   = thaijo_result.get("article_count", 0)
            term             = thaijo_result.get("term", prompt)
            obs_content      = obsidian_result.get("content", "")
            sep = "─" * 44 + "\n\n"

            all_parts: list[str] = []

            # ThaiJo — ใช้ full_text จาก pipeline (รวม summaries + insight แล้ว)
            full_display = thaijo_full_text if thaijo_full_text else (
                f"📚 บทความวิจัย ThaiJo — พบ {total_articles} บทความ\n\n"
            )
            if not full_display.endswith("\n"): full_display += "\n"
            if thaijo_raw:
                all_parts.append(
                    f"=== บทความวิจัย ThaiJo ({total_articles} บทความ) ===\n{thaijo_raw}"
                )

            # Obsidian — แสดง content เต็ม
            if obs_content:
                full_display += (
                    f"\n📖 คลังความรู้สุขภาพ เขต 10\n\n════════════════════════════════════════════\n\n"
                    f"{obs_content}\n\n{sep}"
                )
                all_parts.append(f"=== คลังความรู้สุขภาพ เขต 10 ===\n{obs_content}")

            # Stats — แสดงผลสถิติ
            if stats_final_msg:
                full_display += (
                    f"📊 ข้อมูลสถิติสาธารณสุข\n\n════════════════════════════════════════════\n\n"
                    f"{stats_final_msg}\n\n"
                )
                all_parts.append(f"=== ข้อมูลสถิติสาธารณสุข ===\n{stats_final_msg}")

            # Tavily — ผลการค้นหาจากอินเทอร์เน็ต
            if tavily_final_msg:
                full_display += (
                    f"🔍 ข้อมูลจากอินเทอร์เน็ต (Tavily Search)\n\n════════════════════════════════════════════\n\n"
                    f"{tavily_final_msg}\n\n"
                )
                all_parts.append(f"=== ข้อมูลจากอินเทอร์เน็ต ===\n{tavily_final_msg}")

            # ── Stream combined text to right pane ────────────────────────────
            put({"type": "text_stream_start", "articleCount": total_articles})
            chunk_size = 200
            for i in range(0, len(full_display), chunk_size):
                put({"type": "text_chunk", "text": full_display[i:i + chunk_size]})

            # ── Final event — enables wizard ──────────────────────────────────
            articles_text_combined = "\n\n".join(p for p in all_parts if p)
            put({
                "type":         "final",
                "message":      f"รวบรวมข้อมูลจาก 4 แหล่งสำเร็จ สำหรับ: {report_title}",
                "textResult":   full_display,
                "articlesText": articles_text_combined,
                "reportTitle":  report_title,
                "articleCount": total_articles,
                "agentSteps":   [],
            })
            return

        # ── Tavily mode: ผู้ใช้เลือก "ค้นหาทั่วไป" → ไป Tavily โดยตรงทันที ────
        if mode == "tavily":
            from src.agents.tavily_pipeline import run_tavily_pipeline
            run_tavily_pipeline(prompt=prompt, queue=queue, loop=loop,
                                session_id=session_id, history_section=history_section)
            return

        # ── ThaiJo mode: ผู้ใช้เลือก "วิจัย" → ไป ThaiJo โดยตรงทันที ──────
        if mode == "thaijo":
            run_thaijo_pipeline(prompt=prompt, queue=queue, loop=loop,
                                session_id=session_id, history_context=history_context)
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
                    "",   # province — let agent infer from question
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
