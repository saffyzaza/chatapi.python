"""Accident Chat API router — conversational RTI policy Q&A for Zone 10."""
import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from src.schemas.accident_chat import (
    AccidentChatRequest,
    AccidentChatQuickRequest,
    AccidentChatResponse,
)
from src.agents.progress import create_progress_queue, remove_progress_queue
from src.tools.zone10_accident import ZONE10_PROVINCES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/accident-chat", tags=["accident-chat"])


@router.post("/ask", response_model=AccidentChatResponse)
async def ask_accident_question(request: AccidentChatRequest):
    """Run the full 2-agent Accident Chat pipeline.

    Pipeline: AccidentSQLAgent → AccidentAnswerAgent
    Expected runtime: 30–120 seconds.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="คำถามต้องไม่ว่างเปล่า")

    logger.info(
        "[accident-chat] ask: %s | province=%s year=%s-%s",
        request.question[:80], request.province, request.year_start, request.year_end,
    )

    from src.agents.accident_chat_orchestrator import run_accident_chat
    try:
        result = run_accident_chat(
            question=request.question,
            province=request.province,
            district=request.district,
            year_start=request.year_start,
            year_end=request.year_end,
        )
        return JSONResponse(content=result.model_dump())
    except Exception as exc:
        logger.exception("[accident-chat] ask error: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc)})


@router.post("/ask/stream")
async def ask_accident_stream(request: AccidentChatRequest):
    """Stream the Accident Chat pipeline via Server-Sent Events."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="คำถามต้องไม่ว่างเปล่า")

    request_id = str(uuid.uuid4())
    q = create_progress_queue(request_id)

    async def event_stream():
        yield f"data: {json.dumps({'type': 'start', 'request_id': request_id, 'pipeline': 'accident_chat'}, ensure_ascii=False)}\n\n"

        from src.agents.accident_chat_orchestrator import run_accident_chat_with_progress
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            None,
            lambda: run_accident_chat_with_progress(
                question=request.question,
                province=request.province,
                district=request.district,
                year_start=request.year_start,
                year_end=request.year_end,
                request_id=request_id,
            ),
        )

        while not future.done():
            try:
                import queue as _queue
                event = q.get_nowait()
                payload = {
                    "type": "progress",
                    "agent_name": event.agent_name,
                    "agent_icon": event.agent_icon,
                    "status": event.status,
                    "message": event.message,
                    "elapsed_seconds": event.elapsed_seconds,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except _queue.Empty:
                await asyncio.sleep(0.3)

        import queue as _queue
        while True:
            try:
                event = q.get_nowait()
                payload = {
                    "type": "progress",
                    "agent_name": event.agent_name,
                    "agent_icon": event.agent_icon,
                    "status": event.status,
                    "message": event.message,
                    "elapsed_seconds": event.elapsed_seconds,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except _queue.Empty:
                break

        try:
            result = await future
            yield f"data: {json.dumps({'type': 'result', 'data': result.model_dump()}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
        finally:
            remove_progress_queue(request_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/quick")
async def quick_sql_data(request: AccidentChatQuickRequest):
    """Return raw SQL data from a specific tool — no LLM, fast (<5s)."""
    from src.tools.accident_chat_sql import (
        _query_hotspot_roads, _query_district_road_comparison, _query_fatal_timeband,
        _query_weather_accident_stats, _query_seasonal_comparison, _query_weekend_vs_weekday,
        _query_monthly_vehicle_pattern, _query_late_night_vehicles, _query_kpi_trend,
        _query_serious_injury_ratio, _query_top_cause_shift, _query_district_death_vs_accident,
        _query_district_summary, _query_road_district_breakdown, _query_province_executive_summary,
        _person_data_unavailable,
    )

    tool_map = {
        "hotspot_roads": lambda: _query_hotspot_roads(request.province, request.top_n, request.year_start, request.year_end),
        "district_road_comparison": lambda: _query_district_road_comparison(request.province, request.year_start, request.year_end),
        "fatal_timeband": lambda: _query_fatal_timeband(request.province, request.year_start, request.year_end),
        "weather_accident_stats": lambda: _query_weather_accident_stats(request.province, request.year_start, request.year_end),
        "behavior_stats": lambda: _person_data_unavailable(request.topic),
        "seasonal_comparison": lambda: _query_seasonal_comparison(request.province, request.month1, request.month2, request.year_start, request.year_end),
        "weekend_vs_weekday": lambda: _query_weekend_vs_weekday(request.province, request.year_start, request.year_end),
        "monthly_vehicle_pattern": lambda: _query_monthly_vehicle_pattern(request.province, request.year_start, request.year_end),
        "late_night_vehicles": lambda: _query_late_night_vehicles(request.province, request.year_start, request.year_end),
        "kpi_trend": lambda: _query_kpi_trend(request.province, request.year_start, request.year_end),
        "serious_injury_ratio": lambda: _query_serious_injury_ratio(request.province, request.year),
        "top_cause_shift": lambda: _query_top_cause_shift(request.province, request.year1, request.year2),
        "district_death_vs_accident": lambda: _query_district_death_vs_accident(request.province, request.year_start, request.year_end),
        "district_summary": lambda: _query_district_summary(request.province, request.district, request.year_start, request.year_end),
        "road_district_breakdown": lambda: _query_road_district_breakdown(request.province, request.road_name or "", request.year_start, request.year_end),
        "province_executive_summary": lambda: _query_province_executive_summary(request.province, request.year),
    }

    fn = tool_map.get(request.tool)
    if not fn:
        raise HTTPException(
            status_code=400,
            detail={"error": "unknown_tool", "valid_tools": list(tool_map.keys())},
        )

    try:
        data = fn()
        return JSONResponse(content={"tool": request.tool, "province": request.province or "Zone 10", "data": data})
    except Exception as exc:
        logger.exception("[accident-chat] quick error: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc)})


@router.get("/provinces")
async def list_provinces():
    return JSONResponse(content={"provinces": list(ZONE10_PROVINCES)})


@router.get("/districts")
async def list_districts(province: str = Query(default="")):
    from src.db.pool import query_db
    try:
        if province.strip():
            rows = query_db(
                """SELECT DISTINCT province_name, district_name
                   FROM dim_geography
                   WHERE province_name ILIKE %s AND district_name IS NOT NULL AND LENGTH(district_name) > 0
                   ORDER BY district_name""",
                (f"%{province.strip()}%",),
            )
        else:
            rows = query_db(
                """SELECT DISTINCT province_name, district_name
                   FROM dim_geography
                   WHERE province_name = ANY(%s) AND district_name IS NOT NULL AND LENGTH(district_name) > 0
                   ORDER BY province_name, district_name""",
                (list(ZONE10_PROVINCES),),
            )
        grouped: dict[str, list[str]] = {}
        for r in rows:
            prov = r["province_name"]
            dist = r["district_name"]
            if prov not in grouped:
                grouped[prov] = []
            if dist not in grouped[prov]:
                grouped[prov].append(dist)
        return JSONResponse(content={"grouped": grouped, "province": province})
    except Exception as exc:
        logger.exception("[accident-chat] districts error: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc)})


@router.get("/sample-questions")
async def sample_questions():
    return JSONResponse(content={
        "groups": [
            {
                "id": "hotspot",
                "label": "กลุ่มที่ 1: Hotspot & Engineering",
                "questions": [
                    {"id": "Q1", "text": "ถนนเส้นใดในจังหวัดอุบลราชธานี มีคะแนนจุดเสี่ยง (Hotspot Score) สูงที่สุด 10 อันดับแรก?"},
                    {"id": "Q2", "text": "อำเภอใดในเขตสุขภาพที่ 10 มีผู้เสียชีวิตบนถนนสายรองมากกว่าถนนสายหลัก?"},
                    {"id": "Q3", "text": "จุดเสี่ยงที่มีระดับความรุนแรงถึงขั้นเสียชีวิต มักเกิดในช่วงเวลาใด?"},
                    {"id": "Q4", "text": "สภาพอากาศและลักษณะการเกิดเหตุแบบใดที่ทำให้เกิดผู้บาดเจ็บสาหัสมากที่สุด?"},
                    {"id": "Q5", "text": "สภาพการเกิดเหตุแบบใดสัมพันธ์กับอุบัติเหตุรุนแรงที่สุดในช่วงฤดูฝน?"},
                ],
            },
            {
                "id": "seasonal",
                "label": "กลุ่มที่ 2: Temporal & Seasonal",
                "questions": [
                    {"id": "Q11", "text": "เปรียบเทียบสาเหตุหลักของอุบัติเหตุในเดือนเมษายน (สงกรานต์) กับเดือนพฤศจิกายน?"},
                    {"id": "Q12", "text": "ช่วงเวลาเสี่ยงในวันหยุดเสาร์-อาทิตย์ แตกต่างจากวันธรรมดาอย่างไร?"},
                    {"id": "Q13", "text": "เดือนใดที่มีสัดส่วนอุบัติเหตุจากรถบรรทุก/รถเกษตรพุ่งสูงขึ้นผิดปกติ?"},
                    {"id": "Q15", "text": "ในช่วงหลังเที่ยงคืน (00:00-05:00) ยานพาหนะประเภทใดมีผู้เสียชีวิตมากที่สุด?"},
                ],
            },
            {
                "id": "kpi",
                "label": "กลุ่มที่ 3: KPI & Monitoring",
                "questions": [
                    {"id": "Q16", "text": "แนวโน้มจำนวนผู้เสียชีวิตของจังหวัดอุบลราชธานีในรอบ 3 ปีที่ผ่านมา ลดลงต่อเนื่องหรือไม่?"},
                    {"id": "Q17", "text": "จังหวัดใดในเขตสุขภาพที่ 10 มีอัตราส่วนผู้บาดเจ็บสาหัสต่ออุบัติเหตุ 1 ครั้ง สูงที่สุด?"},
                    {"id": "Q18", "text": "เปรียบเทียบปี 2566 และ 2567 สาเหตุการตายอันดับ 1 เปลี่ยนแปลงไปหรือไม่?"},
                    {"id": "Q20", "text": "สรุปภาพรวม KPI ปีล่าสุดของจังหวัดมุกดาหาร?"},
                ],
            },
        ]
    })
