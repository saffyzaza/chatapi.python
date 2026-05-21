"""Zone 10 Accident Policy API router."""
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from src.schemas.accident_policy import (
    AccidentPolicyRequest,
    AccidentPolicyResponse,
    Zone10DataResponse,
)
from src.tools.zone10_accident import ZONE10_PROVINCES
from src.agents.accident_policy_orchestrator import run_zone10_analysis, run_zone10_data_only

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/accident-policy", tags=["accident-policy"])


@router.get("/zone10/data", response_model=Zone10DataResponse)
async def get_zone10_data(
    provinces: str = Query(default="", description="Comma-separated Zone 10 province names. Empty = all 5.")
):
    """Return raw SQL query results for all 7 policy questions — no LLM, fast (<5s)."""
    province_list = (
        [p.strip() for p in provinces.split(",") if p.strip()]
        if provinces.strip()
        else list(ZONE10_PROVINCES)
    )
    invalid = [p for p in province_list if p not in ZONE10_PROVINCES]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_provinces", "message": f"จังหวัดไม่ถูกต้อง: {invalid}", "valid": list(ZONE10_PROVINCES)},
        )
    try:
        result = run_zone10_data_only(province_list)
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("[accident-policy] /data error: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc)})


@router.post("/zone10", response_model=AccidentPolicyResponse)
async def create_zone10_policy_brief(request: AccidentPolicyRequest):
    """Run the full 3-agent Zone 10 RTI policy analysis pipeline.

    Pipeline: Zone10SqlFetcher → Zone10PolicyAnalyst → Zone10ReportWriter
    Expected runtime: 2–5 minutes.
    """
    logger.info(
        "[accident-policy] provinces=%s questions=%s year_range=%s",
        request.provinces, request.questions, request.year_range,
    )
    try:
        result = run_zone10_analysis(
            provinces=request.provinces,
            questions=request.questions,
            year_range=request.year_range,
        )
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("[accident-policy] Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc)})
