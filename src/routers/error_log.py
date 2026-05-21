"""Error Log Router — REST API for viewing and analysing agent error logs.

Endpoints:
  GET  /api/errors              List recent errors (raw JSON)
  GET  /api/errors/summary      LLM-generated Thai analysis report
  GET  /api/errors/stats        Aggregated counts without LLM
  DELETE /api/errors            Delete all log files (clear logs)
"""
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from src.tools.error_logger import read_all_errors, aggregate_errors, clear_all_logs

router = APIRouter(tags=["error-logs"])


@router.get("/api/errors")
async def list_errors(
    days: int = Query(default=7, ge=1, le=90, description="จำนวนวันย้อนหลัง"),
    error_type: str = Query(default="", description="กรอง error type เช่น auth_error"),
    step: str = Query(default="", description="กรอง pipeline step เช่น code_gen"),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """Return raw error entries newest-first, with optional filters."""
    entries = read_all_errors(days=days)

    if error_type:
        entries = [e for e in entries if e.get("error_type") == error_type]
    if step:
        entries = [e for e in entries if e.get("step") == step]

    return {
        "total": len(entries),
        "returned": min(len(entries), limit),
        "filters": {"days": days, "error_type": error_type or None, "step": step or None},
        "entries": entries[:limit],
    }


@router.get("/api/errors/stats")
async def error_stats(
    days: int = Query(default=7, ge=1, le=90),
):
    """Return aggregated error counts without calling LLM."""
    entries = read_all_errors(days=days)
    agg = aggregate_errors(entries)
    return {"days": days, **agg}


@router.get("/api/errors/summary")
async def error_summary(
    days: int = Query(default=7, ge=1, le=30, description="จำนวนวันที่วิเคราะห์"),
):
    """Run Error Monitor Agent and return Thai markdown analysis report."""
    from src.agents.error_monitor_agent import run_error_monitor

    result = run_error_monitor(days=days)
    return {
        "days": days,
        "aggregate": result["aggregate"],
        "report": result["report"],
        "entry_count": len(result["entries"]),
    }


@router.delete("/api/errors")
async def clear_errors():
    """Delete all error log files (cannot be undone)."""
    deleted = clear_all_logs()
    return {"deleted_files": deleted, "message": f"ลบ log {deleted} ไฟล์เรียบร้อย"}
