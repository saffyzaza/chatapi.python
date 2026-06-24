"""Database Explorer — read-only introspection of the PostgreSQL database."""
import math
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.db.pool import query_db

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/db", tags=["Database Explorer"])


# ── helpers ───────────────────────────────────────────────────────────────────

def _validate_table(table_name: str) -> None:
    """Raise 404 if table_name is not a real public table."""
    rows = query_db(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type   = 'BASE TABLE'
          AND table_name   = %s
        """,
        (table_name,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")


def _validate_column(table_name: str, col_name: str) -> None:
    """Raise 400 if col_name is not a column of table_name."""
    rows = query_db(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name   = %s
          AND column_name  = %s
        """,
        (table_name, col_name),
    )
    if not rows:
        raise HTTPException(status_code=400, detail=f"Column '{col_name}' not found in '{table_name}'")


def _safe_cell(v: object) -> object:
    """Convert values that can't be JSON-serialised or are excessively long."""
    if v is None:
        return None
    s = str(v)
    # pgvector arrays come back as a string like "[0.123,0.456,…]"
    if s.startswith("[") and len(s) > 300:
        return "[…vector…]"
    if len(s) > 2000:
        return s[:2000] + "…"
    return v


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.get("/tables")
async def list_tables():
    """List all public tables with exact row counts."""
    tables = query_db(
        """
        SELECT table_name AS name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type   = 'BASE TABLE'
        ORDER BY table_name
        """
    )
    if not tables:
        return {"tables": []}

    # Single UNION ALL query — table names from information_schema are safe to double-quote
    parts = [
        f"SELECT '{t['name'].replace(chr(39), chr(39)+chr(39))}' AS name,"
        f" COUNT(*)::bigint AS row_count FROM \"{t['name']}\""
        for t in tables
    ]
    counts = query_db(" UNION ALL ".join(parts))
    count_map = {r["name"]: r["row_count"] for r in counts}

    return {
        "tables": [
            {"name": t["name"], "row_count": count_map.get(t["name"], 0)}
            for t in tables
        ]
    }


@router.get("/tables/{table_name}/columns")
async def list_columns(table_name: str):
    """List columns of a table in ordinal order."""
    _validate_table(table_name)
    rows = query_db(
        """
        SELECT
            column_name     AS name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name   = %s
        ORDER BY ordinal_position
        """,
        (table_name,),
    )
    return {"columns": rows}


@router.get("/tables/{table_name}/rows")
async def get_rows(
    table_name: str,
    page:        int           = Query(1,  ge=1),
    page_size:   int           = Query(50, ge=1, le=500),
    filter_col:  Optional[str] = Query(None),
    filter_val:  Optional[str] = Query(None),
):
    """Return paginated rows, with an optional ILIKE text filter."""
    _validate_table(table_name)

    # Build WHERE clause
    where = ""
    params: list = []
    if filter_col and filter_val is not None:
        _validate_column(table_name, filter_col)
        where = f'WHERE CAST("{filter_col}" AS TEXT) ILIKE %s'
        params.append(f"%{filter_val}%")

    # Total count
    cnt_sql = f'SELECT COUNT(*) AS cnt FROM "{table_name}" {where}'
    total: int = (query_db(cnt_sql, params or None)[0] or {}).get("cnt", 0)  # type: ignore[arg-type]

    # Rows
    offset = (page - 1) * page_size
    row_sql = f'SELECT * FROM "{table_name}" {where} LIMIT %s OFFSET %s'
    raw_rows = query_db(row_sql, (params + [page_size, offset]) or None)  # type: ignore[arg-type]

    rows = [{k: _safe_cell(v) for k, v in row.items()} for row in raw_rows]

    return {
        "rows":        rows,
        "total":       total,
        "page":        page,
        "page_size":   page_size,
        "total_pages": max(1, math.ceil(total / page_size)),
    }
