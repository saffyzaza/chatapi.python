"""Obsidian Knowledge Vault search tools — hybrid edition.

Search strategy (two-tier with automatic fallback):
  Tier 1 (preferred): chunk-based search + wikilink graph expansion
    - Searches obsidian_note_chunks with pg_trgm similarity()
    - Returns the matching *section* as snippet (not file start)
    - Expands results with one-hop wikilink neighbors
    - Requires migration 026 (obsidian_note_chunks + obsidian_note_links)
  Tier 2 (fallback): full-note LIKE search
    - Used when chunks table is empty (pre-026 migration)
    - Same behaviour as original implementation
"""
import json
import logging
import re

from crewai.tools import tool

from src.config import get_settings

logger = logging.getLogger("musya.tools.obsidian")
settings = get_settings()


# ── SQL helpers ────────────────────────────────────────────────────────────────

def _build_like_conditions(
    query: str, col: str = "content_stripped"
) -> tuple[list[str], list[str]]:
    """Build LIKE patterns for multi-term keyword search.

    Each term becomes a separate LIKE pattern.  Terms are OR-ed so that a
    chunk containing *any* of the query terms is surfaced.

    Args:
        query: Raw query string (split by whitespace).
        col:   Column expression to match against.

    Returns:
        (conditions, params) — SQL fragments and positional params.
    """
    terms = [t.strip() for t in query.split() if t.strip()]
    if not terms:
        terms = [query.strip()]

    conditions: list[str] = []
    params: list[str] = []
    for term in terms[:5]:
        pattern = f"%{term}%"
        conditions.append(f"{col} LIKE %s")
        params.append(pattern)
    return conditions, params


def _chunks_indexed(vault_id: str) -> bool:
    """Return True if obsidian_note_chunks has rows for this vault."""
    try:
        from src.db.pool import query_db
        rows = query_db(
            """
            SELECT 1 FROM obsidian_note_chunks c
            JOIN obsidian_notes n ON n.note_id = c.note_id
            WHERE n.vault_id = %s
            LIMIT 1
            """,
            (vault_id,),
        )
        return bool(rows)
    except Exception:
        return False


# ── Tier 1: Chunk-based hybrid search ─────────────────────────────────────────

def _search_chunks(
    query: str,
    province: str | None,
    district: str | None,
    tag: str | None,
    note_type: str | None,
    vault_id: str,
    top_k: int,
) -> list[dict]:
    """Search obsidian_note_chunks + expand with wikilink neighbors.

    Returns:
        List of result dicts ready for JSON serialisation.
        Matched results have a real similarity score (0–1).
        Graph-expanded neighbors have score=0.3 and matched_section=None.
    """
    from src.db.pool import query_db

    # ── Build WHERE clause for the chunk search ────────────────────────────
    where_parts = ["n.vault_id = %s"]
    params: list = [vault_id]

    # Keyword LIKE conditions on chunk_content (OR across terms)
    term_conds, term_params = _build_like_conditions(query, col="c.chunk_content")
    if term_conds:
        where_parts.append(f"({' OR '.join(term_conds)})")
        params.extend(term_params)

    if province:
        where_parts.append("n.province = %s")
        params.append(province)
    if district:
        where_parts.append("n.district = %s")
        params.append(district)
    if tag:
        where_parts.append("%s = ANY(n.tags)")
        params.append(tag)
    if note_type:
        where_parts.append("n.note_type = %s")
        params.append(note_type)

    where_sql = " AND ".join(where_parts)

    # ── Step 1: Best matching chunk per note ───────────────────────────────
    # Reserve at least 2 result slots for graph neighbors so they always appear
    # alongside direct matches.  With top_k=1 we still return 1 direct result.
    neighbor_reserve = max(2, top_k // 5)          # e.g. 2 for top_k≤10, 4 for top_k=20
    direct_limit     = max(1, top_k - neighbor_reserve)

    # DISTINCT ON picks the highest-scoring chunk for each note.
    chunk_sql = f"""
        SELECT DISTINCT ON (c.note_id)
            c.note_id,
            c.header                                  AS matched_section,
            SUBSTRING(c.chunk_content, 1, 600)        AS snippet,
            similarity(c.chunk_content, %s)           AS score
        FROM obsidian_note_chunks c
        JOIN obsidian_notes n ON n.note_id = c.note_id
        WHERE {where_sql}
        ORDER BY c.note_id, similarity(c.chunk_content, %s) DESC
        LIMIT %s
    """
    # similarity() params go before and inside ORDER BY
    chunk_params = [query] + params + [query, direct_limit]
    chunk_rows = query_db(chunk_sql, chunk_params)

    if not chunk_rows:
        return []

    matched_note_ids = [r["note_id"] for r in chunk_rows]

    # Compute dynamic neighbor discount: half the best direct match score.
    # Hardcoded 0.3 can EXCEED real trgm scores on long Thai chunks (pg_trgm
    # similarity dilutes with document length), so use a relative discount.
    direct_max = max((float(r.get("score") or 0) for r in chunk_rows), default=0.0)
    neighbor_score = max(round(direct_max * 0.5, 4), 0.01)

    # ── Step 2: One-hop wikilink expansion ────────────────────────────────
    # Fetch notes that are directly linked FROM the matched notes.
    # Only include if not already in matched set; only same vault.
    # Note-type priority: prefer report > policy > research > district.
    link_sql = """
        SELECT DISTINCT ON (n2.note_id)
            n2.note_id,
            NULL::text                                AS matched_section,
            SUBSTRING(n2.content_stripped, 1, 400)   AS snippet,
            0                                         AS score
        FROM obsidian_note_links lnk
        JOIN obsidian_notes n2 ON n2.note_id = lnk.target_note_id
        WHERE lnk.source_note_id = ANY(%s)
          AND n2.note_id <> ALL(%s)
          AND n2.vault_id = %s
        ORDER BY
            n2.note_id,
            CASE n2.note_type
                WHEN 'report'   THEN 0
                WHEN 'policy'   THEN 1
                WHEN 'research' THEN 2
                ELSE 3
            END
        LIMIT %s
    """
    link_rows = query_db(
        link_sql,
        (matched_note_ids, matched_note_ids, vault_id, top_k),
    )

    # ── Step 3: Fetch full metadata for all result note_ids ────────────────
    all_note_ids = matched_note_ids + [r["note_id"] for r in link_rows]
    if not all_note_ids:
        return []

    meta_sql = """
        SELECT note_id, vault_id, title, province, district,
               note_type, tags, source_file, year
        FROM obsidian_notes
        WHERE note_id = ANY(%s)
    """
    meta_rows = {r["note_id"]: r for r in query_db(meta_sql, (all_note_ids,))}

    # ── Step 4: Assemble results (matched first, then neighbors) ───────────
    results: list[dict] = []
    seen: set[str] = set()

    def _make_result(row: dict, is_neighbor: bool = False) -> dict:
        note_id = row["note_id"]
        meta = meta_rows.get(note_id, {})
        return {
            "note_id": note_id,
            "vault_id": meta.get("vault_id") or vault_id,
            "title": meta.get("title") or "",
            "province": meta.get("province") or "",
            "district": meta.get("district") or "",
            "note_type": meta.get("note_type") or "",
            "tags": meta.get("tags") or [],
            "source_file": meta.get("source_file") or "",
            "year": meta.get("year"),
            "snippet": (row.get("snippet") or "")[:600],
            "matched_section": row.get("matched_section"),
            "score": float(row.get("score") or 0.0),
            "is_graph_neighbor": is_neighbor,
        }

    for row in chunk_rows:
        if row["note_id"] not in seen:
            seen.add(row["note_id"])
            results.append(_make_result(row, is_neighbor=False))

    for row in link_rows:
        if row["note_id"] not in seen:
            seen.add(row["note_id"])
            r = _make_result(row, is_neighbor=True)
            r["score"] = neighbor_score  # dynamic discount: max_direct × 0.5
            results.append(r)

    # Cap total to top_k — chunk query + neighbor expansion can each add up to top_k rows
    return results[:top_k]


# ── Tier 2: Full-note fallback (pre-026) ───────────────────────────────────────

def _search_full_notes(
    query: str,
    province: str | None,
    district: str | None,
    tag: str | None,
    note_type: str | None,
    vault_id: str,
    top_k: int,
) -> list[dict]:
    """Original full-note search. Used when chunks are not yet indexed."""
    from src.db.pool import query_db

    where_parts = ["vault_id = %s", "content_stripped IS NOT NULL"]
    params: list = [vault_id]

    term_conds, term_params = _build_like_conditions(
        query, col="content_stripped"
    )
    if term_conds:
        # AND across terms for full-note search (more precise than OR at note level)
        and_conds = [
            f"(content_stripped LIKE %s OR title LIKE %s)" for _ in term_conds
        ]
        params_full: list[str] = []
        for t in [t.strip() for t in query.split() if t.strip()][:5]:
            params_full.extend([f"%{t}%", f"%{t}%"])
        where_parts.append(f"({' OR '.join(and_conds)})")
        params.extend(params_full)

    if province:
        where_parts.append("province = %s")
        params.append(province)
    if district:
        where_parts.append("district = %s")
        params.append(district)
    if tag:
        where_parts.append("%s = ANY(tags)")
        params.append(tag)
    if note_type:
        where_parts.append("note_type = %s")
        params.append(note_type)

    where_sql = " AND ".join(where_parts)
    first_term = f"%{query.split()[0] if query.split() else query}%"
    params.extend([first_term, first_term, top_k])

    sql = f"""
        SELECT
            note_id, vault_id, title, province, district,
            note_type, tags, source_file, year,
            SUBSTRING(content_stripped, 1, 600) AS snippet
        FROM obsidian_notes
        WHERE {where_sql}
        ORDER BY
            CASE WHEN title LIKE %s THEN 0
                 WHEN note_type IN ('report', 'policy', 'research') THEN 1
                 ELSE 2
            END,
            CASE WHEN title LIKE %s THEN LENGTH(content_stripped) ELSE 0 END DESC,
            indexed_at DESC
        LIMIT %s
    """

    rows = query_db(sql, params)
    return [
        {
            "note_id": r["note_id"],
            "vault_id": r["vault_id"],
            "title": r.get("title") or "",
            "province": r.get("province") or "",
            "district": r.get("district") or "",
            "note_type": r.get("note_type") or "",
            "tags": r.get("tags") or [],
            "source_file": r.get("source_file") or "",
            "year": r.get("year"),
            "snippet": (r.get("snippet") or "")[:600],
            "matched_section": None,
            "score": 1.0,
            "is_graph_neighbor": False,
        }
        for r in rows
    ]


# ── Public search impl ─────────────────────────────────────────────────────────

def _search_obsidian_impl(
    query: str,
    province: str | None = None,
    district: str | None = None,
    tag: str | None = None,
    note_type: str | None = None,
    vault_id: str = "health_region_10",
    top_k: int = 5,
) -> str:
    """Core search logic. Auto-selects chunk-based or full-note search."""
    if not settings.OBSIDIAN_ENABLED:
        return json.dumps(
            {"count": 0, "results": [], "note": "Obsidian search disabled"},
            ensure_ascii=False,
        )

    if not query.strip():
        return json.dumps(
            {"count": 0, "results": [], "error": "Empty query"},
            ensure_ascii=False,
        )

    top_k = max(1, min(top_k, 20))

    try:
        if _chunks_indexed(vault_id):
            results = _search_chunks(
                query, province, district, tag, note_type, vault_id, top_k
            )
            tier = "chunk+graph"
        else:
            logger.warning(
                "obsidian_note_chunks empty for vault=%s — using full-note fallback. "
                "Run migration 026 + scripts/index_obsidian.py to enable hybrid search.",
                vault_id,
            )
            results = _search_full_notes(
                query, province, district, tag, note_type, vault_id, top_k
            )
            tier = "full_note_fallback"

        logger.info(
            "Obsidian search [%s]: query=%r vault=%s found=%d",
            tier, query[:50], vault_id, len(results),
        )
        return json.dumps(
            {"count": len(results), "results": results, "tier": tier},
            ensure_ascii=False,
        )

    except Exception as e:
        logger.exception("Obsidian search error: %s", e)
        return json.dumps(
            {"count": 0, "results": [], "error": str(e)[:120]},
            ensure_ascii=False,
        )


# ── Note reader / lister (unchanged) ──────────────────────────────────────────

def _read_note_impl(note_id: str) -> str:
    """Read full content of a note by note_id."""
    try:
        from src.db.pool import query_db
        rows = query_db(
            "SELECT note_id, title, province, district, note_type, tags, "
            "source_file, year, content_stripped FROM obsidian_notes WHERE note_id = %s",
            (note_id,),
        )
        if not rows:
            return json.dumps(
                {"error": f"Note not found: {note_id}"}, ensure_ascii=False
            )
        row = rows[0]
        return json.dumps(
            {
                "note_id": row["note_id"],
                "title": row.get("title") or "",
                "province": row.get("province") or "",
                "district": row.get("district") or "",
                "note_type": row.get("note_type") or "",
                "tags": row.get("tags") or [],
                "source_file": row.get("source_file") or "",
                "year": row.get("year"),
                "content": row.get("content_stripped") or "",
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.exception("Obsidian read note error: %s", e)
        return json.dumps({"error": str(e)[:120]}, ensure_ascii=False)


def _list_notes_impl(
    province: str | None = None,
    district: str | None = None,
    tag: str | None = None,
    note_type: str | None = None,
    vault_id: str = "health_region_10",
) -> str:
    """List notes with optional filters."""
    try:
        from src.db.pool import query_db

        where_parts = ["vault_id = %s"]
        params: list = [vault_id]

        if province:
            where_parts.append("province = %s")
            params.append(province)
        if district:
            where_parts.append("district = %s")
            params.append(district)
        if tag:
            where_parts.append("%s = ANY(tags)")
            params.append(tag)
        if note_type:
            where_parts.append("note_type = %s")
            params.append(note_type)

        where_sql = " AND ".join(where_parts)
        rows = query_db(
            f"SELECT note_id, title, province, district, note_type, tags, "
            f"year, source_file FROM obsidian_notes "
            f"WHERE {where_sql} ORDER BY province, district, title LIMIT 100",
            params,
        )
        results = [
            {
                "note_id": r["note_id"],
                "title": r.get("title") or "",
                "province": r.get("province") or "",
                "district": r.get("district") or "",
                "note_type": r.get("note_type") or "",
                "tags": r.get("tags") or [],
                "year": r.get("year"),
                "source_file": r.get("source_file") or "",
            }
            for r in rows
        ]
        return json.dumps({"count": len(results), "notes": results}, ensure_ascii=False)

    except Exception as e:
        logger.exception("Obsidian list notes error: %s", e)
        return json.dumps({"count": 0, "notes": [], "error": str(e)[:120]}, ensure_ascii=False)


# ── CrewAI Tools ──────────────────────────────────────────────────────────────

@tool("search_obsidian")
def search_obsidian(
    query: str,
    province: str = "",
    district: str = "",
    tag: str = "",
    vault_id: str = "health_region_10",
    top_k: int = 5,
) -> str:
    """ค้นหาความรู้จาก Obsidian Knowledge Vault (ข้อมูลสุขภาพเขตสุขภาพที่ 10)

    ใช้เมื่อต้องการ:
    - ข้อมูล KPI รายอำเภอ เช่น อัตราฆ่าตัวตาย, ผลงาน NCD, วัณโรค
    - เปรียบเทียบผลงานระหว่างจังหวัดหรืออำเภอ
    - นโยบาย แผนปฏิบัติการ ผลการตรวจราชการ สสจ.
    - ข้อมูลเฉพาะพื้นที่เขตสุขภาพที่ 10

    ผลลัพธ์แต่ละรายการมี:
    - snippet: เนื้อหาของ section ที่ตรงกับคำค้น (ไม่ใช่แค่ต้น note)
    - matched_section: ชื่อ ## header ที่พบคำค้น
    - score: ความใกล้เคียงกับคำค้น (0-1)
    - is_graph_neighbor: true = ถูก expand มาจาก wikilink

    Args:
        query: คำค้นหาภาษาไทย เช่น "อัตราฆ่าตัวตาย มุกดาหาร"
        province: กรองตามจังหวัด (ว่าง = ทุกจังหวัด)
        district: กรองตามอำเภอ (ว่าง = ทุกอำเภอ)
        tag: กรองตาม tag เช่น "ตรวจราชการ"
        vault_id: Vault ที่ต้องการค้นหา (default: health_region_10)
        top_k: จำนวนผลลัพธ์ (1-20, default 5)

    Returns:
        JSON: {count, tier, results: [{note_id, title, snippet, matched_section, score, is_graph_neighbor}]}
    """
    return _search_obsidian_impl(
        query=query,
        province=province or None,
        district=district or None,
        tag=tag or None,
        vault_id=vault_id,
        top_k=top_k,
    )


@tool("read_obsidian_note")
def read_obsidian_note(note_id: str) -> str:
    """อ่านเนื้อหาเต็มของ note ใน Obsidian Knowledge Vault

    ใช้เมื่อ snippet จาก search_obsidian ไม่พอ หรือต้องการรายละเอียดครบถ้วน

    Args:
        note_id: ID ของ note ได้จาก search_obsidian results

    Returns:
        JSON: {note_id, title, province, district, note_type, content (full markdown)}
    """
    return _read_note_impl(note_id)


@tool("list_obsidian_notes")
def list_obsidian_notes(
    province: str = "",
    district: str = "",
    tag: str = "",
    vault_id: str = "health_region_10",
) -> str:
    """แสดงรายการ notes ที่มีใน Obsidian Vault พร้อม metadata

    Args:
        province: กรองตามจังหวัด (ว่าง = ทุกจังหวัด)
        district: กรองตามอำเภอ (ว่าง = ทุกอำเภอ)
        tag: กรองตาม tag
        vault_id: Vault ID (default: health_region_10)

    Returns:
        JSON: {count, notes: [{note_id, title, province, district, note_type, tags, year}]}
    """
    return _list_notes_impl(
        province=province or None,
        district=district or None,
        tag=tag or None,
        vault_id=vault_id,
    )
