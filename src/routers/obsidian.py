"""Obsidian Knowledge Vault API router."""
import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from src.config import get_settings
from src.schemas.obsidian import (
    ObsidianSearchRequest,
    ObsidianSearchResponse,
    ObsidianAskRequest,
    ObsidianAskResponse,
    ObsidianStatusResponse,
    ObsidianVaultInfo,
    ObsidianIndexRequest,
    ObsidianIndexResult,
    ObsidianPdfAsset,
    ObsidianPdfSyncResult,
    ObsidianPdfListResponse,
    ObsidianNotePdfPair,
    ObsidianNotePdfPairsResponse,
)
from src.tools.obsidian import _search_obsidian_impl, _read_note_impl, _list_notes_impl
from src.agents.progress import create_progress_queue, remove_progress_queue

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/obsidian", tags=["obsidian"])


# ── Search ─────────────────────────────────────────────────────────────────────

@router.post("/search", response_model=ObsidianSearchResponse)
async def search_obsidian_notes(req: ObsidianSearchRequest):
    """Search Obsidian Knowledge Vault using pg_trgm full-text search.

    No LLM involved — direct database query, fast (<2s).
    """
    if not settings.OBSIDIAN_ENABLED:
        return ObsidianSearchResponse(count=0, results=[], query=req.query, vault_id=req.vault_id, note="Obsidian search disabled")

    raw = _search_obsidian_impl(
        query=req.query,
        province=req.province,
        district=req.district,
        tag=req.tag,
        note_type=req.note_type,
        vault_id=req.vault_id,
        top_k=req.top_k,
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Internal search error")

    if "error" in data:
        return ObsidianSearchResponse(
            count=0, results=[], query=req.query, vault_id=req.vault_id,
            error=data["error"],
        )

    return ObsidianSearchResponse(
        count=data["count"],
        results=data["results"],
        query=req.query,
        vault_id=req.vault_id,
    )


# ── Note CRUD ──────────────────────────────────────────────────────────────────

@router.get("/notes/{note_id:path}")
async def get_obsidian_note(note_id: str):
    """Read full content of a single note by its ID."""
    raw = _read_note_impl(note_id)
    data = json.loads(raw)
    if "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])
    return JSONResponse(content=data)


@router.get("/notes")
async def list_obsidian_notes(
    province: str = Query(default="", description="กรองตามจังหวัด"),
    district: str = Query(default="", description="กรองตามอำเภอ"),
    tag: str = Query(default="", description="กรองตาม tag"),
    note_type: str = Query(default="", description="MOC | district | report | research | policy"),
    vault_id: str = Query(default="health_region_10"),
):
    """List notes with optional filters."""
    raw = _list_notes_impl(
        province=province or None,
        district=district or None,
        tag=tag or None,
        note_type=note_type or None,
        vault_id=vault_id,
    )
    data = json.loads(raw)
    return JSONResponse(content=data)


# ── Ask pipeline ───────────────────────────────────────────────────────────────

@router.post("/ask", response_model=ObsidianAskResponse)
async def ask_obsidian(req: ObsidianAskRequest):
    """Run the 2-agent Obsidian Ask pipeline (synchronous).

    Pipeline: ObsidianSearcher → HealthKnowledgeAnswerWriter
    Expected runtime: 30–120 seconds.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="คำถามต้องไม่ว่างเปล่า")

    if not settings.OBSIDIAN_ENABLED:
        raise HTTPException(status_code=503, detail="Obsidian Knowledge Vault is disabled")

    logger.info("[obsidian] ask: %s | province=%s", req.question[:80], req.province)

    from src.agents.obsidian_fullcontext import run_obsidian_ask_fullcontext
    try:
        result = run_obsidian_ask_fullcontext(
            question=req.question,
            province=req.province or "",
            vault_id=req.vault_id,
        )
        return JSONResponse(content=result.model_dump())
    except Exception as exc:
        logger.exception("[obsidian] ask error: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc)})


@router.post("/ask/stream")
async def ask_obsidian_stream(req: ObsidianAskRequest):
    """Stream the Obsidian Ask pipeline via Server-Sent Events.

    Event types:
      - start: pipeline metadata
      - progress: agent step updates
      - result: final ObsidianAskResponse JSON
      - error: pipeline error
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="คำถามต้องไม่ว่างเปล่า")

    if not settings.OBSIDIAN_ENABLED:
        raise HTTPException(status_code=503, detail="Obsidian Knowledge Vault is disabled")

    request_id = str(uuid.uuid4())
    q = create_progress_queue(request_id)

    async def event_stream():
        yield f"data: {json.dumps({'type': 'start', 'request_id': request_id, 'pipeline': 'obsidian_fullcontext', 'agents': ['📂 Context Loader', '🤖 Gemini Answer Writer']}, ensure_ascii=False)}\n\n"

        from src.agents.obsidian_fullcontext import run_obsidian_ask_fullcontext
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            None,
            lambda: run_obsidian_ask_fullcontext(
                question=req.question,
                province=req.province or "",
                vault_id=req.vault_id,
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
            except Exception:
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
            except Exception:
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


# ── Status & Vault management ──────────────────────────────────────────────────

@router.get("/status", response_model=ObsidianStatusResponse)
async def obsidian_status():
    """Return vault registry and note counts."""
    try:
        from src.db.pool import query_db
        vaults = query_db(
            "SELECT vault_id, name, vault_path, description, note_count, indexed_at "
            "FROM obsidian_vaults ORDER BY vault_id",
            (),
        )
        total = query_db("SELECT COUNT(*) AS cnt FROM obsidian_notes", ())[0]["cnt"]

        vault_list = [
            ObsidianVaultInfo(
                vault_id=v["vault_id"],
                name=v["name"],
                vault_path=v["vault_path"],
                description=v.get("description"),
                note_count=v.get("note_count") or 0,
                indexed_at=str(v["indexed_at"]) if v.get("indexed_at") else None,
            )
            for v in vaults
        ]
        return ObsidianStatusResponse(
            enabled=settings.OBSIDIAN_ENABLED,
            vaults=vault_list,
            total_notes=total,
        )
    except Exception as exc:
        logger.exception("[obsidian] status error: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc)})


@router.get("/vaults")
async def list_vaults():
    """List all registered vaults."""
    try:
        from src.db.pool import query_db
        rows = query_db(
            "SELECT vault_id, name, vault_path, description, note_count, indexed_at "
            "FROM obsidian_vaults ORDER BY vault_id",
            (),
        )
        return JSONResponse(content={"vaults": [dict(r) for r in rows]})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": str(exc)})


# ── Indexer ────────────────────────────────────────────────────────────────────

@router.post("/index", response_model=ObsidianIndexResult)
async def index_vault(req: ObsidianIndexRequest):
    """Trigger vault indexing (walks FS and UPSERTs notes into DB).

    This is a long-running operation (~1-10 minutes for large vaults).
    Runs synchronously — consider calling from CLI script for production.
    """
    logger.info("[obsidian] index requested for vault_id=%s", req.vault_id)
    loop = asyncio.get_event_loop()
    try:
        from src.scripts.index_obsidian import index_vault as _index
        result = await loop.run_in_executor(None, lambda: _index(req.vault_id))
        return JSONResponse(content=result)
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Index script not found. Run: python scripts/index_obsidian.py",
        )
    except Exception as exc:
        logger.exception("[obsidian] index error: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc)})


# ── PDF Assets ─────────────────────────────────────────────────────────────────

@router.post("/pdfs/sync", response_model=ObsidianPdfSyncResult)
async def sync_pdfs_to_minio():
    """Upload all raw PDFs from obsidian_raw_pdf/ to MinIO and record in DB.

    Long-running (~1-5 min for 353 files). Idempotent — skips already-uploaded paths.
    """
    logger.info("[obsidian] pdf sync requested")
    loop = asyncio.get_event_loop()
    try:
        from src.scripts.sync_obsidian_pdfs import sync_pdfs
        result = await loop.run_in_executor(None, sync_pdfs)
        return ObsidianPdfSyncResult(**result)
    except Exception as exc:
        logger.exception("[obsidian] pdf sync error: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc)})


@router.post("/pdfs/sync-from-bucket")
async def sync_pdfs_from_bucket():
    """อ่าน PDF ทั้งหมดจาก pdf-library bucket แล้ว register เข้า obsidian_pdf_assets.

    Idempotent — ใช้ ON CONFLICT DO UPDATE ดังนั้นเรียกซ้ำได้.
    """
    import time
    import urllib.parse

    logger.info("[obsidian] sync-from-bucket requested")
    try:
        from minio import Minio
        from src.config import get_settings
        from src.db.pool import execute_db, query_db

        s = get_settings()
        client = Minio(
            s.minio_endpoint_url,
            access_key=s.MINIO_ACCESS_KEY,
            secret_key=s.MINIO_SECRET_KEY,
            secure=s.MINIO_USE_SSL,
        )
        bucket = s.PDF_BUCKET
        scheme = "https" if s.MINIO_USE_SSL else "http"

        # ensure bucket exists
        if not client.bucket_exists(bucket):
            return {"synced": 0, "skipped": 0, "errors": 0, "total": 0,
                    "note": f"Bucket '{bucket}' does not exist"}

        objects = list(client.list_objects(bucket, recursive=True))
        synced = 0
        skipped = 0
        errors = 0
        start = time.time()

        for obj in objects:
            name = obj.object_name
            # skip metadata/side-car files
            if name.endswith("__apa.json") or name.endswith("__path.json"):
                continue
            try:
                stat = client.stat_object(bucket, name)
                meta = {k.lower(): v for k, v in (stat.metadata or {}).items()}
                ext = meta.get("x-amz-meta-extension", "").lower()
                orig_name = urllib.parse.unquote(meta.get("x-amz-meta-name", name))

                if ext != "pdf" and not orig_name.lower().endswith(".pdf"):
                    skipped += 1
                    continue

                minio_url = f"{scheme}://{s.minio_endpoint_url}/{bucket}/{name}"
                file_size = obj.size or 0

                execute_db(
                    """
                    INSERT INTO obsidian_pdf_assets
                        (province, filename, minio_path, minio_url, file_size, content_type)
                    VALUES (%s, %s, %s, %s, %s, 'application/pdf')
                    ON CONFLICT (minio_path) DO UPDATE SET
                        filename  = EXCLUDED.filename,
                        minio_url = EXCLUDED.minio_url,
                        file_size = EXCLUDED.file_size
                    """,
                    ("ส่วนกลาง", orig_name, name, minio_url, file_size),
                )
                synced += 1
            except Exception as e:
                logger.error("[sync-from-bucket] Error on %s: %s", name, e)
                errors += 1

        elapsed = round(time.time() - start, 2)
        logger.info("[sync-from-bucket] done: synced=%d skipped=%d errors=%d elapsed=%.2fs",
                    synced, skipped, errors, elapsed)
        return {
            "synced": synced,
            "skipped": skipped,
            "errors": errors,
            "total": len(objects),
            "elapsed_seconds": elapsed,
            "bucket": bucket,
        }
    except Exception as exc:
        logger.exception("[obsidian] sync-from-bucket error: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc)})


@router.get("/pdfs", response_model=ObsidianPdfListResponse)
async def list_pdf_assets(
    province: str = Query(default="", description="กรองตามจังหวัด (optional)"),
):
    """List PDF assets stored in MinIO, optionally filtered by province."""
    try:
        from src.db.pool import query_db
        if province:
            rows = query_db(
                "SELECT id, province, note_id, filename, minio_path, minio_url, "
                "file_size, content_type, synced_at "
                "FROM obsidian_pdf_assets WHERE province = %s ORDER BY filename",
                (province,),
            )
        else:
            rows = query_db(
                "SELECT id, province, note_id, filename, minio_path, minio_url, "
                "file_size, content_type, synced_at "
                "FROM obsidian_pdf_assets ORDER BY province, filename",
                (),
            )
        assets = [
            ObsidianPdfAsset(
                id=r["id"],
                province=r["province"],
                note_id=r.get("note_id"),
                filename=r["filename"],
                minio_path=r["minio_path"],
                minio_url=r["minio_url"],
                file_size=r.get("file_size"),
                content_type=r.get("content_type", "application/pdf"),
                synced_at=str(r["synced_at"]) if r.get("synced_at") else None,
            )
            for r in rows
        ]
        return ObsidianPdfListResponse(
            count=len(assets),
            province=province or None,
            assets=assets,
        )
    except Exception as exc:
        logger.exception("[obsidian] pdf list error: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc)})


@router.get("/pdfs/note/{note_id:path}", response_model=ObsidianPdfListResponse)
async def list_pdfs_for_note(note_id: str):
    """List all PDFs for a note's province (province-level grouping).

    Looks up the note's province, then returns all PDF assets for that province.
    """
    try:
        from src.db.pool import query_db
        note_rows = query_db(
            "SELECT province FROM obsidian_notes WHERE note_id = %s LIMIT 1",
            (note_id,),
        )
        if not note_rows:
            raise HTTPException(status_code=404, detail=f"Note not found: {note_id}")

        province = note_rows[0]["province"]
        if not province:
            return ObsidianPdfListResponse(count=0, province=None, assets=[])

        rows = query_db(
            "SELECT id, province, note_id, filename, minio_path, minio_url, "
            "file_size, content_type, synced_at "
            "FROM obsidian_pdf_assets WHERE province = %s ORDER BY filename",
            (province,),
        )
        assets = [
            ObsidianPdfAsset(
                id=r["id"],
                province=r["province"],
                note_id=r.get("note_id"),
                filename=r["filename"],
                minio_path=r["minio_path"],
                minio_url=r["minio_url"],
                file_size=r.get("file_size"),
                content_type=r.get("content_type", "application/pdf"),
                synced_at=str(r["synced_at"]) if r.get("synced_at") else None,
            )
            for r in rows
        ]
        return ObsidianPdfListResponse(
            count=len(assets),
            province=province,
            assets=assets,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[obsidian] pdf note list error: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc)})


@router.get("/pdfs/pairs", response_model=ObsidianNotePdfPairsResponse)
async def list_note_pdf_pairs(
    province: str = Query(default="", description="กรองตามจังหวัด (optional)"),
    vault_id: str = Query(default="health_region_10"),
):
    """Return every note paired with its directly-linked PDFs (via note_id FK).

    Populated by run_update_pdf_note_fk.py which basename-matches source_file
    against obsidian_pdf_assets.filename.  Notes with no FK match still appear
    (direct_pdfs=[], match_type='none') so the UI can show coverage gaps.
    """
    try:
        from src.db.pool import query_db

        # Single LEFT JOIN — one row per (note, pdf); note with no PDF → pdf cols NULL
        sql = """
            SELECT
                n.note_id, n.title, n.province, n.district, n.note_type,
                n.tags, n.source_file, n.year,
                p.id          AS asset_id,
                p.filename    AS pdf_filename,
                p.minio_path,
                p.minio_url,
                p.file_size,
                p.content_type,
                p.synced_at
            FROM obsidian_notes n
            LEFT JOIN obsidian_pdf_assets p ON p.note_id = n.note_id
            WHERE n.vault_id = %s
              AND (%s = '' OR n.province = %s)
            ORDER BY n.province NULLS LAST, n.title NULLS LAST, p.filename NULLS LAST
        """
        rows = query_db(sql, (vault_id, province, province))

        # Group rows by note_id
        from collections import OrderedDict
        pairs_map: dict[str, ObsidianNotePdfPair] = OrderedDict()

        for r in rows:
            nid = r["note_id"]
            if nid not in pairs_map:
                pairs_map[nid] = ObsidianNotePdfPair(
                    note_id=nid,
                    title=r.get("title"),
                    province=r.get("province"),
                    district=r.get("district"),
                    note_type=r.get("note_type"),
                    tags=list(r.get("tags") or []),
                    year=r.get("year"),
                    source_file=r.get("source_file"),
                    direct_pdfs=[],
                    match_type="none",
                )

            # Add PDF if this row has one (LEFT JOIN → asset_id may be None)
            if r.get("asset_id") is not None:
                pairs_map[nid].direct_pdfs.append(ObsidianPdfAsset(
                    id=r["asset_id"],
                    province=r.get("province") or "",
                    note_id=nid,
                    filename=r["pdf_filename"],
                    minio_path=r["minio_path"],
                    minio_url=r["minio_url"],
                    file_size=r.get("file_size"),
                    content_type=r.get("content_type", "application/pdf"),
                    synced_at=str(r["synced_at"]) if r.get("synced_at") else None,
                ))
                pairs_map[nid].match_type = "direct"

        pairs = list(pairs_map.values())
        matched   = sum(1 for p in pairs if p.match_type == "direct")
        unmatched = sum(1 for p in pairs if p.match_type == "none")

        return ObsidianNotePdfPairsResponse(
            count=len(pairs),
            matched=matched,
            unmatched=unmatched,
            province=province or None,
            pairs=pairs,
        )
    except Exception as exc:
        logger.exception("[obsidian] pdf pairs error: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc)})
