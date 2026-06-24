"""Sync raw PDF/DOCX files from obsidian_raw_pdf/ folder to MinIO and record in DB."""
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".xlsx"}


def sync_pdfs() -> dict:
    """Upload all supported files from OBSIDIAN_RAW_PDF_PATH to MinIO.

    Idempotent — skips files already uploaded (by minio_path uniqueness).
    Returns dict matching ObsidianPdfSyncResult schema.
    """
    try:
        from src.config import get_settings
        from src.db.pool import execute_db, query_db
    except ImportError:
        from config import get_settings
        from db.pool import execute_db, query_db

    s = get_settings()

    try:
        from minio import Minio
        from minio.error import S3Error
    except ImportError:
        raise RuntimeError("minio package not installed. Run: pip install minio")

    raw_pdf_root = Path(s.OBSIDIAN_VAULT_PATH).parent / s.OBSIDIAN_RAW_PDF_PATH
    if not raw_pdf_root.exists():
        logger.warning("[sync_pdfs] Raw PDF path not found: %s", raw_pdf_root)
        return {"uploaded": 0, "skipped": 0, "errors": 0, "total_files": 0, "elapsed_seconds": 0.0}

    client = Minio(
        s.minio_endpoint_url,
        access_key=s.MINIO_ACCESS_KEY,
        secret_key=s.MINIO_SECRET_KEY,
        secure=s.MINIO_USE_SSL,
    )

    bucket = s.PDF_BUCKET
    prefix = s.OBSIDIAN_PDF_MINIO_PREFIX

    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            logger.info("[sync_pdfs] Created bucket: %s", bucket)
    except Exception as e:
        logger.warning("[sync_pdfs] Could not ensure bucket %s: %s", bucket, e)

    all_files = [
        p for p in raw_pdf_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    logger.info("[sync_pdfs] Found %d files to sync", len(all_files))

    uploaded = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    for file_path in all_files:
        try:
            rel = file_path.relative_to(raw_pdf_root)
            parts = rel.parts
            province = parts[0] if len(parts) >= 2 else "unknown"
            minio_path = f"{prefix}/{rel.as_posix()}"

            existing = query_db(
                "SELECT id FROM obsidian_pdf_assets WHERE minio_path = %s", (minio_path,)
            )
            if existing:
                skipped += 1
                continue

            content_type = "application/pdf"
            if file_path.suffix.lower() == ".docx":
                content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif file_path.suffix.lower() == ".pptx":
                content_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            elif file_path.suffix.lower() == ".xlsx":
                content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

            client.fput_object(bucket, minio_path, str(file_path), content_type=content_type)

            endpoint = s.minio_endpoint_url
            scheme = "https" if s.MINIO_USE_SSL else "http"
            minio_url = f"{scheme}://{endpoint}/{bucket}/{minio_path}"
            file_size = file_path.stat().st_size

            execute_db(
                """
                INSERT INTO obsidian_pdf_assets (province, filename, minio_path, minio_url, file_size, content_type)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (minio_path) DO NOTHING
                """,
                (province, file_path.name, minio_path, minio_url, file_size, content_type),
            )
            uploaded += 1
            logger.info("[sync_pdfs] Uploaded: %s", minio_path)

        except Exception as e:
            logger.error("[sync_pdfs] Error on %s: %s", file_path, e)
            errors += 1

    elapsed = round(time.time() - start_time, 2)
    logger.info("[sync_pdfs] Done: uploaded=%d skipped=%d errors=%d elapsed=%.2fs",
                uploaded, skipped, errors, elapsed)
    return {
        "uploaded": uploaded,
        "skipped": skipped,
        "errors": errors,
        "total_files": len(all_files),
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    print(sync_pdfs())
