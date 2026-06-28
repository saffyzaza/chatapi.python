"""FastAPI middleware — internal service authentication.

All API endpoints are protected by a shared secret (INTERNAL_API_KEY) that only
the Next.js backend knows. This prevents unauthenticated callers from hitting
Python directly even if they discover the port.

Usage (applied globally in main.py via middleware, not per-router).
"""
import logging
import secrets
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger(__name__)

# Paths that do NOT require the internal key (public / infra)
_PUBLIC_PATHS = frozenset({"/health", "/ui", "/docs", "/openapi.json", "/redoc"})


class InternalKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests that lack a valid X-Internal-Key header."""

    def __init__(self, app, *, internal_key: str) -> None:
        super().__init__(app)
        self._key = internal_key

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Allow static files, health check, Swagger UI without auth
        if path in _PUBLIC_PATHS or path.startswith("/static"):
            return await call_next(request)

        if not self._key:
            log.error("INTERNAL_API_KEY is not set — all API calls are blocked")
            return JSONResponse(
                {"detail": "Server misconfigured: INTERNAL_API_KEY not set"},
                status_code=503,
            )

        provided = request.headers.get("x-internal-key", "")
        # constant-time comparison — กัน timing side-channel ที่เดา key ทีละไบต์
        if not secrets.compare_digest(provided, self._key):
            log.warning("Blocked request to %s — invalid or missing x-internal-key", path)
            return JSONResponse({"detail": "Forbidden"}, status_code=403)

        return await call_next(request)
