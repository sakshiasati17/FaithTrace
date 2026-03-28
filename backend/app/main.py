import logging

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.api.v1 import router as api_v1_router
from app.core.config import settings

logger = logging.getLogger(__name__)

# ─── Sentry (optional) ────────────────────────────────────────────────────────
if settings.SENTRY_DSN:
    import sentry_sdk
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.APP_ENV,
        traces_sample_rate=0.2,
    )

# ─── Rate limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# ─── App ──────────────────────────────────────────────────────────────────────
IS_PROD = settings.APP_ENV == "production"

app = FastAPI(
    title="FaithTrace API",
    description="Temporal + Multimodal RAG Diagnostics Platform",
    version="0.1.0",
    # Disable interactive docs in production — API spec reveals internals
    docs_url=None if IS_PROD else "/docs",
    redoc_url=None if IS_PROD else "/redoc",
    openapi_url=None if IS_PROD else "/openapi.json",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── API key auth middleware ───────────────────────────────────────────────────
# Skip auth if API_KEY is not configured (local dev convenience).
# In production, set API_KEY to a strong random value (openssl rand -hex 32).
_EXEMPT_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if not settings.API_KEY:
        return await call_next(request)

    path = request.url.path
    if path in _EXEMPT_PATHS or path.startswith("/docs") or path.startswith("/redoc"):
        return await call_next(request)

    provided = (
        request.headers.get("X-API-Key")
        or request.query_params.get("api_key")
    )
    if provided != settings.API_KEY:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid or missing API key. Pass X-API-Key header."},
        )
    return await call_next(request)

# ─── Routes ───────────────────────────────────────────────────────────────────
app.include_router(api_v1_router, prefix="/api/v1")


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok", "service": "faithtrace-api", "env": settings.APP_ENV}
