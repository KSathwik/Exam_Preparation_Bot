"""
Exam Prep Bot — FastAPI Backend
Unified application entry point.
"""

import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api import conversations, documents, health, queries
from app.core.config import settings
from app.core.database import init_db
from app.core.rate_limit import limiter
from app.core.security import ACTIVE_API_KEY, require_admin_key

# ── Loguru configuration ──────────────────────────────────────────────
# Rotation/retention/compression/JSON-vs-text are all settings-driven (see
# config.py) rather than hardcoded, so an operator can tune them via .env
# without a code change + redeploy. `enqueue=True` on every file sink makes
# writes thread-safe — route handlers log from a thread-pool worker, not
# just the event loop (see CLAUDE.md's "route handlers offload blocking
# work" note).
_TEXT_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}"

logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)
_file_sink_kwargs = {
    "rotation": settings.log_rotation,
    "retention": settings.log_retention,
    "compression": settings.log_compression,
    "enqueue": True,
    "serialize": settings.log_json,
}
logger.add(
    settings.log_file,  # type: ignore[call-overload]
    level="DEBUG",
    format=_TEXT_FORMAT if not settings.log_json else "{message}",
    **_file_sink_kwargs,
)
# Separate error-only sink — lets ops tail/alert on failures without wading
# through DEBUG/INFO noise from the combined app log.
logger.add(
    settings.error_log_file,  # type: ignore[call-overload]
    level="ERROR",
    format=_TEXT_FORMAT if not settings.log_json else "{message}",
    **_file_sink_kwargs,
)


def _redact_dsn(dsn: str) -> str:
    """Strip embedded credentials from a DB connection string before logging
    it — e.g. postgresql://user:pass@host/db -> postgresql://***@host/db.
    SQLite paths (no "@") pass through unchanged."""
    if "://" not in dsn or "@" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1)
    _, host_part = rest.split("@", 1)
    return f"{scheme}://***@{host_part}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Starting Exam Prep Bot...")
    logger.info(f"LLM Provider : {settings.llm_provider}")
    logger.info(f"Model Name   : {settings.model_name or '(provider default)'}")
    logger.info(f"Embedding    : {settings.embedding_model}")
    logger.info(f"Database URL : {_redact_dsn(settings.database_url)}")
    logger.info(f"Log Level    : {settings.log_level}")
    logger.info(f"Debug Mode   : {settings.debug_mode}")
    logger.info("=" * 60)

    init_db()

    from app.core.dependencies import get_bot

    get_bot()
    logger.info("Bot initialized successfully — ready to serve requests")

    yield

    logger.info("Shutting down Exam Prep Bot...")


app = FastAPI(
    title=settings.app_name,
    description="Intelligent exam preparation chatbot powered by Claude + RAG",
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting (per client IP) on expensive/LLM-calling endpoints
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
app.add_middleware(SlowAPIMiddleware)


def _log_path(path: str, max_len: int = 100) -> str:
    """Truncate a request path before logging it. Most routes are short by
    nature — a long one is almost always a path parameter carrying free-text
    user content (e.g. GET /api/intent/{query}), which would otherwise bypass
    the query-text redaction applied everywhere else (see
    redact_query_for_log) simply by riding in the URL path instead of a
    logged variable."""
    return path if len(path) <= max_len else f"{path[:max_len]}...(len={len(path)})"


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    path = _log_path(request.url.path)
    logger.info(f"→ {request.method} {path}")
    try:
        response = await call_next(request)
        duration = time.time() - start
        logger.info(f"← {request.method} {path}  status={response.status_code}  duration={duration:.3f}s")
        return response
    except Exception as e:
        duration = time.time() - start
        logger.error(f"✗ {request.method} {path}  error={e}  duration={duration:.3f}s")
        raise


# Routers
app.include_router(health.router, tags=["Health"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(queries.router, prefix="/api", tags=["Queries"])
app.include_router(conversations.router, prefix="/api/conversations", tags=["Conversations"])


# ── Root & static ────────────────────────────────────────────────────


@app.get("/", tags=["UI"])
async def root():
    html_path = Path("frontend/index.html")
    if not html_path.exists():
        return {"message": "Exam Prep Bot API is running. Visit /docs for Swagger UI."}

    if not (settings.expose_api_key_to_frontend and ACTIVE_API_KEY):
        return FileResponse(str(html_path))

    # Embed the active key so the bundled first-party UI needs no manual entry.
    # Direct API calls that bypass this page (curl, bots, other clients) still
    # require the X-API-Key header — this only exempts the served page itself.
    html = html_path.read_text(encoding="utf-8")
    injection = f"<script>window.__APP_API_KEY__ = {json.dumps(ACTIVE_API_KEY)};</script>"
    html = html.replace("</head>", f"{injection}\n</head>", 1)
    return HTMLResponse(html)


# System endpoints
@app.get("/api/system/config", tags=["System"])
async def get_system_config():
    return {
        "llm_provider": settings.llm_provider,
        "model": settings.model_name or "(provider default)",
        "embedding_model": settings.embedding_model,
        "max_chunk_size": settings.max_chunk_size,
        "retrieval_top_k": settings.retrieval_top_k,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "debug_mode": settings.debug_mode,
    }


@app.get("/api/metrics", tags=["Monitoring"], dependencies=[Depends(require_admin_key)])
async def get_metrics():
    from app.core.dependencies import get_bot

    bot = get_bot()
    stats = bot.get_stats()
    return {
        "uptime": "ok",
        "vector_store": stats["vector_store"],
        "chat_history_length": stats["chat_history_length"],
        "model_info": {"model": stats["model"], "embedding_model": stats["embedding_model"]},
    }


# Serve frontend static assets (CSS, JS)
frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")
