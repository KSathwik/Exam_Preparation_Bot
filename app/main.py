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
from app.core.security import ACTIVE_API_KEY, require_api_key

# ── Loguru configuration ──────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)
logger.add(
    settings.log_file,
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    enqueue=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Starting Exam Prep Bot...")
    logger.info(f"LLM Provider : {settings.llm_provider}")
    logger.info(f"Model Name   : {settings.model_name or '(provider default)'}")
    logger.info(f"Embedding    : {settings.embedding_model}")
    logger.info(f"Database URL : {settings.database_url}")
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
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info(f"→ {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        duration = time.time() - start
        logger.info(
            f"← {request.method} {request.url.path}  status={response.status_code}  duration={duration:.3f}s"
        )
        return response
    except Exception as e:
        duration = time.time() - start
        logger.error(f"✗ {request.method} {request.url.path}  error={e}  duration={duration:.3f}s")
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
@app.post("/api/system/reset", tags=["System"], dependencies=[Depends(require_api_key)])
async def reset_system():
    from fastapi.concurrency import run_in_threadpool

    from app.core.dependencies import get_bot

    bot = get_bot()
    await run_in_threadpool(bot.reset)
    return {"status": "success", "message": "Bot reset successfully"}


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


@app.get("/api/metrics", tags=["Monitoring"], dependencies=[Depends(require_api_key)])
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
