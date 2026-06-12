"""
Exam Prep Bot — FastAPI Backend
Unified application entry point.
"""

import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from loguru import logger
from pathlib import Path
import time

from app.api import documents, queries, health
from app.core.config import settings
from app.core.database import init_db

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

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info(f"→ {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        duration = time.time() - start
        logger.info(f"← {request.method} {request.url.path}  status={response.status_code}  duration={duration:.3f}s")
        return response
    except Exception as e:
        duration = time.time() - start
        logger.error(f"✗ {request.method} {request.url.path}  error={e}  duration={duration:.3f}s")
        raise


# Routers
app.include_router(health.router, tags=["Health"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(queries.router, prefix="/api", tags=["Queries"])


# ── Root & static ────────────────────────────────────────────────────

@app.get("/", tags=["UI"])
async def root():
    html = Path("frontend/index.html")
    if html.exists():
        return FileResponse(str(html))
    return {"message": "Exam Prep Bot API is running. Visit /docs for Swagger UI."}


# System endpoints
@app.post("/api/system/reset", tags=["System"])
async def reset_system():
    from app.core.dependencies import get_bot
    bot = get_bot()
    bot.reset()
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


@app.get("/api/metrics", tags=["Monitoring"])
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
