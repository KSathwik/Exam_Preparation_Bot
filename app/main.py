"""
Exam Prep Bot — FastAPI Backend
Unified application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from loguru import logger
from pathlib import Path

from app.api import documents, queries, health
from app.core.config import settings
from app.core.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Exam Prep Bot...")
    init_db()

    from app.core.dependencies import get_bot
    get_bot()
    logger.info("Bot initialized successfully")

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
        "model": settings.model_name,
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
