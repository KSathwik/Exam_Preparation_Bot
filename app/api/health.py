"""Health, readiness, and version endpoints."""

from collections import deque
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from pathlib import Path
from app.core.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@router.get("/ready")
async def readiness_check():
    try:
        from app.core.dependencies import get_vector_store_manager, get_intent_classifier

        stats = get_vector_store_manager().get_stats()
        get_intent_classifier()
        return {
            "ready": True,
            "service": settings.app_name,
            "total_vectors": stats["total_vectors"],
        }
    except Exception as e:
        return {"ready": False, "error": str(e)}


@router.get("/version")
async def get_version():
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "api_version": settings.api_version,
        "debug_mode": settings.debug_mode,
    }


@router.get("/config")
async def get_public_config():
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "model": settings.model_name,
        "embedding_model": settings.embedding_model,
        "max_file_size_mb": settings.max_file_size_mb,
        "chunk_size": settings.max_chunk_size,
        "retrieval_top_k": settings.retrieval_top_k,
        "relevance_threshold": settings.relevance_threshold,
    }


@router.get("/logs/tail", response_class=PlainTextResponse)
async def tail_logs(lines: int = 100):
    """Return the last N lines of the log file for quick debugging."""
    log_path = Path(settings.log_file)
    if not log_path.exists():
        return "Log file not found"
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        tail = deque(f, maxlen=lines)
    return "".join(tail)
