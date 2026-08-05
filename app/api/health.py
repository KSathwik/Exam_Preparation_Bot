"""Health, readiness, and version endpoints."""

from collections import deque
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from app.core.config import settings
from app.core.security import require_admin_key

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
        from app.core.dependencies import get_intent_classifier, get_vector_store_manager

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


@router.get("/logs/tail", response_class=PlainTextResponse, dependencies=[Depends(require_admin_key)])
async def tail_logs(lines: int = 100):
    """Return the last N lines of the log file for quick debugging.

    Requires the operator-only ADMIN_API_KEY, not the regular API key — the
    log file can contain other users' (truncated) query text and other
    operational detail that shouldn't be reachable with the same key the
    frontend embeds in every page load.
    """
    log_path = Path(settings.log_file)
    if not log_path.exists():
        return "Log file not found"
    capped_lines = max(1, min(lines, 5000))
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        tail = deque(f, maxlen=capped_lines)
    return "".join(tail)
