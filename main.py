"""
Exam Prep Bot — Application entry point.
Delegates entirely to the structured ``app`` package.
"""

from app.main import app  # noqa: F401 — re-export for uvicorn

if __name__ == "__main__":
    import uvicorn
    from loguru import logger

    from app.core.config import settings

    # Hard safety clamp, not just a default: every heavy resource (the
    # vector store, chat history) is a process-local singleton with no
    # cross-process coordination — >1 worker corrupts the on-disk FAISS
    # index (racing save_index() calls) regardless of what WORKERS is set
    # to in .env. See app/core/config.py's workers field for the full
    # explanation.
    worker_count = settings.workers
    if worker_count > 1:
        logger.warning(
            f"WORKERS={worker_count} requested but this architecture only supports a single "
            "worker process (shared in-memory vector store/chat history) — forcing workers=1. "
            "Fix the config; do not raise this without adding cross-process coordination first."
        )
        worker_count = 1

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug_mode,
        workers=1 if settings.debug_mode else worker_count,
        log_level=settings.log_level.lower(),
    )
