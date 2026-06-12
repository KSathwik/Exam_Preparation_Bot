"""
Exam Prep Bot — Application entry point.
Delegates entirely to the structured ``app`` package.
"""

from app.main import app  # noqa: F401 — re-export for uvicorn

if __name__ == "__main__":
    import uvicorn
    from app.core.config import settings

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug_mode,
        workers=1 if settings.debug_mode else settings.workers,
        log_level=settings.log_level.lower(),
    )
