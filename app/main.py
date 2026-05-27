"""
Exam Prep Bot - FastAPI Backend
Main application entry point
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from loguru import logger
import os
from pathlib import Path

from app.api import documents, queries, health
from app.core.config import settings
from app.core.database import init_db
from app.services.vector_store import vector_store_manager


# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle - startup and shutdown"""
    # Startup
    logger.info("Starting Exam Prep Bot...")
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Exam Prep Bot...")


# Create FastAPI app
app = FastAPI(
    title="Exam Prep Bot API",
    description="Intelligent exam preparation chatbot powered by Claude",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(queries.router, prefix="/api/v1/queries", tags=["Queries"])

# Serve static files (frontend)
frontend_path = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "Exam Prep Bot API"
    }


@app.get("/api/v1/stats")
async def get_stats():
    """Get API statistics"""
    try:
        stats = vector_store_manager.get_stats()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG_MODE,
        workers=settings.WORKERS if not settings.DEBUG_MODE else 1
    )
