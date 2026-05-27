"""Health check and status endpoints"""

from fastapi import APIRouter
from loguru import logger

from app.core.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Exam Prep Bot API",
        "version": settings.APP_VERSION
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check - verify all dependencies are available"""
    try:
        from app.services.vector_store import vector_store_manager
        from app.services.intent_classifier import IntentClassifier
        
        # Check vector store
        stats = vector_store_manager.get_stats()
        
        # Check classifier
        classifier = IntentClassifier()
        
        return {
            "ready": True,
            "service": "Exam Prep Bot API",
            "vector_store": "operational",
            "classifier": "operational",
            "total_vectors": stats['total_vectors']
        }
    
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {
            "ready": False,
            "error": str(e)
        }


@router.get("/version")
async def get_version():
    """Get API version"""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "api_version": settings.API_VERSION,
        "debug_mode": settings.DEBUG_MODE
    }


@router.get("/config")
async def get_config():
    """Get public configuration (sensitive info excluded)"""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "model": settings.MODEL_NAME,
        "embedding_model": settings.EMBEDDING_MODEL,
        "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
        "chunk_size": settings.MAX_CHUNK_SIZE,
        "retrieval_top_k": settings.RETRIEVAL_TOP_K,
        "relevance_threshold": settings.RELEVANCE_THRESHOLD
    }
