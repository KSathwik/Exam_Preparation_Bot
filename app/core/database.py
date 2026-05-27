"""Database initialization and management"""

from loguru import logger
from app.core.config import settings
from pathlib import Path


def init_db():
    """Initialize database"""
    logger.info("Initializing database...")
    
    # Create directories
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.CACHE_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.FAISS_INDEX_PATH).mkdir(parents=True, exist_ok=True)
    Path("./logs").mkdir(parents=True, exist_ok=True)
    
    logger.info("Database initialized successfully")
