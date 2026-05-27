"""Configuration management for FastAPI app"""

from pydantic_settings import BaseSettings
from typing import List
from loguru import logger


class Settings(BaseSettings):
    """Application settings"""
    
    # App Settings
    APP_NAME: str = "Exam Prep Bot"
    APP_VERSION: str = "1.0.0"
    DEBUG_MODE: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # API Settings
    API_VERSION: str = "v1"
    API_PREFIX: str = "/api/v1"
    
    # Anthropic
    ANTHROPIC_API_KEY: str
    MODEL_NAME: str = "claude-3-5-sonnet-20241022"
    MAX_TOKENS: int = 1024
    TEMPERATURE: float = 0.3
    
    # Document Processing
    MAX_CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 100
    MIN_CHUNK_SIZE: int = 100
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_FILE_TYPES: str = "pdf,docx"
    UPLOAD_DIR: str = "./uploads"
    
    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_DIMENSION: int = 384
    BATCH_SIZE: int = 32
    
    # Vector Database
    USE_FAISS: bool = True
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    
    # Retrieval
    RETRIEVAL_TOP_K: int = 5
    RELEVANCE_THRESHOLD: float = 0.5
    MIN_RELEVANCE_SCORE: float = 0.3
    
    # Intent Classification
    INTENT_CLASSIFICATION_THRESHOLD: float = 0.7
    
    # Caching
    ENABLE_QUERY_CACHE: bool = True
    CACHE_TTL_SECONDS: int = 300
    CACHE_DIR: str = "./cache"
    USE_REDIS: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Database
    DATABASE_URL: str = "sqlite:///./exam_prep_bot.db"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    
    # Performance
    ENABLE_ASYNC: bool = True
    WORKER_THREADS: int = 4
    QUERY_TIMEOUT_SECONDS: int = 30
    
    # Development
    DEVELOPMENT_MODE: bool = True
    SAMPLE_DATA_DIR: str = "./data/sample_materials"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Load settings
try:
    settings = Settings()
    logger.info("Settings loaded successfully")
except Exception as e:
    logger.error(f"Failed to load settings: {e}")
    raise
