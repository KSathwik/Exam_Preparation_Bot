"""Configuration management for the Exam Prep Bot."""

from pydantic_settings import BaseSettings
from typing import Optional
from loguru import logger
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # ==================== Anthropic API ====================
    anthropic_api_key: str
    model_name: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.95
    
    # ==================== Document Processing ====================
    max_chunk_size: int = 512
    chunk_overlap: int = 100
    min_chunk_size: int = 100
    max_file_size_mb: int = 50
    allowed_file_types: str = "pdf,docx"
    upload_dir: str = "./uploads"
    
    # ==================== Embeddings ====================
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_dimension: int = 384
    batch_size: int = 32
    
    # ==================== Vector Database - FAISS ====================
    use_faiss: bool = True
    faiss_index_path: str = "./data/faiss_index"
    
    # ==================== Vector Database - Pinecone ====================
    use_pinecone: bool = False
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "exam-prep-bot"
    
    # ==================== Retrieval Settings ====================
    retrieval_top_k: int = 5
    relevance_threshold: float = 0.5
    min_relevance_score: float = 0.3
    
    # ==================== Intent Classification ====================
    intent_classification_threshold: float = 0.7
    intent_embedding_model: str = "all-MiniLM-L6-v2"
    
    # ==================== Caching ====================
    enable_query_cache: bool = True
    cache_ttl_seconds: int = 300
    cache_dir: str = "./cache"
    
    # ==================== Logging ====================
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = "./logs/app.log"
    
    # ==================== Performance ====================
    enable_async: bool = True
    worker_threads: int = 4
    query_timeout_seconds: int = 30
    
    # ==================== Development ====================
    debug_mode: bool = False
    development_mode: bool = True
    sample_data_dir: str = "./data/sample_materials"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def load_settings() -> Settings:
    """Load and validate settings from environment."""
    try:
        settings = Settings()
        logger.info("Settings loaded successfully")
        return settings
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        raise


def create_directories(settings: Settings) -> None:
    """Create necessary directories if they don't exist."""
    directories = [
        settings.upload_dir,
        settings.cache_dir,
        settings.faiss_index_path,
        "./logs",
        "./data",
        "./data/sample_materials",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {directory}")


# Load settings on module import
try:
    settings = load_settings()
    create_directories(settings)
except Exception as e:
    logger.error(f"Configuration error: {e}")
    raise
