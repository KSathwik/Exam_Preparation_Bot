"""Unified configuration management for the Exam Prep Bot."""

from pydantic_settings import BaseSettings
from typing import List, Optional
from loguru import logger
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    app_name: str = "Exam Prep Bot"
    app_version: str = "1.0.0"
    debug_mode: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    # CORS — restrict in production; wildcard kept only for local dev
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # API
    api_version: str = "v1"
    api_prefix: str = "/api/v1"

    # LLM Provider: "gemini", "openai", or "anthropic"
    llm_provider: str = "gemini"

    # API Keys (only the one matching llm_provider is required)
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Model defaults per provider (overridden by model_name if set)
    model_name: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.95

    # Document Processing
    max_chunk_size: int = 512
    chunk_overlap: int = 100
    min_chunk_size: int = 100
    max_file_size_mb: int = 50
    allowed_file_types: str = "pdf,docx"
    upload_dir: str = "./uploads"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_dimension: int = 384
    batch_size: int = 32

    # Vector Database — FAISS
    use_faiss: bool = True
    faiss_index_path: str = "./data/faiss_index"

    # Retrieval
    retrieval_top_k: int = 5
    relevance_threshold: float = 0.2
    min_relevance_score: float = 0.1

    # Intent Classification
    intent_classification_threshold: float = 0.7
    intent_embedding_model: str = "all-MiniLM-L6-v2"

    # Caching
    enable_query_cache: bool = True
    cache_ttl_seconds: int = 300
    cache_dir: str = "./cache"
    use_redis: bool = False
    redis_url: str = "redis://localhost:6379/0"

    # Database
    database_url: str = "sqlite:///./exam_prep_bot.db"

    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"

    # Performance
    enable_async: bool = True
    worker_threads: int = 4
    query_timeout_seconds: int = 30

    # Development
    development_mode: bool = True
    sample_data_dir: str = "./data/sample_materials"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


def _create_directories(s: Settings) -> None:
    """Create necessary directories if they don't exist."""
    for d in [s.upload_dir, s.cache_dir, s.faiss_index_path, "./logs", "./data"]:
        Path(d).mkdir(parents=True, exist_ok=True)


try:
    settings = Settings()
    _create_directories(settings)
    logger.info("Settings loaded successfully")
except Exception as e:
    logger.error(f"Configuration error: {e}")
    raise
