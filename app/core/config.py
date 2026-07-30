"""Unified configuration management for the Exam Prep Bot."""

from pathlib import Path
from typing import List, Optional

from loguru import logger
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


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
    # 700 was cutting off comprehensive/multi-section answers mid-word once
    # the draft actually needed the space (e.g. several categorized lists) —
    # raised to give real headroom before hitting the provider's stop condition.
    max_tokens: int = 2048
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
    # BAAI/bge-small-en-v1.5: same 384 dimensions as the prior all-MiniLM-L6-v2
    # default (no vector_dimension change), but retrieval-tuned rather than
    # general sentence-similarity-tuned, at near-identical size/CPU latency.
    # IMPORTANT: changing this setting on a deployment with an existing FAISS
    # index requires a full reindex — old and new embeddings are not
    # comparable, and load_index() has no way to detect a silent model
    # mismatch (it only validates vector/metadata/embedding *counts* agree).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    vector_dimension: int = 384
    batch_size: int = 32

    # Vector Database — FAISS
    use_faiss: bool = True
    faiss_index_path: str = "./data/faiss_index"

    # Retrieval
    retrieval_top_k: int = 5
    relevance_threshold: float = 0.2
    min_relevance_score: float = 0.1

    # Hybrid retrieval: blend dense (FAISS) and lexical (BM25) scores as
    # hybrid_dense_weight * dense_norm + (1 - hybrid_dense_weight) * bm25_norm.
    # Default favors semantic similarity — exam-prep queries are often
    # conceptual, not exact-keyword.
    hybrid_dense_weight: float = 0.6

    # Cross-encoder reranking of the merged candidate pool — opt-in and
    # off by default; hybrid retrieval alone should capture most of the
    # achievable gain at this project's current scale.
    enable_cross_encoder_rerank: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Conversation memory (Memory Agent) — only meaningful when persist=True.
    # Semantic-memory fallback fires only when document retrieval misses
    # (in_scope=False); a stricter threshold than relevance_threshold since a
    # memory hit fully replaces the "nothing relevant" fallback message.
    # Raised from 0.4 -> 0.6: with it at 0.4, mediocre-but-passable recall of
    # old chat summaries was answering questions that should have stayed
    # grounded in the freshly uploaded document (or fallen through to the
    # plain out-of-scope message) — the currently uploaded material should
    # dominate answers, with memory only stepping in for a near-certain topic
    # match, not "vaguely related to something we discussed before."
    memory_relevance_threshold: float = 0.6
    # Dual summarization trigger — whichever fires first.
    memory_summarize_every_n_turns: int = 10
    memory_summarize_token_threshold: int = 2000

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

    # Authentication — protects mutating/sensitive endpoints with an API key.
    # Secure by default: if enabled and no key is configured, a random key is
    # generated once at process startup and logged (see app.core.security).
    api_auth_enabled: bool = True
    app_api_key: Optional[str] = None

    # If true, the server embeds the active API key into the page it serves at
    # "/" so the bundled first-party UI works with no manual key entry.
    # Trade-off: anyone who can load the page can view-source the key — this
    # only makes sense when "anyone who can reach the server" is already your
    # trust boundary (a private/internal deployment). Set to false and build
    # real per-user authentication before this app has independent customers
    # who shouldn't see each other's — or your — credentials.
    expose_api_key_to_frontend: bool = True

    # Rate limiting (requests per minute per client IP) on LLM-calling endpoints
    rate_limit_per_minute: int = 30

    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


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
