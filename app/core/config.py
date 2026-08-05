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
    # Every heavy resource (VectorStoreManager, its FAISS index + lock,
    # ExamPrepBot.chat_history) is a process-local @lru_cache singleton —
    # >1 worker means >1 independent process each loading its own copy of
    # the index, with no cross-process coordination on save_index()/
    # load_index(). That silently corrupts the on-disk index (two workers'
    # writes racing) and desyncs chat_history/rate-limit counters across
    # requests. Real multi-worker support needs cross-process locking or a
    # proper vector DB — until then this must stay 1.
    workers: int = 1

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

    # Reflection short-circuit — reflection (ReflectionAgent) is a full second
    # LLM round-trip on top of drafting, so paying for it when the draft
    # already looks unrescuable means the user waits through two LLM calls
    # just to land on the same hard-blocked fallback one call would have
    # produced (observed: 60s+ for a query that was always going to end in
    # "couldn't verify this"). Deliberately narrow: citation_rate==0 alone
    # isn't enough to skip on — reflection legitimately re-grounds a mediocre
    # draft into one with real citations often enough that skipping every
    # zero-citation draft would give up genuine rescues, not just latency.
    # Only skip when confidence is *also* already far below the hard-block
    # floor (see orchestrator.py's own 0.3), i.e. the retrieved chunks looked
    # weak too, not just the draft's phrasing — same spirit as that floor's
    # own "leave a wide margin for normal variance" reasoning, just applied
    # one stage earlier and stricter.
    enable_reflection_shortcut: bool = True
    reflection_shortcut_confidence_floor: float = 0.15

    # Hybrid RAG + CAG — when a conversation's resolved document scope is
    # small enough (see context_router.py), skip similarity-ranked retrieval
    # entirely and give the drafting LLM every chunk of every scoped document
    # instead of a ranked top_k subset. Strictly better whenever it fits: no
    # ranking-induced risk of dropping a chunk the answer needed. Falls back
    # to the existing RAG path unchanged whenever the scope exceeds the
    # budget. Budget is conservative on purpose — leaves headroom under
    # typical provider context windows once system prompt, conversation
    # history, and generation budget are accounted for; a word-count proxy
    # (see context_router._estimate_tokens), not a real tokenizer.
    enable_cag: bool = True
    cag_token_budget: int = 6000

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
    # Rotation/retention/compression were previously hardcoded in main.py —
    # moved here so an operator can tune them via .env like every other
    # setting, without a code change + redeploy.
    log_rotation: str = "10 MB"
    log_retention: str = "7 days"
    log_compression: str = "zip"
    # Separate error-only sink so ops can tail/alert on failures without
    # wading through DEBUG/INFO noise from the combined app log.
    error_log_file: str = "./logs/error.log"
    # Plain text is human-readable but not natively parseable by a log
    # aggregation pipeline (Loki/ELK/CloudWatch/Datadog); set true to emit
    # newline-delimited JSON via loguru's serialize=True instead.
    log_json: bool = False
    # Raw user query text was being logged verbatim at INFO across the
    # pipeline, then made retrievable by any holder of the shared,
    # frontend-exposed API key via GET /api/logs/tail — a cross-user
    # disclosure of question content. On by default; only disable in a
    # trusted single-operator environment where full-text debug logs are
    # wanted.
    log_redact_user_content: bool = True

    # A second, operator-only credential for endpoints that must not be
    # reachable with the same key the frontend embeds in every page load
    # (expose_api_key_to_frontend) — currently GET /api/logs/tail and
    # GET /api/metrics. None disables those routes entirely rather than
    # falling back to the public key.
    admin_api_key: Optional[str] = None

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

    # pydantic's ConfigDict typing can be stricter than the runtime fields
    # we pass here; silence mypy for the assignment and keep runtime behavior.
    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="ignore")  # type: ignore


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


def redact_query_for_log(query: Optional[str], max_len: int = 80) -> str:
    """Format a user query for a log line — truncated to `max_len` chars with
    the real length appended, not the full text. Raw query text logged
    verbatim was retrievable by any holder of the shared, frontend-exposed
    API key via GET /api/logs/tail — a cross-user disclosure of question
    content. Controlled by settings.log_redact_user_content (on by default);
    disable only in a trusted single-operator environment that wants
    full-text debug logs."""
    text = query or ""
    if not settings.log_redact_user_content:
        return repr(text)
    shown = text[:max_len]
    suffix = "..." if len(text) > max_len else ""
    return f"{shown!r}{suffix} (len={len(text)})"
