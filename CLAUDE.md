# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate            # Windows (source venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
cp .env.example .env             # then set the API key for whichever LLM_PROVIDER you use

# Run the dev server (serves API + frontend at http://localhost:8000, Swagger at /docs)
uvicorn main:app --reload

# Tests — fully isolated from real project data (see tests/conftest.py)
pytest -v
pytest tests/test_intent_classifier.py -v            # single file
pytest tests/test_intent_classifier.py::test_name -v # single test
pytest --cov=app                                     # with coverage

# Lint/format — config lives in pyproject.toml, not yet enforced pre-commit
black app tests
isort app tests
mypy app                                             # advisory; CI runs it non-blocking

# Docker (FastAPI + PostgreSQL + Redis + Nginx)
docker-compose up -d

# Database migrations (alembic) — schema changes go through a migration, not
# hand-rolled ALTER TABLE or relying on create_all() (which only creates
# missing tables, never alters existing ones)
alembic upgrade head                                 # apply all pending migrations
alembic revision --autogenerate -m "description"     # generate a migration from model changes
alembic downgrade -1                                 # roll back one migration
```

CI (`.github/workflows/ci.yml`) runs `black --check`, `isort --check-only`, the full test suite, and
a Docker build on every push/PR to `main`/`develop`.

See `docs/ARCHITECTURE.md` for diagrams of the pipeline, agent workflow, and memory tiers, and
`docs/PHASE_2_ROADMAP.md` for what's deliberately deferred (Quiz/Flashcard/StudyPlanner agents,
progress tracking, real multi-user auth).

## Architecture

Request flow: `main.py` → `app/main.py` (FastAPI app, lifespan, middleware, routers) → `app/api/*`
routers → `app.core.dependencies` (DI singletons) → `ExamPrepBot` → `OrchestratorAgent` → response.

**`app/core/dependencies.py` is the composition root.** Every expensive resource (SentenceTransformer
models, the FAISS-backed `VectorStoreManager`, `IntentClassifier`, `ExamPrepBot`) is built exactly once
via `@lru_cache(maxsize=1)` factory functions and threaded through constructors — nothing reaches for a
global. When adding a new heavy resource or service, wire it in here rather than instantiating inline.
Tests rely on this: `tests/conftest.py`'s `_reset_singletons` autouse fixture clears every one of these
caches before/after each test, and the `client` fixture patches `SentenceTransformer` at
`app.core.dependencies.SentenceTransformer` — new singleton factories must follow the same pattern to
stay test-isolated.

**Multi-agent pipeline** (`app/services/agents/`): `ExamPrepBot.answer_question()` is a one-line
delegation to `self.orchestrator.run(query, on_stage=...)` — `OrchestratorAgent` (`orchestrator.py`)
is the actual RAG orchestration and the first place to look when tracing a query end-to-end. It's
deterministic Python, not an LLM router — with only one downstream answer-producing agent this phase,
there's nothing to route between yet (see the architecture doc for when that changes). Stages:
1. **Intent classification** (`intent_classifier.py`, called directly) — hybrid classifier: fast
   keyword/rule matching first (`_classify_by_rules`), falling back to semantic similarity against
   template embeddings (`_classify_by_semantics`) when no rule clears `intent_classification_threshold`.
   Classifies into a `QueryType` enum (definition/explain/compare/process/example/diagram/vague/homework).
2. **Retrieval** (`RetrievalAgent` → `retriever.py`'s `HybridRetriever`/`AdaptiveRetriever`) — varies
   `top_k` and reranking strategy by intent (`_TOP_K_STRATEGY`), blends dense (FAISS) and lexical
   (BM25) scores (see Vector storage below), then checks the top result against `relevance_threshold`
   to decide `is_relevant`/`in_scope`. When document retrieval misses, it tries a semantic-memory-only
   search before giving up (see Conversation memory below). Still-out-of-scope queries short-circuit
   with a fixed fallback answer and never reach the LLM.
3. **Draft generation** (`KnowledgeAgent` → `llm_interface.py`'s `generate_structured_answer`) — see
   multi-provider note below. Also extracts factual claims, each traced back to the numbered excerpt(s)
   (`[1]`, `[3]`, ...) it came from, as a JSON array of `{"claim": ..., "chunks": [...]}` objects.
4. **Citation/confidence scoring** (`validator.py`, no LLM call) — `SpanExtractor` maps each claim back
   to a supporting chunk span (searching only the claim's indicated chunks when available, otherwise
   all retrieved chunks), matching against 1–3 consecutive sentences; `ConfidenceScorer` derives
   `overall_confidence` and a `hallucination_risk` level from how many claims got a supporting citation.
5. **Reflection** (`ReflectionAgent` → `llm_interface.py`'s `reflect_on_answer`) — always-on second LLM
   call that checks the draft against the excerpts for hallucination/clarity/structure/completeness,
   returning `{"revised_answer", "materially_changed", "should_block", "issues_found"}`. Any
   failure (provider error, malformed JSON) degrades to "no change" rather than raising.
   `materially_changed=true` triggers a re-validation of stage 4 against the revised text;
   `should_block=true` returns a fixed fallback message instead of the draft.
6. **Memory** (`MemoryAgent`) — records the turn into `chat_history` (always) and, when
   `persist=True` (opt-in, not the default — see Conversation memory below), into the DB and semantic
   index too.

The WebSocket route (`/api/ws`) passes an `on_stage` callback through to the orchestrator for
progressive status/draft/final events — see Conversation memory / WebSocket contract notes below.

**Multi-LLM provider support** (`llm_interface.py`): `ClaudeInterface()` is a *factory function*, not a
class — despite the name (kept for backward compatibility), it dispatches on `settings.llm_provider`
(`gemini`/`openai`/`anthropic`) to return a `_GeminiLLM`/`_OpenAILLM`/`_AnthropicLLM`, all subclassing
`_BaseLLM`. Shared prompt-building, claim extraction, and structured-answer logic lives in `_BaseLLM`;
each subclass only implements `_call()` for its SDK. Per-provider default model names live in
`_DEFAULT_MODELS`, overridable via `MODEL_NAME`. Only the API key matching the active provider is
required at runtime. `_GeminiLLM` uses the `google-genai` SDK (`google.genai.Client(...).models.generate_content`)
— the older `google-generativeai` package it replaces is end-of-life; don't reintroduce it.
`ExamPrepBot.__init__` accepts an optional `llm=` override (same DI pattern as
`vector_store_manager`/`intent_classifier`) specifically so tests can inject a mock LLM instead of
constructing a real provider client.

**Auth and rate limiting**: `app/core/security.py` gates every endpoint except
`/health`/`/ready`/`/version`/`/config`/`/api/system/config` behind an `X-API-Key` header
(`require_api_key`) — secure by default: if `APP_API_KEY` isn't set, a random key is generated once
at import time and logged. The WebSocket route (`/api/ws`) uses `require_api_key_ws` instead, which
reads the key from a `?api_key=` query param since browsers can't set custom WebSocket headers.
`app/core/rate_limit.py` (slowapi) throttles `/api/ask`, `/api/query`, and `/api/batch` to
`RATE_LIMIT_PER_MINUTE` requests/IP — these decorated routes need a `Request` parameter alongside
their Pydantic body model (slowapi's key function reads it) since it can't reuse the body param name.

**Vector storage & hybrid retrieval**: `app/services/embeddings.py` holds the real implementation
(`EmbeddingGenerator`, `FAISSVectorStore`, `VectorStoreManager`) with the FAISS index persisted to
`FAISS_INDEX_PATH` as `index.faiss` + `metadata.pkl` + `embeddings.npy`, reloaded on startup if present.
`app/services/vector_store.py` is a thin singleton shim re-exporting `get_vector_store_manager()` —
don't add logic there. `VectorStoreManager.search()` blends dense (FAISS) and lexical (`rank-bm25`)
scores — `alpha * dense_norm + (1-alpha) * bm25_norm` where `alpha = settings.hybrid_dense_weight`
(default 0.6) — via `_rank_candidates()`; the BM25 index is lazily (re)built in-memory on first search
after any add/remove/reset (`_bm25_dirty` flag), no new persisted artifact. Every chunk carries a
`content_type` (`"document"` or `"memory"`, see Conversation memory below) in the same shared index —
`search()` defaults `content_types=["document"]` so existing call sites never silently start mixing in
memory content. Optional cross-encoder reranking (`settings.enable_cross_encoder_rerank`, off by
default) runs in `AdaptiveRetriever._cross_encoder_rerank()` on the merged candidate pool before the
deterministic `rerank_by_intent` tie-breaking.

Every chunk's metadata carries a `document_id` (set by `VectorStoreManager.add_document`), which is
what makes `remove_document(document_id)` possible — FAISS's flat index has no delete-by-id, so
removal rebuilds the index from the chunks that remain (`FAISSVectorStore.remove_by_document_id`).
All three on-disk files are written via write-temp-then-rename (atomic) and `load_index()` refuses to
load a set of files whose vector/metadata/embedding counts don't agree — a lesson from a real bug
where two processes writing the same on-disk index left them desynced silently. `VectorStoreManager`
serializes every index read/write behind one `threading.Lock` (`_index_lock`) since route handlers
call into it from a thread pool (see below) and FAISS isn't safe for concurrent access. Use
`VectorStoreManager.reset()` (not `vector_store.clear()` directly) when clearing the store — it also
persists the empty state, otherwise a restart would resurrect the pre-reset data from disk.

**Chunking** (`app/services/parser.py`): `DocumentParser._create_chunks()` splits into structural
blocks *before* sentence-accumulation — DOCX via `paragraph.style.name` (exact heading levels), PDF
via a font-size heuristic (`_pdf_page_to_blocks`, falls back to one block per page when no reliable
signal exists). A trailing fragment under `min_chunk_size` is never dropped: it merges into the
previous chunk from the same block, or carries forward into the next block's accumulation, or (no
next block either) is force-emitted — a non-empty document never yields zero chunks. `chunk_position`/
`page_number` are computed only after the final chunk count is known (not against a placeholder), so
they're accurate on every chunk, not just the first.

**Conversation memory** (`app/services/agents/memory_agent.py` + `app/models/db_models.py`): the
`ChatSession`/`ChatMessageRecord` tables and a new `ConversationMemory` table back three tiers —
short-term (the in-memory `chat_history` list, always active), episodic (full session lookup by
`session_id`, not yet wired to any endpoint), and long-term/semantic (`ConversationMemory` summaries
embedded into the shared FAISS index with `content_type="memory"`). `MemoryAgent` defaults to
`persist=False` — exactly today's pure in-memory behavior; the DB/summarization/embedding path only
runs when constructed with `persist=True`, a `session_id`, and a `db_session_factory` (forward-compat
plumbing — there's no per-user session concept wired into `ExamPrepBot`/`get_bot()` yet, since auth is
still the single shared `APP_API_KEY`). Summarization uses a dual trigger, whichever fires first:
every `memory_summarize_every_n_turns` turns, or `memory_summarize_token_threshold` cumulative tokens
since the last summary — both an O(1) check against running-total counters
(`ChatSession.turn_count`/`ChatMessageRecord.token_count`), never a COUNT/SUM scan. Token counts are a
word-count proxy (`_estimate_tokens`), not a real tokenizer. At retrieval time,
`AdaptiveRetriever.retrieve()` only ever tries a semantic-memory search (`content_types=["memory"]`,
`settings.memory_relevance_threshold`) when document retrieval misses (`in_scope=False`) — never
blended into document results, to avoid diluting citations (`ConfidenceScorer` assumes
document-sourced chunks).

**Route handlers offload blocking work to a thread pool.** `answer_question`, `add_document`,
`remove_document`, embedding `encode()`, and LLM provider calls are all synchronous/CPU- or
network-bound; every route in `app/api/documents.py` and `app/api/queries.py` wraps these calls in
`fastapi.concurrency.run_in_threadpool` so a slow request doesn't block the whole event loop (and
therefore every other concurrent request, including health checks). The WebSocket route bridges the
orchestrator's `on_stage` callback (fired from the worker thread) back onto the event loop via
`loop.call_soon_threadsafe` into an `asyncio.Queue`, drained by a concurrent task — see `app/api/queries.py`.

**Configuration** (`app/core/config.py`): a single Pydantic `Settings` object loaded from `.env`,
imported everywhere as `from app.core.config import settings`. Adding a setting means adding a field
here (and to `.env.example`) — there is no other config path.

**`src/` and `config/` at the repo root are empty leftovers from a prior restructuring** (see git log)
and are not part of the current app — all real code lives under `app/`.

## Logging

Loguru is configured in `app/main.py`: colorized stderr at `settings.log_level`, plus a full DEBUG-level
rotating file at `settings.log_file` (`./logs/app.log`, 10 MB rotation, 7-day retention, zipped). Every
pipeline stage and provider call logs structured, greppable lines (e.g. `[ORCHESTRATOR] Stage 2 — ...`,
`[FAISS] ...`, `[SEARCH] ...`, `[INTENT] ...`, `[RERANK] ...`, `[MEMORY] ...`, `[PARSER] ...`) — match
this style when adding logging to new services rather than introducing a new format.
