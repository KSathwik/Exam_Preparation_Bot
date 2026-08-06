# AI Knowledge Assistant — Complete Project Documentation

A comprehensive reference covering what the project is, the full technology stack, every algorithm
and technique used, the system architecture, database schema, API surface, security model, testing
strategy, and deployment setup. For a diagram-first view of the pipeline see `docs/ARCHITECTURE.md`;
for coding-agent-facing conventions see `CLAUDE.md`; for what's intentionally deferred see
`docs/PHASE_2_ROADMAP.md`.

---

## 1. What This Project Is

**AI Knowledge Assistant** is a production-ready Retrieval-Augmented Generation (RAG) platform powered by **Hybrid RAG**, **Context-Aware Generation (CAG)**, and **Multi-Agent Orchestration**. It enables users to upload document collections (PDF/DOCX) and ask natural-language questions about them. Instead of relying purely on an LLM's parametric knowledge, every answer is grounded in the uploaded document context, with citations back to the exact page/section the claim came from, plus a confidence score and a hallucination-risk rating.

> **Demonstration Preset**: The platform includes an out-of-the-box **Exam Preparation & Study Assistant** preset demonstrating intent-driven revision tools (flashcards, MCQs, mark-based exam answers, comparison tables, and summary notes).

Beyond a single-shot Q&A loop, it runs a **multi-agent pipeline**: a deterministic orchestrator coordinates intent classification, hybrid RAG / CAG routing, LLM answer drafting, a mandatory second-pass **self-reflection/quality-control** LLM call, citation/confidence scoring, and persistent conversation memory with semantic recall across sessions.

---

## 2. Technology Stack

### Backend
| Layer | Technology | Purpose |
|---|---|---|
| Web framework | **FastAPI** (`>=0.140.0`) | Async REST + WebSocket API, OpenAPI/Swagger docs at `/docs` |
| ASGI server | **Uvicorn** (`[standard]`) | Dev server; **Gunicorn** (`uvicorn.workers.UvicornWorker`) in production |
| Validation | **Pydantic v2** + **pydantic-settings** | Request/response schemas, typed settings from `.env` |
| ORM | **SQLAlchemy 2.0** | `DocumentRecord`/`QueryRecord`/`ChatSession`/`ChatMessageRecord`/`ConversationMemory` |
| Migrations | **Alembic** | Schema migrations (`alembic/versions/`) — `create_all()` alone can't alter existing tables |
| Database | **SQLite** (dev) / **PostgreSQL 14** (Docker/prod) | Relational metadata store (documents, queries, chat history, memory) |
| Vector index | **FAISS** (`faiss-cpu`, `IndexFlatL2`) | Dense similarity search over embedded chunks |
| Lexical search | **rank-bm25** (`BM25Okapi`) | Keyword/lexical signal blended with dense search (hybrid retrieval) |
| Embeddings | **sentence-transformers** — `BAAI/bge-small-en-v1.5` (384-dim) | Bi-encoder for document/query embeddings |
| Reranking (opt-in) | **sentence-transformers** `CrossEncoder` — `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision reranking of the merged candidate pool |
| Classical ML | **scikit-learn** (`cosine_similarity`) | Semantic-fallback intent classification |
| PDF parsing | **pdfplumber** | Text + per-word font-size/position extraction |
| DOCX parsing | **python-docx** | Paragraph text + heading-style detection |
| LLM SDKs | **anthropic**, **openai**, **google-genai** | Multi-provider LLM access (Claude / GPT / Gemini) |
| Auth | Custom `APIKeyHeader`-based middleware | `X-API-Key` header (REST) / `?api_key=` query param (WebSocket) |
| Rate limiting | **slowapi** | Per-IP throttling on LLM-calling endpoints |
| Caching (available) | **diskcache**, **redis** | Query/result caching infrastructure |
| Background tasks (available) | **celery** | Task queue infrastructure (not yet wired to a route) |
| Logging | **loguru** | Structured console + rotating file logs |
| Testing | **pytest**, **pytest-cov**, **pytest-asyncio** | 210+ tests, fully isolated from real project data |
| Lint/format/type-check | **black**, **isort**, **mypy** (advisory), **pylint** | Enforced in CI (black/isort); mypy non-blocking |

### Frontend
| File | Purpose |
|---|---|
| `frontend/index.html` | Single-page UI shell |
| `frontend/app.js` (~400 lines) | Vanilla JS — WebSocket client, chat rendering, document upload, streaming display |
| `frontend/style.css` (~475 lines) | Styling, no framework/build step |

No React/Vue/build tooling — a deliberately dependency-free single-page app served directly by FastAPI's
`StaticFiles` mount, with the active API key auto-injected into the served HTML so the bundled UI needs
no manual key entry (direct API calls from other clients still require the header).

### Infrastructure / DevOps
| Component | Technology |
|---|---|
| Containerization | **Docker** + **docker-compose** (`api`, `db` (Postgres), `redis`, `nginx`) |
| Reverse proxy | **Nginx** (`nginx.conf`) — TLS termination point for production |
| CI | **GitHub Actions** (`.github/workflows/ci.yml`) — black/isort check, full test suite, Docker build, on every push/PR to `main`/`develop` |
| Process manager (prod) | **Gunicorn** with Uvicorn workers |

---

## 3. System Architecture

```
Client (browser/API) ──REST──▶ /api/ask, /api/query, /api/batch, /api/documents/*
                     ──WS────▶ /api/ws (streaming)
                          │
                          ▼
                 app/api/* routers
                          │
                          ▼
          app.core.dependencies (DI composition root — @lru_cache singletons:
          SentenceTransformer, VectorStoreManager, IntentClassifier, ExamPrepBot)
                          │
                          ▼
                    ExamPrepBot  (app/services/pipeline.py)
                          │  (one-line delegation)
                          ▼
                 OrchestratorAgent  (app/services/agents/orchestrator.py)
              ┌───────────┼────────────┬─────────────┬─────────────┐
              ▼           ▼            ▼             ▼             ▼
   IntentClassifier  RetrievalAgent  KnowledgeAgent  ReflectionAgent  MemoryAgent
```

**Composition root pattern**: every expensive resource (embedding models, the FAISS-backed vector
store, the intent classifier, the bot itself) is constructed exactly once via `@lru_cache(maxsize=1)`
factories in `app/core/dependencies.py` and threaded through constructors — nothing reaches for a
global singleton directly. This is also what makes the test suite fast and isolated: a `_reset_singletons`
autouse fixture clears every cache between tests.

**Multi-agent pipeline** (deterministic Python coordinator, not an LLM router — see rationale below):

| Stage | Agent | What it does |
|---|---|---|
| 1 | *(direct call)* | `IntentClassifier.classify()` — hybrid rule + semantic classification into 8 intent types |
| 2 | `RetrievalAgent` | Hybrid (dense+BM25) search via `HybridRetriever`; falls back to semantic memory if document search misses |
| 3 | `KnowledgeAgent` | LLM drafts the answer + extracts chunk-indexed factual claims |
| 4 | *(plain classes, no LLM)* | `SpanExtractor`/`ConfidenceScorer` — citation matching + confidence/hallucination scoring |
| 5 | `ReflectionAgent` | **Mandatory** second LLM call: reviews the draft for hallucination/clarity/structure, can revise or block it |
| 6 | `MemoryAgent` | Records the turn (in-memory always; DB + semantic embedding when `persist=True`) |

Why the orchestrator is plain Python rather than an LLM-based router: with only one downstream
answer-producing agent this phase (Knowledge), there is nothing for an LLM to meaningfully decide
between — that becomes justified once Phase 2 agents (Quiz/StudyPlanner/...) exist to route to.

---

## 4. Algorithms & Techniques Used

This is the heart of "what algorithms did we use" — enumerated by subsystem.

### 4.1 Retrieval — Hybrid Search
- **Dense vector search**: query and document chunks are embedded with a sentence-transformer
  bi-encoder (`BAAI/bge-small-en-v1.5`, 384 dimensions), indexed in a **FAISS `IndexFlatL2`**
  (brute-force exact L2/Euclidean nearest-neighbor search — chosen over approximate indices like
  HNSW/IVF because query latency is a non-issue at this project's scale; see `docs/PHASE_2_ROADMAP.md`
  for the reassessment trigger at ~200k vectors).
  L2 distance is converted to a bounded similarity score via `similarity = max(0, 1 - distance/2)`.
- **Lexical search — BM25 (Okapi variant)**: a classic probabilistic ranking function
  (`rank-bm25`'s `BM25Okapi`) scores chunks by term-frequency/inverse-document-frequency-weighted
  overlap with the query — catches exact-keyword matches that a purely semantic embedding can miss.
  The index is lazily (re)built in-memory whenever the corpus changes (add/remove/reset).
- **Score fusion (hybrid retrieval)**: dense and BM25 scores are **min-max normalized independently
  across the current candidate set**, then linearly blended:
  `final_score = α · dense_norm + (1−α) · bm25_norm`, with `α = settings.hybrid_dense_weight`
  (default `0.6`, semantic-weighted). Min-max normalization maps each score list to `[0, 1]`; when
  every candidate ties (no differentiating signal), that dimension contributes `0` rather than a
  false "perfect match" of `1`.
- **Adaptive top-k by intent**: retrieval breadth (`top_k`) varies by classified query intent (e.g.
  `DEFINITION → 3`, `EXPLAIN/COMPARE → 8`, `VAGUE → 10`) — a lookup table (`_TOP_K_STRATEGY`), not a
  learned parameter.
- **Cross-encoder reranking (opt-in, off by default)**: a `CrossEncoder`
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) jointly scores `(query, chunk)` pairs — more accurate than
  a bi-encoder's independent embeddings, but O(n) LLM-scale inference per candidate, so it's applied
  only to the small merged candidate pool, only when explicitly enabled.
- **Deterministic intent-based reranking**: a final tie-break pass — `PROCESS` intent sorts
  chronologically (page, then chunk index) to preserve step order; `DEFINITION` intent prefers
  chunks from a document's `"beginning"` position (definitions usually appear early).
- **Semantic-memory retrieval fallback**: when document retrieval's best score misses
  `relevance_threshold`, a *separate* search over `content_type="memory"` embeddings (past
  conversation summaries) is tried against a stricter `memory_relevance_threshold`, before finally
  giving up. Never blended with document results — keeps `ConfidenceScorer`'s assumptions intact.

### 4.2 Intent Classification — Hybrid Rule + Semantic
1. **Rule-based pass** (`_classify_by_rules`): fast keyword/regex matching against per-intent phrase
   lists — near-zero latency, handles the common case.
2. **Semantic fallback** (`_classify_by_semantics`): if no rule clears `intent_classification_threshold`
   (0.7 default), the query is embedded and compared via **cosine similarity**
   (`sklearn.metrics.pairwise.cosine_similarity`) against precomputed template embeddings for each of
   the 8 `QueryType` values (definition/explain/compare/process/example/diagram/vague/homework); the
   closest template wins.

### 4.3 Document Chunking — Structural + Word-Count Windowing
- **Structural block segmentation** *before* sentence accumulation:
  - **DOCX**: `paragraph.style.name` gives exact heading levels (`"Heading 1"`, `"Title"`, ...) — no
    heuristic needed, headings become block boundaries and `section_title`/`section_level` metadata.
  - **PDF**: a **font-size heuristic** — words are grouped into lines by vertical position, a page's
    "body" font size is taken as the statistical mode of word sizes (`collections.Counter`), and any
    line averaging ≥15% larger than body size (and short enough to plausibly be a heading, not a
    large-font paragraph) becomes a new block boundary. Falls back to one block per page when no
    reliable signal exists (common for plain lecture PDFs) — a wrong heading guess is worse than none.
- **Sentence-accumulation chunking**: within each block, sentences (split on `.`/`!`/`?` boundaries)
  are greedily packed up to `max_chunk_size` words, with the last 2 sentences of an emitted chunk
  carried forward as overlap context into the next chunk.
- **No-data-loss trailing-fragment handling**: a final fragment under `min_chunk_size` is never
  dropped — merged into the previous chunk from the same block if one exists, else carried forward
  into the next block's accumulation, else force-emitted as its own chunk. Guarantees a non-empty
  document never yields zero chunks.
- **Deferred metadata computation**: `chunk_position` (beginning/middle/end) and page-number
  estimates are computed only once the *final* total chunk count is known, not against a
  placeholder — otherwise position/page metadata is silently wrong for every chunk after the first.

### 4.4 Citation & Confidence Scoring
- **Claim extraction**: the LLM extracts factual claims from its own draft answer and — critically —
  tags each claim with which numbered source excerpt(s) it came from (`{"claim": ..., "chunks": [1,3]}`),
  turning citation matching from a blind full-corpus search into a narrowed, targeted one.
- **Fuzzy span matching**: `difflib.SequenceMatcher` computes a **ratio-based string similarity**
  (essentially longest-matching-block-based, Ratcliff/Obershelp-style) between each claim and sliding
  windows of **1, 2, and 3 consecutive sentences** from the candidate chunk(s) — a claim often
  paraphrases a short run of sentences, not exactly one.
- **Confidence formula** (`ConfidenceScorer.calculate_answer_confidence`):
  `confidence = avg_relevance × 0.4 + citation_rate × 0.4 + source_consistency × 0.2`, where
  `source_consistency = 1 / (1 + (distinct_pages − 1) × 0.2)` — an answer drawing from many scattered
  pages is penalized slightly relative to one drawing from a focused, consistent source region.
- **Hallucination risk banding**: `confidence > 0.85 and citation_rate > 0.8 → "low"`;
  `confidence > 0.6 and citation_rate > 0.5 → "medium"`; else `"high"` — simple threshold bands, not a
  learned classifier.

### 4.5 Reflection — LLM Self-Critique Pattern
A second, independent LLM call reviews the drafted answer against the same source excerpts for
factual consistency, hallucination, clarity, structure, completeness, and exam relevance, returning a
structured verdict (`revised_answer`, `materially_changed`, `should_block`, `issues_found`). This is
the "LLM-as-judge"/self-refine pattern applied specifically as a safety/quality gate rather than for
iterative improvement — it runs exactly once per answer (bounded cost), not in an unbounded
critique-and-retry loop.

### 4.6 Conversation Memory — Dual-Trigger Summarization
- **Running-total token counters**: each persisted message stores a *cumulative* word-count-proxy
  token total (not per-message length), so checking "how many tokens since the last summary" is an
  O(1) subtraction, never a `SUM()`/`COUNT()` scan over history.
- **Dual trigger, whichever fires first**: every `memory_summarize_every_n_turns` turns (a simple
  modulo check on `turn_count`), or `memory_summarize_token_threshold` cumulative tokens since the
  last summary.
- **Summarize → embed → persist**, in that order with a defensive `embedded` flag: the summary text
  row is written to the DB *before* the embedding call, so a crash between the two never loses the
  summary — an unembedded row is simply invisible to semantic search and can be retried later.

### 4.7 Security & Reliability Primitives
- **Constant-time API key comparison** (`secrets.compare_digest`) — prevents timing-attack key
  recovery.
- **Cryptographically secure random key generation** (`secrets.token_urlsafe(32)`) when no
  `APP_API_KEY` is configured — secure-by-default rather than an open API.
- **Atomic file persistence** (write-temp-then-`os.replace`) for the FAISS index files — rename is
  atomic on both POSIX and NTFS, so a crash or concurrent reader never observes a half-written file.
- **Consistency validation on load**: `load_index()` refuses a set of index/metadata/embedding files
  whose counts don't all agree, rather than silently operating on a desynced index.
- **Thread-safety via a single lock**: every FAISS index read/write is serialized behind one
  `threading.Lock`, since FastAPI route handlers call into it from a thread pool and FAISS isn't
  safe for concurrent mutation.

---

## 5. Multi-LLM Provider Support

`ClaudeInterface()` (a factory function, not a class — kept for backward-compat naming) dispatches on
`settings.llm_provider` to construct `_AnthropicLLM` / `_OpenAILLM` / `_GeminiLLM`, all sharing common
prompt-building, claim-extraction, and structured-answer logic via a `_BaseLLM` superclass — each
subclass implements only its provider's `_call()`. Only the API key matching the active provider is
required at runtime.

| Provider | SDK | Default model |
|---|---|---|
| Anthropic | `anthropic` | `claude-sonnet-5` |
| OpenAI | `openai` | `gpt-4o-mini` |
| Gemini | `google-genai` (`google-generativeai` is EOL, not used) | `gemini-2.0-flash` |

Current-generation Claude models reject `temperature`/`top_p`/`top_k` outright (HTTP 400) rather than
deprecating them, so `_AnthropicLLM._call()` omits them and steers behavior via prompting instead.
Claude responses are also parsed defensively — a `ThinkingBlock` can precede the actual text block in
`response.content`, so text is extracted by filtering for `type == "text"` blocks rather than
indexing `content[0]` directly (a real bug found and fixed via live end-to-end testing this phase).

---

## 6. Database Schema

| Table | Purpose | Key columns |
|---|---|---|
| `documents` | Every uploaded document | `id`, `file_name`, `file_type`, `total_chunks`, `upload_path` |
| `queries` | Every question asked | `document_id` (FK), `query_text`, `answer_text`, `intent`, `overall_confidence`, `hallucination_risk` |
| `chat_sessions` | Persistent chat session (short-term/episodic memory backing) | `user_id` (nullable, forward-compat), `turn_count`, `last_summarized_message_id` |
| `chat_messages` | Individual turns | `session_id` (FK), `role`, `content`, `token_count` (cumulative running total) |
| `conversation_memories` | LLM-generated summaries, embedded for semantic recall | `session_id` (FK), `summary_text`, `covers_from_message_id`/`covers_to_message_id`, `embedded` (bool) |

Migrations are managed with **Alembic** (`alembic/versions/`) — two migrations exist: a baseline
capturing the original 4-table schema, and a second adding the memory-related columns/table.
`Base.metadata.create_all()` (used at app startup) only creates *missing* tables; it never alters
existing ones, which is why real migrations were introduced rather than relying on it alone.

**On-disk vector storage** (not a SQL table): `index.faiss` + `metadata.pkl` + `embeddings.npy` under
`FAISS_INDEX_PATH`, holding both document chunks and memory-summary embeddings in one shared index,
discriminated by a `content_type` field (`"document"` | `"memory"`) in each entry's metadata.

---

## 7. API Reference

### REST
| Method | Path | Auth | Purpose |
|---|---|:---:|---|
| GET | `/` | – | Serves the frontend UI (API key auto-injected if enabled) |
| GET | `/health`, `/ready`, `/version`, `/config` | – | Health/readiness/version/public-config checks |
| GET | `/logs/tail` | ✓ | Tail the application log file |
| POST | `/api/documents/upload` | ✓ | Upload a PDF/DOCX (parsed, chunked, embedded, indexed) |
| GET | `/api/documents/list` | ✓ | List uploaded documents |
| DELETE | `/api/documents/{id}` | ✓ | Remove a document and its vectors |
| GET | `/api/documents/stats` | ✓ | Document/vector-store statistics |
| POST | `/api/ask`, `/api/query` | ✓ | Ask a question (rate-limited) |
| POST | `/api/batch` | ✓ | Ask multiple questions in one call (rate-limited) |
| GET | `/api/intent/{query}` | ✓ | Classify a query's intent without answering it |
| GET | `/api/history` | ✓ | Retrieve chat history |
| DELETE | `/api/history` | ✓ | Clear chat history |
| POST | `/api/search` | ✓ | Raw retrieval (no LLM call) — debugging/inspection |
| GET | `/api/system/config` | – | Public-safe runtime config summary |
| GET | `/api/metrics` | ✓ | Vector store / chat history / model metrics |

### WebSocket
| Path | Auth | Purpose |
|---|:---:|---|
| `/api/ws` | ✓ (`?api_key=`) | Streaming Q&A — see message contract below |

**WebSocket message contract** (additive; REST response shape is untouched):
`intent` → `chunk` (`stage: "draft"`, streamed as the draft answer is ready) → `status`
(`stage: "reflecting"`, sent once before the second LLM call) → `chunk` (`stage: "final"`, the
possibly-revised answer) → `complete` (full structured result) — or `error` at any point. This is an
honest two-stage progressive reveal, not true token-level LLM streaming.

---

## 8. Security Model

- **API-key authentication** (`app/core/security.py`): every endpoint except health/version/config
  requires `X-API-Key` (REST) or `?api_key=` (WebSocket, since browsers can't set custom WS headers).
  Secure by default — a random key is generated and logged if none is configured.
- **Per-IP rate limiting** (`app/core/rate_limit.py`, slowapi): throttles `/api/ask`, `/api/query`,
  `/api/batch` to `RATE_LIMIT_PER_MINUTE` (default 30) — the endpoints that call an LLM provider and
  cost real money per request.
- **Path traversal protection**: uploaded filenames are sanitized (directory-traversal and
  absolute-path components stripped) before being used to build an on-disk upload path.
- **CORS**: configurable allow-list (`CORS_ORIGINS`), wildcard only for local dev.
- **Secrets never in git**: `.env` is gitignored; `.env.example` documents every variable with safe
  placeholder/default values.

---

## 9. Testing Strategy

- **210+ tests**, `pytest` + `pytest-cov` + `pytest-asyncio`, fully isolated from real project data —
  `tests/conftest.py` redirects `FAISS_INDEX_PATH`/`UPLOAD_DIR`/`CACHE_DIR` to a fresh temp directory
  and an autouse fixture clears every `@lru_cache` singleton between tests.
- **No real network calls in the test suite**: LLM provider clients and `SentenceTransformer` are
  always mocked (`MagicMock`/monkeypatched constructors) — tests validate logic and contracts, not
  live model behavior.
- **In-memory SQLite** (`StaticPool`) for database-touching tests (e.g. `MemoryAgent` persistence),
  so DB tests run without a real Postgres instance.
- **Live smoke testing** (manual, not part of CI): the full pipeline was also exercised against the
  real Anthropic API end-to-end over a live WebSocket connection — this is what surfaced and led to
  fixing two real production bugs (a `ThinkingBlock`-vs-text-block parsing bug, and a claims-JSON
  truncation/escaping issue) that no mock-based test could have caught.

---

## 10. Configuration Reference

All settings are a single Pydantic `Settings` object (`app/core/config.py`) loaded from `.env` —
there is no other config path. Full list with defaults lives in `.env.example`; notable groups:

| Group | Key settings |
|---|---|
| LLM | `LLM_PROVIDER`, `*_API_KEY`, `MODEL_NAME`, `MAX_TOKENS`, `TEMPERATURE`, `TOP_P` |
| Chunking | `MAX_CHUNK_SIZE` (512), `CHUNK_OVERLAP` (100), `MIN_CHUNK_SIZE` (100) |
| Embeddings | `EMBEDDING_MODEL` (`BAAI/bge-small-en-v1.5`), `VECTOR_DIMENSION` (384) |
| Retrieval | `RETRIEVAL_TOP_K`, `RELEVANCE_THRESHOLD`, `HYBRID_DENSE_WEIGHT` (0.6) |
| Reranking | `ENABLE_CROSS_ENCODER_RERANK` (false), `CROSS_ENCODER_MODEL` |
| Memory | `MEMORY_RELEVANCE_THRESHOLD`, `MEMORY_SUMMARIZE_EVERY_N_TURNS` (10), `MEMORY_SUMMARIZE_TOKEN_THRESHOLD` (2000) |
| Auth | `APP_API_KEY`, `API_AUTH_ENABLED`, `RATE_LIMIT_PER_MINUTE` |
| Infra | `DATABASE_URL`, `FAISS_INDEX_PATH`, `UPLOAD_DIR`, `CACHE_DIR`, `CORS_ORIGINS` |

---

## 11. Deployment

- **Local dev**: `venv` + `uvicorn main:app --reload`, SQLite, local FAISS index on disk.
- **Docker**: `docker-compose up -d` brings up `api` (FastAPI/Gunicorn), `db` (Postgres 14), `redis`,
  and `nginx` (reverse proxy, TLS termination point) on an internal network — only `api` (8000) and
  `nginx` (80) are published to the host.
- **Migrations in production**: `alembic upgrade head` must be run against the target database before
  starting the app — `init_db()`'s `create_all()` only creates missing tables.
- **CI** (`.github/workflows/ci.yml`): `black --check`, `isort --check-only`, full test suite, and a
  Docker build, on every push/PR to `main`/`develop`.

---

## 12. What's Deliberately Not Built Yet

See `docs/PHASE_2_ROADMAP.md` for full detail. In short: Quiz/Flashcard/NotesGenerator/StudyPlanner
agents, progress tracking, real multi-user login (schema is forward-compatible via nullable `user_id`
columns, but auth today is still a single shared `APP_API_KEY`), a larger FAISS index type
(`IndexIVFFlat`, only worth it past ~200k vectors), multilingual embeddings, and always-on
cross-encoder reranking — all deferred by explicit decision, not oversight.
