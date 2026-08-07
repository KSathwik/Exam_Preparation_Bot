# AI Knowledge Assistant

The **AI Knowledge Assistant** built on **Hybrid RAG**, **Context-Aware Generation (CAG)**, and **Multi-Agent Orchestration**. Upload custom documents (PDF/DOCX) and ask questions in a ChatGPT/Claude-style interface, grounded entirely in your uploaded context — every answer carries claim-level citations, confidence scoring, hallucination-risk assessment, and active reflection quality control.

> **Demonstration Application**: Includes a built-in **Exam Preparation & Study Assistant** domain preset out-of-the-box, demonstrating intent-driven revision tools (flashcards, MCQs, mark-based exam answers, comparison tables, and summary notes).

**Core Experience**
- Persistent, per-browser conversation history — auto-titled, searchable, renameable, deletable
- Streaming answers with live progress ("Searching your documents…" → "Drafting an answer…" →
  "Reviewing the draft…") instead of a silent multi-second wait
- **Intent-driven, not prompt-driven**: a deterministic format classifier detects what shape of
  answer you actually want — key points, summaries, revision notes, comparison tables, flashcards,
  MCQs, one-line/two/five/ten-mark answers, and more — from ordinary phrasing ("in 5 points",
  "TL;DR", "one line answer", "compare X and Y"), no manual format menu required
- Markdown, syntax-highlighted code, tables, and LaTeX math rendering in every response
- Copy / regenerate / edit / stop on every message, plus contextual "Summarize" / "Explain simpler" /
  "Explain in detail" follow-up chips under the latest answer
- **AI Hub** sidebar section (ChatGPT/Claude/Gemini/Perplexity + Wikipedia) and a per-message
  "Open in" action — continue researching a question in another AI or trusted knowledge source in
  one click, opened in a new tab with the question carried over wherever the provider supports it
- Retrieval scoped to the document(s) uploaded in the current conversation first, so answers stay
  grounded in what you just asked about instead of blending in unrelated older uploads
- Small talk ("hi", "thanks", "bye") is answered instantly with a generic reply, never routed through
  the RAG pipeline
- Light/dark/system theme, fully responsive (sidebar becomes an off-canvas drawer on mobile)

**Underneath**
- Multi-agent RAG pipeline (Orchestrator → Retrieval → Knowledge → Reflection → Memory), with an
  always-on reflection pass that re-checks every draft against the source excerpts before it's shown
- **Hybrid RAG + CAG retrieval**: when a conversation's uploaded document(s) are small enough to fit
  an LLM call economically, the whole document goes straight into context instead of a
  similarity-ranked chunk subset — no risk of retrieval dropping a chunk the answer needed. Falls
  back to hybrid dense (FAISS) + lexical (BM25) retrieval, with optional cross-encoder reranking,
  once the scope is too large
- Two orthogonal, fully deterministic classifiers (no extra LLM calls) parameterize every answer:
  intent (what the question is about — drives retrieval tuning) and response format (how it should
  look — drives the drafting prompt and a final formatting pass)
- Per-claim citation extraction, confidence scoring, and hallucination-risk assessment (no LLM call),
  plus a deterministic pre-reflection shortcut that skips the second LLM call outright when a draft
  is already unambiguously ungrounded
- Multi-provider LLM support — Anthropic Claude, OpenAI, Google Gemini, or local Ollama (Llama 3.2 / Llama 3.1 / custom local models), switchable via configuration
- FastAPI backend with async/await and WebSocket streaming
- SQLAlchemy + Alembic migrations for documents, conversations, and semantic conversation memory
- API-key authentication + rate limiting on sensitive/expensive endpoints
- Docker Compose deployment (PostgreSQL + Redis + Nginx)
- 350+ tests, fully isolated from real project data

## Quick Start

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then set LLM_PROVIDER + the matching API key
uvicorn main:app --reload
```

Open http://localhost:8000 for the UI, or http://localhost:8000/docs for Swagger.

On first run with no `APP_API_KEY` set, a temporary API key is generated and printed to the
console/log — copy it, or set `APP_API_KEY` in `.env` for a key that survives restarts (see
[Authentication](#authentication) below).

## Docker

```bash
docker-compose up -d
```

`docker-compose.yml` reads `.env` for `LLM_PROVIDER`, the provider API keys, `APP_API_KEY`, and
every other setting — only `DATABASE_URL`/`REDIS_URL` are overridden to point at the `db`/`redis`
service containers. Nginx (port 80) reverse-proxies everything to the API container; there's no
separate frontend build step, the FastAPI app serves the UI directly.

## Project Structure

```
main.py                    # Entry point (delegates to app/)
app/
  main.py                  # FastAPI app, middleware, routers, lifespan
  core/
    config.py              # Unified Pydantic settings
    database.py             # SQLAlchemy engine + session
    dependencies.py         # Singleton DI (models, vector store, bot)
    security.py             # API-key auth (require_api_key / require_api_key_ws)
    rate_limit.py           # Per-IP request throttling on LLM-calling routes
  api/
    health.py               # /health, /ready, /version, /config, /logs/tail
    documents.py             # /api/documents/upload, list, delete, stats
    queries.py               # /api/ask, /query, /intent, /batch, /history, /search, /ws
    conversations.py          # /api/conversations — list/get/rename/delete chat history
  models/
    schemas.py               # Pydantic request/response schemas
    db_models.py              # SQLAlchemy ORM models
  services/
    pipeline.py               # ExamPrepBot — delegates to OrchestratorAgent
    small_talk.py              # Greeting/thanks/farewell short-circuit (skips the RAG pipeline)
    agents/                     # Multi-agent pipeline (see docs/ARCHITECTURE.md)
      orchestrator.py             # Deterministic coordinator, all stages
      retrieval_agent.py            # Wraps HybridRetriever (RAG search + CAG full-context)
      knowledge_agent.py             # Wraps LLM structured-answer generation
      reflection_agent.py              # Wraps LLM quality-control pass
      memory_agent.py                   # chat_history + opt-in DB persistence
    parser.py                  # PDF/DOCX parsing + structural chunking
    embeddings.py                # Embedding generation + FAISS store + BM25 hybrid
    intent_classifier.py          # Hybrid rule + semantic classifier (what the question is about)
    format_classifier.py           # Deterministic response-format classifier (how the answer should look)
    response_formats.py             # ResponseFormat/ResponseLength enums + per-format prompt templates
    response_formatter.py            # Deterministic post-generation cleanup (length caps, boilerplate)
    context_router.py                 # Deterministic RAG-vs-CAG routing (whole-doc context vs ranked retrieval)
    retriever.py                        # Adaptive retrieval + reranking + document scoping + memory fallback
    llm_interface.py                      # Multi-provider LLM client (Gemini/OpenAI/Anthropic)
    validator.py                           # Citation extraction + confidence scoring
frontend/
  index.html               # Single-page UI shell (sidebar + chat column + settings dialog)
  style.css                # Design system (CSS custom properties, light/dark themes)
  js/                       # ES modules, no build step — CDN-only third-party libs
    main.js                    # Entry point, wires everything together
    state.js                    # Shared in-memory + localStorage state (device/session/theme)
    api.js                       # REST/WebSocket client wrappers
    theme.js                      # Light/dark/system theme toggle
    markdown.js                    # marked.js -> DOMPurify -> KaTeX rendering pipeline
    chat.js                         # Message rendering + streaming + study-tool chips
    input.js                         # Composer: auto-grow, drag-drop upload, send/stop
    sidebar.js                        # Conversation list: search, switch, rename, delete
    settings.js                        # Settings dialog (theme, developer options)
    aiHub.js                            # Sidebar AI Hub section (always-visible provider shortcuts)
    continueResearch.js                  # Per-message "Open in" popup
    researchProviders.js                  # Config-driven provider definitions shared by both
alembic/                   # DB migrations (see CLAUDE.md Commands)
docs/
  ARCHITECTURE.md          # Agent workflow, memory tiers, retrieval flow diagrams
  PHASE_2_ROADMAP.md       # Deliberately deferred scope
tests/                     # pytest test suite
```

## Chat UI

The frontend is deliberately framework-free and build-step-free — plain ES modules
(`<script type="module">`) loaded directly by the browser, with third-party libraries (marked.js,
highlight.js, KaTeX, DOMPurify) pulled from a CDN rather than bundled. Conversations are scoped per
browser, not per real user account: a `device_id` (`crypto.randomUUID()`, persisted in
`localStorage`) identifies "this browser" and a `session_id` identifies one conversation, both sent
alongside every question. There's no login yet — see `docs/PHASE_2_ROADMAP.md` for the multi-tenancy
plan — so anyone who can reach the page can see conversations created from that same browser/device
combination.

**AI Hub**: a configuration-driven provider list (`researchProviders.js`) drives both the
always-visible sidebar section (`aiHub.js`) and the per-message "Open in" popup
(`continueResearch.js`) — adding a new provider (Stack Overflow, MDN, arXiv, ...) is a one-entry
config change, nothing else. Each provider opens in a new, unrelated tab
(`target="_blank"`-equivalent + `noopener,noreferrer`); the current question is carried over via a
verified query parameter where the provider actually supports it (confirmed by hand for each:
Claude and Perplexity do, ChatGPT and Gemini silently strip it) and falls back to the plain homepage
otherwise — never a guessed/unsupported URL shape. None of this touches the running conversation;
it's just a new tab.

## Authentication

Every endpoint except `/health`, `/ready`, `/version`, `/config`, and `/api/system/config` requires
an API key by default (`API_AUTH_ENABLED=true`). Pass it as a header on REST calls:

```bash
curl http://localhost:8000/api/documents/list -H "X-API-Key: <your-key>"
```

The WebSocket endpoint (`/api/ws`) can't receive custom headers from a browser, so it takes the key
as a query parameter instead: `ws://localhost:8000/api/ws?api_key=<your-key>`.

Set `API_AUTH_ENABLED=false` in `.env` to disable auth entirely (e.g. for local-only experimentation).

## Rate Limiting

`/api/ask`, `/api/query`, and `/api/batch` — the endpoints that call the LLM provider — are limited
to `RATE_LIMIT_PER_MINUTE` (default 30) requests per client IP. Exceeding it returns `429 Too Many
Requests`.

## Environment Variables

All settings are loaded from `.env` via Pydantic (`app/core/config.py`) — see `.env.example` for the
full list with defaults. Notable groups:

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | LLM backend to use: `gemini`, `openai`, `anthropic`, or `ollama` |
| `*_API_KEY` | API keys (required for cloud providers `gemini`, `openai`, `anthropic`) |
| `OLLAMA_BASE_URL` | Base URL for Ollama local server (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | Ollama model to use (default: `llama3.2`) |
| `OLLAMA_TIMEOUT` | Ollama request timeout in seconds (default: `120.0`) |
| `OLLAMA_KEEP_ALIVE` | Ollama model keep-alive duration (default: `5m`) |
| `OLLAMA_NUM_CTX` | Context window size for Ollama inference (default: `4096`) |
| `APP_API_KEY`, `API_AUTH_ENABLED` | Auth for this app's own API (not an LLM key) |
| `RATE_LIMIT_PER_MINUTE` | Per-IP throttle on LLM-calling endpoints |
| `DATABASE_URL` | SQLite by default; use a `postgresql://` URL in Docker/production |
| `FAISS_INDEX_PATH`, `UPLOAD_DIR`, `CACHE_DIR` | On-disk data locations |
| `MAX_FILE_SIZE_MB`, `ALLOWED_FILE_TYPES` | Upload constraints |
| `CORS_ORIGINS` | Comma/JSON-list of allowed origins — restrict this in production |
| `EMBEDDING_MODEL` | Changing this on a deployment with an existing index requires a full reindex |
| `HYBRID_DENSE_WEIGHT` | Dense vs. BM25 weight in hybrid retrieval (see `docs/ARCHITECTURE.md`) |
| `ENABLE_CROSS_ENCODER_RERANK` | Opt-in reranking of the merged candidate pool, off by default |
| `MEMORY_RELEVANCE_THRESHOLD`, `MEMORY_SUMMARIZE_*` | Conversation-memory tuning (semantic recall across turns once a conversation summarizes) |
| `ENABLE_CAG`, `CAG_TOKEN_BUDGET` | Whole-document context instead of ranked retrieval when a conversation's document scope fits the budget (word-count proxy, default 6000) |
| `ENABLE_REFLECTION_SHORTCUT`, `REFLECTION_SHORTCUT_CONFIDENCE_FLOOR` | Skip the second (reflection) LLM call when a draft is already unambiguously ungrounded — saves a full round-trip |

## Running with Local Ollama

The Exam Prep Bot supports **Ollama** as a first-class provider alongside cloud LLMs.

1. **Install and start Ollama**:
   Download Ollama from [ollama.com](https://ollama.com) and start the service:
   ```bash
   ollama serve
   ```

2. **Pull your preferred model**:
   ```bash
   ollama pull llama3.2
   # or llama3.1, mistral, gemma2, etc.
   ```

3. **Configure Exam Prep Bot**:
   In your `.env` file, set:
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2
   ```

4. **Verify Health**:
   Start the app (`uvicorn main:app --reload`) and visit `http://localhost:8000/health`. The endpoint reports Ollama connectivity and model availability status.

## API Endpoints

| Method | Path | Auth required | Description |
|--------|------|:---:|-------------|
| GET | `/` | – | Frontend UI |
| GET | `/health` | – | Health check |
| GET | `/ready` | – | Readiness check |
| GET | `/version` | – | Version info |
| GET | `/config` | – | Public config summary |
| GET | `/logs/tail` | ✓ | Tail the application log |
| POST | `/api/documents/upload` | ✓ | Upload PDF/DOCX |
| GET | `/api/documents/list` | ✓ | List documents |
| GET | `/api/documents/stats` | ✓ | Document/vector-store statistics |
| DELETE | `/api/documents/{id}` | ✓ | Delete a document (removes its file + vectors) |
| POST | `/api/ask` / `/api/query` | ✓ | Ask a question (full RAG) — rate-limited |
| GET | `/api/intent/{query}` | ✓ | Classify intent only |
| POST | `/api/batch` | ✓ | Batch queries (max 50) — rate-limited |
| POST | `/api/search` | ✓ | Retrieve matching chunks without answering |
| GET | `/api/history` | ✓ | Chat history (legacy, process-wide) |
| DELETE | `/api/history` | ✓ | Clear chat history |
| WS | `/api/ws` | ✓ (query param) | WebSocket streaming |
| GET | `/api/conversations` | ✓ | List conversations for a `device_id` |
| GET | `/api/conversations/{id}` | ✓ | Full message history for one conversation |
| PATCH | `/api/conversations/{id}` | ✓ | Rename a conversation |
| DELETE | `/api/conversations/{id}` | ✓ | Delete a conversation + its semantic memory vectors |
| GET | `/api/system/config` | – | Internal config summary |
| GET | `/api/metrics` | ✓ | Metrics |

### Example: ask a question

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-key>" \
  -d '{"query": "What is photosynthesis?"}'
```

## Running Tests

```bash
pytest -v                          # full suite (350+ tests)
pytest tests/test_pipeline.py -v   # single file
pytest --cov=app                   # with coverage
```

The test suite is fully isolated from real project data — it uses an in-memory database and a
temp-directory FAISS index/upload folder (see `tests/conftest.py`), so running it never touches
`./data`, `./uploads`, or `./exam_prep_bot.db`.

## Linting & Formatting

```bash
black app tests           # format
isort app tests           # sort imports
mypy app                  # type-check (advisory — not yet enforced in CI)
```

Configuration for all three lives in `pyproject.toml`. CI (`.github/workflows/ci.yml`) runs
`black --check`, `isort --check-only`, the test suite, and a Docker build on every push/PR.

## Production Checklist

- Set a fixed `APP_API_KEY` (don't rely on the auto-generated one across restarts/replicas)
- Set real, unique values for `POSTGRES_PASSWORD` and any other credentials in `docker-compose.yml`
- Restrict `CORS_ORIGINS` to your real frontend origin(s)
- Point `DATABASE_URL` at PostgreSQL (via `docker-compose up`) rather than SQLite
- Run `alembic upgrade head` against that database before starting the app — `init_db()`'s
  `create_all()` only creates missing tables, it never alters existing ones
- Terminate TLS in `nginx.conf` (add a `443` server block + certificates) if exposed to the internet
- Review `RATE_LIMIT_PER_MINUTE` against your expected traffic and LLM provider budget
- If changing `EMBEDDING_MODEL` on a deployment with existing documents, re-upload them after the
  change — old and new embeddings aren't comparable and there's no automatic reindex
- Set `EXPOSE_API_KEY_TO_FRONTEND=false` and build real per-user login before this has independent
  customers who shouldn't see each other's conversations or API key

## Troubleshooting

- **Port already in use** — run with a different port: `uvicorn main:app --port 8001`
- **401 Unauthorized** — pass `X-API-Key` (see [Authentication](#authentication)); check the startup
  log for an auto-generated key if you haven't set `APP_API_KEY`
- **CORS errors** — add your frontend's origin to `CORS_ORIGINS` in `.env`
- **`docker-compose up` fails on `db`/`redis` health checks** — first run can take a few seconds
  longer while Postgres initializes; re-run `docker-compose up -d` or check `docker-compose logs db`
