# Exam Prep Bot

Production-grade exam preparation chatbot powered by an LLM (Anthropic Claude, OpenAI, or Google Gemini) + RAG.

- **FastAPI** backend with async/await and WebSocket streaming
- **FAISS** vector search with sentence-transformer embeddings
- **Intent classification** (8 types) with confidence scoring
- **Source citations** and hallucination risk assessment
- **SQLAlchemy** persistence for documents and queries
- **API-key authentication + rate limiting** on sensitive/expensive endpoints
- **Docker Compose** deployment (PostgreSQL + Redis + Nginx)

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
  models/
    schemas.py               # Pydantic request/response schemas
    db_models.py              # SQLAlchemy ORM models
  services/
    pipeline.py               # ExamPrepBot — delegates to OrchestratorAgent
    agents/                     # Multi-agent pipeline (see docs/ARCHITECTURE.md)
      orchestrator.py             # Deterministic coordinator, all stages
      retrieval_agent.py            # Wraps HybridRetriever
      knowledge_agent.py             # Wraps LLM structured-answer generation
      reflection_agent.py              # Wraps LLM quality-control pass
      memory_agent.py                   # chat_history + opt-in DB persistence
    parser.py                  # PDF/DOCX parsing + structural chunking
    embeddings.py                # Embedding generation + FAISS store + BM25 hybrid
    intent_classifier.py          # Hybrid rule + semantic classifier
    retriever.py                   # Adaptive retrieval + reranking + memory fallback
    llm_interface.py                 # Multi-provider LLM client (Gemini/OpenAI/Anthropic)
    validator.py                      # Citation extraction + confidence scoring
frontend/
  index.html               # Single-page UI
  style.css                # Extracted stylesheet
  app.js                   # Extracted JavaScript
alembic/                   # DB migrations (see CLAUDE.md Commands)
docs/
  ARCHITECTURE.md          # Agent workflow, memory tiers, retrieval flow diagrams
  PHASE_2_ROADMAP.md       # Deliberately deferred scope
tests/                     # pytest test suite
```

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
| `LLM_PROVIDER`, `*_API_KEY` | Which LLM backend to use, and its key (only the matching key is required) |
| `APP_API_KEY`, `API_AUTH_ENABLED` | Auth for this app's own API (not an LLM key) |
| `RATE_LIMIT_PER_MINUTE` | Per-IP throttle on LLM-calling endpoints |
| `DATABASE_URL` | SQLite by default; use a `postgresql://` URL in Docker/production |
| `FAISS_INDEX_PATH`, `UPLOAD_DIR`, `CACHE_DIR` | On-disk data locations |
| `MAX_FILE_SIZE_MB`, `ALLOWED_FILE_TYPES` | Upload constraints |
| `CORS_ORIGINS` | Comma/JSON-list of allowed origins — restrict this in production |
| `EMBEDDING_MODEL` | Changing this on a deployment with an existing index requires a full reindex |
| `HYBRID_DENSE_WEIGHT` | Dense vs. BM25 weight in hybrid retrieval (see `docs/ARCHITECTURE.md`) |
| `ENABLE_CROSS_ENCODER_RERANK` | Opt-in reranking of the merged candidate pool, off by default |
| `MEMORY_RELEVANCE_THRESHOLD`, `MEMORY_SUMMARIZE_*` | Conversation-memory tuning (only relevant once `MemoryAgent(persist=True)` is wired up — see `docs/PHASE_2_ROADMAP.md`) |

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
| GET | `/api/history` | ✓ | Chat history |
| DELETE | `/api/history` | ✓ | Clear chat history |
| WS | `/api/ws` | ✓ (query param) | WebSocket streaming |
| POST | `/api/system/reset` | ✓ | Reset bot (clears history + vector store) |
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
pytest -v                          # full suite
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

## Troubleshooting

- **Port already in use** — run with a different port: `uvicorn main:app --port 8001`
- **401 Unauthorized** — pass `X-API-Key` (see [Authentication](#authentication)); check the startup
  log for an auto-generated key if you haven't set `APP_API_KEY`
- **CORS errors** — add your frontend's origin to `CORS_ORIGINS` in `.env`
- **`docker-compose up` fails on `db`/`redis` health checks** — first run can take a few seconds
  longer while Postgres initializes; re-run `docker-compose up -d` or check `docker-compose logs db`
