# Exam Prep Bot

Production-grade exam preparation chatbot powered by **Claude AI + RAG**.

- **FastAPI** backend with async/await and WebSocket streaming
- **FAISS** vector search with sentence-transformer embeddings
- **Intent classification** (8 types) with confidence scoring
- **Source citations** and hallucination risk assessment
- **SQLAlchemy** persistence for documents and queries
- **Docker Compose** deployment (PostgreSQL + Redis + Nginx)

## Quick Start

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # add your ANTHROPIC_API_KEY
uvicorn main:app --reload
```

Open http://localhost:8000 for the UI, or http://localhost:8000/docs for Swagger.

## Docker

```bash
docker-compose up -d
```

## Project Structure

```
main.py                    # Entry point (delegates to app/)
app/
  main.py                  # FastAPI app, middleware, routers, lifespan
  core/
    config.py              # Unified Pydantic settings
    database.py            # SQLAlchemy engine + session
    dependencies.py        # Singleton DI (models, vector store, bot)
  api/
    health.py              # /health, /ready, /version, /config
    documents.py           # /api/documents/upload, list, delete, stats
    queries.py             # /api/query, /api/ask, batch, history, ws
  models/
    schemas.py             # Pydantic request/response schemas
    db_models.py           # SQLAlchemy ORM models
  services/
    pipeline.py            # ExamPrepBot orchestrator
    parser.py              # PDF/DOCX parsing + chunking
    embeddings.py          # Embedding generation + FAISS store
    intent_classifier.py   # Hybrid rule + semantic classifier
    retriever.py           # Adaptive retrieval + reranking
    llm_interface.py       # Anthropic Claude API client
    validator.py           # Citation extraction + confidence scoring
frontend/
  index.html               # Single-page UI
  style.css                # Extracted stylesheet
  app.js                   # Extracted JavaScript
tests/                     # pytest test suite
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Frontend UI |
| GET | `/health` | Health check |
| GET | `/ready` | Readiness check |
| POST | `/api/documents/upload` | Upload PDF/DOCX |
| GET | `/api/documents/list` | List documents |
| GET | `/api/documents/stats` | Document statistics |
| POST | `/api/query` | Ask a question (full RAG) |
| POST | `/api/ask` | Ask a question (alias) |
| GET | `/api/intent/{query}` | Classify intent only |
| POST | `/api/batch` | Batch queries |
| GET | `/api/history` | Chat history |
| DELETE | `/api/history` | Clear history |
| WS | `/api/ws` | WebSocket streaming |
| POST | `/api/system/reset` | Reset bot |
| GET | `/api/system/config` | System config |
| GET | `/api/metrics` | Metrics |

## Running Tests

```bash
pytest -v
```

## Configuration

All settings are loaded from `.env` via Pydantic. See `.env.example` for all options.
