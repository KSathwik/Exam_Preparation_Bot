# 📚 Exam Prep Bot - FastAPI Edition

**Production-grade API backend with:**
- ✅ **FastAPI** - Modern, fast, production-ready
- ✅ **Async/Await** - High concurrency support
- ✅ **WebSocket** - Real-time streaming
- ✅ **Auto Documentation** - Swagger UI + ReDoc
- ✅ **RESTful API** - Standard REST endpoints
- ✅ **Database** - PostgreSQL integration
- ✅ **Caching** - Redis support
- ✅ **Docker** - Full containerization
- ✅ **Tests** - Comprehensive test suite

## Quick Start

```bash
cd exam-prep-bot-fastapi
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY
uvicorn main:app --reload
```

Visit: http://localhost:8000/docs

## Docker

```bash
docker-compose up -d
```

Includes: FastAPI + PostgreSQL + Redis + Nginx

## API Endpoints

### Documents
- `POST /api/documents/upload` - Upload PDF/DOCX
- `GET /api/documents/stats` - Document info

### Query
- `POST /api/query` - Ask question
- `GET /api/query/intent/{query}` - Classify intent
- `POST /api/batch/query` - Batch queries
- `WS /ws/query` - WebSocket streaming

### Chat
- `GET /api/chat/history` - Get history
- `DELETE /api/chat/history` - Clear history

### System
- `GET /health` - Health check
- `POST /api/system/reset` - Reset bot
- `GET /api/metrics` - Metrics

## Why FastAPI?

✅ **Production-Ready** - Enterprise-grade
✅ **Fast** - Async/await, high concurrency
✅ **Scalable** - Load balancing ready
✅ **Real-time** - WebSocket support
✅ **Documented** - Auto Swagger UI
✅ **Testable** - Easy to test
✅ **Database** - Native ORM/SQL
✅ **Flexible** - Any frontend

See `FASTAPI_GUIDE.md` for complete guide.
