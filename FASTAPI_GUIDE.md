# 🚀 FastAPI Setup Guide - Exam Prep Bot

## Why FastAPI Over Streamlit?

| Feature | Streamlit | FastAPI |
|---------|-----------|---------|
| **Performance** | Moderate | ⭐⭐⭐⭐⭐ (High) |
| **Scalability** | Limited | ⭐⭐⭐⭐⭐ (Unlimited) |
| **Async/Await** | No | ⭐⭐⭐⭐⭐ Yes |
| **WebSocket** | No | ⭐⭐⭐⭐⭐ Yes |
| **Production Ready** | No | ⭐⭐⭐⭐⭐ Yes |
| **API Documentation** | No | ⭐⭐⭐⭐⭐ Auto (Swagger) |
| **Database Integration** | Hard | ⭐⭐⭐⭐⭐ Easy |
| **Load Balancing** | No | ⭐⭐⭐⭐⭐ Yes |
| **Microservices** | No | ⭐⭐⭐⭐⭐ Yes |
| **Custom Frontend** | Limited | ⭐⭐⭐⭐⭐ Any (React/Vue) |

## Architecture

```
┌─────────────────────────────────┐
│   React/Vue Frontend (SPA)      │
│   (Optional - can use any UI)   │
└────────────────┬────────────────┘
                 │ HTTP/WebSocket
     ┌───────────▼───────────┐
     │   FastAPI Backend     │
     │   (main.py)           │
     └───────────┬───────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
[Pipeline]   [Database]   [Cache]
[LLM API]    [Sessions]   [Queue]
```

## Installation & Setup

### 1. Create Project Directory

```bash
mkdir exam-prep-bot-fastapi
cd exam-prep-bot-fastapi
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
# Core FastAPI dependencies
pip install fastapi uvicorn[standard]

# Async support
pip install httpx aiofiles

# Database (if using)
pip install sqlalchemy alembic psycopg2-binary

# Task queue (if needed)
pip install celery redis

# Testing
pip install pytest pytest-asyncio httpx

# All original dependencies
pip install -r requirements.txt
```

### 4. Project Structure

```
exam-prep-bot-fastapi/
├── main.py                    ← FastAPI app entry point
├── requirements.txt           ← All dependencies
├── .env.example              ← Configuration template
├── Dockerfile                ← Docker configuration
├── docker-compose.yml        ← Multi-container setup
│
├── src/
│   ├── models.py             ← Data models (from original)
│   ├── api_models.py         ← API request/response models
│   ├── parser.py             ← PDF/DOCX parsing
│   ├── embeddings.py         ← Vector search
│   ├── intent_classifier.py  ← Query classification
│   ├── retriever.py          ← Document retrieval
│   ├── llm_interface.py      ← Claude API
│   ├── validator.py          ← Citations & scoring
│   └── pipeline.py           ← Main orchestration
│
├── config/
│   ├── __init__.py
│   └── settings.py           ← Configuration management
│
├── frontend/                 ← React/Vue frontend (optional)
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── README.md
│
├── tests/
│   ├── test_api.py           ← API endpoint tests
│   ├── test_documents.py     ← Document upload tests
│   ├── test_queries.py       ← Query processing tests
│   └── test_websocket.py     ← WebSocket tests
│
├── deployment/
│   ├── docker-compose.yml    ← Production setup
│   ├── nginx.conf            ← Reverse proxy
│   └── README.md             ← Deployment guide
│
└── docs/
    ├── API.md                ← API documentation
    ├── ARCHITECTURE.md       ← System design
    └── DEPLOYMENT.md         ← Deployment guide
```

## Running the Application

### Development Mode

```bash
# With auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Then visit:
# - App: http://localhost:8000
# - Docs: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
```

### Production Mode

```bash
# Using gunicorn + uvicorn workers
pip install gunicorn

gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

## API Endpoints

### Health & Status
```
GET  /              - Status & links
GET  /health        - Health check
GET  /api/metrics   - System metrics
```

### Documents
```
POST /api/documents/upload   - Upload PDF/DOCX
GET  /api/documents/stats    - Document statistics
```

### Query & Answering
```
POST /api/query              - Ask question
GET  /api/query/intent/{q}   - Classify intent only
POST /api/batch/query        - Process multiple queries
```

### Chat
```
GET  /api/chat/history       - Get chat history
DELETE /api/chat/history     - Clear chat history
```

### WebSocket
```
WS   /ws/query               - Real-time streaming
```

### System
```
POST /api/system/reset       - Reset bot
GET  /api/system/config      - Get configuration
```

## Using the API

### Example 1: Upload Document

```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "accept: application/json" \
  -F "file=@biology_notes.pdf"
```

**Response:**
```json
{
  "success": true,
  "file_name": "biology_notes.pdf",
  "file_type": "pdf",
  "total_chunks": 42,
  "file_size_mb": 2.5,
  "processing_time_seconds": 3.2,
  "message": "Successfully uploaded..."
}
```

### Example 2: Ask Question

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is photosynthesis?"
  }'
```

**Response:**
```json
{
  "success": true,
  "query": "What is photosynthesis?",
  "answer": "Photosynthesis is the process...",
  "intent": "definition",
  "intent_confidence": 0.98,
  "sources": [
    {
      "page": 42,
      "section": "Chapter 3",
      "quote": "Photosynthesis is...",
      "confidence": 0.95,
      "relevance": 0.98
    }
  ],
  "confidence": 0.96,
  "hallucination_risk": "low",
  "response_time_seconds": 1.8,
  "timestamp": "2024-05-23T10:30:00"
}
```

### Example 3: Batch Query

```bash
curl -X POST "http://localhost:8000/api/batch/query" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "What is photosynthesis?",
      "Compare mitosis and meiosis",
      "Explain DNA replication"
    ]
  }'
```

### Example 4: WebSocket Streaming

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/query');

ws.onopen = () => {
  ws.send(JSON.stringify({
    query: "What is photosynthesis?",
    stream: true
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  if (message.type === 'intent') {
    console.log('Intent:', message.intent);
  } else if (message.type === 'chunk') {
    console.log('Chunk:', message.text);
  } else if (message.type === 'complete') {
    console.log('Complete answer:', message.answer);
    console.log('Sources:', message.sources);
  }
};
```

## Docker Deployment

### Simple Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy files
COPY requirements.txt .
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build & Run

```bash
# Build image
docker build -t exam-prep-bot .

# Run container
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  exam-prep-bot
```

## Frontend Integration (React)

### 1. Create React App

```bash
npx create-react-app frontend
cd frontend
npm install axios react-query zustand
```

### 2. API Service

```javascript
// frontend/src/services/api.js
import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

export const api = {
  // Documents
  uploadDocument: (file) => 
    axios.post(`${API_BASE}/documents/upload`, 
      { file }, 
      { headers: { 'Content-Type': 'multipart/form-data' } }
    ),
  
  getDocStats: () => 
    axios.get(`${API_BASE}/documents/stats`),
  
  // Query
  query: (query) => 
    axios.post(`${API_BASE}/query`, { query }),
  
  classifyIntent: (query) => 
    axios.get(`${API_BASE}/query/intent/${query}`),
  
  // Chat
  getChatHistory: () => 
    axios.get(`${API_BASE}/chat/history`),
  
  clearChat: () => 
    axios.delete(`${API_BASE}/chat/history`),
};
```

### 3. React Component

```javascript
// frontend/src/components/ChatInterface.jsx
import { useState } from 'react';
import { api } from '../services/api';

export function ChatInterface() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const handleQuery = async () => {
    setLoading(true);
    try {
      const result = await api.query(query);
      setResponse(result.data);
    } catch (error) {
      console.error('Query failed:', error);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="chat-interface">
      <input 
        value={query} 
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask a question..."
      />
      <button onClick={handleQuery} disabled={loading}>
        {loading ? 'Loading...' : 'Ask'}
      </button>
      
      {response && (
        <div className="response">
          <p>{response.answer}</p>
          <p>Confidence: {response.confidence.toFixed(2)}</p>
          <div className="sources">
            {response.sources.map((source, i) => (
              <div key={i}>
                <p>Page {source.page}: {source.quote}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

## Testing

### Unit Tests

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_query():
    # Upload document first
    # Then test query
    response = client.post(
        "/api/query",
        json={"query": "What is photosynthesis?"}
    )
    assert response.status_code == 200
    assert "answer" in response.json()
```

### Run Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src
```

## Environment Variables

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
MODEL_NAME=claude-3-5-sonnet-20241022
MAX_TOKENS=1024
TEMPERATURE=0.3

# FastAPI
DEBUG_MODE=false
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://user:password@localhost/dbname

# Redis (if using caching)
REDIS_URL=redis://localhost:6379

# CORS
ALLOWED_ORIGINS=["http://localhost:3000", "https://yourdomain.com"]
```

## Performance Optimization

### 1. Async Document Upload

```python
from fastapi import BackgroundTasks

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile, 
    background_tasks: BackgroundTasks
):
    # Save file immediately
    # Process in background
    background_tasks.add_task(process_document, file_path)
    return {"status": "processing"}
```

### 2. Caching with Redis

```python
from redis import Redis
import json

redis = Redis(host='localhost', port=6379)

@app.get("/api/query/intent/{query}")
async def classify_intent(query: str):
    # Check cache
    cached = redis.get(f"intent:{query}")
    if cached:
        return json.loads(cached)
    
    # Process
    result = bot_instance.intent_classifier.classify(query)
    
    # Cache for 1 hour
    redis.setex(f"intent:{query}", 3600, json.dumps(result))
    
    return result
```

### 3. Database for Persistence

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine)

@app.get("/api/chat/history")
async def get_chat_history():
    session = SessionLocal()
    messages = session.query(ChatMessage).all()
    return messages
```

## Production Deployment

### Using Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: exam-prep-bot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: exam-prep-bot
  template:
    metadata:
      labels:
        app: exam-prep-bot
    spec:
      containers:
      - name: api
        image: exam-prep-bot:latest
        ports:
        - containerPort: 8000
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: anthropic-secrets
              key: api-key
```

### Using Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_URL=postgresql://user:password@db:5432/exam-prep
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:14
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=exam-prep
  
  redis:
    image: redis:7
  
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
```

## Monitoring & Logging

### Structured Logging

```python
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger(__name__)

logger.info("query_processed", 
    query=query,
    confidence=confidence,
    response_time=response_time
)
```

### Metrics with Prometheus

```python
from prometheus_client import Counter, Histogram
from fastapi import Request
import time

query_counter = Counter('queries_total', 'Total queries')
query_duration = Histogram('query_duration_seconds', 'Query duration')

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    query_counter.inc()
    query_duration.observe(duration)
    
    return response
```

## Comparison: Streamlit vs FastAPI

### When to Use Streamlit
- Quick prototypes
- Data visualization dashboards
- Internal tools
- Non-production apps

### When to Use FastAPI ✅ (OUR CHOICE)
- Production APIs
- Microservices
- High-traffic applications
- Mobile app backends
- Custom frontends
- Real-time features (WebSocket)
- Database integration
- Team collaboration

## Next Steps

1. ✅ Copy all core modules from original project
2. ✅ Use main.py as FastAPI entry point
3. ✅ Install FastAPI: `pip install fastapi uvicorn`
4. ✅ Run: `uvicorn main:app --reload`
5. ✅ Visit: http://localhost:8000/docs

## Summary

**FastAPI is MUCH better than Streamlit for production because:**

✅ **Performance** - Async/await, high concurrency
✅ **Scalability** - Load balancing, microservices
✅ **APIs** - RESTful + WebSocket + GraphQL ready
✅ **Real-time** - WebSocket streaming support
✅ **Documentation** - Auto-generated Swagger UI
✅ **Testing** - Easy to test with pytest
✅ **Databases** - Native SQL/ORM support
✅ **Enterprise** - Used by major companies
✅ **Flexibility** - Any frontend (React, Vue, Mobile)
✅ **Standards** - OpenAPI/REST standards

---

**Ready to build a production-grade API? Let's go! 🚀**
