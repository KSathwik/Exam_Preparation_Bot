"""
FastAPI Request/Response Models
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# ==================== DOCUMENT MODELS ====================

class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    success: bool
    file_name: str
    file_type: str
    total_chunks: int
    file_size_mb: float
    processing_time_seconds: float
    message: str

# ==================== QUERY MODELS ====================

class QueryRequest(BaseModel):
    """Request model for query"""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")

class SourceCitation(BaseModel):
    """Citation source"""
    page: int
    section: Optional[str] = None
    quote: str
    confidence: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)

class QueryResponse(BaseModel):
    """Response model for query"""
    success: bool
    query: str
    answer: str
    intent: str
    intent_confidence: float = Field(ge=0, le=1)
    sources: List[SourceCitation]
    confidence: float = Field(ge=0, le=1)
    hallucination_risk: str
    response_time_seconds: float
    timestamp: str

# ==================== CHAT MODELS ====================

class ChatMessage(BaseModel):
    """Chat message model"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    intent: Optional[str] = None

class ChatHistory(BaseModel):
    """Chat history model"""
    messages: List[ChatMessage]
    total: int

# ==================== BATCH MODELS ====================

class BatchQueryRequest(BaseModel):
    """Batch query request"""
    queries: List[str] = Field(..., min_items=1, max_items=100)

class BatchQueryResult(BaseModel):
    """Single batch query result"""
    query: str
    success: bool
    answer: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None

class BatchQueryResponse(BaseModel):
    """Batch query response"""
    total: int
    successful: int
    failed: int
    results: List[BatchQueryResult]

# ==================== SYSTEM MODELS ====================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    bot_initialized: bool

class ConfigResponse(BaseModel):
    """System configuration response"""
    model: str
    embedding_model: str
    max_chunk_size: int
    retrieval_top_k: int
    temperature: float
    max_tokens: int
    debug_mode: bool

class MetricsResponse(BaseModel):
    """System metrics response"""
    uptime: str
    vector_store: Dict[str, Any]
    chat_history_length: int
    model_info: Dict[str, str]

# ==================== ERROR MODELS ====================

class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: str

# ==================== WEBSOCKET MODELS ====================

class WebSocketMessage(BaseModel):
    """WebSocket message"""
    type: str  # "intent", "chunk", "complete", "error"
    data: Dict[str, Any]
