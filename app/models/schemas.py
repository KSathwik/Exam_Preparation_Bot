"""Pydantic schemas for FastAPI"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime


class QueryType(str, Enum):
    """Intent types for queries"""
    DEFINITION = "definition"
    EXPLAIN = "explain"
    COMPARE = "compare"
    PROCESS = "process"
    VAGUE = "vague"
    EXAMPLE = "example"
    DIAGRAM = "diagram"


# ==================== Document Models ====================

class DocumentUploadResponse(BaseModel):
    """Response after document upload"""
    success: bool
    file_name: str
    file_type: str
    total_chunks: int
    file_size_mb: float
    processing_time_seconds: float
    document_id: str
    message: str


class DocumentInfo(BaseModel):
    """Document information"""
    document_id: str
    file_name: str
    file_type: str
    total_chunks: int
    upload_timestamp: datetime
    file_size_mb: float


class DocumentListResponse(BaseModel):
    """List of documents"""
    success: bool
    total_documents: int
    documents: List[DocumentInfo]


# ==================== Query Models ====================

class QueryRequest(BaseModel):
    """User query request"""
    query: str = Field(..., min_length=1, max_length=1000)
    document_id: Optional[str] = None  # If None, search all documents
    top_k: Optional[int] = None


class SourceCitation(BaseModel):
    """Source citation"""
    page_number: int
    section_title: Optional[str] = None
    quoted_text: str
    confidence: float
    relevance_score: float


class QueryResponse(BaseModel):
    """Response to user query"""
    success: bool
    query: str
    answer: str
    query_intent: QueryType
    intent_confidence: float
    sources: List[SourceCitation]
    overall_confidence: float
    hallucination_risk: str  # low, medium, high
    response_time_seconds: float
    format_type: str


class QueryHistoryItem(BaseModel):
    """Query history item"""
    query_id: str
    query: str
    answer: str
    intent: QueryType
    confidence: float
    timestamp: datetime


class ChatMessage(BaseModel):
    """Chat message"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    intent_type: Optional[QueryType] = None


class ChatSession(BaseModel):
    """Chat session"""
    session_id: str
    document_id: Optional[str] = None
    messages: List[ChatMessage]
    created_at: datetime
    updated_at: datetime


# ==================== Health & Status ====================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    service: str


class StatsResponse(BaseModel):
    """Statistics response"""
    total_documents: int
    total_chunks: int
    embedding_dimension: int
    embedding_model: str
    total_queries_processed: int
    average_response_time: float


# ==================== Error Models ====================

class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[dict] = None
