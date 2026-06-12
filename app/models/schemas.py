"""Pydantic schemas for FastAPI request/response models."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class QueryType(str, Enum):
    DEFINITION = "definition"
    EXPLAIN = "explain"
    COMPARE = "compare"
    PROCESS = "process"
    VAGUE = "vague"
    EXAMPLE = "example"
    DIAGRAM = "diagram"
    HOMEWORK = "homework"


# ── Documents ────────────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    success: bool
    file_name: str
    file_type: str
    total_chunks: int
    file_size_mb: float
    processing_time_seconds: float
    document_id: Optional[str] = None
    message: str


class DocumentInfo(BaseModel):
    document_id: str
    file_name: str
    file_type: str
    total_chunks: int
    upload_timestamp: datetime
    file_size_mb: float


class DocumentListResponse(BaseModel):
    success: bool
    total_documents: int
    documents: List[DocumentInfo]


# ── Queries ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    document_id: Optional[str] = None
    session_id: Optional[str] = None
    top_k: Optional[int] = None


class SourceCitationOut(BaseModel):
    page_number: int
    section_title: Optional[str] = None
    quoted_text: str
    confidence: float = Field(ge=0, le=1)
    relevance_score: float = Field(ge=0, le=1)


class QueryResponse(BaseModel):
    success: bool
    query: str
    answer: str
    query_intent: str
    intent_confidence: float = Field(ge=0, le=1)
    sources: List[SourceCitationOut]
    overall_confidence: float = Field(ge=0, le=1)
    hallucination_risk: str
    response_time_seconds: float
    format_type: str
    timestamp: Optional[str] = None


# ── Chat ─────────────────────────────────────────────────────────────

class ChatMessageOut(BaseModel):
    role: str
    content: str
    timestamp: str
    intent: Optional[str] = None


# ── Batch ────────────────────────────────────────────────────────────

class BatchQueryResult(BaseModel):
    query: str
    success: bool
    answer: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None


class BatchQueryResponse(BaseModel):
    total: int
    successful: int
    failed: int
    results: List[BatchQueryResult]


# ── System / Health ──────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    service: str
    bot_initialized: bool = True


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
