"""Pydantic schemas for FastAPI request/response models.

Note: query intent is represented here as a plain ``str`` (see
``QueryResponse.query_intent``), not a duplicate enum — the single source of
truth for intent types is ``app.services.models.QueryType``, used throughout
the pipeline. Keeping only one definition avoids the two drifting apart.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

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
    device_id: Optional[str] = None
    # Scopes retrieval to just these documents first, falling back to the
    # full index if that misses — see AdaptiveRetriever.retrieve. Lets the
    # frontend keep a conversation grounded in the document(s) actually
    # uploaded there instead of searching every document ever uploaded.
    document_ids: Optional[List[str]] = None
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
    session_id: Optional[str] = None


# ── Chat ─────────────────────────────────────────────────────────────


class ChatMessageOut(BaseModel):
    role: str
    content: str
    timestamp: str
    intent: Optional[str] = None
    format_type: Optional[str] = None


# ── Conversations ────────────────────────────────────────────────────


class ConversationSummaryOut(BaseModel):
    session_id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationListResponse(BaseModel):
    success: bool
    total: int
    conversations: List[ConversationSummaryOut]


class ConversationDetailResponse(BaseModel):
    success: bool
    session_id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    messages: List[ChatMessageOut]


class ConversationRenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)


class ConversationRenameResponse(BaseModel):
    success: bool
    conversation: ConversationSummaryOut


class ConversationDeleteResponse(BaseModel):
    success: bool
    message: str
    session_id: str


# ── Batch ────────────────────────────────────────────────────────────


class BatchQueryRequest(BaseModel):
    queries: List[str] = Field(..., min_length=1, max_length=50)

    @field_validator("queries")
    @classmethod
    def _validate_each_query(cls, queries: List[str]) -> List[str]:
        for q in queries:
            if not q or not q.strip():
                raise ValueError("Batch queries cannot be empty")
            if len(q) > 1000:
                raise ValueError("Each query must be 1000 characters or fewer")
        return queries


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
