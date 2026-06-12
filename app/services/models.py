"""Domain models for the Exam Prep Bot."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class QueryType(str, Enum):
    """Intent types for queries."""
    DEFINITION = "definition"
    EXPLAIN = "explain"
    COMPARE = "compare"
    PROCESS = "process"
    VAGUE = "vague"
    EXAMPLE = "example"
    DIAGRAM = "diagram"
    HOMEWORK = "homework"


class ChunkMetadata(BaseModel):
    """Metadata for document chunks."""
    page_number: int
    section_title: Optional[str] = None
    section_level: Optional[int] = None
    chunk_index: int
    total_chunks: int
    file_name: str
    chunk_position: str = Field(default="unknown")


class DocumentChunk(BaseModel):
    """A chunk of text from a document."""
    content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None

    class Config:
        arbitrary_types_allowed = True


class Document(BaseModel):
    """Uploaded document with metadata."""
    file_name: str
    file_type: str
    file_size_bytes: int
    total_chunks: int
    chunks: List[DocumentChunk]
    upload_timestamp: str

    class Config:
        arbitrary_types_allowed = True


class RetrievedChunk(BaseModel):
    """Retrieved chunk with relevance score."""
    content: str
    metadata: ChunkMetadata
    relevance_score: float = Field(ge=0.0, le=1.0)
    rank: int

    class Config:
        arbitrary_types_allowed = True


class IntentClassificationResult(BaseModel):
    """Result of intent classification."""
    query: str
    primary_intent: QueryType
    confidence: float = Field(ge=0.0, le=1.0)
    alternative_intents: Optional[Dict[QueryType, float]] = None
    reasoning: Optional[str] = None


class SourceCitation(BaseModel):
    """Source citation for an answer."""
    page_number: int
    section_title: Optional[str] = None
    quoted_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)


class AnswerWithSources(BaseModel):
    """Complete answer with sources."""
    answer: str
    query_intent: QueryType
    intent_confidence: float = Field(ge=0.0, le=1.0)
    sources: List[SourceCitation]
    overall_confidence: float = Field(ge=0.0, le=1.0)
    hallucination_risk: str = Field(default="low")
    response_time_seconds: float
    format_type: str


class ChatMessage(BaseModel):
    """A message in the chat history."""
    role: str
    content: str
    timestamp: str
    intent_type: Optional[QueryType] = None
    metadata: Optional[Dict[str, Any]] = None
