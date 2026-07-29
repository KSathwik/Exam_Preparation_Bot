"""Tests for domain models and schemas."""

import pytest

from app.services.models import (
    AnswerWithSources,
    ChatMessage,
    ChunkMetadata,
    Document,
    DocumentChunk,
    IntentClassificationResult,
    QueryType,
    RetrievedChunk,
    SourceCitation,
)


def test_query_type_values():
    assert QueryType.DEFINITION.value == "definition"
    assert QueryType.HOMEWORK.value == "homework"
    assert len(QueryType) == 8


def test_chunk_metadata_defaults():
    meta = ChunkMetadata(page_number=1, chunk_index=0, total_chunks=5, file_name="test.pdf")
    assert meta.chunk_position == "unknown"
    assert meta.section_title is None


def test_document_chunk_creation():
    meta = ChunkMetadata(page_number=1, chunk_index=0, total_chunks=1, file_name="a.pdf")
    chunk = DocumentChunk(content="hello world", metadata=meta)
    assert chunk.content == "hello world"
    assert chunk.embedding is None


def test_retrieved_chunk_validation():
    meta = ChunkMetadata(page_number=1, chunk_index=0, total_chunks=1, file_name="a.pdf")
    rc = RetrievedChunk(content="text", metadata=meta, relevance_score=0.8, rank=1)
    assert rc.relevance_score == 0.8

    with pytest.raises(Exception):
        RetrievedChunk(content="text", metadata=meta, relevance_score=1.5, rank=1)


def test_intent_classification_result():
    result = IntentClassificationResult(
        query="What is X?",
        primary_intent=QueryType.DEFINITION,
        confidence=0.9,
    )
    assert result.primary_intent == QueryType.DEFINITION


def test_source_citation():
    sc = SourceCitation(page_number=3, quoted_text="some text", confidence=0.85, relevance_score=0.7)
    assert sc.page_number == 3


def test_answer_with_sources():
    ans = AnswerWithSources(
        answer="Test answer",
        query_intent=QueryType.EXPLAIN,
        intent_confidence=0.9,
        sources=[],
        overall_confidence=0.8,
        hallucination_risk="low",
        response_time_seconds=1.2,
        format_type="comprehensive",
    )
    assert ans.hallucination_risk == "low"


def test_chat_message():
    msg = ChatMessage(role="user", content="hello", timestamp="2024-01-01")
    assert msg.intent_type is None
