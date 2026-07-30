"""Tests for the ExamPrepBot RAG orchestration pipeline."""

from unittest.mock import MagicMock

import pytest

from app.services.models import (
    AnswerWithSources,
    ChunkMetadata,
    IntentClassificationResult,
    QueryType,
    RetrievedChunk,
)
from app.services.pipeline import ExamPrepBot


def _make_chunk(content="Photosynthesis converts light into chemical energy."):
    meta = ChunkMetadata(page_number=1, chunk_index=0, total_chunks=1, file_name="bio.pdf")
    return RetrievedChunk(content=content, metadata=meta, relevance_score=0.9, rank=1)


@pytest.fixture()
def mock_intent_classifier():
    clf = MagicMock()
    clf.classify.return_value = IntentClassificationResult(
        query="What is photosynthesis?",
        primary_intent=QueryType.DEFINITION,
        confidence=0.95,
        reasoning="test",
    )
    return clf


@pytest.fixture()
def mock_vector_store_manager():
    vsm = MagicMock()
    vsm.get_stats.return_value = {
        "total_vectors": 1,
        "embedding_dimension": 384,
        "embedding_model": "all-MiniLM-L6-v2",
        "index_path": "./data/faiss_index",
    }
    return vsm


@pytest.fixture()
def mock_llm():
    llm = MagicMock()
    llm.generate_structured_answer.return_value = {
        "answer": "Photosynthesis converts light into chemical energy.",
        "format_type": "definition",
        "claims": ["Photosynthesis converts light into chemical energy."],
        "intent": "definition",
    }
    llm.reflect_on_answer.return_value = {
        "revised_answer": "Photosynthesis converts light into chemical energy.",
        "materially_changed": False,
        "should_block": False,
        "issues_found": [],
    }
    return llm


@pytest.fixture()
def bot(mock_vector_store_manager, mock_intent_classifier, mock_llm, monkeypatch):
    b = ExamPrepBot(
        vector_store_manager=mock_vector_store_manager,
        intent_classifier=mock_intent_classifier,
        llm=mock_llm,
    )
    monkeypatch.setattr(
        b.retriever,
        "search",
        MagicMock(
            return_value={
                "query": "What is photosynthesis?",
                "intent": QueryType.DEFINITION,
                "chunks": [_make_chunk()],
                "in_scope": True,
                "is_relevant": True,
                "relevance_score": 0.9,
                "total_retrieved": 1,
            }
        ),
    )
    return b


def test_answer_question_happy_path(bot):
    result = bot.answer_question("What is photosynthesis?")
    assert isinstance(result, AnswerWithSources)
    assert result.query_intent == QueryType.DEFINITION
    assert result.format_type == "definition"
    assert result.hallucination_risk in {"low", "medium", "high"}
    assert len(bot.chat_history) == 2
    assert bot.chat_history[0].role == "user"
    assert bot.chat_history[1].role == "assistant"


def test_answer_question_finds_citation_for_matching_claim(bot):
    result = bot.answer_question("What is photosynthesis?")
    assert len(result.sources) == 1
    assert result.sources[0].page_number == 1


def test_answer_question_out_of_scope_short_circuits_before_llm(bot, mock_llm):
    bot.retriever.search.return_value = {
        "query": "irrelevant",
        "intent": QueryType.VAGUE,
        "chunks": [],
        "in_scope": False,
        "is_relevant": False,
        "relevance_score": 0.0,
        "total_retrieved": 0,
    }
    result = bot.answer_question("Something totally unrelated")
    assert result.format_type == "out_of_scope"
    assert result.hallucination_risk == "high"
    assert result.sources == []
    mock_llm.generate_structured_answer.assert_not_called()


def test_answer_question_handles_llm_failure_gracefully(bot, mock_llm):
    mock_llm.generate_structured_answer.side_effect = RuntimeError("provider unavailable")
    result = bot.answer_question("What is photosynthesis?")
    assert result.format_type == "error"
    assert result.hallucination_risk == "high"
    assert result.overall_confidence == 0.0


def test_reset_clears_history_and_delegates_to_vector_store(bot, mock_vector_store_manager):
    bot.chat_history.append(MagicMock())
    bot.reset()
    assert bot.chat_history == []
    mock_vector_store_manager.reset.assert_called_once()


def test_get_stats(bot, mock_vector_store_manager):
    stats = bot.get_stats()
    assert stats["vector_store"] == mock_vector_store_manager.get_stats.return_value
    assert stats["chat_history_length"] == len(bot.chat_history)


def test_answer_question_passes_through_session_id_and_device_id(bot, monkeypatch):
    monkeypatch.setattr(bot.orchestrator, "run", MagicMock(return_value=MagicMock()))

    bot.answer_question("What is photosynthesis?", session_id="sess-1", device_id="device-1")

    bot.orchestrator.run.assert_called_once_with(
        "What is photosynthesis?",
        session_id="sess-1",
        device_id="device-1",
        document_ids=None,
        on_stage=None,
    )


def test_exam_prep_bot_wires_memory_agent_for_persistence(bot, mock_llm, mock_vector_store_manager):
    from app.core.database import SessionLocal

    memory_agent = bot.orchestrator.memory_agent
    assert memory_agent.persist is True
    assert memory_agent.llm is mock_llm
    assert memory_agent.vector_store_manager is mock_vector_store_manager
    assert memory_agent.db_session_factory is SessionLocal


def test_upload_document_generates_document_id(bot, mock_vector_store_manager, tmp_path, monkeypatch):
    from app.services import pipeline as pipeline_module
    from app.services.models import Document

    fake_document = Document(
        file_name="notes.pdf",
        file_type="pdf",
        file_size_bytes=1024,
        total_chunks=2,
        chunks=[],
        upload_timestamp="2024-01-01T00:00:00",
    )
    monkeypatch.setattr(pipeline_module, "parse_file", MagicMock(return_value=fake_document))

    result = bot.upload_document(str(tmp_path / "notes.pdf"))
    assert result["success"] is True
    assert result["document_id"]
    mock_vector_store_manager.add_document.assert_called_once()
    call_args = mock_vector_store_manager.add_document.call_args[0]
    assert call_args[0] is fake_document
    assert call_args[1] == result["document_id"]
