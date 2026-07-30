"""Tests for the multi-agent pipeline: Retrieval/Knowledge/Reflection/Memory
agents plus the OrchestratorAgent that coordinates them.

All agents wrap mocked collaborators here — nothing in this file touches a
real vector store, LLM provider, or document.
"""

from unittest.mock import MagicMock

import pytest

from app.services.agents.knowledge_agent import KnowledgeAgent
from app.services.agents.memory_agent import MemoryAgent
from app.services.agents.orchestrator import OrchestratorAgent
from app.services.agents.reflection_agent import ReflectionAgent
from app.services.agents.retrieval_agent import RetrievalAgent
from app.services.models import (
    AnswerWithSources,
    ChunkMetadata,
    IntentClassificationResult,
    QueryType,
    RetrievedChunk,
)


def _make_chunk(content="Photosynthesis converts light into chemical energy."):
    meta = ChunkMetadata(page_number=1, chunk_index=0, total_chunks=1, file_name="bio.pdf")
    return RetrievedChunk(content=content, metadata=meta, relevance_score=0.9, rank=1)


def _reflection_no_change(draft_answer):
    return {
        "revised_answer": draft_answer,
        "materially_changed": False,
        "should_block": False,
        "issues_found": [],
    }


# ----------------------------------------------------------------------
# Individual agent wrappers
# ----------------------------------------------------------------------


def test_retrieval_agent_delegates_to_retriever():
    retriever = MagicMock()
    retriever.search.return_value = {"chunks": [], "is_relevant": False}
    agent = RetrievalAgent(retriever)

    result = agent.search("q", QueryType.DEFINITION, top_k=5)

    retriever.search.assert_called_once_with("q", QueryType.DEFINITION, 5, document_ids=None, session_id=None)
    assert result == {"chunks": [], "is_relevant": False}


def test_retrieval_agent_picks_up_monkeypatched_search(monkeypatch):
    retriever = MagicMock()
    agent = RetrievalAgent(retriever)

    replacement = MagicMock(return_value={"chunks": [_make_chunk()], "is_relevant": True})
    monkeypatch.setattr(retriever, "search", replacement)

    result = agent.search("q", QueryType.DEFINITION)
    assert result["is_relevant"] is True
    replacement.assert_called_once()


def test_knowledge_agent_delegates_to_llm():
    llm = MagicMock()
    llm.generate_structured_answer.return_value = {"answer": "42", "claims": [], "format_type": "definition"}
    agent = KnowledgeAgent(llm)

    result = agent.generate("q", [_make_chunk()], QueryType.DEFINITION)

    assert result["answer"] == "42"
    llm.generate_structured_answer.assert_called_once()


def test_reflection_agent_delegates_to_llm():
    llm = MagicMock()
    llm.reflect_on_answer.return_value = {
        "revised_answer": "Better.",
        "materially_changed": True,
        "should_block": False,
        "issues_found": [],
    }
    agent = ReflectionAgent(llm)

    result = agent.reflect("q", [_make_chunk()], "draft", "summary", QueryType.DEFINITION)

    assert result["revised_answer"] == "Better."
    llm.reflect_on_answer.assert_called_once_with(
        "q", [_make_chunk()], "draft", "summary", QueryType.DEFINITION
    )


def test_reflection_agent_falls_back_when_llm_raises():
    llm = MagicMock()
    llm.reflect_on_answer.side_effect = RuntimeError("boom")
    agent = ReflectionAgent(llm)

    result = agent.reflect("q", [_make_chunk()], "draft answer", "summary", QueryType.DEFINITION)

    assert result == {
        "revised_answer": "draft answer",
        "materially_changed": False,
        "should_block": False,
        "issues_found": [],
    }


def test_memory_agent_records_user_and_assistant_turns():
    history = []
    agent = MemoryAgent(history)

    agent.record_turn("What is X?", "X is Y.", QueryType.DEFINITION)

    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "What is X?"
    assert history[1].role == "assistant"
    assert history[1].content == "X is Y."


def test_memory_agent_shares_the_same_list_reference():
    history = []
    agent = MemoryAgent(history)
    agent.record_turn("q", "a")
    assert agent.chat_history is history


# ----------------------------------------------------------------------
# OrchestratorAgent
# ----------------------------------------------------------------------


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
def mock_llm():
    llm = MagicMock()
    llm.generate_structured_answer.return_value = {
        "answer": "Photosynthesis converts light into chemical energy.",
        "format_type": "definition",
        "claims": ["Photosynthesis converts light into chemical energy."],
        "intent": "definition",
    }
    llm.reflect_on_answer.return_value = _reflection_no_change(
        "Photosynthesis converts light into chemical energy."
    )
    return llm


@pytest.fixture()
def mock_retriever():
    retriever = MagicMock()
    retriever.search.return_value = {
        "query": "What is photosynthesis?",
        "intent": QueryType.DEFINITION,
        "chunks": [_make_chunk()],
        "in_scope": True,
        "is_relevant": True,
        "relevance_score": 0.9,
        "total_retrieved": 1,
    }
    return retriever


@pytest.fixture()
def orchestrator(mock_intent_classifier, mock_llm, mock_retriever):
    history = []
    return OrchestratorAgent(
        intent_classifier=mock_intent_classifier,
        retrieval_agent=RetrievalAgent(mock_retriever),
        knowledge_agent=KnowledgeAgent(mock_llm),
        reflection_agent=ReflectionAgent(mock_llm),
        memory_agent=MemoryAgent(history),
    )


def test_orchestrator_happy_path(orchestrator):
    result = orchestrator.run("What is photosynthesis?")
    assert isinstance(result, AnswerWithSources)
    assert result.query_intent == QueryType.DEFINITION
    assert result.format_type == "definition"
    assert len(result.sources) == 1
    assert orchestrator.memory_agent.chat_history[0].role == "user"
    assert orchestrator.memory_agent.chat_history[1].content == result.answer


def test_orchestrator_out_of_scope_short_circuits_before_llm(orchestrator, mock_retriever, mock_llm):
    mock_retriever.search.return_value = {
        "query": "irrelevant",
        "intent": QueryType.VAGUE,
        "chunks": [],
        "in_scope": False,
        "is_relevant": False,
        "relevance_score": 0.0,
        "total_retrieved": 0,
    }
    stages = []
    result = orchestrator.run(
        "Something totally unrelated", on_stage=lambda stage, payload: stages.append((stage, payload))
    )
    assert result.format_type == "out_of_scope"
    assert result.hallucination_risk == "high"
    assert result.answer.startswith("I couldn't find this information in the uploaded documents.")
    mock_llm.generate_structured_answer.assert_not_called()
    mock_llm.reflect_on_answer.assert_not_called()
    # Out-of-scope streams to the client exactly like every other path.
    assert stages == [("answer_ready", {"answer": result.answer})]


def test_orchestrator_reflection_materially_changed_triggers_revalidation(orchestrator, mock_llm):
    """Reflection returns its own claims for the revised answer in the same
    call — no separate re-extraction round trip. Proven here with a claim
    that deliberately doesn't match the chunk (unlike the draft's claim,
    which does): if the code incorrectly fell back to the draft's claims,
    a citation would appear."""
    mock_llm.reflect_on_answer.return_value = {
        "revised_answer": "Revised: photosynthesis converts light into stored chemical energy.",
        "materially_changed": True,
        "should_block": False,
        "issues_found": ["clarified wording"],
        "claims": [{"claim": "A completely unrelated claim about volcanoes.", "chunks": []}],
    }

    result = orchestrator.run("What is photosynthesis?")

    assert result.answer == "Revised: photosynthesis converts light into stored chemical energy."
    assert result.sources == []


def test_orchestrator_reflection_should_block_returns_fixed_fallback(orchestrator, mock_llm):
    mock_llm.reflect_on_answer.return_value = {
        "revised_answer": "",
        "materially_changed": True,
        "should_block": True,
        "issues_found": ["hallucinated throughout"],
    }

    stages = []
    result = orchestrator.run(
        "What is photosynthesis?", on_stage=lambda stage, payload: stages.append((stage, payload))
    )

    assert result.format_type == "reflection_blocked"
    assert result.hallucination_risk == "high"
    assert result.sources == []
    # The user never saw the blocked draft — only this fixed fallback streams.
    assert stages == [("answer_ready", {"answer": result.answer})]


def test_orchestrator_handles_llm_failure_gracefully(orchestrator, mock_llm):
    mock_llm.generate_structured_answer.side_effect = RuntimeError("provider unavailable")
    stages = []
    result = orchestrator.run(
        "What is photosynthesis?", on_stage=lambda stage, payload: stages.append((stage, payload))
    )
    assert result.format_type == "error"
    assert result.hallucination_risk == "high"
    assert result.overall_confidence == 0.0
    assert stages == [("answer_ready", {"answer": result.answer})]


def test_orchestrator_invokes_on_stage_callback(orchestrator):
    """Drafting and reflection both happen silently — on_stage only ever
    fires once, carrying the final, already-reflected answer text."""
    stages = []
    orchestrator.run("What is photosynthesis?", on_stage=lambda stage, payload: stages.append(stage))
    assert stages == ["answer_ready"]


def test_orchestrator_short_circuits_small_talk_before_retrieval(orchestrator, mock_retriever, mock_llm):
    stages = []
    result = orchestrator.run("hi", on_stage=lambda stage, payload: stages.append(stage))

    assert result.format_type == "greeting"
    assert result.sources == []
    # Never reached retrieval/drafting/reflection, but the single
    # answer_ready event still fires so every response path streams the
    # same way to the client.
    assert stages == ["answer_ready"]
    mock_retriever.search.assert_not_called()
    mock_llm.generate_structured_answer.assert_not_called()
    mock_llm.reflect_on_answer.assert_not_called()
    assert orchestrator.memory_agent.chat_history[0].content == "hi"
    assert orchestrator.memory_agent.chat_history[1].content == result.answer


def test_orchestrator_forwards_session_id_and_device_id_to_memory_agent(
    mock_intent_classifier, mock_llm, mock_retriever
):
    memory_agent = MagicMock()
    orchestrator = OrchestratorAgent(
        intent_classifier=mock_intent_classifier,
        retrieval_agent=RetrievalAgent(mock_retriever),
        knowledge_agent=KnowledgeAgent(mock_llm),
        reflection_agent=ReflectionAgent(mock_llm),
        memory_agent=memory_agent,
    )

    result = orchestrator.run("What is photosynthesis?", session_id="sess-1", device_id="device-1")

    memory_agent.record_turn.assert_called_once_with(
        "What is photosynthesis?",
        result.answer,
        QueryType.DEFINITION,
        session_id="sess-1",
        device_id="device-1",
    )


def test_orchestrator_forwards_document_ids_to_retrieval_agent(mock_intent_classifier, mock_llm):
    retrieval_agent = MagicMock()
    retrieval_agent.search.return_value = {
        "query": "What is photosynthesis?",
        "intent": QueryType.DEFINITION,
        "chunks": [_make_chunk()],
        "in_scope": True,
        "is_relevant": True,
        "relevance_score": 0.9,
        "total_retrieved": 1,
    }
    orchestrator = OrchestratorAgent(
        intent_classifier=mock_intent_classifier,
        retrieval_agent=retrieval_agent,
        knowledge_agent=KnowledgeAgent(mock_llm),
        reflection_agent=ReflectionAgent(mock_llm),
        memory_agent=MemoryAgent([]),
    )

    orchestrator.run("What is photosynthesis?", document_ids=["doc-1", "doc-2"])

    retrieval_agent.search.assert_called_once_with(
        "What is photosynthesis?", QueryType.DEFINITION, document_ids=["doc-1", "doc-2"], session_id=None
    )


def test_orchestrator_resolves_document_ids_from_db_when_session_id_given(
    monkeypatch, mock_intent_classifier, mock_llm
):
    """With a db_session_factory configured, the orchestrator resolves the
    conversation's authoritative document scope instead of trusting the
    caller-supplied document_ids verbatim — see resolve_document_scope."""
    import app.services.agents.orchestrator as orchestrator_module

    resolve_mock = MagicMock(return_value=["doc-server-1"])
    monkeypatch.setattr(orchestrator_module, "resolve_document_scope", resolve_mock)

    retrieval_agent = MagicMock()
    retrieval_agent.search.return_value = {
        "query": "What is photosynthesis?",
        "intent": QueryType.DEFINITION,
        "chunks": [_make_chunk()],
        "in_scope": True,
        "is_relevant": True,
        "relevance_score": 0.9,
        "total_retrieved": 1,
    }
    fake_db_session_factory = MagicMock()
    orchestrator = OrchestratorAgent(
        intent_classifier=mock_intent_classifier,
        retrieval_agent=retrieval_agent,
        knowledge_agent=KnowledgeAgent(mock_llm),
        reflection_agent=ReflectionAgent(mock_llm),
        memory_agent=MemoryAgent([]),
        db_session_factory=fake_db_session_factory,
    )

    orchestrator.run("What is photosynthesis?", session_id="sess-1", document_ids=["client-side-stale-id"])

    resolve_mock.assert_called_once_with(fake_db_session_factory, "sess-1", ["client-side-stale-id"])
    retrieval_agent.search.assert_called_once_with(
        "What is photosynthesis?", QueryType.DEFINITION, document_ids=["doc-server-1"], session_id="sess-1"
    )


def test_orchestrator_global_search_request_overrides_scope_to_none(
    monkeypatch, mock_intent_classifier, mock_llm
):
    """An explicit 'search across all my documents'-style query bypasses
    conversation scoping entirely, even when a session_id/db factory would
    otherwise resolve a narrower scope."""
    import app.services.agents.orchestrator as orchestrator_module

    resolve_mock = MagicMock(return_value=["doc-server-1"])
    monkeypatch.setattr(orchestrator_module, "resolve_document_scope", resolve_mock)

    retrieval_agent = MagicMock()
    retrieval_agent.search.return_value = {
        "query": "search across all my documents for X",
        "intent": QueryType.DEFINITION,
        "chunks": [_make_chunk()],
        "in_scope": True,
        "is_relevant": True,
        "relevance_score": 0.9,
        "total_retrieved": 1,
    }
    orchestrator = OrchestratorAgent(
        intent_classifier=mock_intent_classifier,
        retrieval_agent=retrieval_agent,
        knowledge_agent=KnowledgeAgent(mock_llm),
        reflection_agent=ReflectionAgent(mock_llm),
        memory_agent=MemoryAgent([]),
        db_session_factory=MagicMock(),
    )

    orchestrator.run("search across all my documents for X", session_id="sess-1")

    resolve_mock.assert_not_called()
    retrieval_agent.search.assert_called_once_with(
        "search across all my documents for X", QueryType.DEFINITION, document_ids=None, session_id="sess-1"
    )
