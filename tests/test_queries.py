"""Tests for the query-processing API endpoints.

The bot/classifier/vector-store singletons are replaced with mocks at the
`app.api.queries` module level so these tests never construct a real
ExamPrepBot (which would otherwise try to reach a real LLM provider).
"""

from unittest.mock import MagicMock

import pytest

import app.api.queries as queries_module
from app.services.models import AnswerWithSources, ChatMessage, IntentClassificationResult, QueryType
from tests.conftest import TEST_API_KEY


def _fake_answer(text="42", intent=QueryType.DEFINITION):
    return AnswerWithSources(
        answer=text,
        query_intent=intent,
        intent_confidence=0.9,
        sources=[],
        overall_confidence=0.85,
        hallucination_risk="low",
        response_time_seconds=0.1,
        format_type="definition",
    )


@pytest.fixture()
def mock_bot(monkeypatch):
    bot = MagicMock()
    bot.answer_question.return_value = _fake_answer()
    bot.chat_history = []
    monkeypatch.setattr(queries_module, "get_bot", lambda: bot)
    return bot


@pytest.fixture()
def mock_intent_classifier(monkeypatch):
    clf = MagicMock()
    clf.classify.return_value = IntentClassificationResult(
        query="q", primary_intent=QueryType.EXPLAIN, confidence=0.8
    )
    monkeypatch.setattr(queries_module, "get_intent_classifier", lambda: clf)
    return clf


def test_ask_question(client, mock_bot):
    resp = client.post("/api/ask", json={"query": "What is X?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["answer"] == "42"
    assert data["query_intent"] == "definition"
    mock_bot.answer_question.assert_called_once_with("What is X?")


def test_query_alias_matches_ask(client, mock_bot):
    resp = client.post("/api/query", json={"query": "What is X?"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "42"


def test_ask_rejects_empty_query(client, mock_bot):
    resp = client.post("/api/ask", json={"query": ""})
    assert resp.status_code == 422


def test_ask_rejects_overlong_query(client, mock_bot):
    resp = client.post("/api/ask", json={"query": "x" * 1001})
    assert resp.status_code == 422


def test_ask_handles_pipeline_exception(client, mock_bot):
    mock_bot.answer_question.side_effect = RuntimeError("boom")
    resp = client.post("/api/ask", json={"query": "What is X?"})
    assert resp.status_code == 500


def test_classify_intent(client, mock_intent_classifier):
    resp = client.get("/api/intent/what%20is%20x")
    assert resp.status_code == 200
    data = resp.json()
    assert data["primary_intent"] == "explain"
    assert data["confidence"] == 0.8


def test_batch_queries(client, mock_bot):
    resp = client.post("/api/batch", json={"queries": ["a?", "b?"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["successful"] == 2
    assert mock_bot.answer_question.call_count == 2


def test_batch_rejects_too_many_queries(client, mock_bot):
    resp = client.post("/api/batch", json={"queries": [f"q{i}?" for i in range(51)]})
    assert resp.status_code == 422


def test_batch_rejects_empty_query_in_list(client, mock_bot):
    resp = client.post("/api/batch", json={"queries": ["valid?", "   "]})
    assert resp.status_code == 422


def test_batch_rejects_overlong_query_in_list(client, mock_bot):
    resp = client.post("/api/batch", json={"queries": ["x" * 1001]})
    assert resp.status_code == 422


def test_batch_partial_failure_reported(client, mock_bot):
    mock_bot.answer_question.side_effect = [_fake_answer(), RuntimeError("provider down")]
    resp = client.post("/api/batch", json={"queries": ["ok?", "fails?"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["successful"] == 1
    assert data["failed"] == 1


def test_history_roundtrip(client, mock_bot):
    mock_bot.chat_history = [
        ChatMessage(role="user", content="hi", timestamp="2024-01-01", intent_type=QueryType.VAGUE)
    ]
    resp = client.get("/api/history")
    assert resp.status_code == 200
    assert resp.json()["total"] == 1

    resp = client.delete("/api/history")
    assert resp.status_code == 200
    assert mock_bot.chat_history == []


def test_websocket_streams_stage_events_and_completes(client, mock_bot):
    mock_bot.intent_classifier.classify.return_value = IntentClassificationResult(
        query="What is X?", primary_intent=QueryType.DEFINITION, confidence=0.9
    )

    def fake_answer_question(query, on_stage=None):
        if on_stage:
            on_stage("retrieving", {})
            on_stage("drafting", {})
            on_stage("draft_ready", {"answer": "Draft answer text."})
            on_stage("reflecting", {"message": "Reviewing the draft answer for accuracy..."})
            on_stage("final_ready", {"answer": "Final answer text."})
        return _fake_answer(text="Final answer text.")

    mock_bot.answer_question.side_effect = fake_answer_question

    with client.websocket_connect(f"/api/ws?api_key={TEST_API_KEY}") as ws:
        ws.send_text('{"query": "What is X?"}')

        intent_msg = ws.receive_json()
        assert intent_msg["type"] == "intent"

        # Stage order matches OrchestratorAgent.run: draft_ready fires before
        # reflecting, so the draft chunk(s) arrive before the status message.
        msg = ws.receive_json()
        draft_chunks = []
        while msg["type"] == "chunk" and msg["stage"] == "draft":
            draft_chunks.append(msg["text"])
            msg = ws.receive_json()
        assert "".join(draft_chunks) == "Draft answer text."

        assert msg == {
            "type": "status",
            "stage": "reflecting",
            "message": "Reviewing the draft answer for accuracy...",
        }

        msg = ws.receive_json()
        final_chunks = []
        while msg["type"] == "chunk" and msg["stage"] == "final":
            final_chunks.append(msg["text"])
            msg = ws.receive_json()
        assert "".join(final_chunks) == "Final answer text."

        assert msg["type"] == "complete"
        assert msg["answer"] == "Final answer text."


def test_search_documents(client, monkeypatch, mock_intent_classifier):
    monkeypatch.setattr(queries_module, "get_vector_store_manager", lambda: MagicMock())

    fake_retriever = MagicMock()
    fake_retriever.search.return_value = {
        "query": "find something",
        "intent": QueryType.EXAMPLE,
        "chunks": [],
        "in_scope": False,
        "is_relevant": False,
        "relevance_score": 0.0,
        "total_retrieved": 0,
    }
    mock_intent_classifier.classify.return_value = IntentClassificationResult(
        query="find something", primary_intent=QueryType.EXAMPLE, confidence=0.7
    )

    import app.services.retriever as retriever_module

    monkeypatch.setattr(retriever_module, "HybridRetriever", lambda vs: fake_retriever)

    resp = client.post("/api/search", params={"query": "find something"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "example"
    assert data["total_results"] == 0
