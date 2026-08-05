"""Tests for MemoryAgent's opt-in persistence: session/message rows, the
dual summarization trigger, and semantic-memory embedding.

persist=False (the default used by ExamPrepBot today) is covered in
tests/test_orchestrator.py; this file exercises the persist=True path in
isolation against a real in-memory SQLite database.
"""

from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.models.db_models import Base, ChatMessageRecord, ChatSession, ConversationMemory
from app.services.agents.memory_agent import MemoryAgent
from app.services.models import QueryType, ChatMessage


@pytest.fixture()
def db_session_factory():
    # StaticPool + check_same_thread=False so every db_session_factory() call
    # shares the same in-memory SQLite database instead of each getting its
    # own empty one (SQLite ":memory:" is otherwise per-connection).
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


def test_record_turn_without_persist_only_updates_chat_history(db_session_factory):
    history: list[ChatMessage] = []
    agent = MemoryAgent(history, persist=False, session_id="sess-1", db_session_factory=db_session_factory)

    agent.record_turn("What is X?", "X is Y.")

    assert len(history) == 2
    db = db_session_factory()
    assert db.query(ChatSession).count() == 0
    db.close()


def test_record_turn_with_persist_creates_session_and_messages(db_session_factory):
    history: list[ChatMessage] = []
    agent = MemoryAgent(history, persist=True, session_id="sess-1", db_session_factory=db_session_factory)

    agent.record_turn("What is X?", "X is Y.", intent=QueryType.DEFINITION)

    db = db_session_factory()
    session_row = db.get(ChatSession, "sess-1")
    assert session_row is not None
    assert session_row.turn_count == 1
    # A real turn is exactly the kind of "activity" that should move this
    # conversation to the top of the sidebar (see ChatSession.last_activity_at).
    assert session_row.last_activity_at is not None

    messages = db.query(ChatMessageRecord).filter_by(session_id="sess-1").order_by(ChatMessageRecord.id).all()
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "What is X?"
    assert messages[0].token_count == 3  # "What is X?" -> 3 words
    assert messages[1].role == "assistant"
    assert messages[1].token_count == 6  # running total: 3 + "X is Y." (3 words)
    db.close()


def test_record_turn_persist_without_session_id_skips_gracefully():
    history: list[ChatMessage] = []
    agent = MemoryAgent(history, persist=True, session_id=None, db_session_factory=MagicMock())

    agent.record_turn("q", "a")  # must not raise

    assert len(history) == 2


def test_persist_turn_rolls_back_and_keeps_history_on_db_failure(db_session_factory):
    history: list[ChatMessage] = []
    agent = MemoryAgent(history, persist=True, session_id="sess-1", db_session_factory=db_session_factory)
    broken_session = MagicMock()
    broken_session.get.side_effect = RuntimeError("db down")
    agent.db_session_factory = MagicMock(return_value=broken_session)

    agent.record_turn("q", "a")  # must not raise

    assert len(history) == 2
    broken_session.rollback.assert_called_once()
    broken_session.close.assert_called_once()


def test_summarization_triggers_every_n_turns(db_session_factory, monkeypatch):
    monkeypatch.setattr(settings, "memory_summarize_every_n_turns", 2)
    monkeypatch.setattr(settings, "memory_summarize_token_threshold", 10_000)
    mock_llm = MagicMock()
    mock_llm.summarize_conversation.return_value = "Summary of the conversation."
    mock_vsm = MagicMock()
    agent = MemoryAgent(
        [],
        persist=True,
        session_id="sess-1",
        db_session_factory=db_session_factory,
        llm=mock_llm,
        vector_store_manager=mock_vsm,
    )

    agent.record_turn("q1", "a1")
    mock_llm.summarize_conversation.assert_not_called()

    agent.record_turn("q2", "a2")
    mock_llm.summarize_conversation.assert_called_once()
    mock_vsm.add_memory.assert_called_once()

    db = db_session_factory()
    memories = db.query(ConversationMemory).filter_by(session_id="sess-1").all()
    assert len(memories) == 1
    assert memories[0].summary_text == "Summary of the conversation."
    assert memories[0].embedded is True
    session_row = db.get(ChatSession, "sess-1")
    assert session_row.last_summarized_message_id == memories[0].covers_to_message_id
    db.close()


def test_summarization_triggers_on_token_threshold(db_session_factory, monkeypatch):
    monkeypatch.setattr(settings, "memory_summarize_every_n_turns", 1000)
    monkeypatch.setattr(settings, "memory_summarize_token_threshold", 5)
    mock_llm = MagicMock()
    mock_llm.summarize_conversation.return_value = "Summary."
    agent = MemoryAgent(
        [], persist=True, session_id="sess-1", db_session_factory=db_session_factory, llm=mock_llm
    )

    agent.record_turn("one two three", "four five six")  # cumulative tokens = 6 >= 5

    mock_llm.summarize_conversation.assert_called_once()


def test_summarization_skipped_without_llm(db_session_factory, monkeypatch):
    monkeypatch.setattr(settings, "memory_summarize_every_n_turns", 1)
    agent = MemoryAgent(
        [], persist=True, session_id="sess-1", db_session_factory=db_session_factory, llm=None
    )

    agent.record_turn("q", "a")  # would trigger, but no llm configured

    db = db_session_factory()
    assert db.query(ConversationMemory).count() == 0
    db.close()


def test_summarization_keeps_memory_row_when_embedding_fails(db_session_factory, monkeypatch):
    monkeypatch.setattr(settings, "memory_summarize_every_n_turns", 1)
    mock_llm = MagicMock()
    mock_llm.summarize_conversation.return_value = "Summary."
    mock_vsm = MagicMock()
    mock_vsm.add_memory.side_effect = RuntimeError("embedding failed")
    agent = MemoryAgent(
        [],
        persist=True,
        session_id="sess-1",
        db_session_factory=db_session_factory,
        llm=mock_llm,
        vector_store_manager=mock_vsm,
    )

    agent.record_turn("q", "a")  # must not raise

    db = db_session_factory()
    memories = db.query(ConversationMemory).filter_by(session_id="sess-1").all()
    assert len(memories) == 1
    assert memories[0].embedded is False
    db.close()


def test_summarization_does_not_resummarize_same_turns(db_session_factory, monkeypatch):
    monkeypatch.setattr(settings, "memory_summarize_every_n_turns", 1)
    mock_llm = MagicMock()
    mock_llm.summarize_conversation.return_value = "Summary."
    agent = MemoryAgent(
        [], persist=True, session_id="sess-1", db_session_factory=db_session_factory, llm=mock_llm
    )

    agent.record_turn("q1", "a1")
    agent.record_turn("q2", "a2")

    assert mock_llm.summarize_conversation.call_count == 2
    second_transcript = mock_llm.summarize_conversation.call_args_list[1][0][0]
    assert all(m.content in ("q2", "a2") for m in second_transcript)


# ── Conversation title + device_id (chat-history sidebar support) ────


def test_persist_turn_sets_title_from_first_message(db_session_factory):
    agent = MemoryAgent([], persist=True, session_id="sess-1", db_session_factory=db_session_factory)

    agent.record_turn("What is photosynthesis?", "It's how plants make energy.")

    db = db_session_factory()
    session_row = db.get(ChatSession, "sess-1")
    assert session_row.title == "What is photosynthesis?"
    db.close()


def test_persist_turn_truncates_long_title(db_session_factory):
    agent = MemoryAgent([], persist=True, session_id="sess-1", db_session_factory=db_session_factory)
    long_query = "Can you explain in great detail how the process of cellular respiration works?"

    agent.record_turn(long_query, "Sure, here's an explanation.")

    db = db_session_factory()
    session_row = db.get(ChatSession, "sess-1")
    assert session_row.title == long_query[:50].rstrip() + "…"
    assert len(session_row.title) <= 51
    db.close()


def test_persist_turn_stores_device_id_on_new_session(db_session_factory):
    agent = MemoryAgent([], persist=True, session_id="sess-1", db_session_factory=db_session_factory)

    agent.record_turn("q", "a", device_id="device-abc")

    db = db_session_factory()
    session_row = db.get(ChatSession, "sess-1")
    assert session_row.device_id == "device-abc"
    db.close()


def test_persist_turn_backfills_device_id_when_missing(db_session_factory):
    agent = MemoryAgent([], persist=True, session_id="sess-1", db_session_factory=db_session_factory)

    agent.record_turn("q1", "a1")  # no device_id yet
    agent.record_turn("q2", "a2", device_id="device-abc")  # backfilled here

    db = db_session_factory()
    session_row = db.get(ChatSession, "sess-1")
    assert session_row.device_id == "device-abc"
    db.close()


def test_persist_turn_does_not_overwrite_existing_title_on_subsequent_turns(db_session_factory):
    agent = MemoryAgent([], persist=True, session_id="sess-1", db_session_factory=db_session_factory)

    agent.record_turn("First question", "First answer")
    agent.record_turn("Second question", "Second answer")

    db = db_session_factory()
    session_row = db.get(ChatSession, "sess-1")
    assert session_row.title == "First question"
    db.close()


def test_record_turn_session_id_override_takes_precedence_over_constructor_value(db_session_factory):
    """Proves the concurrency-safety fix actually works: session_id must flow
    as a per-call parameter, not be read from shared mutable instance state
    (this agent is a process-wide singleton shared across concurrent
    requests for different conversations)."""
    agent = MemoryAgent([], persist=True, session_id="ctor-sess", db_session_factory=db_session_factory)

    agent.record_turn("q", "a", session_id="call-sess")

    db = db_session_factory()
    assert db.get(ChatSession, "call-sess") is not None
    assert db.get(ChatSession, "ctor-sess") is None
    db.close()


# ── get_recent_turns (short-term conversation context for the LLM calls) ──


def test_get_recent_turns_returns_chronological_order(db_session_factory):
    agent = MemoryAgent([], persist=True, db_session_factory=db_session_factory)
    agent.record_turn("First question", "First answer", session_id="sess-1")
    agent.record_turn("Second question", "Second answer", session_id="sess-1")

    turns = agent.get_recent_turns("sess-1")

    assert [t.content for t in turns] == [
        "First question",
        "First answer",
        "Second question",
        "Second answer",
    ]
    assert [t.role for t in turns] == ["user", "assistant", "user", "assistant"]


def test_get_recent_turns_respects_max_messages(db_session_factory):
    agent = MemoryAgent([], persist=True, db_session_factory=db_session_factory)
    for i in range(5):
        agent.record_turn(f"question {i}", f"answer {i}", session_id="sess-1")

    turns = agent.get_recent_turns("sess-1", max_messages=4)

    assert len(turns) == 4
    # Most recent 2 turns (4 messages), still oldest-first within that window.
    assert [t.content for t in turns] == ["question 3", "answer 3", "question 4", "answer 4"]


def test_get_recent_turns_scoped_to_the_right_session(db_session_factory):
    agent = MemoryAgent([], persist=True, db_session_factory=db_session_factory)
    agent.record_turn("Session A question", "Session A answer", session_id="sess-a")
    agent.record_turn("Session B question", "Session B answer", session_id="sess-b")

    turns = agent.get_recent_turns("sess-a")

    assert [t.content for t in turns] == ["Session A question", "Session A answer"]


def test_get_recent_turns_returns_empty_without_persist_or_session_id(db_session_factory):
    persisted_agent = MemoryAgent([], persist=True, db_session_factory=db_session_factory)
    agent_not_persisting = MemoryAgent([], persist=False, db_session_factory=db_session_factory)

    assert persisted_agent.get_recent_turns(None) == []
    assert agent_not_persisting.get_recent_turns("sess-1") == []


def test_get_recent_turns_handles_db_failure_gracefully():
    agent = MemoryAgent([], persist=True, session_id="sess-1", db_session_factory=MagicMock(side_effect=RuntimeError("db down")))

    assert agent.get_recent_turns("sess-1") == []
