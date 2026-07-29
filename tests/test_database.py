"""Tests for SQLAlchemy models and database setup."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.db_models import (
    Base,
    ChatMessageRecord,
    ChatSession,
    ConversationMemory,
    DocumentRecord,
    QueryRecord,
)


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_create_tables(db_session):
    """Tables should be created without errors."""
    assert db_session.query(DocumentRecord).count() == 0


def test_insert_document(db_session):
    doc = DocumentRecord(
        id="doc-1",
        file_name="test.pdf",
        file_type="pdf",
        file_size_mb=1.5,
        total_chunks=10,
    )
    db_session.add(doc)
    db_session.commit()

    fetched = db_session.query(DocumentRecord).filter_by(id="doc-1").first()
    assert fetched is not None
    assert fetched.file_name == "test.pdf"
    assert fetched.total_chunks == 10


def test_insert_query_record(db_session):
    doc = DocumentRecord(id="doc-2", file_name="a.pdf", file_type="pdf", file_size_mb=1.0, total_chunks=5)
    db_session.add(doc)
    db_session.commit()

    query = QueryRecord(
        id="q-1",
        document_id="doc-2",
        query_text="What is X?",
        answer_text="X is ...",
        intent="definition",
        overall_confidence=0.85,
    )
    db_session.add(query)
    db_session.commit()

    assert db_session.query(QueryRecord).count() == 1
    assert db_session.query(QueryRecord).first().intent == "definition"


def test_cascade_delete(db_session):
    doc = DocumentRecord(id="doc-3", file_name="b.pdf", file_type="pdf", file_size_mb=2.0, total_chunks=3)
    db_session.add(doc)
    db_session.flush()

    q = QueryRecord(id="q-2", document_id="doc-3", query_text="Test?")
    db_session.add(q)
    db_session.commit()

    db_session.delete(doc)
    db_session.commit()
    assert db_session.query(QueryRecord).count() == 0


# ── Conversation memory schema (ChatSession/ChatMessageRecord extensions,
#    ConversationMemory) ────────────────────────────────────────────────


def test_chat_session_turn_count_defaults_to_zero(db_session):
    session = ChatSession(id="sess-1")
    db_session.add(session)
    db_session.commit()

    fetched = db_session.query(ChatSession).filter_by(id="sess-1").first()
    assert fetched.turn_count == 0
    assert fetched.user_id is None
    assert fetched.last_summarized_message_id is None


def test_chat_message_round_trip_with_token_count(db_session):
    session = ChatSession(id="sess-2")
    db_session.add(session)
    db_session.flush()

    msg = ChatMessageRecord(session_id="sess-2", role="user", content="What is X?", token_count=12)
    db_session.add(msg)
    db_session.commit()

    fetched = db_session.query(ChatMessageRecord).filter_by(session_id="sess-2").first()
    assert fetched.token_count == 12
    assert fetched.session.id == "sess-2"


def test_conversation_memory_round_trip(db_session):
    session = ChatSession(id="sess-3")
    db_session.add(session)
    db_session.flush()

    m1 = ChatMessageRecord(session_id="sess-3", role="user", content="q1")
    m2 = ChatMessageRecord(session_id="sess-3", role="assistant", content="a1")
    db_session.add_all([m1, m2])
    db_session.flush()

    memory = ConversationMemory(
        id="mem-1",
        session_id="sess-3",
        summary_text="Discussed X.",
        covers_from_message_id=m1.id,
        covers_to_message_id=m2.id,
        embedded=False,
    )
    db_session.add(memory)
    db_session.commit()

    fetched = db_session.query(ConversationMemory).filter_by(id="mem-1").first()
    assert fetched.summary_text == "Discussed X."
    assert fetched.embedded is False
    assert fetched.session.id == "sess-3"


def test_deleting_session_cascades_to_messages_and_memories(db_session):
    session = ChatSession(id="sess-4")
    db_session.add(session)
    db_session.flush()

    m1 = ChatMessageRecord(session_id="sess-4", role="user", content="q1")
    db_session.add(m1)
    db_session.flush()

    memory = ConversationMemory(
        id="mem-2",
        session_id="sess-4",
        summary_text="Discussed Y.",
        covers_from_message_id=m1.id,
        covers_to_message_id=m1.id,
    )
    db_session.add(memory)
    db_session.commit()

    db_session.delete(session)
    db_session.commit()

    assert db_session.query(ChatMessageRecord).filter_by(session_id="sess-4").count() == 0
    assert db_session.query(ConversationMemory).filter_by(session_id="sess-4").count() == 0
