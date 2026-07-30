"""Tests for resolve_document_scope — the authoritative, DB-backed
conversation document scope that client-sent document_ids can only narrow
within, never expand beyond (see app/services/conversation_scope.py)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.db_models import Base, DocumentRecord
from app.services.conversation_scope import resolve_document_scope


@pytest.fixture()
def db_session_factory():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine)

    db = session_local()
    db.add(
        DocumentRecord(id="doc-1", file_name="a.pdf", file_type="pdf", file_size_mb=1, session_id="sess-1")
    )
    db.add(
        DocumentRecord(id="doc-2", file_name="b.pdf", file_type="pdf", file_size_mb=1, session_id="sess-1")
    )
    db.add(
        DocumentRecord(id="doc-3", file_name="c.pdf", file_type="pdf", file_size_mb=1, session_id="sess-2")
    )
    db.commit()
    db.close()

    return session_local


def test_no_session_id_returns_client_ids_unchanged(db_session_factory):
    assert resolve_document_scope(db_session_factory, None, ["client-1"]) == ["client-1"]
    assert resolve_document_scope(db_session_factory, None, None) is None


def test_session_id_with_no_client_ids_returns_full_conversation_scope(db_session_factory):
    result = resolve_document_scope(db_session_factory, "sess-1", None)
    assert set(result) == {"doc-1", "doc-2"}


def test_session_with_zero_documents_returns_empty_list_not_none(db_session_factory):
    result = resolve_document_scope(db_session_factory, "sess-with-no-docs", None)
    assert result == []


def test_client_ids_narrow_within_the_conversations_documents(db_session_factory):
    result = resolve_document_scope(db_session_factory, "sess-1", ["doc-1"])
    assert result == ["doc-1"]


def test_client_ids_cannot_expand_beyond_the_conversations_documents(db_session_factory):
    """A stale/foreign document_id (e.g. from another conversation, or a
    reload that resurrected an old client-side list) is silently dropped —
    it can never widen the scope beyond what this conversation actually owns."""
    result = resolve_document_scope(db_session_factory, "sess-1", ["doc-3"])
    assert set(result) == {"doc-1", "doc-2"}


def test_client_ids_entirely_foreign_falls_back_to_full_conversation_scope(db_session_factory):
    result = resolve_document_scope(db_session_factory, "sess-1", ["doc-3", "doc-nonexistent"])
    assert set(result) == {"doc-1", "doc-2"}
