"""Tests for conversation (chat history) management endpoints.

Overrides the `get_db` FastAPI dependency with a StaticPool-backed in-memory
SQLite engine, so rows seeded directly in the test function and the
TestClient's app-handled requests share the same connection regardless of
which thread handles them — TestClient dispatches through a background
thread, and a bare `sqlite:///:memory:` connection is otherwise per-thread
(see the equivalent note in tests/test_documents.py).
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.api.conversations as conversations_module
from app.core.database import get_db
from app.main import app
from app.models.db_models import Base, ChatMessageRecord, ChatSession, ConversationMemory, DocumentRecord


@pytest.fixture()
def db_session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    test_session_local = sessionmaker(bind=engine)

    def _override_get_db():
        db = test_session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    yield test_session_local
    app.dependency_overrides.pop(get_db, None)


def _seed_session(
    db_session_factory, session_id, device_id="device-a", title=None, turn_count=1, last_activity_at=None
):
    db = db_session_factory()
    session = ChatSession(id=session_id, device_id=device_id, title=title, turn_count=turn_count)
    if last_activity_at is not None:
        session.last_activity_at = last_activity_at
    db.add(session)
    db.flush()
    db.add(ChatMessageRecord(session_id=session_id, role="user", content="hello", token_count=1))
    db.add(ChatMessageRecord(session_id=session_id, role="assistant", content="hi there", token_count=3))
    db.commit()
    db.close()


def test_list_conversations_requires_device_id(client, db_session_factory):
    resp = client.get("/api/conversations")
    assert resp.status_code == 422


def test_list_conversations_empty_for_unknown_device(client, db_session_factory):
    resp = client.get("/api/conversations", params={"device_id": "no-such-device"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["total"] == 0
    assert data["conversations"] == []


def test_list_conversations_filters_by_device_and_orders_by_recency(client, db_session_factory):
    now = datetime.now()
    _seed_session(db_session_factory, "sess-a1", device_id="device-a", title="First", last_activity_at=now)
    _seed_session(
        db_session_factory,
        "sess-a2",
        device_id="device-a",
        title="Second",
        last_activity_at=now + timedelta(minutes=5),
    )
    _seed_session(db_session_factory, "sess-b1", device_id="device-b", title="Other device")

    resp = client.get("/api/conversations", params={"device_id": "device-a"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    ids = {c["session_id"] for c in data["conversations"]}
    assert ids == {"sess-a1", "sess-a2"}
    assert data["conversations"][0]["message_count"] == 2
    # sess-a2 has the more recent last_activity_at, so it must lead despite
    # being created after sess-a1 — this is the "recency" the test name claims.
    assert [c["session_id"] for c in data["conversations"]] == ["sess-a2", "sess-a1"]


def test_renaming_conversation_does_not_change_sidebar_order(client, db_session_factory):
    """Regression test: renaming used to bump ChatSession.updated_at, which
    list_conversations sorted by — so a rename silently jumped a conversation
    to the top. Renaming is metadata, not activity, and must never reorder
    the sidebar (see ChatSession.last_activity_at)."""
    now = datetime.now()
    _seed_session(db_session_factory, "sess-older", title="Older", last_activity_at=now)
    _seed_session(
        db_session_factory, "sess-newer", title="Newer", last_activity_at=now + timedelta(minutes=5)
    )

    resp = client.get("/api/conversations", params={"device_id": "device-a"})
    before = [c["session_id"] for c in resp.json()["conversations"]]
    assert before == ["sess-newer", "sess-older"]

    rename_resp = client.patch("/api/conversations/sess-older", json={"title": "Renamed"})
    assert rename_resp.status_code == 200

    resp2 = client.get("/api/conversations", params={"device_id": "device-a"})
    after = [c["session_id"] for c in resp2.json()["conversations"]]
    assert after == before


def test_get_conversation_returns_messages_in_order(client, db_session_factory):
    _seed_session(db_session_factory, "sess-a1", title="First")

    resp = client.get("/api/conversations/sess-a1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess-a1"
    assert data["title"] == "First"
    assert [m["role"] for m in data["messages"]] == ["user", "assistant"]
    assert data["messages"][0]["content"] == "hello"


def test_get_conversation_404_for_unknown_id(client, db_session_factory):
    resp = client.get("/api/conversations/does-not-exist")
    assert resp.status_code == 404


def test_rename_conversation_updates_title(client, db_session_factory):
    _seed_session(db_session_factory, "sess-a1", title="Old title")

    resp = client.patch("/api/conversations/sess-a1", json={"title": "New title"})
    assert resp.status_code == 200
    assert resp.json()["conversation"]["title"] == "New title"

    resp2 = client.get("/api/conversations/sess-a1")
    assert resp2.json()["title"] == "New title"


def test_rename_conversation_404_for_unknown_id(client, db_session_factory):
    resp = client.patch("/api/conversations/does-not-exist", json={"title": "New title"})
    assert resp.status_code == 404


def test_rename_conversation_rejects_empty_title(client, db_session_factory):
    _seed_session(db_session_factory, "sess-a1")
    resp = client.patch("/api/conversations/sess-a1", json={"title": ""})
    assert resp.status_code == 422


def test_delete_conversation_cascades_messages_and_memories(client, db_session_factory, monkeypatch):
    _seed_session(db_session_factory, "sess-a1")
    db = db_session_factory()
    db.add(
        ConversationMemory(
            id="mem-1",
            session_id="sess-a1",
            summary_text="summary",
            covers_from_message_id=1,
            covers_to_message_id=2,
            embedded=True,
        )
    )
    db.commit()
    db.close()

    mock_bot = MagicMock()
    monkeypatch.setattr(conversations_module, "get_bot", lambda: mock_bot)

    resp = client.delete("/api/conversations/sess-a1")
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    mock_bot.vector_store_manager.remove_document.assert_called_once_with("mem-1")

    db = db_session_factory()
    assert db.get(ChatSession, "sess-a1") is None
    assert db.query(ChatMessageRecord).filter_by(session_id="sess-a1").count() == 0
    assert db.query(ConversationMemory).filter_by(session_id="sess-a1").count() == 0
    db.close()


def test_delete_conversation_cascades_documents(client, db_session_factory, monkeypatch):
    """Each conversation is its own isolated notebook — deleting it deletes
    the document(s) that belonged only to it, not just messages/memories."""
    _seed_session(db_session_factory, "sess-a1")
    db = db_session_factory()
    db.add(
        DocumentRecord(
            id="doc-1",
            file_name="resume.pdf",
            file_type="pdf",
            file_size_mb=0.1,
            total_chunks=2,
            session_id="sess-a1",
        )
    )
    db.commit()
    db.close()

    mock_bot = MagicMock()
    monkeypatch.setattr(conversations_module, "get_bot", lambda: mock_bot)

    resp = client.delete("/api/conversations/sess-a1")
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    mock_bot.vector_store_manager.remove_document.assert_called_once_with("doc-1")

    db = db_session_factory()
    assert db.query(DocumentRecord).filter_by(id="doc-1").first() is None
    db.close()


def test_delete_conversation_skips_vector_cleanup_when_no_embedded_memories(
    client, db_session_factory, monkeypatch
):
    _seed_session(db_session_factory, "sess-a1")
    mock_bot = MagicMock()
    monkeypatch.setattr(conversations_module, "get_bot", lambda: mock_bot)

    resp = client.delete("/api/conversations/sess-a1")
    assert resp.status_code == 200
    mock_bot.vector_store_manager.remove_document.assert_not_called()


def test_delete_conversation_404_for_unknown_id(client, db_session_factory):
    resp = client.delete("/api/conversations/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/conversations/some-id"),
        ("patch", "/api/conversations/some-id"),
        ("delete", "/api/conversations/some-id"),
    ],
)
def test_conversation_detail_endpoints_reject_missing_api_key(method, path, db_session_factory):
    with TestClient(app) as unauthenticated_client:
        call = getattr(unauthenticated_client, method)
        resp = call(path, json={"title": "x"}) if method == "patch" else call(path)
        assert resp.status_code == 401
