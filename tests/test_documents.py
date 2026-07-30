"""Tests for document upload and management endpoints."""

import io
from pathlib import Path

from app.api.documents import _safe_upload_path
from app.core.config import settings


def test_upload_rejects_unsupported_type(client):
    fake = io.BytesIO(b"hello")
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", fake, "text/plain")},
    )
    assert resp.status_code == 400
    assert "Unsupported" in resp.json()["detail"]


def test_upload_rejects_oversize_file(client, monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "max_file_size_mb", 0)  # 0 MB limit

    fake = io.BytesIO(b"x" * 1024)
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("test.pdf", fake, "application/pdf")},
    )
    assert resp.status_code == 413


def test_list_documents_empty(client):
    resp = client.get("/api/documents/list")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["total_documents"] == 0


def test_delete_nonexistent_document(client):
    resp = client.delete("/api/documents/nonexistent-id")
    assert resp.status_code == 404


def test_document_stats(client):
    resp = client.get("/api/documents/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "vector_store" in data
    assert "timestamp" in data


# ── Path traversal hardening ─────────────────────────────────────────


def test_safe_upload_path_strips_directory_traversal():
    path = _safe_upload_path("../../../etc/passwd", "abc123")
    assert path.parent == Path(settings.upload_dir)
    assert ".." not in path.name
    assert path.name == "abc123_passwd"


def test_safe_upload_path_strips_absolute_path():
    path = _safe_upload_path("/etc/passwd", "abc123")
    assert path.parent == Path(settings.upload_dir)
    assert path.name == "abc123_passwd"


def test_safe_upload_path_strips_windows_style_traversal():
    path = _safe_upload_path("..\\..\\windows\\system32\\evil.pdf", "abc123")
    assert path.parent == Path(settings.upload_dir)
    assert ".." not in str(path.relative_to(Path(settings.upload_dir)))


def test_safe_upload_path_handles_missing_filename():
    path = _safe_upload_path(None, "abc123")
    assert path.parent == Path(settings.upload_dir)
    assert path.name == "abc123_upload"


def test_safe_upload_path_handles_dot_dot_filename():
    path = _safe_upload_path("..", "abc123")
    assert path.name == "abc123_upload"


# ── Upload happy path + delete cleanup ───────────────────────────────
# Goes through the real endpoints (not a raw DB session) since TestClient
# runs the app on its own thread — a SQLite `:memory:` session opened
# directly from the test thread would not see the app's tables at all.


def _make_docx_bytes() -> bytes:
    from io import BytesIO

    from docx import Document as DocxDocument

    sentence = (
        "Photosynthesis is the process by which green plants convert sunlight "
        "into chemical energy stored in glucose molecules. "
    )
    doc = DocxDocument()
    doc.add_paragraph(sentence * 15)  # comfortably over min_chunk_size (100 words)
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_upload_docx_happy_path(client):
    resp = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "notes.docx",
                _make_docx_bytes(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["total_chunks"] >= 1
    assert data["document_id"]

    # Clean up so this doesn't leak into other tests' document counts.
    client.delete(f"/api/documents/{data['document_id']}")


def test_upload_with_session_id_scopes_document_to_that_conversation(client):
    resp = client.post(
        "/api/documents/upload",
        data={"session_id": "sess-doc-1", "device_id": "device-1"},
        files={
            "file": (
                "notes.docx",
                _make_docx_bytes(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert resp.status_code == 200
    document_id = resp.json()["document_id"]

    try:
        scoped = client.get("/api/documents/list", params={"session_id": "sess-doc-1"})
        assert document_id in [d["document_id"] for d in scoped.json()["documents"]]

        other_session = client.get("/api/documents/list", params={"session_id": "some-other-session"})
        assert document_id not in [d["document_id"] for d in other_session.json()["documents"]]
    finally:
        client.delete(f"/api/documents/{document_id}")


def test_upload_without_session_id_still_works(client):
    """session_id is optional — uploads with no conversation concept in play
    keep working exactly as before, just unscoped (session_id=None)."""
    resp = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "notes.docx",
                _make_docx_bytes(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert resp.status_code == 200
    document_id = resp.json()["document_id"]
    client.delete(f"/api/documents/{document_id}")


def test_delete_document_removes_file_and_vectors(client):
    upload_resp = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "notes.docx",
                _make_docx_bytes(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert upload_resp.status_code == 200
    document_id = upload_resp.json()["document_id"]
    assert upload_resp.json()["total_chunks"] >= 1

    list_resp = client.get("/api/documents/list")
    ids_before = [d["document_id"] for d in list_resp.json()["documents"]]
    assert document_id in ids_before

    delete_resp = client.delete(f"/api/documents/{document_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["removed_vectors"] >= 1

    list_resp_after = client.get("/api/documents/list")
    ids_after = [d["document_id"] for d in list_resp_after.json()["documents"]]
    assert document_id not in ids_after
