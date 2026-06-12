"""Tests for document upload and management endpoints."""

import io


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
