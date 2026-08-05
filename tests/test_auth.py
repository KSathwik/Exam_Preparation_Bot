"""Tests for X-API-Key authentication on sensitive endpoints.

Uses its own TestClient (no default auth header) so it can probe both the
rejection and acceptance paths — the shared `client` fixture in conftest.py
always sends a valid key, which would defeat these tests.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import TEST_ADMIN_API_KEY, TEST_API_KEY

PROTECTED_ENDPOINTS = [
    ("get", "/api/documents/list"),
    ("get", "/api/documents/stats"),
    ("get", "/api/conversations?device_id=test-device"),
]

# Gated behind ADMIN_API_KEY, not the regular APP_API_KEY — see
# require_admin_key in app/core/security.py. The regular key is deliberately
# embedded in every page load (expose_api_key_to_frontend), so it isn't a
# real boundary for operator-only endpoints like log tail/metrics.
ADMIN_ENDPOINTS = [
    ("get", "/api/metrics"),
    ("get", "/logs/tail"),
]

PUBLIC_ENDPOINTS = [
    ("get", "/health"),
    ("get", "/ready"),
    ("get", "/version"),
    ("get", "/config"),
    ("get", "/api/system/config"),
]


@pytest.fixture()
def unauthenticated_client(mock_sentence_transformer):
    with patch("app.core.dependencies.SentenceTransformer", return_value=mock_sentence_transformer):
        from app.main import app

        with TestClient(app) as c:
            yield c


@pytest.mark.parametrize("method,path", PROTECTED_ENDPOINTS)
def test_protected_endpoint_rejects_missing_key(unauthenticated_client, method, path):
    resp = getattr(unauthenticated_client, method)(path)
    assert resp.status_code == 401


@pytest.mark.parametrize("method,path", PROTECTED_ENDPOINTS)
def test_protected_endpoint_rejects_wrong_key(unauthenticated_client, method, path):
    resp = getattr(unauthenticated_client, method)(path, headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


@pytest.mark.parametrize("method,path", PROTECTED_ENDPOINTS)
def test_protected_endpoint_accepts_valid_key(unauthenticated_client, method, path):
    resp = getattr(unauthenticated_client, method)(path, headers={"X-API-Key": TEST_API_KEY})
    assert resp.status_code == 200


@pytest.mark.parametrize("method,path", PUBLIC_ENDPOINTS)
def test_public_endpoint_requires_no_key(unauthenticated_client, method, path):
    resp = getattr(unauthenticated_client, method)(path)
    assert resp.status_code == 200


@pytest.mark.parametrize("method,path", ADMIN_ENDPOINTS)
def test_admin_endpoint_rejects_missing_key(unauthenticated_client, method, path):
    resp = getattr(unauthenticated_client, method)(path)
    assert resp.status_code == 401


@pytest.mark.parametrize("method,path", ADMIN_ENDPOINTS)
def test_admin_endpoint_rejects_wrong_key(unauthenticated_client, method, path):
    resp = getattr(unauthenticated_client, method)(path, headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


@pytest.mark.parametrize("method,path", ADMIN_ENDPOINTS)
def test_admin_endpoint_rejects_the_regular_public_key(unauthenticated_client, method, path):
    """The key the frontend embeds in every page must not unlock operator-
    only endpoints — that's the entire point of splitting the two keys."""
    resp = getattr(unauthenticated_client, method)(path, headers={"X-API-Key": TEST_API_KEY})
    assert resp.status_code == 401


@pytest.mark.parametrize("method,path", ADMIN_ENDPOINTS)
def test_admin_endpoint_accepts_valid_admin_key(unauthenticated_client, method, path):
    resp = getattr(unauthenticated_client, method)(path, headers={"X-API-Key": TEST_ADMIN_API_KEY})
    assert resp.status_code == 200


@pytest.mark.parametrize("method,path", ADMIN_ENDPOINTS)
def test_admin_endpoint_disabled_when_admin_key_unset(unauthenticated_client, method, path, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "admin_api_key", None)
    resp = getattr(unauthenticated_client, method)(path, headers={"X-API-Key": TEST_ADMIN_API_KEY})
    assert resp.status_code == 404


def test_auth_can_be_disabled(unauthenticated_client, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "api_auth_enabled", False)
    resp = unauthenticated_client.get("/api/documents/list")
    assert resp.status_code == 200


def test_websocket_rejects_missing_key(unauthenticated_client):
    with pytest.raises(Exception):
        with unauthenticated_client.websocket_connect("/api/ws") as ws:
            ws.send_text('{"query": "hello"}')
            ws.receive_json()


def test_websocket_accepts_key_via_query_param(unauthenticated_client):
    with unauthenticated_client.websocket_connect(f"/api/ws?api_key={TEST_API_KEY}") as ws:
        ws.send_text('{"query": ""}')
        message = ws.receive_json()
        assert message["type"] == "error"
