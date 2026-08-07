"""Tests for health and system endpoints."""

from tests.conftest import TEST_ADMIN_API_KEY


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "degraded", "unhealthy")
    assert "version" in data
    assert "llm_provider" in data
    assert "llm_health" in data


def test_version_endpoint(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    data = resp.json()
    assert data["app_name"] in ("AI Knowledge Assistant", "Exam Prep Bot")
    assert "version" in data


def test_config_endpoint(client):
    resp = client.get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "model" in data
    assert "embedding_model" in data


def test_root_serves_response(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_system_config(client):
    resp = client.get("/api/system/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "model" in data
    assert "temperature" in data


def test_metrics_endpoint(client):
    # /api/metrics is admin-gated, not covered by the `client` fixture's
    # default (regular) X-API-Key header — see require_admin_key.
    resp = client.get("/api/metrics", headers={"X-API-Key": TEST_ADMIN_API_KEY})
    assert resp.status_code == 200
    data = resp.json()
    assert "vector_store" in data
    assert "model_info" in data
