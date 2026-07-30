"""Tests for health and system endpoints."""


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_version_endpoint(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    data = resp.json()
    assert data["app_name"] == "Exam Prep Bot"
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
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "vector_store" in data
    assert "model_info" in data
