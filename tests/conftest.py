"""Shared fixtures for the test suite."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
# Must be set explicitly: settings.llm_provider defaults to "gemini" (see
# app/core/config.py), and unlike local dev there is no .env file in CI to
# override it — without this, every test that builds the app (get_bot())
# tries to construct a real Gemini client with no GEMINI_API_KEY and fails
# at startup, not at the assertion.
os.environ.setdefault("LLM_PROVIDER", "anthropic")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key-for-tests")
os.environ.setdefault("APP_API_KEY", "test-api-key-for-tests")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key-for-tests")
# CI sets APP_API_KEY as a real job-step env var (see ci.yml), which wins
# over setdefault() above — read back whatever actually won so the fixtures
# below always send a key that matches settings.app_api_key, instead of a
# literal that's only correct when nothing else has already set it.

# Tests must never read from or write to the real project data directories —
# a shared on-disk FAISS index/upload folder across test runs (and across
# concurrent pytest invocations) is exactly how index.faiss/metadata.pkl/
# embeddings.npy end up desynced. Point everything at a fresh temp dir.
_TEST_DATA_ROOT = tempfile.mkdtemp(prefix="exam_prep_bot_tests_")
os.environ.setdefault("FAISS_INDEX_PATH", os.path.join(_TEST_DATA_ROOT, "faiss_index"))
os.environ.setdefault("UPLOAD_DIR", os.path.join(_TEST_DATA_ROOT, "uploads"))
os.environ.setdefault("CACHE_DIR", os.path.join(_TEST_DATA_ROOT, "cache"))

TEST_API_KEY = os.environ["APP_API_KEY"]
TEST_ADMIN_API_KEY = os.environ["ADMIN_API_KEY"]


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Clear lru_cache singletons between tests to avoid cross-contamination."""
    from app.core import dependencies as deps

    deps.get_embedding_model.cache_clear()
    deps.get_intent_model.cache_clear()
    deps.get_vector_store_manager.cache_clear()
    deps.get_intent_classifier.cache_clear()
    deps.get_bot.cache_clear()
    yield
    deps.get_embedding_model.cache_clear()
    deps.get_intent_model.cache_clear()
    deps.get_vector_store_manager.cache_clear()
    deps.get_intent_classifier.cache_clear()
    deps.get_bot.cache_clear()


@pytest.fixture()
def mock_sentence_transformer():
    """Return a mock SentenceTransformer that produces deterministic embeddings."""
    import numpy as np

    mock = MagicMock()
    mock.get_sentence_embedding_dimension.return_value = 384
    mock.encode.side_effect = lambda texts, **kw: (
        np.random.default_rng(42)
        .random((len(texts) if isinstance(texts, list) else 1, 384))
        .astype("float32")
        if isinstance(texts, list)
        else np.random.default_rng(42).random(384).astype("float32")
    )
    return mock


@pytest.fixture()
def client(mock_sentence_transformer):
    """FastAPI TestClient with all heavy models mocked out.

    Sends a valid X-API-Key on every request by default so existing tests
    exercise the authenticated path without each needing to know about auth.
    Tests that specifically probe auth behavior should build their own
    TestClient (see test_auth.py) instead of using this fixture.
    """
    with patch("app.core.dependencies.SentenceTransformer", return_value=mock_sentence_transformer):
        from app.main import app

        with TestClient(app, headers={"X-API-Key": TEST_API_KEY}) as c:
            yield c
