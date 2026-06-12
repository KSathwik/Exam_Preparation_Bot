"""Shared fixtures for the test suite."""

import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key-for-tests")


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
        np.random.default_rng(42).random((len(texts) if isinstance(texts, list) else 1, 384)).astype("float32")
        if isinstance(texts, list)
        else np.random.default_rng(42).random(384).astype("float32")
    )
    return mock


@pytest.fixture()
def client(mock_sentence_transformer):
    """FastAPI TestClient with all heavy models mocked out."""
    with patch("app.core.dependencies.SentenceTransformer", return_value=mock_sentence_transformer):
        from app.main import app

        with TestClient(app) as c:
            yield c
