"""Tests for FAISS vector store operations (no real model needed)."""

import numpy as np
import pytest
from app.services.embeddings import FAISSVectorStore


@pytest.fixture()
def store(tmp_path):
    s = FAISSVectorStore(dimension=4, index_path=str(tmp_path / "faiss"))
    s.create_index()
    return s


def test_create_empty_index(store):
    assert store.get_size() == 0


def test_add_and_search(store):
    embeddings = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    meta = [{"content": "a", "metadata": {}}, {"content": "b", "metadata": {}}]
    store.add_embeddings(embeddings, meta)
    assert store.get_size() == 2

    distances, indices = store.search(np.array([1, 0, 0, 0], dtype=np.float32), k=1)
    assert len(distances) == 1
    assert indices[0] == 0  # closest to first vector


def test_get_chunk_by_index(store):
    embeddings = np.array([[1, 0, 0, 0]], dtype=np.float32)
    store.add_embeddings(embeddings, [{"content": "hello"}])
    assert store.get_chunk_by_index(0)["content"] == "hello"
    assert store.get_chunk_by_index(99) is None


def test_save_and_load(store, tmp_path):
    embeddings = np.array([[1, 0, 0, 0]], dtype=np.float32)
    store.add_embeddings(embeddings, [{"content": "persisted"}])
    store.save_index()

    store2 = FAISSVectorStore(dimension=4, index_path=str(tmp_path / "faiss"))
    assert store2.load_index() is True
    assert store2.get_size() == 1


def test_clear_index(store):
    embeddings = np.array([[1, 0, 0, 0]], dtype=np.float32)
    store.add_embeddings(embeddings, [{"content": "x"}])
    store.clear()
    assert store.get_size() == 0


def test_search_empty_returns_empty(store):
    distances, indices = store.search(np.array([1, 0, 0, 0], dtype=np.float32), k=1)
    assert distances == []
    assert indices == []
