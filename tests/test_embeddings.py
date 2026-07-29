"""Tests for FAISS vector store operations (no real model needed)."""

import pickle
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.services.embeddings import FAISSVectorStore, VectorStoreManager, _min_max_normalize
from app.services.models import ChunkMetadata, Document, DocumentChunk


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


# ── Persistence: atomicity + consistency validation ──────────────────
# Regression coverage for a real desync found during this audit: two
# concurrent processes (or a crash mid-save) writing index.faiss,
# metadata.pkl, and embeddings.npy as three separate non-atomic file writes
# left them out of sync, and load_index() loaded the mismatched state
# without complaint — the next add/remove then crashed on a shape mismatch.


def test_save_index_leaves_no_tmp_files(store, tmp_path):
    store.add_embeddings(np.array([[1, 0, 0, 0]], dtype=np.float32), [{"content": "a"}])
    store.save_index()

    faiss_dir = tmp_path / "faiss"
    assert not any(p.suffix == ".tmp" for p in faiss_dir.iterdir())
    assert (faiss_dir / "index.faiss").exists()
    assert (faiss_dir / "metadata.pkl").exists()
    assert (faiss_dir / "embeddings.npy").exists()


def test_load_index_rejects_inconsistent_state(store, tmp_path):
    store.add_embeddings(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
        [{"content": "a"}, {"content": "b"}],
    )
    store.save_index()

    # Simulate a desync: metadata.pkl written by a different, older save
    # than index.faiss/embeddings.npy (fewer entries than vectors).
    metadata_file = tmp_path / "faiss" / "metadata.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump([{"content": "a"}], f)

    reloaded = FAISSVectorStore(dimension=4, index_path=str(tmp_path / "faiss"))
    assert reloaded.load_index() is False
    assert reloaded.get_size() == 0


def test_remove_by_document_id_deletes_only_matching_chunks(store):
    embeddings = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    meta = [
        {"content": "a", "document_id": "doc-1"},
        {"content": "b", "document_id": "doc-2"},
        {"content": "c", "document_id": "doc-1"},
    ]
    store.add_embeddings(embeddings, meta)

    removed = store.remove_by_document_id("doc-1")
    assert removed == 2
    assert store.get_size() == 1
    assert store.get_chunk_by_index(0)["content"] == "b"


def test_remove_by_document_id_no_match_returns_zero(store):
    embeddings = np.array([[1, 0, 0, 0]], dtype=np.float32)
    store.add_embeddings(embeddings, [{"content": "a", "document_id": "doc-1"}])

    removed = store.remove_by_document_id("nonexistent")
    assert removed == 0
    assert store.get_size() == 1


def test_remove_by_document_id_on_empty_index_returns_zero(store):
    assert store.remove_by_document_id("doc-1") == 0


def test_remove_by_document_id_persists_to_disk(store, tmp_path):
    embeddings = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    meta = [{"content": "keep", "document_id": "doc-2"}, {"content": "gone", "document_id": "doc-1"}]
    store.add_embeddings(embeddings, meta)
    store.save_index()

    store.remove_by_document_id("doc-1")

    reloaded = type(store)(dimension=4, index_path=store.index_path)
    assert reloaded.load_index() is True
    assert reloaded.get_size() == 1
    assert reloaded.get_chunk_by_index(0)["content"] == "keep"


# ── VectorStoreManager: document tagging, deletion, reset ────────────


@pytest.fixture()
def manager(mock_sentence_transformer, monkeypatch, tmp_path):
    from app.core.config import settings

    monkeypatch.setattr(settings, "faiss_index_path", str(tmp_path / "faiss"))
    return VectorStoreManager(embedding_model=mock_sentence_transformer)


def _document(chunk_count=2):
    chunks = [
        DocumentChunk(
            content=f"chunk {i}",
            metadata=ChunkMetadata(page_number=1, chunk_index=i, total_chunks=chunk_count, file_name="d.pdf"),
        )
        for i in range(chunk_count)
    ]
    return Document(
        file_name="d.pdf",
        file_type="pdf",
        file_size_bytes=100,
        total_chunks=chunk_count,
        chunks=chunks,
        upload_timestamp="2024-01-01T00:00:00",
    )


def test_add_document_tags_chunks_with_document_id(manager):
    manager.add_document(_document(), document_id="doc-42")
    assert manager.vector_store.get_size() == 2
    assert manager.vector_store.get_chunk_by_index(0)["document_id"] == "doc-42"


def test_remove_document_delegates_to_vector_store(manager):
    manager.add_document(_document(), document_id="doc-1")
    removed = manager.remove_document("doc-1")
    assert removed == 2
    assert manager.vector_store.get_size() == 0


def test_reset_clears_and_persists_across_reload(manager):
    manager.add_document(_document(), document_id="doc-1")
    assert manager.vector_store.get_size() == 2

    manager.reset()
    assert manager.vector_store.get_size() == 0

    reloaded = VectorStoreManager(embedding_model=manager.embedding_gen.model)
    assert reloaded.vector_store.get_size() == 0


# ── VectorStoreManager: memory (content_type discriminator) ──────────


def test_add_memory_tags_chunk_with_memory_content_type(manager):
    manager.add_memory("Discussed mitochondria last session.", session_id="sess-1", memory_id="mem-1")
    assert manager.vector_store.get_size() == 1
    chunk = manager.vector_store.get_chunk_by_index(0)
    assert chunk["metadata"]["content_type"] == "memory"
    assert chunk["document_id"] == "mem-1"


def test_search_defaults_to_documents_only(manager):
    manager.add_document(_document(chunk_count=1), document_id="doc-1")
    manager.add_memory("A prior conversation summary.", session_id="sess-1", memory_id="mem-1")

    results = manager.search("anything", top_k=5)
    content_types = {r[0]["metadata"]["content_type"] for r in results}
    assert content_types == {"document"}


def test_search_with_memory_content_types_returns_only_memories(manager):
    manager.add_document(_document(chunk_count=1), document_id="doc-1")
    manager.add_memory("A prior conversation summary.", session_id="sess-1", memory_id="mem-1")

    results = manager.search("anything", top_k=5, content_types=["memory"])
    assert len(results) == 1
    assert results[0][0]["metadata"]["content_type"] == "memory"


def test_search_on_empty_index_returns_empty(manager):
    assert manager.search("anything", top_k=5) == []


# ── Hybrid retrieval: BM25 lexical signal blended with dense similarity ──


def test_min_max_normalize_scales_to_unit_range():
    assert _min_max_normalize([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]


def test_min_max_normalize_all_equal_returns_zero_not_one():
    assert _min_max_normalize([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]


def test_min_max_normalize_empty_returns_empty():
    assert _min_max_normalize([]) == []


def test_rank_candidates_blends_bm25_scores_with_dense_similarity(manager, monkeypatch):
    # rank_bm25's classic IDF formula degenerates for tiny corpora (e.g. a
    # term in exactly 1-of-2 documents can score zero), so the blending math
    # itself is tested against a stubbed BM25 index rather than a real one
    # built from a two-chunk corpus.
    fake_bm25 = MagicMock()
    fake_bm25.get_scores.return_value = [10.0, 0.0]
    monkeypatch.setattr(manager, "_get_bm25_index", lambda: fake_bm25)

    chunk_a = {"content": "Mitochondria produce ATP through cellular respiration."}
    chunk_b = {"content": "Volcanoes erupt due to magma pressure buildup underground."}
    candidates = [(0, chunk_a, 0.5), (1, chunk_b, 0.5)]  # tied dense scores

    ranked = manager._rank_candidates("mitochondria cellular respiration ATP", candidates)

    assert ranked[0][0] is chunk_a
    assert ranked[0][1] > ranked[1][1]


def test_rank_candidates_falls_back_to_dense_order_without_bm25(manager, monkeypatch):
    monkeypatch.setattr(manager, "_get_bm25_index", lambda: None)
    chunk_a = {"content": "a"}
    chunk_b = {"content": "b"}
    candidates = [(0, chunk_a, 0.3), (1, chunk_b, 0.9)]

    ranked = manager._rank_candidates("anything", candidates)

    assert ranked == [(chunk_b, 0.9), (chunk_a, 0.3)]


def test_rank_candidates_empty_returns_empty(manager):
    assert manager._rank_candidates("anything", []) == []


def test_bm25_index_invalidated_on_add_document(manager):
    manager.add_document(_document(chunk_count=1), document_id="doc-1")
    assert manager._get_bm25_index() is not None
    assert manager._bm25_dirty is False

    manager.add_document(_document(chunk_count=1), document_id="doc-2")
    assert manager._bm25_dirty is True


def test_bm25_index_rebuilds_after_remove_document(manager):
    manager.add_document(_document(chunk_count=2), document_id="doc-1")
    manager._get_bm25_index()
    assert manager._bm25_dirty is False

    manager.remove_document("doc-1")
    assert manager._bm25_dirty is True
    assert manager._get_bm25_index() is None  # empty corpus after removal


def test_bm25_index_rebuilds_after_reset(manager):
    manager.add_document(_document(chunk_count=1), document_id="doc-1")
    manager._get_bm25_index()
    manager.reset()
    assert manager._bm25_dirty is True
    assert manager._get_bm25_index() is None
