"""Tests for the deterministic RAG-vs-CAG router.

No LLM/embeddings involved — ContextRouter is a pure size comparison over an
already-fetched chunk list, so these tests need no mocking beyond monkeypatching
the settings that define the decision threshold.
"""

import pytest

from app.core.config import settings
from app.services.context_router import ContextMode, ContextRouter
from app.services.models import ChunkMetadata, RetrievedChunk


def _chunk(content, chunk_index=0):
    meta = ChunkMetadata(page_number=1, chunk_index=chunk_index, total_chunks=1, file_name="doc.pdf")
    return RetrievedChunk(content=content, metadata=meta, relevance_score=1.0, rank=chunk_index)


@pytest.fixture()
def router():
    return ContextRouter()


def test_small_scope_chooses_cag(router, monkeypatch):
    monkeypatch.setattr(settings, "enable_cag", True)
    monkeypatch.setattr(settings, "cag_token_budget", 100)
    chunks = [_chunk("A short chunk of text."), _chunk("Another short chunk.")]
    assert router.decide(chunks) == ContextMode.CAG


def test_oversized_scope_falls_back_to_rag(router, monkeypatch):
    monkeypatch.setattr(settings, "enable_cag", True)
    monkeypatch.setattr(settings, "cag_token_budget", 5)
    chunks = [_chunk("This chunk alone has more than five words in it.")]
    assert router.decide(chunks) == ContextMode.RAG


def test_exactly_at_budget_chooses_cag(router, monkeypatch):
    """The boundary is inclusive (<=) — a scope that exactly fits the budget
    should still get the CAG win, not be pushed to RAG by an off-by-one."""
    monkeypatch.setattr(settings, "enable_cag", True)
    monkeypatch.setattr(settings, "cag_token_budget", 4)
    chunks = [_chunk("one two three four")]  # exactly 4 words
    assert router.decide(chunks) == ContextMode.CAG


def test_no_chunks_falls_back_to_rag(router, monkeypatch):
    monkeypatch.setattr(settings, "enable_cag", True)
    monkeypatch.setattr(settings, "cag_token_budget", 6000)
    assert router.decide([]) == ContextMode.RAG


def test_disabled_setting_always_falls_back_to_rag(router, monkeypatch):
    """Even a tiny, well-within-budget scope must stay on RAG when the
    feature is turned off — the setting is a hard override, not a hint."""
    monkeypatch.setattr(settings, "enable_cag", False)
    monkeypatch.setattr(settings, "cag_token_budget", 6000)
    chunks = [_chunk("tiny")]
    assert router.decide(chunks) == ContextMode.RAG


def test_multi_document_scope_sums_across_all_chunks(router, monkeypatch):
    """The budget applies to the whole resolved scope, not per document —
    two small documents that individually fit can still combine to exceed
    the budget together."""
    monkeypatch.setattr(settings, "enable_cag", True)
    monkeypatch.setattr(settings, "cag_token_budget", 6)
    chunks = [_chunk("one two three"), _chunk("four five six seven")]  # 3 + 4 = 7 words total
    assert router.decide(chunks) == ContextMode.RAG
