"""Tests for adaptive/hybrid document retrieval."""

from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.services.models import QueryType
from app.services.retriever import AdaptiveRetriever, HybridRetriever


def _raw_result(content, page, chunk_index, similarity, rank):
    return (
        {
            "content": content,
            "metadata": {
                "page_number": page,
                "chunk_index": chunk_index,
                "total_chunks": 3,
                "file_name": "doc.pdf",
                "chunk_position": "beginning" if chunk_index == 0 else "middle",
            },
        },
        similarity,
        rank,
    )


def _memory_result(content, similarity, rank=1):
    return (
        {
            "content": content,
            "metadata": {
                "page_number": 0,
                "chunk_index": 0,
                "total_chunks": 1,
                "file_name": "memory:sess-1",
                "content_type": "memory",
            },
        },
        similarity,
        rank,
    )


@pytest.fixture()
def mock_vector_store():
    vs = MagicMock()
    vs.search.return_value = [
        _raw_result("First chunk", 1, 0, 0.9, 1),
        _raw_result("Second chunk", 2, 1, 0.6, 2),
    ]
    return vs


def test_retrieve_returns_chunks_and_in_scope(mock_vector_store):
    retriever = AdaptiveRetriever(mock_vector_store)
    chunks, in_scope = retriever.retrieve("What is X?", QueryType.DEFINITION)
    assert len(chunks) == 2
    assert in_scope is True
    assert chunks[0].relevance_score == 0.9


def test_retrieve_uses_intent_specific_top_k(mock_vector_store):
    retriever = AdaptiveRetriever(mock_vector_store)
    retriever.retrieve("Explain in detail", QueryType.EXPLAIN)
    called_top_k = mock_vector_store.search.call_args.kwargs.get(
        "top_k"
    ) or mock_vector_store.search.call_args[1].get("top_k")
    # EXPLAIN requests 8 top_k, doubled internally for post-filtering headroom
    assert called_top_k == 16


def test_retrieve_out_of_scope_below_threshold(mock_vector_store):
    mock_vector_store.search.return_value = [_raw_result("Barely related", 1, 0, 0.01, 1)]
    retriever = AdaptiveRetriever(mock_vector_store)
    chunks, in_scope = retriever.retrieve("unrelated query", QueryType.VAGUE)
    assert in_scope is False
    assert len(chunks) == 1  # still returned — scope check is separate from filtering


def test_retrieve_empty_results():
    vs = MagicMock()
    vs.search.return_value = []
    retriever = AdaptiveRetriever(vs)
    chunks, in_scope = retriever.retrieve("anything", QueryType.VAGUE)
    assert chunks == []
    assert in_scope is False


# ── document_ids scoping (prefer the conversation's own documents) ────


def test_retrieve_uses_scoped_document_ids_when_they_hit():
    vs = MagicMock()

    def fake_search(query, top_k=None, content_types=None, document_ids=None):
        if document_ids == ["doc-1"]:
            return [_raw_result("Scoped chunk", 1, 0, 0.9, 1)]
        return [_raw_result("Global chunk", 1, 0, 0.9, 1)]

    vs.search.side_effect = fake_search
    retriever = AdaptiveRetriever(vs)

    chunks, in_scope = retriever.retrieve("What is X?", QueryType.DEFINITION, document_ids=["doc-1"])

    assert in_scope is True
    assert chunks[0].content == "Scoped chunk"
    # The scoped search hit — no need to also search the full index.
    assert vs.search.call_count == 1


def test_retrieve_treats_weak_scoped_match_as_in_scope_without_falling_back():
    """A vague meta-question ("what's in this document?") often has weak raw
    embedding similarity to the document's own text even when it's exactly
    the right document — falling back to the full index in that case let a
    coincidentally higher-scoring, unrelated document win (the "confusing
    between different uploads" bug). An explicit document scope must be
    authoritative: never overridden by a below-relevance_threshold score."""
    vs = MagicMock()

    def fake_search(query, top_k=None, content_types=None, document_ids=None):
        if document_ids == ["doc-1"]:
            return [_raw_result("Weak scoped chunk", 1, 0, 0.01, 1)]
        return [_raw_result("Global chunk", 1, 0, 0.9, 1)]

    vs.search.side_effect = fake_search
    retriever = AdaptiveRetriever(vs)

    chunks, in_scope = retriever.retrieve("What is X?", QueryType.DEFINITION, document_ids=["doc-1"])

    assert in_scope is True
    assert chunks[0].content == "Weak scoped chunk"
    # Only the scoped search should have run — a non-empty scoped result
    # never falls through to the full index.
    assert vs.search.call_count == 1


def test_retrieve_missed_scoped_search_never_falls_back_to_full_index():
    """Per the conversation-isolation redesign, a document scope that misses
    must never silently search the full index — that was exactly how one
    conversation's documents used to leak into another's answers. It falls
    through only to this same conversation's memory (session-scoped), never
    to unrelated documents."""
    vs = MagicMock()

    def fake_search(query, top_k=None, content_types=None, document_ids=None, session_ids=None):
        if document_ids == ["doc-1"]:
            return []
        if content_types == ["memory"]:
            return [_memory_result("Recalled from this same conversation.", 0.9)]
        return [_raw_result("A different conversation's document", 1, 0, 0.9, 1)]

    vs.search.side_effect = fake_search
    retriever = AdaptiveRetriever(vs)

    chunks, in_scope = retriever.retrieve(
        "What is X?", QueryType.DEFINITION, document_ids=["doc-1"], session_id="sess-1"
    )

    assert in_scope is True
    assert chunks[0].content == "Recalled from this same conversation."
    # Scoped document search + session-scoped memory search only — the
    # full/unscoped index is never queried.
    assert vs.search.call_count == 2


def test_retrieve_empty_document_scope_returns_nothing_without_querying():
    """document_ids=[] (this conversation has zero documents) is distinct
    from document_ids=None (no scope requested) — it must search nothing,
    not fall back to a global search, and without a session_id there's no
    conversation memory to fall back to either."""
    vs = MagicMock()
    vs.search.return_value = [_raw_result("Should never be returned", 1, 0, 0.9, 1)]
    retriever = AdaptiveRetriever(vs)

    chunks, in_scope = retriever.retrieve("What is X?", QueryType.DEFINITION, document_ids=[])

    assert chunks == []
    assert in_scope is False
    vs.search.assert_not_called()


def test_retrieve_without_document_ids_only_searches_once(mock_vector_store):
    retriever = AdaptiveRetriever(mock_vector_store)
    retriever.retrieve("What is X?", QueryType.DEFINITION)
    assert mock_vector_store.search.call_count == 1


def test_rerank_by_intent_process_orders_by_page_then_chunk_index(mock_vector_store):
    retriever = AdaptiveRetriever(mock_vector_store)
    chunks, _ = retriever.retrieve("steps", QueryType.PROCESS)
    reranked = AdaptiveRetriever.rerank_by_intent(list(reversed(chunks)), QueryType.PROCESS)
    assert reranked[0].metadata.page_number <= reranked[-1].metadata.page_number


def test_rerank_by_intent_noop_for_unhandled_types(mock_vector_store):
    retriever = AdaptiveRetriever(mock_vector_store)
    chunks, _ = retriever.retrieve("example", QueryType.EXAMPLE)
    reranked = AdaptiveRetriever.rerank_by_intent(chunks, QueryType.EXAMPLE)
    assert reranked == chunks


def test_hybrid_retriever_search_returns_full_result_dict(mock_vector_store):
    hybrid = HybridRetriever(mock_vector_store)
    result = hybrid.search("What is X?", QueryType.DEFINITION)
    assert result["query"] == "What is X?"
    assert result["intent"] == QueryType.DEFINITION
    assert result["is_relevant"] is True
    assert result["relevance_score"] == 0.9
    assert result["total_retrieved"] == 2


def test_hybrid_retriever_not_relevant_below_threshold():
    vs = MagicMock()
    vs.search.return_value = [_raw_result("weak match", 1, 0, 0.01, 1)]
    hybrid = HybridRetriever(vs)
    result = hybrid.search("unrelated", QueryType.VAGUE)
    assert result["is_relevant"] is False


# ── Cross-encoder reranking (opt-in, off by default) ──────────────────


def test_cross_encoder_rerank_disabled_by_default(mock_vector_store):
    mock_cross_encoder = MagicMock()
    retriever = AdaptiveRetriever(mock_vector_store, cross_encoder=mock_cross_encoder)
    retriever.retrieve("What is X?", QueryType.DEFINITION)
    mock_cross_encoder.predict.assert_not_called()


def test_cross_encoder_rerank_reorders_when_enabled(mock_vector_store, monkeypatch):
    monkeypatch.setattr(settings, "enable_cross_encoder_rerank", True)
    mock_cross_encoder = MagicMock()
    # "First chunk" is dense-favored (0.9 vs 0.6) but the cross-encoder
    # scores it lower, so the rerank should flip their order.
    mock_cross_encoder.predict.return_value = [0.1, 0.9]
    retriever = AdaptiveRetriever(mock_vector_store, cross_encoder=mock_cross_encoder)

    chunks, _ = retriever.retrieve("What is X?", QueryType.DEFINITION)

    assert [c.content for c in chunks] == ["Second chunk", "First chunk"]


def test_cross_encoder_rerank_falls_back_on_failure(mock_vector_store, monkeypatch):
    monkeypatch.setattr(settings, "enable_cross_encoder_rerank", True)
    mock_cross_encoder = MagicMock()
    mock_cross_encoder.predict.side_effect = RuntimeError("model error")
    retriever = AdaptiveRetriever(mock_vector_store, cross_encoder=mock_cross_encoder)

    chunks, _ = retriever.retrieve("What is X?", QueryType.DEFINITION)

    assert [c.content for c in chunks] == ["First chunk", "Second chunk"]


def test_cross_encoder_rerank_skipped_for_single_chunk(mock_vector_store, monkeypatch):
    monkeypatch.setattr(settings, "enable_cross_encoder_rerank", True)
    mock_vector_store.search.return_value = [_raw_result("Only chunk", 1, 0, 0.9, 1)]
    mock_cross_encoder = MagicMock()
    retriever = AdaptiveRetriever(mock_vector_store, cross_encoder=mock_cross_encoder)

    retriever.retrieve("What is X?", QueryType.DEFINITION)

    mock_cross_encoder.predict.assert_not_called()


# ── Semantic memory fallback (only when document retrieval misses) ───


def test_memory_fallback_used_when_document_retrieval_out_of_scope():
    vs = MagicMock()

    def fake_search(query, top_k=None, content_types=None, session_ids=None):
        if content_types == ["memory"]:
            return [_memory_result("We discussed photosynthesis last time.", 0.8)]
        return [_raw_result("Barely related", 1, 0, 0.01, 1)]

    vs.search.side_effect = fake_search
    retriever = AdaptiveRetriever(vs)

    chunks, in_scope = retriever.retrieve("what did we discuss?", QueryType.VAGUE, session_id="sess-1")

    assert in_scope is True
    assert chunks[0].content == "We discussed photosynthesis last time."


def test_memory_fallback_not_used_without_session_id():
    """No session_id means no conversation to scope memory by — searching
    every conversation's summaries would be exactly the cross-session leak
    this scoping exists to prevent, so the fallback is skipped entirely
    rather than guessing."""
    vs = MagicMock()

    def fake_search(query, top_k=None, content_types=None, session_ids=None):
        if content_types == ["memory"]:
            return [_memory_result("We discussed photosynthesis last time.", 0.8)]
        return [_raw_result("Barely related", 1, 0, 0.01, 1)]

    vs.search.side_effect = fake_search
    retriever = AdaptiveRetriever(vs)

    chunks, in_scope = retriever.retrieve("what did we discuss?", QueryType.VAGUE)

    assert in_scope is False
    assert chunks[0].content == "Barely related"


def test_memory_fallback_not_used_when_document_retrieval_in_scope(mock_vector_store):
    retriever = AdaptiveRetriever(mock_vector_store)

    chunks, in_scope = retriever.retrieve("What is X?", QueryType.DEFINITION)

    assert in_scope is True
    assert chunks[0].content == "First chunk"
    # Only the document search should have run — memory fallback never
    # triggers once document retrieval already scored in scope.
    assert mock_vector_store.search.call_count == 1


def test_memory_fallback_below_threshold_stays_out_of_scope():
    vs = MagicMock()

    def fake_search(query, top_k=None, content_types=None, session_ids=None):
        if content_types == ["memory"]:
            return [_memory_result("Weak memory match.", 0.1)]
        return [_raw_result("Barely related", 1, 0, 0.01, 1)]

    vs.search.side_effect = fake_search
    retriever = AdaptiveRetriever(vs)

    chunks, in_scope = retriever.retrieve("unrelated", QueryType.VAGUE, session_id="sess-1")

    assert in_scope is False
    assert chunks[0].content == "Barely related"


def test_memory_fallback_when_document_retrieval_returns_no_results():
    vs = MagicMock()

    def fake_search(query, top_k=None, content_types=None, session_ids=None):
        if content_types == ["memory"]:
            return [_memory_result("Recalled from a past session.", 0.9)]
        return []

    vs.search.side_effect = fake_search
    retriever = AdaptiveRetriever(vs)

    chunks, in_scope = retriever.retrieve("anything", QueryType.VAGUE, session_id="sess-1")

    assert in_scope is True
    assert chunks[0].content == "Recalled from a past session."


def test_hybrid_retriever_is_relevant_reflects_memory_fallback():
    vs = MagicMock()

    def fake_search(query, top_k=None, content_types=None, session_ids=None):
        if content_types == ["memory"]:
            return [_memory_result("We discussed photosynthesis last time.", 0.8)]
        return [_raw_result("Barely related", 1, 0, 0.01, 1)]

    vs.search.side_effect = fake_search
    hybrid = HybridRetriever(vs)

    result = hybrid.search("what did we discuss?", QueryType.VAGUE, session_id="sess-1")

    assert result["is_relevant"] is True
    assert result["in_scope"] is True
