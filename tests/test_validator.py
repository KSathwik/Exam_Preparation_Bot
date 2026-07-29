"""Tests for span extraction and confidence scoring."""

import pytest

from app.services.models import ChunkMetadata, RetrievedChunk, SourceCitation
from app.services.validator import ConfidenceScorer, SpanExtractor


def _make_chunk(content: str, page: int = 1, relevance: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        metadata=ChunkMetadata(page_number=page, chunk_index=0, total_chunks=1, file_name="test.pdf"),
        relevance_score=relevance,
        rank=1,
    )


class TestSpanExtractor:
    def test_exact_match_found(self):
        extractor = SpanExtractor(min_similarity=0.5)
        chunks = [_make_chunk("Photosynthesis converts CO2 to glucose using sunlight.")]
        citation = extractor.extract_supporting_span("Photosynthesis converts CO2 to glucose", chunks)
        assert citation is not None
        assert citation.page_number == 1

    def test_no_match_below_threshold(self):
        extractor = SpanExtractor(min_similarity=0.99)
        chunks = [_make_chunk("Completely unrelated text about volcanoes.")]
        citation = extractor.extract_supporting_span("Photosynthesis produces oxygen", chunks)
        assert citation is None

    def test_span_truncated_when_long(self):
        extractor = SpanExtractor(min_similarity=0.3)
        long_text = "A " * 200 + "important claim here."
        chunks = [_make_chunk(long_text)]
        citation = extractor.extract_supporting_span("important claim", chunks)
        if citation:
            assert len(citation.quoted_text) <= 203  # 200 + "..."

    def test_matches_span_across_three_consecutive_sentences(self):
        extractor = SpanExtractor(min_similarity=0.5)
        chunks = [
            _make_chunk(
                "Photosynthesis begins in the chloroplast. Light energy splits water molecules. "
                "The released electrons power glucose synthesis."
            )
        ]
        citation = extractor.extract_supporting_span(
            "Photosynthesis begins in the chloroplast, light energy splits water, "
            "and the released electrons power glucose synthesis",
            chunks,
        )
        assert citation is not None
        assert "chloroplast" in citation.quoted_text
        assert "glucose synthesis" in citation.quoted_text

    def test_chunk_indexed_claim_narrows_search_to_indicated_chunk(self):
        extractor = SpanExtractor(min_similarity=0.5)
        chunks = [
            _make_chunk("Photosynthesis converts CO2 to glucose using sunlight.", page=1),
            _make_chunk("Completely unrelated text about volcanoes.", page=2),
        ]
        claim = {"claim": "Photosynthesis converts CO2 to glucose", "chunks": [1]}
        citation = extractor.extract_supporting_span(claim, chunks)
        assert citation is not None
        assert citation.page_number == 1

    def test_chunk_indexed_claim_falls_back_to_all_chunks_when_indices_invalid(self):
        extractor = SpanExtractor(min_similarity=0.5)
        chunks = [_make_chunk("Photosynthesis converts CO2 to glucose using sunlight.", page=1)]
        claim = {"claim": "Photosynthesis converts CO2 to glucose", "chunks": [99]}
        citation = extractor.extract_supporting_span(claim, chunks)
        assert citation is not None
        assert citation.page_number == 1

    def test_chunk_indexed_claim_with_no_indices_searches_all_chunks(self):
        extractor = SpanExtractor(min_similarity=0.5)
        chunks = [_make_chunk("Photosynthesis converts CO2 to glucose using sunlight.", page=1)]
        claim = {"claim": "Photosynthesis converts CO2 to glucose", "chunks": []}
        citation = extractor.extract_supporting_span(claim, chunks)
        assert citation is not None
        assert citation.page_number == 1


class TestConfidenceScorer:
    def test_empty_chunks_returns_zero(self):
        scorer = ConfidenceScorer()
        assert scorer.calculate_answer_confidence([], [], 0) == 0.0

    def test_high_confidence(self):
        scorer = ConfidenceScorer()
        chunks = [_make_chunk("text", relevance=0.95)]
        citations = [SourceCitation(page_number=1, quoted_text="q", confidence=0.9, relevance_score=0.9)]
        conf = scorer.calculate_answer_confidence(chunks, citations, 1)
        assert conf > 0.7

    def test_hallucination_risk_low(self):
        scorer = ConfidenceScorer()
        chunks = [_make_chunk("text", relevance=0.95)]
        citations = [SourceCitation(page_number=1, quoted_text="q", confidence=0.9, relevance_score=0.9)]
        risk = scorer.assess_hallucination_risk(chunks, citations, 1)
        assert risk in ("low", "medium")

    def test_hallucination_risk_high_no_citations(self):
        scorer = ConfidenceScorer()
        chunks = [_make_chunk("text", relevance=0.2)]
        risk = scorer.assess_hallucination_risk(chunks, [], 5)
        assert risk == "high"
