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

    def test_reordered_claim_matches_via_token_overlap(self):
        """Extracted claims are the drafting LLM's own paraphrase of the
        source, not a verbatim quote — a claim that restructures the same
        clauses in a different order has low character-sequence similarity
        (SequenceMatcher alone: ~0.44 here) but shares nearly all the same
        words, so it must still be recognized as grounded rather than
        flagged as an unsupported/hallucinated claim."""
        extractor = SpanExtractor(min_similarity=0.65)
        chunks = [
            _make_chunk(
                "Water splits into oxygen and hydrogen during electrolysis, "
                "releasing energy in the process."
            )
        ]
        citation = extractor.extract_supporting_span(
            "During electrolysis, energy is released as water splits into oxygen and hydrogen.",
            chunks,
        )
        assert citation is not None


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

    def test_zero_claims_not_automatically_high_risk(self):
        """assess_hallucination_risk's citation_rate default for num_claims=0
        must agree with calculate_answer_confidence's (0.5, neutral) — it
        used to default to 0.0 here specifically, which silently forced
        every zero-claim answer (nothing to fact-check, not necessarily
        ungrounded) into "high" risk regardless of chunk relevance."""
        scorer = ConfidenceScorer()
        chunks = [_make_chunk("text", relevance=0.9)]
        risk = scorer.assess_hallucination_risk(chunks, [], 0)
        assert risk != "high"

    def test_multi_page_answer_not_harshly_penalized(self):
        """Drawing on several distinct pages is breadth, not a sign of poor
        grounding — a well-cited answer spanning 5 pages should still land
        solidly in the "medium" band, not get crushed into "high" purely for
        being comprehensive."""
        scorer = ConfidenceScorer()
        chunks = [_make_chunk("text", page=p, relevance=0.7) for p in range(1, 6)]
        citations = [
            SourceCitation(page_number=p, quoted_text="q", confidence=0.8, relevance_score=0.7)
            for p in range(1, 6)
        ]
        conf = scorer.calculate_answer_confidence(chunks, citations, 5)
        assert conf > 0.6
        assert scorer.assess_hallucination_risk(chunks, citations, 5) in ("low", "medium")

    def test_hallucination_risk_thresholds_reachable_by_solid_answer(self):
        """A realistically-good (not perfect) answer — decent relevance,
        most-but-not-all claims cited, a couple of source pages — should
        clear "medium," not fall into "high" by default the way the original
        0.6-confidence/0.5-citation-rate cutoffs effectively did."""
        scorer = ConfidenceScorer()
        chunks = [_make_chunk("text", page=1, relevance=0.6), _make_chunk("text", page=2, relevance=0.6)]
        citations = [
            SourceCitation(page_number=1, quoted_text="q", confidence=0.8, relevance_score=0.6),
            SourceCitation(page_number=2, quoted_text="q", confidence=0.8, relevance_score=0.6),
        ]
        risk = scorer.assess_hallucination_risk(chunks, citations, 3)  # 2/3 claims cited
        assert risk == "medium"
