"""Span extraction, confidence scoring, and citation formatting."""

import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple, Union

from loguru import logger

from .models import RetrievedChunk, SourceCitation

_SPAN_WINDOW_SIZES = (1, 2, 3)


class SpanExtractor:
    """Extract exact supporting text spans from retrieved chunks."""

    def __init__(self, min_similarity: float = 0.7):
        self.min_similarity = min_similarity

    def extract_supporting_span(
        self, claim: Union[str, dict], chunks: List[RetrievedChunk]
    ) -> Optional[SourceCitation]:
        claim_text, candidate_chunks = self._resolve_claim(claim, chunks)
        best_match = None
        best_sim = 0.0

        for chunk in candidate_chunks:
            span, sim = self._find_span_in_text(claim_text, chunk.content)
            if sim > best_sim:
                best_sim = sim
                best_match = (chunk, span, sim)

        if best_match and best_sim >= self.min_similarity:
            chunk, span, sim = best_match
            return SourceCitation(
                page_number=chunk.metadata.page_number,
                section_title=chunk.metadata.section_title,
                quoted_text=span,
                confidence=min(0.95, sim),
                relevance_score=chunk.relevance_score,
            )
        return None

    @staticmethod
    def _resolve_claim(
        claim: Union[str, dict], chunks: List[RetrievedChunk]
    ) -> Tuple[str, List[RetrievedChunk]]:
        """Accept either a plain claim string (legacy shape) or the
        chunk-indexed dict produced by ``_BaseLLM.extract_claims``
        (``{"claim": str, "chunks": [1, 3]}``, 1-based indices matching the
        ``[1]``/``[3]`` labels in ``_format_context``). When indices are
        present, narrow the search to those chunks instead of fuzzy-matching
        blindly across every retrieved chunk."""
        if isinstance(claim, dict):
            text = claim.get("claim", "")
            indices = claim.get("chunks") or []
            narrowed = [chunks[i - 1] for i in indices if isinstance(i, int) and 0 < i <= len(chunks)]
            return text, (narrowed or chunks)
        return claim, chunks

    def _find_span_in_text(self, claim: str, text: str) -> Tuple[str, float]:
        sentences = _split_sentences(text)
        best_span, best_ratio = "", 0.0

        # Try windows of 1, 2, and 3 consecutive sentences — a claim often
        # paraphrases a short run of sentences rather than exactly one.
        for window_size in _SPAN_WINDOW_SIZES:
            for i in range(len(sentences) - window_size + 1):
                combined = " ".join(sentences[i : i + window_size])
                ratio = _similarity(claim, combined)
                if ratio > best_ratio:
                    best_ratio, best_span = ratio, combined

        best_span = best_span.strip()
        if len(best_span) > 200:
            best_span = best_span[:200] + "..."
        return best_span, best_ratio


class ConfidenceScorer:
    """Calculate confidence scores for answers."""

    def calculate_answer_confidence(
        self,
        retrieved_chunks: List[RetrievedChunk],
        citations: List[SourceCitation],
        num_claims: int,
    ) -> float:
        if not retrieved_chunks:
            return 0.0

        avg_relevance = sum(c.relevance_score for c in retrieved_chunks) / len(retrieved_chunks)
        citation_rate = min(1.0, len(citations) / max(1, num_claims)) if num_claims > 0 else 0.5

        # A thorough answer legitimately draws on several pages — that's
        # breadth, not a sign of poor grounding, so a wide source spread
        # should cost a little, not most, of this term (was 0.2/extra page,
        # e.g. 10 pages -> 0.36; now -> 0.53).
        pages = [c.metadata.page_number for c in retrieved_chunks]
        source_consistency = 1.0 / (1.0 + (len(set(pages)) - 1) * 0.1)

        return min(
            1.0,
            avg_relevance * 0.4 + citation_rate * 0.4 + source_consistency * 0.2,
        )

    def assess_hallucination_risk(
        self,
        retrieved_chunks: List[RetrievedChunk],
        citations: List[SourceCitation],
        num_claims: int,
    ) -> str:
        # Thresholds calibrated against calculate_answer_confidence's actual
        # achievable range: it's a weighted blend of three sub-1.0 terms, so
        # even a well-grounded answer rarely clears ~0.75-0.8 overall (that
        # would need avg_relevance, citation_rate, *and* source_consistency
        # all near-perfect simultaneously) — the original 0.85/0.6 cutoffs
        # were calibrated as if confidence commonly approached 1.0, which
        # pushed nearly everything into "high" regardless of actual quality.
        confidence = self.calculate_answer_confidence(retrieved_chunks, citations, num_claims)
        citation_rate = len(citations) / max(1, num_claims) if num_claims > 0 else 0.0
        if confidence > 0.7 and citation_rate > 0.7:
            return "low"
        if confidence > 0.45 and citation_rate > 0.4:
            return "medium"
        return "high"


class CitationFormatter:
    @staticmethod
    def format_citations(citations: List[SourceCitation]) -> str:
        if not citations:
            return "No sources cited."
        lines = ["**Sources:**"]
        for i, c in enumerate(citations, 1):
            section = c.section_title or "Text"
            lines.append(f'{i}. Page {c.page_number} ({section}): "{c.quoted_text}"')
        return "\n".join(lines)


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _token_overlap(a: str, b: str) -> float:
    """Word-level Dice coefficient — robust to clause reordering and minor
    rewording between a claim and its source span, which raw character-
    sequence matching (below) is not. A claim that restructures a sentence
    while keeping the same key terms/numbers still overlaps heavily here
    even when its character sequence looks quite different."""
    words_a = set(re.findall(r"\w+", a.lower()))
    words_b = set(re.findall(r"\w+", b.lower()))
    if not words_a or not words_b:
        return 0.0
    return 2 * len(words_a & words_b) / (len(words_a) + len(words_b))


def _similarity(a: str, b: str) -> float:
    """Best of two complementary measures: character-sequence ratio (best
    for near-verbatim quotes) and word-overlap (best for a paraphrase that
    reorders or lightly rewords the same source content). Using only the
    former was flagging plenty of genuinely-grounded, merely-paraphrased
    claims as unsupported — extracted claims are the drafting LLM's own
    summary of the source, never a verbatim quote, so requiring high
    character-sequence similarity was the wrong bar for what this is
    actually checking."""
    return max(SequenceMatcher(None, a.lower(), b.lower()).ratio(), _token_overlap(a, b))
