"""Span extraction and citation formatting."""

from typing import List, Tuple, Optional
from difflib import SequenceMatcher
from loguru import logger
from .models import RetrievedChunk, SourceCitation
import re


class SpanExtractor:
    """Extract exact supporting text spans from retrieved chunks."""
    
    def __init__(self, min_similarity: float = 0.7):
        """Initialize span extractor."""
        self.min_similarity = min_similarity
    
    def extract_supporting_span(
        self,
        claim: str,
        chunks: List[RetrievedChunk]
    ) -> Optional[SourceCitation]:
        """
        Find exact supporting text span for a claim.
        
        Args:
            claim: Factual claim to verify
            chunks: Retrieved chunks
            
        Returns:
            SourceCitation with quoted text or None
        """
        logger.debug(f"Extracting span for claim: {claim}")
        
        best_match = None
        best_similarity = 0
        
        for chunk in chunks:
            # Try to find span in chunk
            span, similarity = self._find_span_in_text(claim, chunk.content)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (chunk, span, similarity)
        
        if best_match and best_similarity >= self.min_similarity:
            chunk, span, similarity = best_match
            
            logger.debug(f"Found supporting span: {span[:50]}... (similarity: {similarity:.2f})")
            
            return SourceCitation(
                page_number=chunk.metadata.page_number,
                section_title=chunk.metadata.section_title,
                quoted_text=span,
                confidence=min(0.95, similarity),  # Cap at 0.95
                relevance_score=chunk.relevance_score
            )
        
        logger.debug(f"No supporting span found (best similarity: {best_similarity:.2f})")
        return None
    
    def _find_span_in_text(self, claim: str, text: str) -> Tuple[str, float]:
        """
        Find best matching span in text for claim.
        
        Returns:
            (span_text, similarity_score)
        """
        # Split text into sentences
        sentences = self._split_sentences(text)
        
        best_span = ""
        best_ratio = 0
        
        # Try sentences
        for sentence in sentences:
            ratio = self._similarity(claim, sentence)
            if ratio > best_ratio:
                best_ratio = ratio
                best_span = sentence
        
        # Try sentence combinations (for multi-sentence claims)
        if len(sentences) > 1:
            for i in range(len(sentences) - 1):
                combined = sentences[i] + " " + sentences[i + 1]
                ratio = self._similarity(claim, combined)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_span = combined
        
        # Clean span
        best_span = best_span.strip()
        if len(best_span) > 200:
            best_span = best_span[:200] + "..."
        
        return best_span, best_ratio
    
    def _similarity(self, a: str, b: str) -> float:
        """Calculate similarity between two strings (0-1)."""
        matcher = SequenceMatcher(None, a.lower(), b.lower())
        return matcher.ratio()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple regex-based sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences


class ConfidenceScorer:
    """Calculate confidence scores for answers."""
    
    def __init__(self):
        """Initialize confidence scorer."""
        self.logger = logger
    
    def calculate_answer_confidence(
        self,
        retrieved_chunks: List[RetrievedChunk],
        citations: List[SourceCitation],
        num_claims: int
    ) -> float:
        """
        Calculate overall confidence for answer.
        
        Factors:
        - Retrieval quality (top chunk relevance)
        - Citation quality (how many claims are cited)
        - Source agreement (all sources consistent?)
        
        Returns:
            Confidence score 0-1
        """
        if not retrieved_chunks:
            return 0.0
        
        scores = []
        
        # Factor 1: Retrieval quality (40%)
        avg_relevance = sum(c.relevance_score for c in retrieved_chunks) / len(retrieved_chunks)
        scores.append(('retrieval', avg_relevance, 0.4))
        
        # Factor 2: Citation quality (40%)
        citation_rate = len(citations) / max(1, num_claims) if num_claims > 0 else 0.5
        citation_rate = min(1.0, citation_rate)
        scores.append(('citation', citation_rate, 0.4))
        
        # Factor 3: Source concentration (20%)
        # If all chunks are from same page, they're likely consistent
        if len(retrieved_chunks) > 0:
            pages = [c.metadata.page_number for c in retrieved_chunks]
            unique_pages = len(set(pages))
            source_consistency = 1.0 / (1.0 + (unique_pages - 1) * 0.2)
            scores.append(('source_consistency', source_consistency, 0.2))
        
        # Weighted average
        total_confidence = sum(score * weight for _, score, weight in scores)
        
        self.logger.debug(f"Confidence calculation: {scores}, total: {total_confidence:.2f}")
        
        return min(1.0, total_confidence)
    
    def assess_hallucination_risk(
        self,
        retrieved_chunks: List[RetrievedChunk],
        citations: List[SourceCitation],
        num_claims: int
    ) -> str:
        """
        Assess risk of hallucination.
        
        Returns:
            "low", "medium", or "high"
        """
        confidence = self.calculate_answer_confidence(retrieved_chunks, citations, num_claims)
        citation_rate = len(citations) / max(1, num_claims) if num_claims > 0 else 0.0
        
        if confidence > 0.85 and citation_rate > 0.8:
            return "low"
        elif confidence > 0.6 and citation_rate > 0.5:
            return "medium"
        else:
            return "high"


class CitationFormatter:
    """Format citations for display."""
    
    @staticmethod
    def format_citation(citation: SourceCitation) -> str:
        """Format single citation."""
        page = citation.page_number
        section = citation.section_title or "Text"
        quote = citation.quoted_text
        
        return f"Page {page} ({section}): \"{quote}\""
    
    @staticmethod
    def format_citations(citations: List[SourceCitation]) -> str:
        """Format multiple citations."""
        if not citations:
            return "No sources cited."
        
        formatted = ["**Sources:**"]
        for i, citation in enumerate(citations, 1):
            formatted.append(f"{i}. {CitationFormatter.format_citation(citation)}")
        
        return "\n".join(formatted)
    
    @staticmethod
    def format_as_json(citations: List[SourceCitation]) -> dict:
        """Format citations as structured JSON."""
        return {
            'total_citations': len(citations),
            'sources': [
                {
                    'page': c.page_number,
                    'section': c.section_title,
                    'quote': c.quoted_text,
                    'confidence': c.confidence,
                    'relevance': c.relevance_score
                }
                for c in citations
            ]
        }
