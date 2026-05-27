"""Document retrieval and ranking module."""

from typing import List, Tuple, Optional
from .models import QueryType, RetrievedChunk, ChunkMetadata
from .embeddings import VectorStoreManager
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import numpy as np
from config.settings import settings


class AdaptiveRetriever:
    """Adaptive retriever that adjusts strategy based on query intent."""
    
    def __init__(self):
        """Initialize retriever."""
        self.vector_store = VectorStoreManager()
        self.relevance_threshold = settings.relevance_threshold
        self.min_relevance_score = settings.min_relevance_score
    
    def retrieve(
        self,
        query: str,
        intent: QueryType,
        top_k: Optional[int] = None
    ) -> Tuple[List[RetrievedChunk], bool]:
        """
        Retrieve relevant chunks based on query and intent.
        
        Args:
            query: User query
            intent: Classified query intent
            top_k: Number of results (adjusted by intent)
            
        Returns:
            (retrieved_chunks, in_scope) - list of chunks and scope flag
        """
        # Adjust k based on intent
        if top_k is None:
            top_k = self._get_top_k_for_intent(intent)
        
        logger.debug(f"Retrieving top {top_k} chunks for intent: {intent}")
        
        # Search vector store
        raw_results = self.vector_store.search(query, top_k=top_k * 2)  # Get more, then filter
        
        if not raw_results:
            logger.warning("No results from vector store")
            return [], False
        
        # Convert to RetrievedChunk objects
        chunks = []
        for chunk_info, similarity, rank in raw_results[:top_k]:
            # Create metadata
            metadata_dict = chunk_info.get('metadata', {})
            metadata = ChunkMetadata(**metadata_dict)
            
            chunk = RetrievedChunk(
                content=chunk_info.get('content', ''),
                metadata=metadata,
                relevance_score=similarity,
                rank=rank
            )
            chunks.append(chunk)
        
        # Check if in scope
        in_scope = self._is_in_scope(chunks)
        
        logger.debug(f"Retrieved {len(chunks)} chunks, in_scope: {in_scope}")
        
        return chunks, in_scope
    
    def _get_top_k_for_intent(self, intent: QueryType) -> int:
        """Get recommended top_k based on intent."""
        strategy = {
            QueryType.DEFINITION: 3,      # Need only 1-2 definitions
            QueryType.EXPLAIN: 8,         # Need comprehensive coverage
            QueryType.COMPARE: 8,         # Need both entities
            QueryType.PROCESS: 6,         # Need sequential steps
            QueryType.EXAMPLE: 4,         # Need 1-2 examples
            QueryType.DIAGRAM: 4,         # Need description + context
            QueryType.VAGUE: 10,          # Cast wider net
            QueryType.HOMEWORK: 5,        # Standard retrieval
        }
        return strategy.get(intent, settings.retrieval_top_k)
    
    def _is_in_scope(self, chunks: List[RetrievedChunk]) -> bool:
        """Determine if query is in-scope based on top relevance score."""
        if not chunks:
            return False
        
        top_score = chunks[0].relevance_score
        
        # Query is in scope if top relevance > threshold
        in_scope = top_score >= self.relevance_threshold
        
        logger.debug(f"In-scope check: top_score={top_score:.3f}, threshold={self.relevance_threshold}, in_scope={in_scope}")
        
        return in_scope
    
    def rerank_by_intent(
        self,
        chunks: List[RetrievedChunk],
        intent: QueryType
    ) -> List[RetrievedChunk]:
        """
        Rerank chunks based on intent-specific signals.
        
        For example:
        - DEFINITION: Rank chunks that contain the term definition early
        - PROCESS: Rank chunks in order
        - COMPARE: Keep both entities
        """
        if not chunks:
            return chunks
        
        if intent == QueryType.PROCESS:
            # Sort by page/position to maintain order
            return sorted(chunks, key=lambda x: (x.metadata.page_number, x.metadata.chunk_index))
        
        elif intent == QueryType.DEFINITION:
            # Prefer chunks with lower position in document (definitions early)
            return sorted(chunks, key=lambda x: x.metadata.chunk_position != "beginning")
        
        # Default: keep original ranking
        return chunks


class RelevanceChecker:
    """Check if query is relevant to document."""
    
    def __init__(self):
        """Initialize relevance checker."""
        self.threshold = settings.relevance_threshold
    
    def check_relevance(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk]
    ) -> Tuple[bool, float]:
        """
        Check if query is relevant to retrieved chunks.
        
        Returns:
            (is_relevant, score)
        """
        if not retrieved_chunks:
            return False, 0.0
        
        # Use top chunk relevance score
        top_score = retrieved_chunks[0].relevance_score
        is_relevant = top_score >= self.threshold
        
        logger.debug(f"Relevance check: score={top_score:.3f}, relevant={is_relevant}")
        
        return is_relevant, top_score


class HybridRetriever:
    """Hybrid retriever combining semantic and keyword search."""
    
    def __init__(self):
        """Initialize hybrid retriever."""
        self.adaptive_retriever = AdaptiveRetriever()
        self.relevance_checker = RelevanceChecker()
    
    def search(
        self,
        query: str,
        intent: QueryType,
        top_k: Optional[int] = None
    ) -> dict:
        """
        Full search pipeline.
        
        Returns:
            {
                'query': str,
                'intent': QueryType,
                'chunks': List[RetrievedChunk],
                'in_scope': bool,
                'relevance_score': float,
                'total_retrieved': int
            }
        """
        logger.info(f"Hybrid search for query: {query}")
        
        # Retrieve chunks
        chunks, in_scope = self.adaptive_retriever.retrieve(query, intent, top_k)
        
        # Check relevance
        is_relevant, relevance_score = self.relevance_checker.check_relevance(query, chunks)
        
        # Rerank by intent
        chunks = self.adaptive_retriever.rerank_by_intent(chunks, intent)
        
        return {
            'query': query,
            'intent': intent,
            'chunks': chunks,
            'in_scope': in_scope,
            'is_relevant': is_relevant,
            'relevance_score': relevance_score,
            'total_retrieved': len(chunks)
        }


def retrieve_for_query(
    query: str,
    intent: QueryType,
    top_k: Optional[int] = None
) -> dict:
    """Convenience function for retrieval."""
    retriever = HybridRetriever()
    return retriever.search(query, intent, top_k)
