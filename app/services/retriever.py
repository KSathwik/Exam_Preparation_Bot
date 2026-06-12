"""Document retrieval and ranking module."""

from typing import List, Tuple, Optional
from .models import QueryType, RetrievedChunk, ChunkMetadata
from .embeddings import VectorStoreManager
from loguru import logger
from app.core.config import settings


class AdaptiveRetriever:
    """Adaptive retriever that adjusts strategy based on query intent."""

    _TOP_K_STRATEGY = {
        QueryType.DEFINITION: 3,
        QueryType.EXPLAIN: 8,
        QueryType.COMPARE: 8,
        QueryType.PROCESS: 6,
        QueryType.EXAMPLE: 4,
        QueryType.DIAGRAM: 4,
        QueryType.VAGUE: 10,
        QueryType.HOMEWORK: 5,
    }

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.vector_store = vector_store or VectorStoreManager()
        self.relevance_threshold = settings.relevance_threshold

    def retrieve(
        self, query: str, intent: QueryType, top_k: Optional[int] = None
    ) -> Tuple[List[RetrievedChunk], bool]:
        if top_k is None:
            top_k = self._TOP_K_STRATEGY.get(intent, settings.retrieval_top_k)

        raw_results = self.vector_store.search(query, top_k=top_k * 2)
        if not raw_results:
            logger.debug(f"No raw results for query: {query!r}")
            return [], False

        chunks = []
        for chunk_info, similarity, rank in raw_results[:top_k]:
            meta = ChunkMetadata(**chunk_info.get("metadata", {}))
            chunks.append(
                RetrievedChunk(
                    content=chunk_info.get("content", ""),
                    metadata=meta,
                    relevance_score=similarity,
                    rank=rank,
                )
            )

        best = chunks[0].relevance_score if chunks else 0.0
        logger.info(
            f"Retrieval: query={query!r}  best_score={best:.4f}  "
            f"threshold={self.relevance_threshold}  in_scope={best >= self.relevance_threshold}  "
            f"n_chunks={len(chunks)}"
        )
        in_scope = bool(chunks) and best >= self.relevance_threshold
        return chunks, in_scope

    @staticmethod
    def rerank_by_intent(
        chunks: List[RetrievedChunk], intent: QueryType
    ) -> List[RetrievedChunk]:
        if not chunks:
            return chunks
        if intent == QueryType.PROCESS:
            return sorted(chunks, key=lambda x: (x.metadata.page_number, x.metadata.chunk_index))
        if intent == QueryType.DEFINITION:
            return sorted(chunks, key=lambda x: x.metadata.chunk_position != "beginning")
        return chunks


class HybridRetriever:
    """Hybrid retriever combining semantic search and relevance checking."""

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.adaptive = AdaptiveRetriever(vector_store)

    def search(self, query: str, intent: QueryType, top_k: Optional[int] = None) -> dict:
        chunks, in_scope = self.adaptive.retrieve(query, intent, top_k)
        relevance_score = chunks[0].relevance_score if chunks else 0.0
        is_relevant = relevance_score >= settings.relevance_threshold
        chunks = AdaptiveRetriever.rerank_by_intent(chunks, intent)
        return {
            "query": query,
            "intent": intent,
            "chunks": chunks,
            "in_scope": in_scope,
            "is_relevant": is_relevant,
            "relevance_score": relevance_score,
            "total_retrieved": len(chunks),
        }
