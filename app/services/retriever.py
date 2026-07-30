"""Document retrieval and ranking module."""

from typing import List, Optional, Tuple

from loguru import logger

from app.core.config import settings

from .embeddings import VectorStoreManager
from .models import ChunkMetadata, QueryType, RetrievedChunk


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

    def __init__(self, vector_store: Optional[VectorStoreManager] = None, cross_encoder=None):
        self.vector_store = vector_store or VectorStoreManager()
        self.relevance_threshold = settings.relevance_threshold
        # Pre-injected (tests) or lazily built on first use — see
        # _get_cross_encoder. Only ever touched when reranking is enabled.
        self._cross_encoder = cross_encoder

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"[RERANK] Loading cross-encoder: {settings.cross_encoder_model}")
            self._cross_encoder = CrossEncoder(settings.cross_encoder_model)
        return self._cross_encoder

    def _cross_encoder_rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Rerank the merged candidate pool with a cross-encoder — opt-in
        (settings.enable_cross_encoder_rerank) since hybrid retrieval alone
        should capture most of the achievable gain at this project's current
        scale. Falls back to the original order on any failure rather than
        blocking retrieval on a reranking problem."""
        try:
            cross_encoder = self._get_cross_encoder()
            scores = cross_encoder.predict([(query, c.content) for c in chunks])
            order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
            return [chunks[i] for i in order]
        except Exception as exc:
            logger.warning(
                f"[RERANK] Cross-encoder rerank failed, keeping original order: {type(exc).__name__}: {exc}"
            )
            return chunks

    def _score_results(
        self, query: str, raw_results: List[Tuple[dict, float, int]], top_k: int
    ) -> Tuple[List[RetrievedChunk], bool]:
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

        if settings.enable_cross_encoder_rerank and len(chunks) > 1:
            chunks = self._cross_encoder_rerank(query, chunks)

        best = chunks[0].relevance_score if chunks else 0.0
        in_scope = bool(chunks) and best >= self.relevance_threshold
        logger.info(
            f"Retrieval: query={query!r}  best_score={best:.4f}  "
            f"threshold={self.relevance_threshold}  in_scope={in_scope}  "
            f"n_chunks={len(chunks)}"
        )
        return chunks, in_scope

    def retrieve(
        self,
        query: str,
        intent: QueryType,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
    ) -> Tuple[List[RetrievedChunk], bool]:
        if top_k is None:
            top_k = self._TOP_K_STRATEGY.get(intent, settings.retrieval_top_k)

        if document_ids:
            # Restrict the candidate pool up front to the document(s) the
            # user actually uploaded in this conversation. Deliberately
            # unconditional once any chunk exists there — NOT gated behind
            # relevance_threshold like the global path below. A vague
            # meta-question ("what's in this document?", "summarize this")
            # often has weak raw embedding similarity to the document's own
            # text (a meta-question doesn't share vocabulary with the
            # content it's asking about), which would otherwise read as a
            # "miss" and fall through to the full index — where a
            # completely unrelated document can coincidentally score higher
            # and win. That's the exact "confusing between different
            # uploads" failure mode this scoping exists to prevent, so an
            # explicit document scope is treated as authoritative: only an
            # empty result set (the id has no indexed chunks at all, e.g. a
            # deleted document) falls through to the full-index search.
            scoped_raw = self.vector_store.search(query, top_k=top_k * 2, document_ids=document_ids)
            if scoped_raw:
                chunks, _ = self._score_results(query, scoped_raw, top_k)
                logger.info(f"[RETRIEVAL] Scoped to document_ids={document_ids} — {len(chunks)} chunk(s)")
                return chunks, True
            logger.debug(f"[RETRIEVAL] No indexed chunks for document_ids={document_ids} — trying full index")

        raw_results = self.vector_store.search(query, top_k=top_k * 2)
        if not raw_results:
            logger.debug(f"No raw results for query: {query!r}")
            return self._try_memory_fallback(query, top_k)

        chunks, in_scope = self._score_results(query, raw_results, top_k)

        if not in_scope:
            # Document retrieval missed — try semantic memory before giving
            # up entirely. Not blended into document retrieval (would risk
            # diluting citations; ConfidenceScorer assumes document-sourced
            # chunks), so this only ever replaces a would-be "nothing
            # relevant" result, never mixes with document chunks.
            memory_chunks, memory_in_scope = self._try_memory_fallback(query, top_k)
            if memory_in_scope:
                return memory_chunks, True

        return chunks, in_scope

    def _try_memory_fallback(self, query: str, top_k: int) -> Tuple[List[RetrievedChunk], bool]:
        memory_results = self.vector_store.search(query, top_k=top_k, content_types=["memory"])
        if not memory_results:
            return [], False

        chunks = []
        for chunk_info, similarity, rank in memory_results:
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
        in_scope = best >= settings.memory_relevance_threshold
        if in_scope:
            logger.info(
                f"[RETRIEVAL] Document retrieval missed — semantic memory hit: "
                f"query={query!r}  best_score={best:.4f}"
            )
            return chunks, True
        return [], False

    @staticmethod
    def rerank_by_intent(chunks: List[RetrievedChunk], intent: QueryType) -> List[RetrievedChunk]:
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

    def search(
        self,
        query: str,
        intent: QueryType,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
    ) -> dict:
        chunks, in_scope = self.adaptive.retrieve(query, intent, top_k, document_ids=document_ids)
        relevance_score = chunks[0].relevance_score if chunks else 0.0
        # in_scope already reflects the full decision — including a semantic
        # memory hit replacing a missed document search — so is_relevant
        # mirrors it rather than re-deriving from relevance_score against
        # settings.relevance_threshold, which would ignore memory fallback.
        is_relevant = in_scope
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
