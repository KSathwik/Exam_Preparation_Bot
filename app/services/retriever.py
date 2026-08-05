"""Document retrieval and ranking module."""

from typing import List, Optional, Tuple

from loguru import logger

from app.core.config import redact_query_for_log, settings

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
            f"Retrieval: query={redact_query_for_log(query)}  best_score={best:.4f}  "
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
        session_id: Optional[str] = None,
    ) -> Tuple[List[RetrievedChunk], bool]:
        if top_k is None:
            top_k = self._TOP_K_STRATEGY.get(intent, settings.retrieval_top_k)

        if document_ids is not None:
            # An explicit scope (even an empty one) is authoritative — see
            # resolve_document_scope. Restricting the candidate pool up front
            # to the document(s) that belong to this conversation is
            # deliberately unconditional once any chunk exists there, NOT
            # gated behind relevance_threshold like the unscoped path below.
            # A vague meta-question ("what's in this document?", "summarize
            # this") often has weak raw embedding similarity to the
            # document's own text (a meta-question doesn't share vocabulary
            # with the content it's asking about), which would otherwise read
            # as a "miss". Unlike before, there is no fallback to the full
            # index here: an empty or missed scoped search means "nothing
            # relevant in this conversation," full stop — never a silent
            # global search (see is_global_search_request for the explicit
            # opt-in path instead).
            scoped_raw = (
                self.vector_store.search(query, top_k=top_k * 2, document_ids=document_ids)
                if document_ids
                else []
            )
            if scoped_raw:
                chunks, _ = self._score_results(query, scoped_raw, top_k)
                logger.info(f"[RETRIEVAL] Scoped to document_ids={document_ids} — {len(chunks)} chunk(s)")
                return chunks, True
            logger.debug(
                f"[RETRIEVAL] No chunks for document_ids={document_ids} — trying conversation memory only"
            )
            return self._try_memory_fallback(query, top_k, session_id)

        raw_results = self.vector_store.search(query, top_k=top_k * 2)
        if not raw_results:
            logger.debug(f"No raw results for query: {redact_query_for_log(query)}")
            return self._try_memory_fallback(query, top_k, session_id)

        chunks, in_scope = self._score_results(query, raw_results, top_k)

        if not in_scope:
            # Document retrieval missed — try semantic memory before giving
            # up entirely. Not blended into document retrieval (would risk
            # diluting citations; ConfidenceScorer assumes document-sourced
            # chunks), so this only ever replaces a would-be "nothing
            # relevant" result, never mixes with document chunks.
            memory_chunks, memory_in_scope = self._try_memory_fallback(query, top_k, session_id)
            if memory_in_scope:
                return memory_chunks, True

        return chunks, in_scope

    def _try_memory_fallback(
        self, query: str, top_k: int, session_id: Optional[str] = None
    ) -> Tuple[List[RetrievedChunk], bool]:
        if not session_id:
            # No conversation context to scope by — searching every
            # conversation's summaries would be exactly the cross-session
            # leak this scoping exists to prevent, so skip the fallback
            # entirely rather than guess.
            return [], False
        memory_results = self.vector_store.search(
            query, top_k=top_k, content_types=["memory"], session_ids=[session_id]
        )
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
                f"query={redact_query_for_log(query)}  best_score={best:.4f}"
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
        session_id: Optional[str] = None,
    ) -> dict:
        chunks, in_scope = self.adaptive.retrieve(
            query, intent, top_k, document_ids=document_ids, session_id=session_id
        )
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

    def get_full_context(self, document_ids: List[str]) -> dict:
        """CAG path — every chunk of every document in ``document_ids``, in
        reading order, no ranking/truncation. Mirrors search()'s return shape
        so callers (OrchestratorAgent) don't need a different code path to
        handle the result, just a different call to produce it. is_relevant
        is True whenever the scope has any indexed content at all — no
        relevance_threshold applies, since these documents were explicitly
        uploaded to this conversation, not found by a similarity search."""
        chunks = self.adaptive.vector_store.get_chunks_by_document_ids(document_ids)
        is_relevant = bool(chunks)
        return {
            "query": None,
            "intent": None,
            "chunks": chunks,
            "in_scope": is_relevant,
            "is_relevant": is_relevant,
            "relevance_score": 1.0 if chunks else 0.0,
            "total_retrieved": len(chunks),
        }
