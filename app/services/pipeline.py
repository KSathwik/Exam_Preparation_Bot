"""Main pipeline orchestration."""

import time
from typing import Optional
from loguru import logger
from datetime import datetime

from .models import QueryType, AnswerWithSources, ChatMessage
from .parser import parse_file
from .embeddings import VectorStoreManager
from .intent_classifier import IntentClassifier
from .retriever import HybridRetriever
from .llm_interface import ClaudeInterface
from .validator import SpanExtractor, ConfidenceScorer
from app.core.config import settings


class ExamPrepBot:
    """Main exam prep bot pipeline.

    Accepts pre-built singletons via constructor so that the DI layer
    (``app.core.dependencies``) controls resource lifetime.
    """

    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        intent_classifier: Optional[IntentClassifier] = None,
    ):
        logger.info("Initializing ExamPrepBot")
        self.vector_store_manager = vector_store_manager or VectorStoreManager()
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.retriever = HybridRetriever(self.vector_store_manager)
        self.llm = ClaudeInterface(intent_classifier=self.intent_classifier)
        self.span_extractor = SpanExtractor()
        self.confidence_scorer = ConfidenceScorer()
        self.chat_history: list[ChatMessage] = []

    # ------------------------------------------------------------------
    # Document upload
    # ------------------------------------------------------------------
    def upload_document(self, file_path: str, file_type: str = None) -> dict:
        logger.info(f"Uploading document: {file_path}")
        start = time.time()
        try:
            document = parse_file(file_path)
            self.vector_store_manager.add_document(document)
            duration = time.time() - start
            return {
                "success": True,
                "file_name": document.file_name,
                "file_type": document.file_type,
                "total_chunks": document.total_chunks,
                "file_size_mb": document.file_size_bytes / (1024 * 1024),
                "processing_time_seconds": duration,
                "message": f"Successfully uploaded {document.file_name} with {document.total_chunks} chunks",
            }
        except Exception as e:
            logger.error(f"Error uploading document: {e}")
            return {"success": False, "error": str(e), "message": f"Failed to upload: {e}"}

    # ------------------------------------------------------------------
    # Question answering (full RAG pipeline)
    # ------------------------------------------------------------------
    def answer_question(self, query: str) -> AnswerWithSources:
        logger.info(f"[PIPELINE] ===== New query: {query!r} =====")
        start = time.time()

        try:
            # Stage 1: Intent classification
            t0 = time.time()
            intent_result = self.intent_classifier.classify(query)
            intent = intent_result.primary_intent
            intent_confidence = intent_result.confidence
            logger.info(
                f"[PIPELINE] Stage 1 — Intent: {intent.value}  confidence={intent_confidence:.3f}  "
                f"method={intent_result.reasoning}  duration={time.time()-t0:.3f}s"
            )
            if intent_result.alternative_intents:
                logger.debug(f"[PIPELINE] Alternative intents: {intent_result.alternative_intents}")

            # Stage 2: Retrieval
            t0 = time.time()
            retrieval_result = self.retriever.search(query, intent)
            chunks = retrieval_result["chunks"]
            is_relevant = retrieval_result["is_relevant"]
            logger.info(
                f"[PIPELINE] Stage 2 — Retrieval: chunks={len(chunks)}  "
                f"relevance={retrieval_result['relevance_score']:.4f}  "
                f"is_relevant={is_relevant}  in_scope={retrieval_result['in_scope']}  "
                f"duration={time.time()-t0:.3f}s"
            )
            for i, c in enumerate(chunks[:3]):
                logger.debug(
                    f"[PIPELINE]   chunk[{i}]: page={c.metadata.page_number}  "
                    f"score={c.relevance_score:.4f}  content={c.content[:80]!r}..."
                )

            if not is_relevant:
                duration = time.time() - start
                logger.warning(f"[PIPELINE] Query out of scope — returning fallback  total_duration={duration:.3f}s")
                return AnswerWithSources(
                    answer="I couldn't find relevant information about this topic in your uploaded materials. Could you rephrase your question or provide more context?",
                    query_intent=intent,
                    intent_confidence=intent_confidence,
                    sources=[],
                    overall_confidence=0.0,
                    hallucination_risk="high",
                    response_time_seconds=duration,
                    format_type="out_of_scope",
                )

            # Stage 3: LLM answer generation
            t0 = time.time()
            structured = self.llm.generate_structured_answer(query, chunks, intent)
            answer_text = structured["answer"]
            claims = structured["claims"]
            logger.info(
                f"[PIPELINE] Stage 3 — LLM: answer_length={len(answer_text)}  "
                f"claims={len(claims)}  format={structured.get('format_type')}  "
                f"duration={time.time()-t0:.3f}s"
            )

            # Stage 4: Citation extraction & confidence scoring
            t0 = time.time()
            citations = [
                c
                for claim in claims
                if (c := self.span_extractor.extract_supporting_span(claim, chunks))
            ]

            overall_confidence = self.confidence_scorer.calculate_answer_confidence(
                chunks, citations, len(claims)
            )
            hallucination_risk = self.confidence_scorer.assess_hallucination_risk(
                chunks, citations, len(claims)
            )
            logger.info(
                f"[PIPELINE] Stage 4 — Validation: citations={len(citations)}/{len(claims)}  "
                f"confidence={overall_confidence:.3f}  hallucination_risk={hallucination_risk}  "
                f"duration={time.time()-t0:.3f}s"
            )

            response_time = time.time() - start
            result = AnswerWithSources(
                answer=answer_text,
                query_intent=intent,
                intent_confidence=intent_confidence,
                sources=citations,
                overall_confidence=overall_confidence,
                hallucination_risk=hallucination_risk,
                response_time_seconds=response_time,
                format_type=structured.get("format_type", "general"),
            )

            now = datetime.now().isoformat()
            self.chat_history.append(ChatMessage(role="user", content=query, timestamp=now, intent_type=intent))
            self.chat_history.append(ChatMessage(role="assistant", content=answer_text, timestamp=now, intent_type=intent))

            logger.info(f"[PIPELINE] ===== Query complete: total_duration={response_time:.3f}s =====")
            return result

        except Exception as e:
            duration = time.time() - start
            logger.error(f"[PIPELINE] ===== Query FAILED: error={type(e).__name__}: {e}  duration={duration:.3f}s =====")
            logger.exception("[PIPELINE] Full traceback:")
            return AnswerWithSources(
                answer=f"An error occurred: {e}",
                query_intent=QueryType.VAGUE,
                intent_confidence=0.0,
                sources=[],
                overall_confidence=0.0,
                hallucination_risk="high",
                response_time_seconds=duration,
                format_type="error",
            )

    # ------------------------------------------------------------------
    # Stats / Reset
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        return {
            "vector_store": self.vector_store_manager.get_stats(),
            "chat_history_length": len(self.chat_history),
            "model": settings.model_name,
            "embedding_model": settings.embedding_model,
        }

    def reset(self) -> None:
        logger.info("Resetting bot")
        self.chat_history.clear()
        self.vector_store_manager.vector_store.clear()
        self.vector_store_manager.vector_store.create_index()


def create_bot() -> ExamPrepBot:
    """Factory — prefer ``get_bot()`` from dependencies for DI."""
    return ExamPrepBot()
