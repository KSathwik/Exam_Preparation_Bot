"""Main pipeline orchestration."""

import time
import uuid
from typing import Callable, List, Optional

from loguru import logger

from app.core.config import settings
from app.core.database import SessionLocal

from .agents import KnowledgeAgent, MemoryAgent, OrchestratorAgent, ReflectionAgent, RetrievalAgent
from .embeddings import VectorStoreManager
from .intent_classifier import IntentClassifier
from .llm_interface import ClaudeInterface
from .models import AnswerWithSources, ChatMessage
from .parser import parse_file
from .retriever import HybridRetriever
from .validator import ConfidenceScorer, SpanExtractor


class ExamPrepBot:
    """Main exam prep bot pipeline.

    Accepts pre-built singletons via constructor so that the DI layer
    (``app.core.dependencies``) controls resource lifetime. Query answering
    itself is delegated to ``self.orchestrator`` (see ``app.services.agents``)
    — this class stays the composition root so every existing attribute
    (``retriever``, ``llm``, ``chat_history``, ...) keeps working exactly as
    before for both callers and tests.
    """

    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        intent_classifier: Optional[IntentClassifier] = None,
        llm=None,
    ):
        logger.info("Initializing ExamPrepBot")
        self.vector_store_manager = vector_store_manager or VectorStoreManager()
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.retriever = HybridRetriever(self.vector_store_manager)
        self.llm = llm or ClaudeInterface()
        self.span_extractor = SpanExtractor()
        self.confidence_scorer = ConfidenceScorer()
        self.chat_history: list[ChatMessage] = []

        # Agents wrap self.retriever/self.llm/self.chat_history by reference
        # (not bound methods), so monkeypatching e.g. ``bot.retriever.search``
        # after construction — as the test suite does — still takes effect.
        self.orchestrator = OrchestratorAgent(
            intent_classifier=self.intent_classifier,
            retrieval_agent=RetrievalAgent(self.retriever),
            knowledge_agent=KnowledgeAgent(self.llm),
            reflection_agent=ReflectionAgent(self.llm),
            memory_agent=MemoryAgent(
                self.chat_history,
                persist=True,
                db_session_factory=SessionLocal,
                llm=self.llm,
                vector_store_manager=self.vector_store_manager,
            ),
            span_extractor=self.span_extractor,
            confidence_scorer=self.confidence_scorer,
            db_session_factory=SessionLocal,
        )

    # ------------------------------------------------------------------
    # Document upload
    # ------------------------------------------------------------------
    def upload_document(self, file_path: str, file_type: Optional[str] = None) -> dict:
        logger.info(f"Uploading document: {file_path}")
        start = time.time()
        try:
            document = parse_file(file_path)
            document_id = str(uuid.uuid4())
            self.vector_store_manager.add_document(document, document_id)
            duration = time.time() - start
            return {
                "success": True,
                "document_id": document_id,
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
    def answer_question(
        self,
        query: str,
        session_id: Optional[str] = None,
        device_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        on_stage: Optional[Callable[[str, dict], None]] = None,
    ) -> AnswerWithSources:
        """Run the multi-agent pipeline (intent -> retrieval -> draft ->
        reflection -> memory) — see ``OrchestratorAgent.run`` for the stage
        breakdown. ``session_id``/``device_id`` scope conversation persistence
        (see ``MemoryAgent``) to one conversation/browser; omit for the legacy
        global in-memory-only behavior. ``document_ids``, when given, scopes
        retrieval to just those documents first (falling back to the full
        index if that misses) — see ``AdaptiveRetriever.retrieve``. ``on_stage``
        is an optional progress callback (used by the WebSocket route for
        progressive status/draft/final events); REST and batch callers simply
        omit it."""
        return self.orchestrator.run(
            query,
            session_id=session_id,
            device_id=device_id,
            document_ids=document_ids,
            on_stage=on_stage,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        return {
            "vector_store": self.vector_store_manager.get_stats(),
            "chat_history_length": len(self.chat_history),
            "model": settings.model_name,
            "embedding_model": settings.embedding_model,
        }


def create_bot() -> ExamPrepBot:
    """Factory — prefer ``get_bot()`` from dependencies for DI."""
    return ExamPrepBot()
