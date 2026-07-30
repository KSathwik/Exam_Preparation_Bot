"""OrchestratorAgent — deterministic coordinator for the multi-agent RAG pipeline.

Not an LLM router: with only one downstream answer-producing agent this phase
(Knowledge), there is nothing meaningful to route between yet. `IntentClassifier`
already parameterizes the pipeline (top_k, prompt template, format_type) —
that's not the same thing as agent selection. Real LLM-based routing becomes
justified once Phase 2 agents (Quiz/StudyPlanner/...) exist to route between.
"""

import time
from typing import Callable, List, Optional, Tuple

from loguru import logger

from ..conversation_scope import resolve_document_scope
from ..global_search import is_global_search_request
from ..intent_classifier import IntentClassifier
from ..models import AnswerWithSources, QueryType, RetrievedChunk, SourceCitation
from ..small_talk import match_small_talk
from ..validator import ConfidenceScorer, SpanExtractor
from .knowledge_agent import KnowledgeAgent
from .memory_agent import MemoryAgent
from .reflection_agent import ReflectionAgent
from .retrieval_agent import RetrievalAgent

OnStage = Callable[[str, dict], None]


def _noop_on_stage(stage: str, payload: dict) -> None:
    pass


class OrchestratorAgent:
    """Runs intent classification -> retrieval -> draft -> reflection -> memory.

    Mirrors `ExamPrepBot.answer_question`'s prior single-method behavior stage
    for stage, with an added always-on reflection pass and an optional
    `on_stage` callback for progressive (WebSocket) status updates.
    """

    def __init__(
        self,
        intent_classifier: IntentClassifier,
        retrieval_agent: RetrievalAgent,
        knowledge_agent: KnowledgeAgent,
        reflection_agent: ReflectionAgent,
        memory_agent: MemoryAgent,
        span_extractor: Optional[SpanExtractor] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        db_session_factory: Optional[Callable] = None,
    ):
        self.intent_classifier = intent_classifier
        self.retrieval_agent = retrieval_agent
        self.knowledge_agent = knowledge_agent
        self.reflection_agent = reflection_agent
        self.memory_agent = memory_agent
        self.span_extractor = span_extractor or SpanExtractor()
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        # Enables resolving a conversation's authoritative document scope
        # from the DB (see resolve_document_scope) — optional so tests that
        # construct this agent directly with mocks keep today's
        # pass-document_ids-straight-through behavior unchanged.
        self.db_session_factory = db_session_factory

    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        device_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        on_stage: Optional[OnStage] = None,
    ) -> AnswerWithSources:
        on_stage = on_stage or _noop_on_stage
        logger.info(f"[ORCHESTRATOR] ===== New query: {query!r} =====")
        start = time.time()

        try:
            # Stage 0: Small talk (greetings/thanks/farewells) short-circuits
            # before intent classification even runs — retrieval and the LLM
            # have nothing useful to do with "hi", and routing it through the
            # full pipeline is exactly what produced an unprompted document
            # summary instead of a plain reply.
            small_talk_reply = match_small_talk(query)
            if small_talk_reply is not None:
                duration = time.time() - start
                logger.info(
                    f"[ORCHESTRATOR] Small talk detected — skipping retrieval/LLM  duration={duration:.3f}s"
                )
                result = AnswerWithSources(
                    answer=small_talk_reply,
                    query_intent=QueryType.VAGUE,
                    intent_confidence=1.0,
                    sources=[],
                    overall_confidence=1.0,
                    hallucination_risk="low",
                    response_time_seconds=duration,
                    format_type="greeting",
                )
                self.memory_agent.record_turn(
                    query, small_talk_reply, QueryType.VAGUE, session_id=session_id, device_id=device_id
                )
                on_stage("answer_ready", {"answer": result.answer})
                return result

            # Stage 1: Intent classification
            t0 = time.time()
            intent_result = self.intent_classifier.classify(query)
            intent = intent_result.primary_intent
            intent_confidence = intent_result.confidence
            logger.info(
                f"[ORCHESTRATOR] Stage 1 — Intent: {intent.value}  confidence={intent_confidence:.3f}  "
                f"duration={time.time()-t0:.3f}s"
            )

            # Stage 2: Retrieval
            t0 = time.time()

            if is_global_search_request(query):
                # Explicit opt-out of conversation scoping — see
                # is_global_search_request. document_ids=None tells
                # AdaptiveRetriever.retrieve to search the full index.
                resolved_document_ids = None
                logger.info(f"[ORCHESTRATOR] Explicit global search request detected: {query!r}")
            elif self.db_session_factory is not None:
                resolved_document_ids = resolve_document_scope(
                    self.db_session_factory, session_id, document_ids
                )
            else:
                # No DB access configured (e.g. a test constructing this
                # agent directly with mocks) — keep the caller-supplied value
                # exactly as before.
                resolved_document_ids = document_ids

            retrieval_result = self.retrieval_agent.search(
                query, intent, document_ids=resolved_document_ids, session_id=session_id
            )
            chunks = retrieval_result["chunks"]
            is_relevant = retrieval_result["is_relevant"]
            logger.info(
                f"[ORCHESTRATOR] Stage 2 — Retrieval: chunks={len(chunks)}  "
                f"is_relevant={is_relevant}  duration={time.time()-t0:.3f}s"
            )

            if not is_relevant:
                duration = time.time() - start
                logger.warning(
                    f"[ORCHESTRATOR] Query out of scope — returning fallback  total_duration={duration:.3f}s"
                )
                result = AnswerWithSources(
                    answer="I couldn't find this information in the uploaded documents. Try asking something more specific about their content — see the loaded documents below.",
                    query_intent=intent,
                    intent_confidence=intent_confidence,
                    sources=[],
                    overall_confidence=0.0,
                    hallucination_risk="high",
                    response_time_seconds=duration,
                    format_type="out_of_scope",
                )
                on_stage("answer_ready", {"answer": result.answer})
                return result

            # Stage 3: Draft generation (Knowledge Agent) — runs fully
            # server-side; the draft is never shown to the user, only the
            # post-reflection text below (see Stage 5).
            t0 = time.time()
            structured = self.knowledge_agent.generate(query, chunks, intent)
            draft_answer = structured["answer"]
            claims = structured["claims"]
            logger.info(
                f"[ORCHESTRATOR] Stage 3 — Draft: answer_length={len(draft_answer)}  "
                f"claims={len(claims)}  duration={time.time()-t0:.3f}s"
            )

            # Stage 4: Citation extraction & confidence scoring on the draft
            citations, overall_confidence, hallucination_risk = self._score(claims, chunks)
            logger.info(
                f"[ORCHESTRATOR] Stage 4 — Validation: citations={len(citations)}/{len(claims)}  "
                f"confidence={overall_confidence:.3f}  hallucination_risk={hallucination_risk}"
            )

            # Stage 5: Reflection (always-on) — silent quality-control pass;
            # never surfaced to the user (see on_stage below, which only
            # fires once the final, already-reflected text is ready).
            t0 = time.time()
            validator_summary = (
                f"{len(citations)}/{len(claims)} claims cited, "
                f"confidence={overall_confidence:.2f}, hallucination_risk={hallucination_risk}"
            )
            reflection = self.reflection_agent.reflect(query, chunks, draft_answer, validator_summary, intent)
            logger.info(
                f"[ORCHESTRATOR] Stage 5 — Reflection: materially_changed={reflection['materially_changed']}  "
                f"should_block={reflection['should_block']}  issues={len(reflection['issues_found'])}  "
                f"duration={time.time()-t0:.3f}s"
            )

            if reflection["should_block"]:
                duration = time.time() - start
                logger.warning(
                    f"[ORCHESTRATOR] Reflection blocked the draft answer  total_duration={duration:.3f}s"
                )
                result = AnswerWithSources(
                    answer="I found information related to your question, but couldn't produce a reliable answer from it. Try rephrasing your question or checking the source material directly.",
                    query_intent=intent,
                    intent_confidence=intent_confidence,
                    sources=[],
                    overall_confidence=0.0,
                    hallucination_risk="high",
                    response_time_seconds=duration,
                    format_type="reflection_blocked",
                )
                on_stage("answer_ready", {"answer": result.answer})
                return result

            final_answer = reflection["revised_answer"]
            # Stage 6: Final scoring — reflection returns its own claims for
            # the (possibly revised) answer in the same call, so no separate
            # re-extraction round trip is needed even when materially_changed.
            # A None (reflection failed, or genuinely returned nothing) falls
            # back to the draft's own claims — correct either way, since an
            # unchanged answer's claims are unchanged too.
            final_claims = reflection.get("claims") or claims
            citations, overall_confidence, hallucination_risk = self._score(final_claims, chunks)
            logger.info(
                f"[ORCHESTRATOR] Stage 6 — Final scoring: claims={len(final_claims)}  "
                f"citations={len(citations)}/{len(final_claims)}  confidence={overall_confidence:.3f}"
            )

            response_time = time.time() - start
            result = AnswerWithSources(
                answer=final_answer,
                query_intent=intent,
                intent_confidence=intent_confidence,
                sources=citations,
                overall_confidence=overall_confidence,
                hallucination_risk=hallucination_risk,
                response_time_seconds=response_time,
                format_type=structured.get("format_type", "general"),
            )

            # Stage 7: Memory
            self.memory_agent.record_turn(
                query, final_answer, intent, session_id=session_id, device_id=device_id
            )

            on_stage("answer_ready", {"answer": result.answer})
            logger.info(f"[ORCHESTRATOR] ===== Query complete: total_duration={response_time:.3f}s =====")
            return result

        except Exception as e:
            duration = time.time() - start
            logger.error(
                f"[ORCHESTRATOR] ===== Query FAILED: error={type(e).__name__}: {e}  duration={duration:.3f}s ====="
            )
            logger.exception("[ORCHESTRATOR] Full traceback:")
            result = AnswerWithSources(
                answer=f"An error occurred: {e}",
                query_intent=QueryType.VAGUE,
                intent_confidence=0.0,
                sources=[],
                overall_confidence=0.0,
                hallucination_risk="high",
                response_time_seconds=duration,
                format_type="error",
            )
            on_stage("answer_ready", {"answer": result.answer})
            return result

    def _score(
        self, claims: List[str], chunks: List[RetrievedChunk]
    ) -> Tuple[List[SourceCitation], float, str]:
        citations = [
            c for claim in claims if (c := self.span_extractor.extract_supporting_span(claim, chunks))
        ]
        overall_confidence = self.confidence_scorer.calculate_answer_confidence(
            chunks, citations, len(claims)
        )
        hallucination_risk = self.confidence_scorer.assess_hallucination_risk(chunks, citations, len(claims))
        return citations, overall_confidence, hallucination_risk
