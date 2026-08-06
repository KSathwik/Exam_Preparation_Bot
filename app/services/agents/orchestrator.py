"""OrchestratorAgent — deterministic coordinator for the multi-agent RAG pipeline.

Not an LLM router: with only one downstream answer-producing agent this phase
(Knowledge), there is nothing meaningful to route between yet. Two orthogonal,
deterministic classifiers parameterize the pipeline instead: `IntentClassifier`
(what the question is about — drives retrieval top_k/reranking) and
`FormatClassifier` (how the answer should look — drives the drafting prompt
template and the final formatting pass). Neither is agent selection. Real
LLM-based routing becomes justified once Phase 2 agents (Quiz/StudyPlanner/...)
exist to route between.
"""

import time
from typing import Callable, List, Optional, Tuple

from loguru import logger

from app.core.config import redact_query_for_log, settings

from ..context_router import ContextMode, ContextRouter
from ..conversation_scope import resolve_document_scope
from ..format_classifier import FormatClassifier
from ..global_search import is_global_search_request
from ..intent_classifier import IntentClassifier
from ..models import AnswerWithSources, QueryType, RetrievedChunk, SourceCitation
from ..response_formatter import format_response
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
        format_classifier: Optional[FormatClassifier] = None,
        context_router: Optional[ContextRouter] = None,
        db_session_factory: Optional[Callable] = None,
    ):
        self.intent_classifier = intent_classifier
        self.retrieval_agent = retrieval_agent
        self.knowledge_agent = knowledge_agent
        self.reflection_agent = reflection_agent
        self.memory_agent = memory_agent
        self.span_extractor = span_extractor or SpanExtractor()
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        # Deterministic, stateless (no embeddings/LLM) — see format_classifier.py.
        self.format_classifier = format_classifier or FormatClassifier()
        # Deterministic, stateless (no embeddings/LLM) — see context_router.py.
        self.context_router = context_router or ContextRouter()
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
        logger.info(f"[ORCHESTRATOR] ===== New query: {redact_query_for_log(query)} =====")
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
                    query,
                    small_talk_reply,
                    QueryType.VAGUE,
                    format_type="greeting",
                    session_id=session_id,
                    device_id=device_id,
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

            # Stage 1b: Response format classification — the presentation
            # axis (how the answer should look), orthogonal to intent (what
            # it's about, used only for retrieval tuning above/below). Fully
            # deterministic and LLM-free — see format_classifier.py.
            t0 = time.time()
            response_format = self.format_classifier.classify(query, intent)
            logger.info(
                f"[ORCHESTRATOR] Stage 1b — Format: {response_format.value}  "
                f"duration={time.time()-t0:.3f}s"
            )

            # Recent conversation turns (this conversation only, from the DB —
            # NOT self.chat_history, which is a single list shared across
            # every conversation this process has ever handled), fetched
            # before retrieval so a follow-up's pronoun/reference ("explain
            # THAT more simply", "the second ONE") can be resolved for the
            # retrieval query itself, not just the drafting/reflection
            # prompts below — a follow-up's raw text alone often doesn't
            # share enough vocabulary with the right chunks to match at all.
            recent_history = self.memory_agent.get_recent_turns(session_id)
            retrieval_query = query
            if recent_history:
                last_user_turn = next((m.content for m in reversed(recent_history) if m.role == "user"), None)
                if last_user_turn:
                    retrieval_query = f"{last_user_turn} {query}"

            # Stage 2: Retrieval
            t0 = time.time()

            if is_global_search_request(query):
                # Explicit opt-out of conversation scoping — see
                # is_global_search_request. document_ids=None tells
                # AdaptiveRetriever.retrieve to search the full index.
                resolved_document_ids = None
                logger.info(
                    f"[ORCHESTRATOR] Explicit global search request detected: {redact_query_for_log(query)}"
                )
            elif self.db_session_factory is not None:
                resolved_document_ids = resolve_document_scope(
                    self.db_session_factory, session_id, document_ids
                )
            else:
                # No DB access configured (e.g. a test constructing this
                # agent directly with mocks) — keep the caller-supplied value
                # exactly as before.
                resolved_document_ids = document_ids

            # Hybrid RAG + CAG: a real, non-empty conversation-scoped document
            # set is a candidate for CAG — fetch it once (cheap, in-memory
            # metadata filter, no FAISS/embedding cost) and let ContextRouter
            # decide whether it's small enough to use directly, reusing it
            # for the CAG path or discarding it in favor of a ranked search()
            # call for the RAG path. An unscoped/global search always stays
            # on the RAG path — there's no bounded candidate set to give the
            # model wholesale.
            context_mode = ContextMode.RAG
            retrieval_result = None
            if resolved_document_ids:
                full_context = self.retrieval_agent.get_full_context(resolved_document_ids)
                context_mode = self.context_router.decide(full_context["chunks"])
                if context_mode == ContextMode.CAG:
                    retrieval_result = full_context

            if retrieval_result is None:
                retrieval_result = self.retrieval_agent.search(
                    retrieval_query, intent, document_ids=resolved_document_ids, session_id=session_id
                )

            chunks = retrieval_result["chunks"]
            is_relevant = retrieval_result["is_relevant"]
            logger.info(
                f"[ORCHESTRATOR] Stage 2 — Retrieval ({context_mode.value}): chunks={len(chunks)}  "
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
            structured = self.knowledge_agent.generate(
                query, chunks, intent, response_format=response_format, history=recent_history
            )
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

            # Pre-reflection short-circuit — see config.py's
            # enable_reflection_shortcut for the full reasoning. Narrow on
            # purpose: only skips the (expensive) reflection round-trip when
            # the draft found literally zero supporting citations AND
            # confidence is already far below the hard-block floor, i.e. the
            # retrieved chunks looked weak too, not just the draft's
            # phrasing — reflection re-grounding a merely-mediocre draft into
            # a properly-cited one is a real, common outcome this must not
            # give up for the sake of latency.
            draft_citation_rate = len(citations) / len(claims) if claims else None
            if (
                settings.enable_reflection_shortcut
                and draft_citation_rate == 0
                and overall_confidence < settings.reflection_shortcut_confidence_floor
            ):
                duration = time.time() - start
                logger.warning(
                    f"[ORCHESTRATOR] Pre-reflection shortcut — draft cited none of its claims and "
                    f"confidence={overall_confidence:.3f} is already far below the block floor; "
                    f"skipping reflection  total_duration={duration:.3f}s"
                )
                return self._blocked_result(
                    query,
                    intent,
                    intent_confidence,
                    overall_confidence,
                    duration,
                    session_id,
                    device_id,
                    on_stage,
                )

            # Stage 5: Reflection (always-on) — silent quality-control pass;
            # never surfaced to the user (see on_stage below, which only
            # fires once the final, already-reflected text is ready).
            t0 = time.time()
            validator_summary = (
                f"{len(citations)}/{len(claims)} claims cited, "
                f"confidence={overall_confidence:.2f}, hallucination_risk={hallucination_risk}"
            )
            reflection = self.reflection_agent.reflect(
                query,
                chunks,
                draft_answer,
                validator_summary,
                intent,
                response_format=response_format,
                history=recent_history,
            )
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
            # Only a genuinely missing/failed reflection (claims=None) falls
            # back to the draft's own claims — an explicit empty list means
            # reflection ran and found nothing traceable, which must score as
            # zero citations, not silently reuse the draft's (possibly
            # different-answer's) claims.
            final_claims = claims if reflection.get("claims") is None else reflection["claims"]
            citations, overall_confidence, hallucination_risk = self._score(final_claims, chunks)
            logger.info(
                f"[ORCHESTRATOR] Stage 6 — Final scoring: claims={len(final_claims)}  "
                f"citations={len(citations)}/{len(final_claims)}  confidence={overall_confidence:.3f}"
            )

            # Deterministic safety net, not just an LLM's opinion of itself:
            # `should_block` above only fires when the reflection LLM flags
            # its own (or a fresh) draft — the same model that could
            # hallucinate the answer could just as easily hallucinate a clean
            # bill of health. This gate is code-level and non-LLM.
            #
            # Deliberately stricter than the "high" risk *badge* threshold
            # (0.45 confidence / 0.4 citation rate) rather than reusing it:
            # live testing showed confidence naturally varies +/-0.1 run-to-
            # run for the identical question against identical source
            # material (ordinary LLM sampling variance in which claims get
            # extracted/phrased), which straddles that boundary often enough
            # to hard-block a perfectly fine answer roughly half the time.
            # Hard-blocking needs a much wider margin so normal variance
            # never crosses it — only confidence well below the "medium"
            # floor, or an answer that made claims but verified literally
            # none of them, is unambiguous enough to withhold outright.
            citation_rate = len(citations) / len(final_claims) if final_claims else None
            if overall_confidence < 0.3 or citation_rate == 0:
                duration = time.time() - start
                logger.warning(
                    f"[ORCHESTRATOR] Hallucination-risk gate blocked the answer "
                    f"(confidence={overall_confidence:.3f}  citations={len(citations)}/{len(final_claims)})  "
                    f"total_duration={duration:.3f}s"
                )
                return self._blocked_result(
                    query,
                    intent,
                    intent_confidence,
                    overall_confidence,
                    duration,
                    session_id,
                    device_id,
                    on_stage,
                )

            # Stage 7: Response formatting — deterministic, presentation-only
            # cleanup (length caps, boilerplate stripping) on top of the
            # already-reflected text; see response_formatter.py for why this
            # doesn't attempt to restructure prose after the fact.
            final_answer = format_response(final_answer, response_format)

            response_time = time.time() - start
            result = AnswerWithSources(
                answer=final_answer,
                query_intent=intent,
                intent_confidence=intent_confidence,
                sources=citations,
                overall_confidence=overall_confidence,
                hallucination_risk=hallucination_risk,
                response_time_seconds=response_time,
                format_type=response_format.value,
            )

            # Stage 8: Memory
            self.memory_agent.record_turn(
                query,
                final_answer,
                intent,
                format_type=response_format.value,
                session_id=session_id,
                device_id=device_id,
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
                answer="Something went wrong while preparing your answer. Please try again.",
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

    def _blocked_result(
        self,
        query: str,
        intent: QueryType,
        intent_confidence: float,
        overall_confidence: float,
        duration: float,
        session_id: Optional[str],
        device_id: Optional[str],
        on_stage: OnStage,
    ) -> AnswerWithSources:
        """Shared by the pre-reflection shortcut and the post-reflection
        hallucination-risk gate — both hard-block for the same reason (the
        draft's claims don't hold up against the retrieved chunks) and must
        produce identical, consistently-recorded output regardless of which
        stage caught it."""
        result = AnswerWithSources(
            answer="I found information related to your question, but couldn't verify enough of it against your uploaded material to answer reliably. Try rephrasing your question, or asking about a more specific part of the document.",
            query_intent=intent,
            intent_confidence=intent_confidence,
            sources=[],
            overall_confidence=overall_confidence,
            hallucination_risk="high",
            response_time_seconds=duration,
            format_type="reflection_blocked",
        )
        on_stage("answer_ready", {"answer": result.answer})
        self.memory_agent.record_turn(
            query,
            result.answer,
            intent,
            format_type="reflection_blocked",
            session_id=session_id,
            device_id=device_id,
        )
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
