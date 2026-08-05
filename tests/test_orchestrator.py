"""Tests for the multi-agent pipeline: Retrieval/Knowledge/Reflection/Memory
agents plus the OrchestratorAgent that coordinates them.

All agents wrap mocked collaborators here — nothing in this file touches a
real vector store, LLM provider, or document.
"""

from unittest.mock import MagicMock

import pytest

from app.services.agents.knowledge_agent import KnowledgeAgent
from app.services.agents.memory_agent import MemoryAgent
from app.services.agents.orchestrator import OrchestratorAgent
from app.services.agents.reflection_agent import ReflectionAgent
from app.services.agents.retrieval_agent import RetrievalAgent
from app.services.models import (
    AnswerWithSources,
    ChunkMetadata,
    IntentClassificationResult,
    QueryType,
    RetrievedChunk,
    ChatMessage,
)
from app.services.response_formats import ResponseFormat


def _make_chunk(content="Photosynthesis converts light into chemical energy."):
    meta = ChunkMetadata(page_number=1, chunk_index=0, total_chunks=1, file_name="bio.pdf")
    return RetrievedChunk(content=content, metadata=meta, relevance_score=0.9, rank=1)


def _reflection_no_change(draft_answer):
    return {
        "revised_answer": draft_answer,
        "materially_changed": False,
        "should_block": False,
        "issues_found": [],
    }


# ----------------------------------------------------------------------
# Individual agent wrappers
# ----------------------------------------------------------------------


def test_retrieval_agent_delegates_to_retriever():
    retriever = MagicMock()
    retriever.search.return_value = {"chunks": [], "is_relevant": False}
    agent = RetrievalAgent(retriever)

    result = agent.search("q", QueryType.DEFINITION, top_k=5)

    retriever.search.assert_called_once_with("q", QueryType.DEFINITION, 5, document_ids=None, session_id=None)
    assert result == {"chunks": [], "is_relevant": False}


def test_retrieval_agent_picks_up_monkeypatched_search(monkeypatch):
    retriever = MagicMock()
    agent = RetrievalAgent(retriever)

    replacement = MagicMock(return_value={"chunks": [_make_chunk()], "is_relevant": True})
    monkeypatch.setattr(retriever, "search", replacement)

    result = agent.search("q", QueryType.DEFINITION)
    assert result["is_relevant"] is True
    replacement.assert_called_once()


def test_knowledge_agent_delegates_to_llm():
    llm = MagicMock()
    llm.generate_structured_answer.return_value = {"answer": "42", "claims": [], "format_type": "definition"}
    agent = KnowledgeAgent(llm)

    result = agent.generate("q", [_make_chunk()], QueryType.DEFINITION)

    assert result["answer"] == "42"
    llm.generate_structured_answer.assert_called_once()


def test_reflection_agent_delegates_to_llm():
    llm = MagicMock()
    llm.reflect_on_answer.return_value = {
        "revised_answer": "Better.",
        "materially_changed": True,
        "should_block": False,
        "issues_found": [],
    }
    agent = ReflectionAgent(llm)

    result = agent.reflect("q", [_make_chunk()], "draft", "summary", QueryType.DEFINITION)

    assert result["revised_answer"] == "Better."
    llm.reflect_on_answer.assert_called_once_with(
        "q",
        [_make_chunk()],
        "draft",
        "summary",
        QueryType.DEFINITION,
        response_format=ResponseFormat.GENERAL,
        history=None,
    )


def test_reflection_agent_falls_back_when_llm_raises():
    llm = MagicMock()
    llm.reflect_on_answer.side_effect = RuntimeError("boom")
    agent = ReflectionAgent(llm)

    result = agent.reflect("q", [_make_chunk()], "draft answer", "summary", QueryType.DEFINITION)

    assert result == {
        "revised_answer": "draft answer",
        "materially_changed": False,
        "should_block": False,
        "issues_found": [],
    }


def test_memory_agent_records_user_and_assistant_turns():
    history: list[ChatMessage] = []
    agent = MemoryAgent(history)

    agent.record_turn("What is X?", "X is Y.", QueryType.DEFINITION)

    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "What is X?"
    assert history[1].role == "assistant"
    assert history[1].content == "X is Y."


def test_memory_agent_shares_the_same_list_reference():
    history: list[ChatMessage] = []
    agent = MemoryAgent(history)
    agent.record_turn("q", "a")
    assert agent.chat_history is history


# ----------------------------------------------------------------------
# OrchestratorAgent
# ----------------------------------------------------------------------


@pytest.fixture()
def mock_intent_classifier():
    clf = MagicMock()
    clf.classify.return_value = IntentClassificationResult(
        query="What is photosynthesis?",
        primary_intent=QueryType.DEFINITION,
        confidence=0.95,
        reasoning="test",
    )
    return clf


@pytest.fixture()
def mock_llm():
    llm = MagicMock()
    llm.generate_structured_answer.return_value = {
        "answer": "Photosynthesis converts light into chemical energy.",
        "format_type": "definition",
        "claims": ["Photosynthesis converts light into chemical energy."],
        "intent": "definition",
    }
    llm.reflect_on_answer.return_value = _reflection_no_change(
        "Photosynthesis converts light into chemical energy."
    )
    return llm


@pytest.fixture()
def mock_retriever():
    retriever = MagicMock()
    retriever.search.return_value = {
        "query": "What is photosynthesis?",
        "intent": QueryType.DEFINITION,
        "chunks": [_make_chunk()],
        "in_scope": True,
        "is_relevant": True,
        "relevance_score": 0.9,
        "total_retrieved": 1,
    }
    # Empty by default so ContextRouter.decide() sees no chunks and falls
    # back to RAG (today's behavior) for every test that doesn't explicitly
    # opt into exercising the CAG path — see test_context_router.py and the
    # dedicated CAG-path orchestrator tests for cases that override this.
    retriever.get_full_context.return_value = {
        "query": None,
        "intent": None,
        "chunks": [],
        "in_scope": False,
        "is_relevant": False,
        "relevance_score": 0.0,
        "total_retrieved": 0,
    }
    return retriever


@pytest.fixture()
def orchestrator(mock_intent_classifier, mock_llm, mock_retriever):
    history: list[ChatMessage] = []
    return OrchestratorAgent(
        intent_classifier=mock_intent_classifier,
        retrieval_agent=RetrievalAgent(mock_retriever),
        knowledge_agent=KnowledgeAgent(mock_llm),
        reflection_agent=ReflectionAgent(mock_llm),
        memory_agent=MemoryAgent(history),
    )


def test_orchestrator_happy_path(orchestrator):
    result = orchestrator.run("What is photosynthesis?")
    assert isinstance(result, AnswerWithSources)
    assert result.query_intent == QueryType.DEFINITION
    assert result.format_type == "definition"
    assert len(result.sources) == 1
    assert orchestrator.memory_agent.chat_history[0].role == "user"
    assert orchestrator.memory_agent.chat_history[1].content == result.answer


def test_orchestrator_uses_cag_path_when_scope_fits_the_budget(orchestrator, mock_retriever, mock_llm):
    """A small, in-budget document scope must skip ranked search() entirely
    and hand the drafting LLM every chunk get_full_context() returns, in the
    order it returns them — not a similarity-ranked/truncated subset."""
    # First chunk matches mock_llm's default claim text so citation matching
    # succeeds (keeps this test clear of the hallucination-risk gate, which
    # has its own dedicated tests) — the other two exist purely to prove
    # multiple chunks pass through untouched.
    all_chunks = [
        _make_chunk("Photosynthesis converts light into chemical energy."),
        _make_chunk("Chunk two."),
        _make_chunk("Chunk three."),
    ]
    mock_retriever.get_full_context.return_value = {
        "query": None,
        "intent": None,
        "chunks": all_chunks,
        "in_scope": True,
        "is_relevant": True,
        "relevance_score": 1.0,
        "total_retrieved": 3,
    }

    result = orchestrator.run("What is photosynthesis?", document_ids=["doc-1"])

    mock_retriever.search.assert_not_called()
    mock_retriever.get_full_context.assert_called_once_with(["doc-1"])
    draft_args, _ = mock_llm.generate_structured_answer.call_args
    assert draft_args[1] == all_chunks  # exactly what get_full_context returned, unranked/untruncated
    assert result.format_type == "definition"


def test_orchestrator_falls_back_to_rag_when_scope_exceeds_budget(orchestrator, mock_retriever, monkeypatch):
    """An oversized document scope (per ContextRouter's token budget) must
    fall back to the existing ranked search() path unchanged, proving Phase 1
    is purely additive rather than replacing RAG outright."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "cag_token_budget", 1)
    mock_retriever.get_full_context.return_value = {
        "chunks": [_make_chunk("A chunk with several words in it.")],
        "is_relevant": True,
    }

    orchestrator.run("What is photosynthesis?", document_ids=["doc-1"])

    mock_retriever.get_full_context.assert_called_once_with(["doc-1"])
    mock_retriever.search.assert_called_once()


def test_orchestrator_threads_explicit_format_directive_through_the_pipeline(orchestrator, mock_llm):
    """ "in 5 key points" must classify as KEY_POINTS regardless of intent
    (DEFINITION here, from the fixture) and reach both the drafting and
    reflection LLM calls — proving format is a genuinely separate axis from
    intent, not just relabeled intent."""
    result = orchestrator.run("Explain photosynthesis in 5 key points")

    assert result.format_type == "key_points"
    mock_llm.generate_structured_answer.assert_called_once()
    _, kwargs = mock_llm.generate_structured_answer.call_args
    assert kwargs["response_format"] == ResponseFormat.KEY_POINTS
    mock_llm.reflect_on_answer.assert_called_once()
    _, reflect_kwargs = mock_llm.reflect_on_answer.call_args
    assert reflect_kwargs["response_format"] == ResponseFormat.KEY_POINTS


def test_orchestrator_applies_response_formatting_to_the_final_answer(orchestrator, mock_llm):
    """A ONE_LINE-format request must come back truncated to one sentence —
    proving the Stage 7 formatting pass actually runs on the real pipeline,
    not just in response_formatter.py's own unit tests. The claim text must
    stay on-topic with the fixture chunk's content ("Photosynthesis converts
    light into chemical energy.") so it actually gets cited and clears the
    hallucination-risk gate — an off-topic claim would get hard-blocked
    before formatting ever runs, which is a different code path already
    covered by other tests."""
    answer = (
        "Photosynthesis converts light into chemical energy. It occurs in chloroplasts. "
        "It produces oxygen as a byproduct."
    )
    mock_llm.generate_structured_answer.return_value = {
        "answer": answer,
        "format_type": "one_line",
        "claims": ["Photosynthesis converts light into chemical energy."],
        "intent": "definition",
    }
    mock_llm.reflect_on_answer.return_value = _reflection_no_change(answer)

    result = orchestrator.run("One line answer: what is photosynthesis?")

    assert result.format_type == "one_line"
    assert result.answer == "Photosynthesis converts light into chemical energy."


def test_orchestrator_out_of_scope_short_circuits_before_llm(orchestrator, mock_retriever, mock_llm):
    mock_retriever.search.return_value = {
        "query": "irrelevant",
        "intent": QueryType.VAGUE,
        "chunks": [],
        "in_scope": False,
        "is_relevant": False,
        "relevance_score": 0.0,
        "total_retrieved": 0,
    }
    stages = []
    result = orchestrator.run(
        "Something totally unrelated", on_stage=lambda stage, payload: stages.append((stage, payload))
    )
    assert result.format_type == "out_of_scope"
    assert result.hallucination_risk == "high"
    assert result.answer.startswith("I couldn't find this information in the uploaded documents.")
    mock_llm.generate_structured_answer.assert_not_called()
    mock_llm.reflect_on_answer.assert_not_called()
    # Out-of-scope streams to the client exactly like every other path.
    assert stages == [("answer_ready", {"answer": result.answer})]


def test_orchestrator_reflection_materially_changed_triggers_revalidation(orchestrator, mock_llm):
    """Reflection returns its own claims for the revised answer in the same
    call — no separate re-extraction round trip. Proven here via claim
    count: the draft's mock always produces exactly one claim, so two
    resulting citations can only happen if reflection's own two-claim list
    was actually used for scoring, not silently replaced by the draft's
    (single-claim) list. Both claims are well-grounded (matching the only
    chunk) so this stays well clear of the hallucination-risk gate — that
    gate has its own dedicated test below."""
    mock_llm.reflect_on_answer.return_value = {
        "revised_answer": "Revised: photosynthesis converts light into stored chemical energy.",
        "materially_changed": True,
        "should_block": False,
        "issues_found": ["clarified wording"],
        "claims": [
            {"claim": "Photosynthesis converts light into chemical energy.", "chunks": [1]},
            {"claim": "Light energy is converted into chemical energy by photosynthesis.", "chunks": [1]},
        ],
    }

    result = orchestrator.run("What is photosynthesis?")

    assert result.answer == "Revised: photosynthesis converts light into stored chemical energy."
    assert len(result.sources) == 2


def test_orchestrator_reflection_should_block_returns_fixed_fallback(orchestrator, mock_llm):
    mock_llm.reflect_on_answer.return_value = {
        "revised_answer": "",
        "materially_changed": True,
        "should_block": True,
        "issues_found": ["hallucinated throughout"],
    }

    stages = []
    result = orchestrator.run(
        "What is photosynthesis?", on_stage=lambda stage, payload: stages.append((stage, payload))
    )

    assert result.format_type == "reflection_blocked"
    assert result.hallucination_risk == "high"
    assert result.sources == []
    # The user never saw the blocked draft — only this fixed fallback streams.
    assert stages == [("answer_ready", {"answer": result.answer})]


def test_orchestrator_hard_gates_on_high_hallucination_risk_even_without_should_block(
    orchestrator, mock_llm, mock_retriever
):
    """The deterministic gate must fire even when the reflection LLM itself
    saw nothing wrong (should_block=False) — it's a non-LLM backstop against
    exactly the case where the model that could hallucinate the answer also
    hallucinates a clean self-assessment."""
    mock_llm.reflect_on_answer.return_value = {
        "revised_answer": "Photosynthesis also explains volcanic activity on Io.",
        "materially_changed": True,
        "should_block": False,
        "issues_found": [],
        "claims": [{"claim": "A completely unrelated claim about volcanoes.", "chunks": []}],
    }

    result = orchestrator.run("What is photosynthesis?")

    assert result.format_type == "reflection_blocked"
    assert result.hallucination_risk == "high"
    assert result.sources == []
    assert "couldn't verify" in result.answer
    mock_retriever.search.assert_called_once()  # only ran retrieval once, no retry


def test_orchestrator_reflection_shortcut_skips_reflection_when_draft_is_hopeless(
    mock_intent_classifier, mock_retriever
):
    """When the draft finds zero supporting citations AND confidence is
    already far below the shortcut floor, the expensive reflection round
    trip must be skipped entirely — proven via reflect_on_answer never being
    called, not just via the final answer's shape (which looks identical to
    the post-reflection hallucination-risk gate's fallback)."""
    llm = MagicMock()
    llm.generate_structured_answer.return_value = {
        "answer": "An unrelated draft answer.",
        "format_type": "definition",
        "claims": ["A completely unrelated claim."],
        "intent": "definition",
    }
    span_extractor = MagicMock()
    span_extractor.extract_supporting_span.return_value = None  # citation_rate == 0
    confidence_scorer = MagicMock()
    confidence_scorer.calculate_answer_confidence.return_value = 0.05  # far below the 0.15 floor
    confidence_scorer.assess_hallucination_risk.return_value = "high"

    history: list[ChatMessage] = []
    orch = OrchestratorAgent(
        intent_classifier=mock_intent_classifier,
        retrieval_agent=RetrievalAgent(mock_retriever),
        knowledge_agent=KnowledgeAgent(llm),
        reflection_agent=ReflectionAgent(llm),
        memory_agent=MemoryAgent(history),
        span_extractor=span_extractor,
        confidence_scorer=confidence_scorer,
    )

    stages = []
    result = orch.run(
        "What is photosynthesis?", on_stage=lambda stage, payload: stages.append((stage, payload))
    )

    assert result.format_type == "reflection_blocked"
    assert result.hallucination_risk == "high"
    assert result.overall_confidence == 0.05
    assert "couldn't verify" in result.answer
    llm.reflect_on_answer.assert_not_called()  # the whole point: skipped the second LLM round trip
    assert stages == [("answer_ready", {"answer": result.answer})]
    assert (
        orch.memory_agent.chat_history[-1].content == result.answer
    )  # still recorded, same as the post-reflection gate


def test_orchestrator_reflection_shortcut_does_not_fire_above_confidence_floor(orchestrator, mock_llm):
    """Zero citations alone must NOT skip reflection — only zero citations
    *combined with* confidence already far below the shortcut floor. Here the
    single retrieved chunk's relevance_score (0.9, from the fixture) alone
    puts confidence at 0.56 even with citation_rate=0 (see
    calculate_answer_confidence's 0.4*avg_relevance + 0.2*source_consistency
    floor), comfortably above the 0.15 shortcut floor — reflection must still
    get its real chance to rescue this draft, matching the existing
    hallucination-risk-gate test's premise that reflection commonly does."""
    mock_llm.generate_structured_answer.return_value = {
        "answer": "An unrelated draft answer.",
        "format_type": "definition",
        "claims": ["A completely unrelated claim about volcanoes."],
        "intent": "definition",
    }

    orchestrator.run("What is photosynthesis?")

    mock_llm.reflect_on_answer.assert_called_once()


def test_orchestrator_reflection_claims_empty_list_scores_as_zero_not_draft_fallback(orchestrator, mock_llm):
    """An explicit empty claims list from reflection (it ran and genuinely
    found nothing traceable) must score as zero citations — not be confused
    with claims=None (reflection failed/missing) and silently fall back to
    reusing the draft's own (different-answer's) claims."""
    mock_llm.reflect_on_answer.return_value = {
        "revised_answer": "Revised: a purely stylistic rewrite with no new facts.",
        "materially_changed": True,
        "should_block": False,
        "issues_found": [],
        "claims": [],
    }

    result = orchestrator.run("What is photosynthesis?")

    # Zero claims -> num_claims=0 -> ConfidenceScorer's citation_rate defaults
    # to 0.5 (see calculate_answer_confidence), landing as "medium" risk, not
    # "high" — confirms the hard gate did NOT fire and this is genuinely
    # scored as zero claims, not silently replaced by the draft's claims.
    assert result.sources == []
    assert result.format_type != "reflection_blocked"
    assert result.hallucination_risk == "medium"


def test_orchestrator_handles_llm_failure_gracefully(orchestrator, mock_llm):
    mock_llm.generate_structured_answer.side_effect = RuntimeError("provider unavailable")
    stages = []
    result = orchestrator.run(
        "What is photosynthesis?", on_stage=lambda stage, payload: stages.append((stage, payload))
    )
    assert result.format_type == "error"
    assert result.hallucination_risk == "high"
    assert result.overall_confidence == 0.0
    assert stages == [("answer_ready", {"answer": result.answer})]


def test_orchestrator_invokes_on_stage_callback(orchestrator):
    """Drafting and reflection both happen silently — on_stage only ever
    fires once, carrying the final, already-reflected answer text."""
    stages = []
    orchestrator.run("What is photosynthesis?", on_stage=lambda stage, payload: stages.append(stage))
    assert stages == ["answer_ready"]


def test_orchestrator_short_circuits_small_talk_before_retrieval(orchestrator, mock_retriever, mock_llm):
    stages = []
    result = orchestrator.run("hi", on_stage=lambda stage, payload: stages.append(stage))

    assert result.format_type == "greeting"
    assert result.sources == []
    # Never reached retrieval/drafting/reflection, but the single
    # answer_ready event still fires so every response path streams the
    # same way to the client.
    assert stages == ["answer_ready"]
    mock_retriever.search.assert_not_called()
    mock_llm.generate_structured_answer.assert_not_called()
    mock_llm.reflect_on_answer.assert_not_called()
    assert orchestrator.memory_agent.chat_history[0].content == "hi"
    assert orchestrator.memory_agent.chat_history[1].content == result.answer


def test_orchestrator_forwards_session_id_and_device_id_to_memory_agent(
    mock_intent_classifier, mock_llm, mock_retriever
):
    memory_agent = MagicMock()
    orchestrator = OrchestratorAgent(
        intent_classifier=mock_intent_classifier,
        retrieval_agent=RetrievalAgent(mock_retriever),
        knowledge_agent=KnowledgeAgent(mock_llm),
        reflection_agent=ReflectionAgent(mock_llm),
        memory_agent=memory_agent,
    )

    result = orchestrator.run("What is photosynthesis?", session_id="sess-1", device_id="device-1")

    memory_agent.record_turn.assert_called_once_with(
        "What is photosynthesis?",
        result.answer,
        QueryType.DEFINITION,
        format_type="definition",
        session_id="sess-1",
        device_id="device-1",
    )


def test_orchestrator_forwards_document_ids_to_retrieval_agent(mock_intent_classifier, mock_llm):
    retrieval_agent = MagicMock()
    retrieval_agent.search.return_value = {
        "query": "What is photosynthesis?",
        "intent": QueryType.DEFINITION,
        "chunks": [_make_chunk()],
        "in_scope": True,
        "is_relevant": True,
        "relevance_score": 0.9,
        "total_retrieved": 1,
    }
    # Empty so ContextRouter.decide() falls back to RAG — this test is
    # specifically about the RAG search() call, not CAG routing.
    retrieval_agent.get_full_context.return_value = {"chunks": [], "is_relevant": False}
    orchestrator = OrchestratorAgent(
        intent_classifier=mock_intent_classifier,
        retrieval_agent=retrieval_agent,
        knowledge_agent=KnowledgeAgent(mock_llm),
        reflection_agent=ReflectionAgent(mock_llm),
        memory_agent=MemoryAgent([]),
    )

    orchestrator.run("What is photosynthesis?", document_ids=["doc-1", "doc-2"])

    retrieval_agent.search.assert_called_once_with(
        "What is photosynthesis?", QueryType.DEFINITION, document_ids=["doc-1", "doc-2"], session_id=None
    )


def test_orchestrator_threads_recent_history_into_draft_and_reflection(
    mock_intent_classifier, mock_llm, mock_retriever
):
    """Follow-up questions ("explain that more") need the drafting and
    reflection LLM calls to actually see recent turns — not just chat_history
    being recorded and never read anywhere."""
    memory_agent = MagicMock()
    fake_history = [MagicMock(role="user", content="What is photosynthesis?")]
    memory_agent.get_recent_turns.return_value = fake_history
    knowledge_agent = MagicMock()
    knowledge_agent.generate.return_value = {
        "answer": "Photosynthesis converts light into chemical energy.",
        "format_type": "definition",
        "claims": ["Photosynthesis converts light into chemical energy."],
        "intent": "definition",
    }
    reflection_agent = MagicMock()
    reflection_agent.reflect.return_value = {
        "revised_answer": "Photosynthesis converts light into chemical energy.",
        "materially_changed": False,
        "should_block": False,
        "issues_found": [],
        "claims": None,
    }
    orchestrator = OrchestratorAgent(
        intent_classifier=mock_intent_classifier,
        retrieval_agent=RetrievalAgent(mock_retriever),
        knowledge_agent=knowledge_agent,
        reflection_agent=reflection_agent,
        memory_agent=memory_agent,
    )

    orchestrator.run("Explain that more", session_id="sess-1")

    memory_agent.get_recent_turns.assert_called_once_with("sess-1")
    assert knowledge_agent.generate.call_args.kwargs["history"] is fake_history
    assert reflection_agent.reflect.call_args.kwargs["history"] is fake_history
    # Retrieval itself must also see the prior question — "Explain that more"
    # alone rarely shares enough vocabulary with the right chunks to match at
    # all; the draft LLM having history doesn't help if retrieval found
    # nothing relevant to begin with.
    retrieval_query = mock_retriever.search.call_args[0][0]
    assert retrieval_query == "What is photosynthesis? Explain that more"


def test_orchestrator_retrieval_query_unaugmented_without_history(
    mock_intent_classifier, mock_llm, mock_retriever
):
    """No prior turns (a fresh conversation, or persistence not configured)
    -> retrieval sees the query exactly as asked, unchanged."""
    orchestrator = OrchestratorAgent(
        intent_classifier=mock_intent_classifier,
        retrieval_agent=RetrievalAgent(mock_retriever),
        knowledge_agent=KnowledgeAgent(mock_llm),
        reflection_agent=ReflectionAgent(mock_llm),
        memory_agent=MemoryAgent([]),  # persist=False -> get_recent_turns always []
    )

    orchestrator.run("What is photosynthesis?")

    retrieval_query = mock_retriever.search.call_args[0][0]
    assert retrieval_query == "What is photosynthesis?"


def test_orchestrator_resolves_document_ids_from_db_when_session_id_given(
    monkeypatch, mock_intent_classifier, mock_llm
):
    """With a db_session_factory configured, the orchestrator resolves the
    conversation's authoritative document scope instead of trusting the
    caller-supplied document_ids verbatim — see resolve_document_scope."""
    import app.services.agents.orchestrator as orchestrator_module

    resolve_mock = MagicMock(return_value=["doc-server-1"])
    monkeypatch.setattr(orchestrator_module, "resolve_document_scope", resolve_mock)

    retrieval_agent = MagicMock()
    retrieval_agent.search.return_value = {
        "query": "What is photosynthesis?",
        "intent": QueryType.DEFINITION,
        "chunks": [_make_chunk()],
        "in_scope": True,
        "is_relevant": True,
        "relevance_score": 0.9,
        "total_retrieved": 1,
    }
    # Empty so ContextRouter.decide() falls back to RAG — this test is
    # specifically about the RAG search() call, not CAG routing.
    retrieval_agent.get_full_context.return_value = {"chunks": [], "is_relevant": False}
    fake_db_session_factory = MagicMock()
    orchestrator = OrchestratorAgent(
        intent_classifier=mock_intent_classifier,
        retrieval_agent=retrieval_agent,
        knowledge_agent=KnowledgeAgent(mock_llm),
        reflection_agent=ReflectionAgent(mock_llm),
        memory_agent=MemoryAgent([]),
        db_session_factory=fake_db_session_factory,
    )

    orchestrator.run("What is photosynthesis?", session_id="sess-1", document_ids=["client-side-stale-id"])

    resolve_mock.assert_called_once_with(fake_db_session_factory, "sess-1", ["client-side-stale-id"])
    retrieval_agent.search.assert_called_once_with(
        "What is photosynthesis?", QueryType.DEFINITION, document_ids=["doc-server-1"], session_id="sess-1"
    )


def test_orchestrator_global_search_request_overrides_scope_to_none(
    monkeypatch, mock_intent_classifier, mock_llm
):
    """An explicit 'search across all my documents'-style query bypasses
    conversation scoping entirely, even when a session_id/db factory would
    otherwise resolve a narrower scope."""
    import app.services.agents.orchestrator as orchestrator_module

    resolve_mock = MagicMock(return_value=["doc-server-1"])
    monkeypatch.setattr(orchestrator_module, "resolve_document_scope", resolve_mock)

    retrieval_agent = MagicMock()
    retrieval_agent.search.return_value = {
        "query": "search across all my documents for X",
        "intent": QueryType.DEFINITION,
        "chunks": [_make_chunk()],
        "in_scope": True,
        "is_relevant": True,
        "relevance_score": 0.9,
        "total_retrieved": 1,
    }
    orchestrator = OrchestratorAgent(
        intent_classifier=mock_intent_classifier,
        retrieval_agent=retrieval_agent,
        knowledge_agent=KnowledgeAgent(mock_llm),
        reflection_agent=ReflectionAgent(mock_llm),
        memory_agent=MemoryAgent([]),
        db_session_factory=MagicMock(),
    )

    orchestrator.run("search across all my documents for X", session_id="sess-1")

    resolve_mock.assert_not_called()
    retrieval_agent.search.assert_called_once_with(
        "search across all my documents for X", QueryType.DEFINITION, document_ids=None, session_id="sess-1"
    )
