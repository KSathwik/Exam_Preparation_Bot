"""ReflectionAgent — always-on quality-control pass wrapping `_BaseLLM.reflect_on_answer`."""

from typing import List

from loguru import logger

from ..models import QueryType, RetrievedChunk


class ReflectionAgent:
    """Delegates to an `_BaseLLM`-like object, stored by reference (see `RetrievalAgent`).

    `_BaseLLM.reflect_on_answer` already degrades to a safe fallback on any
    provider/parse failure — this wrapper adds one more layer of defense so an
    unexpected exception (e.g. from a test double or a future LLM
    implementation) can never break the pipeline stage that calls it.
    """

    def __init__(self, llm):
        self.llm = llm

    def reflect(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        draft_answer: str,
        validator_summary: str,
        intent: QueryType,
    ) -> dict:
        try:
            return self.llm.reflect_on_answer(
                query, retrieved_chunks, draft_answer, validator_summary, intent
            )
        except Exception as exc:
            logger.warning(
                f"[REFLECTION] reflect_on_answer raised, falling back to draft: {type(exc).__name__}: {exc}"
            )
            return {
                "revised_answer": draft_answer,
                "materially_changed": False,
                "should_block": False,
                "issues_found": [],
            }
