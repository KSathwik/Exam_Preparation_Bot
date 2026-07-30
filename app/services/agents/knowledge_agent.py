"""KnowledgeAgent — thin wrapper over the LLM's structured-answer generation."""

from typing import List

from ..models import QueryType, RetrievedChunk


class KnowledgeAgent:
    """Delegates to an `_BaseLLM`-like object, stored by reference (see `RetrievalAgent`)."""

    def __init__(self, llm):
        self.llm = llm

    def generate(self, query: str, chunks: List[RetrievedChunk], intent: QueryType) -> dict:
        return self.llm.generate_structured_answer(query, chunks, intent)
