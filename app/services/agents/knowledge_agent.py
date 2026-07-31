"""KnowledgeAgent — thin wrapper over the LLM's structured-answer generation."""

from typing import List, Optional

from ..models import ChatMessage, QueryType, RetrievedChunk
from ..response_formats import ResponseFormat


class KnowledgeAgent:
    """Delegates to an `_BaseLLM`-like object, stored by reference (see `RetrievalAgent`)."""

    def __init__(self, llm):
        self.llm = llm

    def generate(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        intent: QueryType,
        response_format: ResponseFormat = ResponseFormat.GENERAL,
        history: Optional[List[ChatMessage]] = None,
    ) -> dict:
        return self.llm.generate_structured_answer(
            query, chunks, intent, response_format=response_format, history=history
        )
