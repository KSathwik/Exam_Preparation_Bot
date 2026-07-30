"""RetrievalAgent — thin wrapper over HybridRetriever for the multi-agent pipeline."""

from typing import List, Optional

from ..models import QueryType
from ..retriever import HybridRetriever


class RetrievalAgent:
    """Delegates to a `HybridRetriever` instance, stored by reference.

    Storing the retriever object (not a bound `.search` method) keeps
    `monkeypatch.setattr(bot.retriever, "search", ...)` effective in tests
    that patch the retriever after construction.
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def search(
        self,
        query: str,
        intent: QueryType,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        return self.retriever.search(query, intent, top_k, document_ids=document_ids, session_id=session_id)
