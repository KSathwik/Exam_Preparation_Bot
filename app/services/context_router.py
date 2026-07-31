"""Deterministic RAG-vs-CAG routing.

Answers "should this turn get the whole document or a ranked subset" —
orthogonal to `IntentClassifier` (what the question is about) and
`FormatClassifier` (how the answer should look). No LLM involved, same
philosophy as `FormatClassifier`: the decision only needs a size comparison,
so there's nothing an LLM call would improve on.

RAG (similarity-ranked, truncated to top_k) is the right strategy for a large
corpus. CAG (every chunk of every scoped document, in reading order) is
strictly better whenever the whole scope is small enough to fit an LLM call's
context economically — no ranking-induced risk of dropping a chunk the answer
actually needed. This app's real usage skews heavily toward small documents
(single-digit chunk counts per upload — see the architecture-review context
this was built from), so CAG is the common case, not the exception.
"""

from enum import Enum
from typing import List

from loguru import logger

from app.core.config import settings

from .models import RetrievedChunk


class ContextMode(str, Enum):
    RAG = "rag"
    CAG = "cag"


def _estimate_tokens(text: str) -> int:
    """Word-count proxy — consistent with the estimator already used for
    memory-summarization triggers (see memory_agent.py's _estimate_tokens);
    not a real tokenizer, just a cheap, good-enough size signal."""
    return len(text.split())


class ContextRouter:
    """Decides RAG vs CAG for an already-fetched candidate chunk list.

    Takes chunks rather than document_ids so the caller (OrchestratorAgent)
    fetches the full scope exactly once — reused directly for the CAG path,
    discarded in favor of a ranked search() call for the RAG path — instead
    of this class re-fetching the same data independently.
    """

    def decide(self, chunks: List[RetrievedChunk]) -> ContextMode:
        if not settings.enable_cag or not chunks:
            # CAG disabled, or nothing indexed for this scope — either way,
            # the existing search()/out-of-scope path is the right fallback.
            return ContextMode.RAG

        total_tokens = sum(_estimate_tokens(c.content) for c in chunks)
        mode = ContextMode.CAG if total_tokens <= settings.cag_token_budget else ContextMode.RAG
        logger.info(
            f"[CONTEXT_ROUTER] chunks={len(chunks)}  estimated_tokens={total_tokens}  "
            f"budget={settings.cag_token_budget}  mode={mode.value}"
        )
        return mode
