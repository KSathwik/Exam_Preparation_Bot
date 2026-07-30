"""Explicit "search everywhere" detection.

Retrieval defaults to strict per-conversation scope (see
``conversation_scope.resolve_document_scope``) — documents from other
conversations are never searched unless the user explicitly asks for it.
Matched here as a small set of phrase patterns, the same narrow/conservative
philosophy as ``small_talk.match_small_talk``: a real in-scope question must
never misfire as a request to search every conversation's documents.
"""

import re

_DOC_NOUN = r"(documents?|pdfs?|files?|notes?|uploads?)"

_GLOBAL_SEARCH_PATTERNS = [
    rf"\bacross all\b.*\b{_DOC_NOUN}\b",
    rf"\bevery\b.*\b{_DOC_NOUN}\b",
    rf"\ball my\b.*\b{_DOC_NOUN}\b",
    r"\b(entire|whole) knowledge base\b",
    rf"\bcompare all\b.*\b{_DOC_NOUN}\b",
    rf"\bsearch (my )?(entire|whole)\b.*\b(knowledge base|{_DOC_NOUN})\b",
]

_GLOBAL_SEARCH_RE = re.compile("|".join(_GLOBAL_SEARCH_PATTERNS), re.IGNORECASE)


def is_global_search_request(query: str) -> bool:
    """Return True if ``query`` explicitly asks to search beyond the current
    conversation's documents (e.g. "search across all my documents",
    "compare all my notes", "find this in every document")."""
    return bool(_GLOBAL_SEARCH_RE.search(query))
