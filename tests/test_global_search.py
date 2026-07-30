"""Tests for the explicit "search everywhere" opt-in detector — a real
question about the current conversation's own documents must never misfire
as a request to search across every conversation."""

import pytest

from app.services.global_search import is_global_search_request


@pytest.mark.parametrize(
    "query",
    [
        "Search across all my uploaded PDFs",
        "Compare all my notes",
        "Find this in every document",
        "Search my entire knowledge base",
        "search across all my files for interest rates",
    ],
)
def test_matches_global_search_requests(query):
    assert is_global_search_request(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "What is photosynthesis?",
        "Compare all the branches of government",
        "Explain every step of the DBMS transaction process",
        "What does this document say about interest rates?",
        "Summarize the resume",
        "",
    ],
)
def test_does_not_match_real_questions(query):
    assert is_global_search_request(query) is False
