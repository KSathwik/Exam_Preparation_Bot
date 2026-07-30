"""Tests for small-talk detection — greetings/thanks/farewells short-circuit
before the RAG pipeline runs; anything else must fall through untouched."""

import pytest

from app.services.small_talk import match_small_talk


@pytest.mark.parametrize("query", ["hi", "Hi!", "HIE", "hello.", "hey", "sup", "hiya"])
def test_matches_greetings(query):
    assert match_small_talk(query) is not None


@pytest.mark.parametrize("query", ["how are you", "How are you?", "what's up", "hows it going"])
def test_matches_how_are_you(query):
    assert match_small_talk(query) is not None


@pytest.mark.parametrize("query", ["thanks", "thank you!", "thx"])
def test_matches_thanks(query):
    assert match_small_talk(query) is not None


@pytest.mark.parametrize("query", ["bye", "goodbye.", "see ya"])
def test_matches_farewells(query):
    assert match_small_talk(query) is not None


@pytest.mark.parametrize(
    "query",
    [
        "What is photosynthesis?",
        "hi, what does this document say about interest rates?",
        "Explain the process in the document",
        "",
        "   ",
    ],
)
def test_does_not_match_real_questions(query):
    assert match_small_talk(query) is None
