"""Tests for the deterministic response-format classifier.

No embeddings/LLM involved (unlike IntentClassifier) — FormatClassifier is
pure keyword matching + an intent-based default, so these tests need no
mocking at all.
"""

import pytest

from app.services.format_classifier import FormatClassifier
from app.services.models import QueryType
from app.services.response_formats import ResponseFormat


@pytest.fixture()
def classifier():
    return FormatClassifier()


# ── Explicit format directives override the intent's default ───────────


def test_key_points_directive(classifier):
    result = classifier.classify("Explain photosynthesis in 5 key points", QueryType.EXPLAIN)
    assert result == ResponseFormat.KEY_POINTS


def test_summary_directive(classifier):
    assert classifier.classify("Summarize this chapter", QueryType.VAGUE) == ResponseFormat.SUMMARY
    assert classifier.classify("TL;DR on entropy", QueryType.VAGUE) == ResponseFormat.SUMMARY


def test_one_line_directive(classifier):
    result = classifier.classify("One line answer: what is a deadlock?", QueryType.DEFINITION)
    assert result == ResponseFormat.ONE_LINE


def test_mark_based_directives(classifier):
    assert classifier.classify("Two mark answer on osmosis", QueryType.EXPLAIN) == ResponseFormat.TWO_MARK
    assert classifier.classify("5 mark question on mitosis", QueryType.EXPLAIN) == ResponseFormat.FIVE_MARK
    assert (
        classifier.classify("Ten-mark answer on the water cycle", QueryType.EXPLAIN)
        == ResponseFormat.TEN_MARK
    )


def test_comparison_directive_overrides_non_compare_intent(classifier):
    """Even when intent classification didn't land on COMPARE, an explicit
    "vs"/"difference between" directive must still win — this is the whole
    point of a separate presentation axis."""
    result = classifier.classify("TCP vs UDP", QueryType.EXPLAIN)
    assert result == ResponseFormat.COMPARISON


def test_flashcards_and_mcq_directives(classifier):
    assert (
        classifier.classify("Create flashcards for this chapter", QueryType.VAGUE)
        == ResponseFormat.FLASHCARDS
    )
    assert (
        classifier.classify("Generate a quiz on operating systems", QueryType.HOMEWORK) == ResponseFormat.MCQ
    )


def test_bare_quiz_word_triggers_mcq(classifier):
    """"Prepare me a quiz" — a very natural student phrasing — found via live
    testing to not match any prior MCQ trigger ("quiz me"/"generate a quiz"),
    falling through to the HOMEWORK-intent default (EXAM_QUESTIONS, a
    study-guide-style list) instead of an actual quiz. Any mention of the
    bare word "quiz" must trigger MCQ regardless of the surrounding verb."""
    assert classifier.classify("Prepare me a quiz", QueryType.HOMEWORK) == ResponseFormat.MCQ
    assert classifier.classify("Can you make a quiz on this?", QueryType.VAGUE) == ResponseFormat.MCQ


def test_simple_explanation_directive(classifier):
    result = classifier.classify("Explain simply what mitosis is", QueryType.EXPLAIN)
    assert result == ResponseFormat.SIMPLE_EXPLANATION


def test_revision_notes_directive(classifier):
    result = classifier.classify("Give me revision notes on deadlocks", QueryType.EXPLAIN)
    assert result == ResponseFormat.REVISION_NOTES


# ── No directive -> falls back to the intent's default format ──────────


@pytest.mark.parametrize(
    "intent,expected_default",
    [
        (QueryType.DEFINITION, ResponseFormat.DEFINITION),
        (QueryType.EXPLAIN, ResponseFormat.DETAILED_EXPLANATION),
        (QueryType.COMPARE, ResponseFormat.COMPARISON),
        (QueryType.PROCESS, ResponseFormat.STEPS),
        (QueryType.EXAMPLE, ResponseFormat.DETAILED_EXPLANATION),
        (QueryType.DIAGRAM, ResponseFormat.FLOWCHART),
        (QueryType.VAGUE, ResponseFormat.GENERAL),
        (QueryType.HOMEWORK, ResponseFormat.EXAM_QUESTIONS),
    ],
)
def test_intent_default_fallback(classifier, intent, expected_default):
    result = classifier.classify("some query with no format keywords at all", intent)
    assert result == expected_default


# ── False-positive avoidance ─────────────────────────────────────────────


def test_substring_of_define_does_not_false_positive(classifier):
    """ "well-defined"/"definition" as an incidental word must not force
    DEFINITION format on an unrelated question — only an explicit "define "/
    "definition of" directive should."""
    result = classifier.classify("This document is well-defined and complete", QueryType.VAGUE)
    assert result == ResponseFormat.GENERAL


def test_definition_directive_still_matches(classifier):
    assert classifier.classify("Define entropy", QueryType.VAGUE) == ResponseFormat.DEFINITION
    assert (
        classifier.classify("What is the meaning of entropy?", QueryType.VAGUE) == ResponseFormat.DEFINITION
    )


def test_classification_is_case_insensitive(classifier):
    result = classifier.classify("SUMMARIZE THIS FOR ME", QueryType.VAGUE)
    assert result == ResponseFormat.SUMMARY
