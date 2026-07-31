"""Tests for the deterministic post-generation response formatter.

Deliberately narrow scope — see response_formatter.py's module docstring:
this only ever strips boilerplate and truncates length, never restructures
content, so tests stay focused on exactly those two behaviors.
"""

from app.services.response_formats import ResponseFormat
from app.services.response_formatter import format_response


def test_strips_certainly_preamble():
    text = "Certainly! Mitochondria produce ATP."
    assert format_response(text, ResponseFormat.GENERAL) == "Mitochondria produce ATP."


def test_strips_heres_the_preamble():
    text = "Here's the answer: entropy measures disorder."
    result = format_response(text, ResponseFormat.GENERAL)
    assert not result.lower().startswith("here")


def test_strips_based_on_document_preamble():
    text = "Based on the document, deadlocks occur when processes wait circularly."
    result = format_response(text, ResponseFormat.GENERAL)
    assert result == "Deadlocks occur when processes wait circularly."


def test_no_preamble_left_untouched():
    text = "Deadlocks occur when processes wait circularly."
    assert format_response(text, ResponseFormat.GENERAL) == text


def test_one_line_truncates_to_first_sentence():
    text = "A deadlock is a circular wait. It can be prevented. Detection is also possible."
    result = format_response(text, ResponseFormat.ONE_LINE)
    assert result == "A deadlock is a circular wait."


def test_one_line_leaves_single_sentence_untouched():
    text = "A deadlock is a circular wait."
    assert format_response(text, ResponseFormat.ONE_LINE) == text


def test_two_mark_truncates_to_three_sentences():
    text = "One. Two. Three. Four. Five."
    result = format_response(text, ResponseFormat.TWO_MARK)
    assert result == "One. Two. Three."


def test_two_mark_leaves_shorter_answer_untouched():
    text = "One. Two."
    assert format_response(text, ResponseFormat.TWO_MARK) == text


def test_key_points_format_does_not_truncate():
    """Only ONE_LINE/TWO_MARK have a sentence cap — a bulleted list format
    must never get chopped mid-list by the same mechanism."""
    text = "- Point one about the topic.\n- Point two about the topic.\n- Point three about the topic."
    assert format_response(text, ResponseFormat.KEY_POINTS) == text


def test_combines_preamble_stripping_and_truncation():
    text = "Certainly! A deadlock is a circular wait. It can be prevented."
    result = format_response(text, ResponseFormat.ONE_LINE)
    assert result == "A deadlock is a circular wait."
