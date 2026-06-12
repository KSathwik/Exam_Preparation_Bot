"""Tests for document parser utilities (no real files needed)."""

import pytest
from app.services.parser import _split_sentences, _make_chunk


def test_split_sentences_basic():
    text = "First sentence. Second sentence! Third sentence?"
    result = _split_sentences(text)
    assert len(result) == 3
    assert result[0] == "First sentence."


def test_split_sentences_empty():
    assert _split_sentences("") == []


def test_split_sentences_no_punctuation():
    result = _split_sentences("Just some text without a period")
    assert len(result) == 1


def test_make_chunk_beginning_position():
    chunk = _make_chunk("Some content", "test.pdf", chunk_index=0, total_chunks=10, total_pages=5)
    assert chunk.metadata.chunk_position == "beginning"
    assert chunk.metadata.file_name == "test.pdf"


def test_make_chunk_end_position():
    chunk = _make_chunk("Some content", "test.pdf", chunk_index=9, total_chunks=10, total_pages=5)
    assert chunk.metadata.chunk_position == "end"


def test_make_chunk_middle_position():
    chunk = _make_chunk("Some content", "test.pdf", chunk_index=5, total_chunks=10, total_pages=5)
    assert chunk.metadata.chunk_position == "middle"


def test_make_chunk_unknown_when_few():
    chunk = _make_chunk("Content", "test.pdf", chunk_index=1, total_chunks=2, total_pages=2)
    assert chunk.metadata.chunk_position == "unknown"


def test_make_chunk_page_number_bounded():
    chunk = _make_chunk("Content", "test.pdf", chunk_index=99, total_chunks=100, total_pages=5)
    assert chunk.metadata.page_number <= 5
