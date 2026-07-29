"""Tests for the intent classification module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture()
def classifier(mock_sentence_transformer):
    with patch(
        "app.services.intent_classifier.SentenceTransformer",
        return_value=mock_sentence_transformer,
    ):
        from app.services.intent_classifier import IntentClassifier

        return IntentClassifier(model=mock_sentence_transformer)


def test_rule_based_definition(classifier):
    result = classifier.classify("What is the definition of photosynthesis?")
    assert result.primary_intent.value == "definition"
    assert result.confidence > 0


def test_rule_based_compare(classifier):
    result = classifier.classify("Compare mitosis and meiosis")
    assert result.primary_intent.value == "compare"


def test_rule_based_process(classifier):
    result = classifier.classify("What are the steps of cell division?")
    assert result.primary_intent.value == "process"


def test_rule_based_example(classifier):
    result = classifier.classify("Give me an example of osmosis")
    assert result.primary_intent.value == "example"


def test_semantic_fallback(classifier):
    """A query with no keywords should still get a classification via semantics."""
    result = classifier.classify("XYZ arbitrary text")
    assert result.primary_intent is not None
    assert 0 <= result.confidence <= 1


def test_classification_prompt_returned(classifier):
    from app.services.models import QueryType

    prompt = classifier.get_classification_prompt(QueryType.DEFINITION)
    assert "definition" in prompt.lower()

    prompt_fallback = classifier.get_classification_prompt(QueryType.HOMEWORK)
    assert len(prompt_fallback) > 0


def test_alternative_intents_populated(classifier):
    result = classifier.classify("Define the term osmosis")
    assert result.alternative_intents is not None
    assert len(result.alternative_intents) > 0
