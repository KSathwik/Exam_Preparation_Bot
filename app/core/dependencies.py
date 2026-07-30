"""
Dependency injection for heavy resources.

SentenceTransformer models and the FAISS vector store are expensive to
initialise.  This module loads them once at import time and exposes
FastAPI-compatible ``Depends`` callables so every request handler shares
the same instances.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from loguru import logger
from sentence_transformers import SentenceTransformer

from app.core.config import settings

if TYPE_CHECKING:
    from app.services.embeddings import VectorStoreManager
    from app.services.intent_classifier import IntentClassifier
    from app.services.pipeline import ExamPrepBot


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer used for document/query embeddings."""
    logger.info(f"Loading embedding model (singleton): {settings.embedding_model}")
    return SentenceTransformer(settings.embedding_model)


@lru_cache(maxsize=1)
def get_intent_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer used for intent classification.

    When the intent model name equals the embedding model name the same
    object is returned — no memory wasted on a duplicate.
    """
    if settings.intent_embedding_model == settings.embedding_model:
        return get_embedding_model()
    logger.info(f"Loading intent model (singleton): {settings.intent_embedding_model}")
    return SentenceTransformer(settings.intent_embedding_model)


@lru_cache(maxsize=1)
def get_vector_store_manager() -> "VectorStoreManager":
    from app.services.embeddings import VectorStoreManager

    logger.info("Creating VectorStoreManager (singleton)")
    return VectorStoreManager(embedding_model=get_embedding_model())


@lru_cache(maxsize=1)
def get_intent_classifier() -> "IntentClassifier":
    from app.services.intent_classifier import IntentClassifier

    logger.info("Creating IntentClassifier (singleton)")
    return IntentClassifier(model=get_intent_model())


@lru_cache(maxsize=1)
def get_bot() -> "ExamPrepBot":
    from app.services.pipeline import ExamPrepBot

    logger.info("Creating ExamPrepBot (singleton)")
    return ExamPrepBot(
        vector_store_manager=get_vector_store_manager(),
        intent_classifier=get_intent_classifier(),
    )
