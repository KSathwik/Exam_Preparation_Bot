"""Vector store singleton — use ``get_vector_store_manager()`` from dependencies instead."""

from app.core.dependencies import get_vector_store_manager

vector_store_manager = get_vector_store_manager()

__all__ = ["vector_store_manager"]
