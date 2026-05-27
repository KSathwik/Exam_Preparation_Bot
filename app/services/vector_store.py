"""Vector store manager for FastAPI"""

# Import from actual services
from app.services.embeddings import VectorStoreManager

# Create singleton instance
vector_store_manager = VectorStoreManager()

__all__ = ['vector_store_manager']
