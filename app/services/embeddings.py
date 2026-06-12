"""Embedding generation and vector database management."""

import numpy as np
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from loguru import logger
import faiss
import pickle
from pathlib import Path
from .models import DocumentChunk, Document
from app.core.config import settings


class EmbeddingGenerator:
    """Generate embeddings for text chunks."""

    def __init__(self, model: Optional[SentenceTransformer] = None):
        """Initialise with an *already-loaded* model when possible (DI)."""
        if model is not None:
            self.model = model
        else:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self.model = SentenceTransformer(settings.embedding_model)

        self.model_name = settings.embedding_model
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        batch_size = batch_size or settings.batch_size
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def encode_single(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)


class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search."""

    def __init__(self, dimension: int = None, index_path: str = None):
        self.dimension = dimension or settings.vector_dimension
        self.index_path = index_path or settings.faiss_index_path
        self.index = None
        self.chunk_metadata: List[dict] = []
        self.embeddings: Optional[np.ndarray] = None
        Path(self.index_path).mkdir(parents=True, exist_ok=True)

    def create_index(self) -> None:
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunk_metadata = []
        self.embeddings = None

    def load_index(self) -> bool:
        index_file = Path(self.index_path) / "index.faiss"
        metadata_file = Path(self.index_path) / "metadata.pkl"
        embeddings_file = Path(self.index_path) / "embeddings.npy"

        if not index_file.exists():
            return False
        try:
            self.index = faiss.read_index(str(index_file))
            if metadata_file.exists():
                with open(metadata_file, "rb") as f:
                    self.chunk_metadata = pickle.load(f)
            if embeddings_file.exists():
                self.embeddings = np.load(embeddings_file)
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

    def save_index(self) -> None:
        if self.index is None:
            return
        index_file = Path(self.index_path) / "index.faiss"
        metadata_file = Path(self.index_path) / "metadata.pkl"
        embeddings_file = Path(self.index_path) / "embeddings.npy"

        faiss.write_index(self.index, str(index_file))
        with open(metadata_file, "wb") as f:
            pickle.dump(self.chunk_metadata, f)
        if self.embeddings is not None:
            np.save(embeddings_file, self.embeddings)

    def add_embeddings(self, embeddings: np.ndarray, metadata_list: List[dict]) -> None:
        if self.index is None:
            self.create_index()
        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.chunk_metadata.extend(metadata_list)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[float], List[int]]:
        if self.index is None or self.index.ntotal == 0:
            return [], []
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        return distances[0].tolist(), indices[0].tolist()

    def get_chunk_by_index(self, idx: int) -> Optional[dict]:
        if 0 <= idx < len(self.chunk_metadata):
            return self.chunk_metadata[idx]
        return None

    def clear(self) -> None:
        self.index = None
        self.chunk_metadata = []
        self.embeddings = None

    def get_size(self) -> int:
        return self.index.ntotal if self.index else 0


class VectorStoreManager:
    """Manager for vector store operations."""

    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        self.embedding_gen = EmbeddingGenerator(model=embedding_model)
        self.vector_store = FAISSVectorStore()
        if not self.vector_store.load_index():
            self.vector_store.create_index()

    def add_document(self, document: Document) -> None:
        texts = [chunk.content for chunk in document.chunks]
        embeddings = self.embedding_gen.encode(texts)
        metadata_list = [
            {
                "content": chunk.content,
                "metadata": chunk.metadata.model_dump(),
                "file_name": chunk.metadata.file_name,
                "page_number": chunk.metadata.page_number,
            }
            for chunk in document.chunks
        ]
        self.vector_store.add_embeddings(embeddings, metadata_list)
        self.vector_store.save_index()

    def search(self, query: str, top_k: int = None) -> List[Tuple[dict, float, int]]:
        top_k = top_k or settings.retrieval_top_k
        query_embedding = self.embedding_gen.encode_single(query)
        distances, indices = self.vector_store.search(query_embedding, top_k)
        results = []
        for rank, (distance, idx) in enumerate(zip(distances, indices)):
            chunk_info = self.vector_store.get_chunk_by_index(idx)
            if chunk_info:
                similarity = 1 / (1 + distance)
                results.append((chunk_info, similarity, rank + 1))
        return results

    def get_stats(self) -> dict:
        return {
            "total_vectors": self.vector_store.get_size(),
            "embedding_dimension": self.vector_store.dimension,
            "embedding_model": self.embedding_gen.model_name,
            "index_path": self.vector_store.index_path,
        }
