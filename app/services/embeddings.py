"""Embedding generation and vector database management."""

import numpy as np
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from loguru import logger
import faiss
import pickle
from pathlib import Path
from .models import DocumentChunk, Document
from config.settings import settings


class EmbeddingGenerator:
    """Generate embeddings for text chunks."""
    
    def __init__(self, model_name: str = None):
        """Initialize embedding generator."""
        self.model_name = model_name or settings.embedding_model
        logger.info(f"Loading embedding model: {self.model_name}")
        
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def encode(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        batch_size = batch_size or settings.batch_size
        
        logger.debug(f"Encoding {len(texts)} texts with batch size {batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string."""
        return self.model.encode(text, normalize_embeddings=True)


class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search."""
    
    def __init__(self, dimension: int = None, index_path: str = None):
        """Initialize FAISS vector store."""
        self.dimension = dimension or settings.vector_dimension
        self.index_path = index_path or settings.faiss_index_path
        self.index = None
        self.chunk_metadata = []
        self.embeddings = None
        
        self._ensure_index_directory()
    
    def _ensure_index_directory(self):
        """Ensure index directory exists."""
        Path(self.index_path).mkdir(parents=True, exist_ok=True)
    
    def create_index(self) -> None:
        """Create a new FAISS index."""
        logger.info(f"Creating new FAISS index with dimension {self.dimension}")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunk_metadata = []
        self.embeddings = None
    
    def load_index(self) -> bool:
        """Load existing FAISS index."""
        index_file = Path(self.index_path) / "index.faiss"
        metadata_file = Path(self.index_path) / "metadata.pkl"
        embeddings_file = Path(self.index_path) / "embeddings.npy"
        
        if not index_file.exists():
            logger.warning(f"Index file not found at {index_file}")
            return False
        
        try:
            logger.info(f"Loading FAISS index from {index_file}")
            self.index = faiss.read_index(str(index_file))
            
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    self.chunk_metadata = pickle.load(f)
                logger.info(f"Loaded metadata for {len(self.chunk_metadata)} chunks")
            
            if embeddings_file.exists():
                self.embeddings = np.load(embeddings_file)
                logger.info(f"Loaded {len(self.embeddings)} embeddings")
            
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def save_index(self) -> None:
        """Save FAISS index to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return
        
        try:
            index_file = Path(self.index_path) / "index.faiss"
            metadata_file = Path(self.index_path) / "metadata.pkl"
            embeddings_file = Path(self.index_path) / "embeddings.npy"
            
            faiss.write_index(self.index, str(index_file))
            logger.info(f"Saved FAISS index to {index_file}")
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.chunk_metadata, f)
            logger.info(f"Saved metadata for {len(self.chunk_metadata)} chunks")
            
            if self.embeddings is not None:
                np.save(embeddings_file, self.embeddings)
                logger.info(f"Saved {len(self.embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    def add_embeddings(self, embeddings: np.ndarray, metadata_list: List[dict]) -> None:
        """
        Add embeddings and metadata to index.
        
        Args:
            embeddings: numpy array of shape (n, dimension)
            metadata_list: List of metadata dicts (should be DocumentChunk objects)
        """
        if self.index is None:
            self.create_index()
        
        logger.info(f"Adding {len(embeddings)} embeddings to index")
        
        # Convert to float32 if needed
        embeddings = embeddings.astype(np.float32)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        self.chunk_metadata.extend(metadata_list)
        
        # Store embeddings
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        logger.info(f"Index now contains {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[float], List[int]]:
        """
        Search for nearest neighbors.
        
        Args:
            query_embedding: Query embedding (1D array)
            k: Number of results to return
            
        Returns:
            (distances, indices) - lists of distances and chunk indices
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return [], []
        
        # Reshape to (1, dimension) for FAISS
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert to lists
        distances = distances[0].tolist()
        indices = indices[0].tolist()
        
        return distances, indices
    
    def get_chunk_by_index(self, idx: int) -> Optional[dict]:
        """Get chunk metadata by index."""
        if idx < 0 or idx >= len(self.chunk_metadata):
            return None
        return self.chunk_metadata[idx]
    
    def get_chunks_by_indices(self, indices: List[int]) -> List[dict]:
        """Get multiple chunks by indices."""
        return [self.get_chunk_by_index(idx) for idx in indices if idx >= 0]
    
    def clear(self) -> None:
        """Clear the index."""
        self.index = None
        self.chunk_metadata = []
        self.embeddings = None
        logger.info("Index cleared")
    
    def get_size(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal if self.index else 0


class VectorStoreManager:
    """Manager for vector store operations."""
    
    def __init__(self):
        """Initialize vector store manager."""
        self.embedding_gen = EmbeddingGenerator()
        self.vector_store = FAISSVectorStore()
        
        # Try to load existing index
        if not self.vector_store.load_index():
            self.vector_store.create_index()
    
    def add_document(self, document: Document) -> None:
        """Add document chunks to vector store."""
        logger.info(f"Adding document to vector store: {document.file_name}")
        
        # Extract text from chunks
        texts = [chunk.content for chunk in document.chunks]
        
        # Generate embeddings
        embeddings = self.embedding_gen.encode(texts)
        
        # Prepare metadata (convert DocumentChunk to dict)
        metadata_list = [
            {
                'content': chunk.content,
                'metadata': chunk.metadata.model_dump(),
                'file_name': chunk.metadata.file_name,
                'page_number': chunk.metadata.page_number
            }
            for chunk in document.chunks
        ]
        
        # Add to vector store
        self.vector_store.add_embeddings(embeddings, metadata_list)
        
        # Save index
        self.vector_store.save_index()
        
        logger.info(f"Document added. Vector store now has {self.vector_store.get_size()} vectors")
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[dict, float, int]]:
        """
        Search for similar chunks.
        
        Args:
            query: Query string
            top_k: Number of results
            
        Returns:
            List of (chunk_metadata, distance, rank) tuples
        """
        top_k = top_k or settings.retrieval_top_k
        
        # Encode query
        query_embedding = self.embedding_gen.encode_single(query)
        
        # Search in vector store
        distances, indices = self.vector_store.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for rank, (distance, idx) in enumerate(zip(distances, indices)):
            chunk_info = self.vector_store.get_chunk_by_index(idx)
            if chunk_info:
                # Convert L2 distance to similarity score (0-1)
                similarity = 1 / (1 + distance)
                results.append((chunk_info, similarity, rank + 1))
        
        return results
    
    def get_stats(self) -> dict:
        """Get vector store statistics."""
        return {
            'total_vectors': self.vector_store.get_size(),
            'embedding_dimension': self.vector_store.dimension,
            'embedding_model': self.embedding_gen.model_name,
            'index_path': self.vector_store.index_path
        }
