"""Embedding generation and vector database management."""

import pickle
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.core.config import settings

from .models import ChunkMetadata, Document, DocumentChunk

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - exercised only when the optional dep is absent
    BM25Okapi = None


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _min_max_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        # No differentiating signal (e.g. every candidate scored zero) —
        # contribute nothing rather than artificially boosting every
        # candidate to a "perfect" lexical match.
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


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
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
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
            logger.info(f"[FAISS] No existing index at {index_file} — will create new")
            return False
        try:
            index = faiss.read_index(str(index_file))
            chunk_metadata: List[dict] = []
            if metadata_file.exists():
                with open(metadata_file, "rb") as f:
                    chunk_metadata = pickle.load(f)
            embeddings = np.load(embeddings_file) if embeddings_file.exists() else None

            # The three files are written together but not atomically as a
            # unit — a crash or a second process writing concurrently between
            # writes can leave them out of sync. Loading a mismatched set
            # would silently corrupt every future add/remove, so refuse it
            # and start from a clean empty index instead.
            embeddings_count = embeddings.shape[0] if embeddings is not None else 0
            if not (index.ntotal == len(chunk_metadata) == embeddings_count):
                logger.error(
                    f"[FAISS] Refusing to load inconsistent index at {self.index_path}: "
                    f"index.ntotal={index.ntotal}  metadata={len(chunk_metadata)}  embeddings={embeddings_count}. "
                    "Starting from an empty index instead — re-upload your documents."
                )
                self.create_index()
                return False

            self.index = index
            self.chunk_metadata = chunk_metadata
            self.embeddings = embeddings
            logger.info(
                f"[FAISS] Loaded index: vectors={self.index.ntotal}  metadata={len(self.chunk_metadata)}  path={self.index_path}"
            )
            return True
        except Exception as e:
            logger.error(f"[FAISS] Failed to load index: {type(e).__name__}: {e}")
            return False

    def save_index(self) -> None:
        if self.index is None:
            return
        index_file = Path(self.index_path) / "index.faiss"
        metadata_file = Path(self.index_path) / "metadata.pkl"
        embeddings_file = Path(self.index_path) / "embeddings.npy"

        # Write to temp files and rename into place — rename is atomic on both
        # POSIX and Windows (NTFS), so a crash or a concurrent reader never
        # observes a half-written file; readers see either the old or the
        # fully-written new version of each file, never a partial one.
        index_tmp = index_file.with_suffix(index_file.suffix + ".tmp")
        metadata_tmp = metadata_file.with_suffix(metadata_file.suffix + ".tmp")
        embeddings_tmp = embeddings_file.with_suffix(embeddings_file.suffix + ".tmp")

        faiss.write_index(self.index, str(index_tmp))
        with open(metadata_tmp, "wb") as f:
            pickle.dump(self.chunk_metadata, f)
        if self.embeddings is not None:
            # np.save appends a .npy suffix to string/Path targets that lack
            # one — passing an open file handle instead saves to the exact
            # "*.npy.tmp" path so the later atomic rename lands correctly.
            with open(embeddings_tmp, "wb") as f:
                np.save(f, self.embeddings)

        index_tmp.replace(index_file)
        metadata_tmp.replace(metadata_file)
        if self.embeddings is not None:
            embeddings_tmp.replace(embeddings_file)
        elif embeddings_file.exists():
            embeddings_file.unlink()

        logger.debug(f"[FAISS] Index saved: vectors={self.index.ntotal}  path={self.index_path}")

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

    def remove_by_document_id(self, document_id: str) -> int:
        """Remove every chunk belonging to ``document_id``.

        FAISS's flat index has no native delete-by-id, so removal means
        filtering the parallel metadata/embeddings arrays and rebuilding the
        index from what remains. Returns the number of chunks removed.
        """
        if self.index is None or self.index.ntotal == 0:
            return 0
        keep_mask = [m.get("document_id") != document_id for m in self.chunk_metadata]
        removed = len(keep_mask) - sum(keep_mask)
        if removed == 0:
            return 0

        kept_metadata = [m for m, keep in zip(self.chunk_metadata, keep_mask) if keep]
        kept_embeddings = (
            self.embeddings[np.array(keep_mask, dtype=bool)] if self.embeddings is not None else None
        )

        self.create_index()
        if kept_embeddings is not None and len(kept_metadata):
            self.add_embeddings(kept_embeddings, kept_metadata)
        self.save_index()
        logger.info(
            f"[FAISS] Removed document {document_id}: chunks_removed={removed}  remaining={self.get_size()}"
        )
        return removed

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
    """Manager for vector store operations.

    All mutating and read operations serialize on ``_index_lock`` — the FAISS
    index and its parallel metadata list are not safe for concurrent access,
    and route handlers run this manager's methods from a thread pool.
    """

    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        self.embedding_gen = EmbeddingGenerator(model=embedding_model)
        self.vector_store = FAISSVectorStore()
        self._index_lock = threading.Lock()
        if not self.vector_store.load_index():
            self.vector_store.create_index()
        # Lazily (re)built on first search after any add/remove/reset — see
        # _get_bm25_index. Rebuilt in-memory only, no new persisted artifact.
        self._bm25_index = None
        self._bm25_dirty = True

    def add_document(self, document: Document, document_id: Optional[str] = None) -> None:
        logger.info(f"[EMBED] Adding document: {document.file_name}  chunks={document.total_chunks}")
        texts = [chunk.content for chunk in document.chunks]
        embeddings = self.embedding_gen.encode(texts)
        logger.debug(f"[EMBED] Encoded {len(texts)} chunks → shape={embeddings.shape}")
        metadata_list = [
            {
                "content": chunk.content,
                "metadata": chunk.metadata.model_dump(),
                "file_name": chunk.metadata.file_name,
                "page_number": chunk.metadata.page_number,
                "document_id": document_id,
            }
            for chunk in document.chunks
        ]
        with self._index_lock:
            self.vector_store.add_embeddings(embeddings, metadata_list)
            self.vector_store.save_index()
            self._bm25_dirty = True
        logger.info(f"[EMBED] Document indexed: total_vectors={self.vector_store.get_size()}")

    def add_memory(self, text: str, session_id: str, memory_id: str) -> None:
        """Embed and index a conversation-memory summary on the *same* FAISS
        index as documents, discriminated via ``content_type="memory"`` —
        avoids duplicating the persistence/locking/consistency-check
        machinery a second index would need (see architecture plan)."""
        logger.info(f"[EMBED] Adding memory: session={session_id}  memory_id={memory_id}")
        embedding = self.embedding_gen.encode([text])
        chunk_metadata = ChunkMetadata(
            page_number=0,
            chunk_index=0,
            total_chunks=1,
            file_name=f"memory:{session_id}",
            content_type="memory",
        )
        metadata_list = [
            {
                "content": text,
                "metadata": chunk_metadata.model_dump(),
                "file_name": chunk_metadata.file_name,
                "page_number": 0,
                "document_id": memory_id,
            }
        ]
        with self._index_lock:
            self.vector_store.add_embeddings(embedding, metadata_list)
            self.vector_store.save_index()
            self._bm25_dirty = True
        logger.info(f"[EMBED] Memory indexed: total_vectors={self.vector_store.get_size()}")

    def remove_document(self, document_id: str) -> int:
        """Remove all indexed chunks belonging to ``document_id``. Returns the count removed."""
        with self._index_lock:
            removed = self.vector_store.remove_by_document_id(document_id)
            if removed:
                self._bm25_dirty = True
            return removed

    def reset(self) -> None:
        """Clear the index and persist the empty state so a restart doesn't
        resurrect the previous (pre-reset) data from disk."""
        with self._index_lock:
            self.vector_store.clear()
            self.vector_store.create_index()
            self.vector_store.save_index()
            self._bm25_dirty = True
        logger.info("[EMBED] Vector store reset and persisted")

    def _get_bm25_index(self):
        """Return the lazily-(re)built BM25Okapi index over every indexed
        chunk's text, or ``None`` when unavailable (rank-bm25 not installed,
        or the index is empty) — callers fall back to pure dense ranking."""
        if self._bm25_dirty or self._bm25_index is None:
            corpus = self.vector_store.chunk_metadata
            if BM25Okapi is None or not corpus:
                self._bm25_index = None
            else:
                self._bm25_index = BM25Okapi([_tokenize(c.get("content", "")) for c in corpus])
            self._bm25_dirty = False
        return self._bm25_index

    def _rank_candidates(
        self, query: str, candidates: List[Tuple[int, dict, float]]
    ) -> List[Tuple[dict, float]]:
        """Blend dense similarity with a BM25 lexical score across the same
        candidate set (hybrid retrieval), normalizing BM25 scores min-max
        across just this candidate set before blending. Falls back to pure
        dense ranking when no lexical signal is available."""
        if not candidates:
            return []
        bm25_index = self._get_bm25_index()
        if bm25_index is None:
            ranked = sorted(candidates, key=lambda c: c[2], reverse=True)
            return [(chunk_info, dense_sim) for _, chunk_info, dense_sim in ranked]

        tokenized_query = _tokenize(query)
        bm25_scores_all = bm25_index.get_scores(tokenized_query)
        bm25_norm = _min_max_normalize([bm25_scores_all[idx] for idx, _, _ in candidates])

        alpha = settings.hybrid_dense_weight
        blended = [
            (chunk_info, alpha * dense_sim + (1 - alpha) * bm25_n)
            for (_, chunk_info, dense_sim), bm25_n in zip(candidates, bm25_norm)
        ]
        blended.sort(key=lambda x: x[1], reverse=True)
        return blended

    def search(
        self, query: str, top_k: int = None, content_types: Optional[List[str]] = None
    ) -> List[Tuple[dict, float, int]]:
        """Search the shared index. Defaults to ``["document"]`` so existing
        document-retrieval call sites never silently start mixing in memory
        content — pass ``content_types=["memory"]`` explicitly for semantic
        memory lookups."""
        top_k = top_k or settings.retrieval_top_k
        content_types = ["document"] if content_types is None else content_types
        logger.debug(
            f"[SEARCH] query={query!r}  top_k={top_k}  content_types={content_types}  "
            f"index_size={self.vector_store.get_size()}"
        )
        query_embedding = self.embedding_gen.encode_single(query)
        with self._index_lock:
            # content_type isn't part of the vector space, so filtering is
            # post-hoc: over-fetch candidates from FAISS, then filter and
            # re-rank in Python. Cheap at this project's scale (see plan).
            fetch_k = min(self.vector_store.get_size(), max(top_k * 5, 50))
            candidates: List[Tuple[int, dict, float]] = []
            if fetch_k > 0:
                distances, indices = self.vector_store.search(query_embedding, fetch_k)
                for distance, idx in zip(distances, indices):
                    chunk_info = self.vector_store.get_chunk_by_index(idx)
                    if not chunk_info:
                        continue
                    chunk_content_type = chunk_info.get("metadata", {}).get("content_type", "document")
                    if chunk_content_type not in content_types:
                        continue
                    similarity = max(0.0, 1.0 - (distance / 2.0))
                    candidates.append((idx, chunk_info, similarity))

            ranked = self._rank_candidates(query, candidates)
            results = [
                (chunk_info, score, rank + 1) for rank, (chunk_info, score) in enumerate(ranked[:top_k])
            ]
        best_sim = f"{results[0][1]:.4f}" if results else "0.0"
        logger.info(f"[SEARCH] Returning {len(results)} results  best_sim={best_sim}")
        return results

    def get_stats(self) -> dict:
        return {
            "total_vectors": self.vector_store.get_size(),
            "embedding_dimension": self.vector_store.dimension,
            "embedding_model": self.embedding_gen.model_name,
            "index_path": self.vector_store.index_path,
        }
