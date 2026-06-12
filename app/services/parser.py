"""Document parsing module for PDF and DOCX files."""

import pdfplumber
from docx import Document as DocxDocument
from pathlib import Path
from typing import List, Optional
from loguru import logger
from .models import DocumentChunk, ChunkMetadata, Document
from app.core.config import settings
import re
from datetime import datetime


class DocumentParser:
    """Parse documents and extract text with metadata."""

    def __init__(self, max_chunk_size: int = None, chunk_overlap: int = None):
        self.max_chunk_size = max_chunk_size or settings.max_chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_size = settings.min_chunk_size

    def parse_document(self, file_path: str, file_type: str) -> Document:
        if file_type.lower() == "pdf":
            return self._parse_pdf(file_path)
        elif file_type.lower() == "docx":
            return self._parse_docx(file_path)
        raise ValueError(f"Unsupported file type: {file_type}")

    def _parse_pdf(self, file_path: str) -> Document:
        logger.info(f"[PARSER] Parsing PDF: {file_path}")
        file_size_bytes = Path(file_path).stat().st_size
        file_name = Path(file_path).name

        with pdfplumber.open(file_path) as pdf:
            logger.info(f"[PARSER] PDF opened: pages={len(pdf.pages)}  size={file_size_bytes/1024:.1f} KB")
            full_text = ""
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                logger.debug(f"[PARSER] Page {page_num}: {len(text)} chars extracted")
                if full_text:
                    full_text += f"\n\n--- Page {page_num} ---\n\n"
                full_text += text

            logger.info(f"[PARSER] Total text extracted: {len(full_text)} chars")
            chunks = self._create_chunks(full_text, file_name, len(pdf.pages))

        logger.info(f"[PARSER] PDF parsed: {file_name}  chunks={len(chunks)}")
        return Document(
            file_name=file_name,
            file_type="pdf",
            file_size_bytes=file_size_bytes,
            total_chunks=len(chunks),
            chunks=chunks,
            upload_timestamp=datetime.now().isoformat(),
        )

    def _parse_docx(self, file_path: str) -> Document:
        logger.info(f"[PARSER] Parsing DOCX: {file_path}")
        file_size_bytes = Path(file_path).stat().st_size
        file_name = Path(file_path).name

        doc = DocxDocument(file_path)
        full_text = ""
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                full_text += text + "\n"

        logger.info(f"[PARSER] DOCX text extracted: {len(full_text)} chars  paragraphs={len(doc.paragraphs)}")
        chunks = self._create_chunks(full_text, file_name, len(doc.paragraphs))

        logger.info(f"[PARSER] DOCX parsed: {file_name}  chunks={len(chunks)}")
        return Document(
            file_name=file_name,
            file_type="docx",
            file_size_bytes=file_size_bytes,
            total_chunks=len(chunks),
            chunks=chunks,
            upload_timestamp=datetime.now().isoformat(),
        )

    def _create_chunks(
        self, text: str, file_name: str, total_pages: int
    ) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        sentences = _split_sentences(text)
        current_chunk = ""
        chunk_sentences: List[str] = []
        chunk_index = 0

        for sentence in sentences:
            test_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
            if len(test_chunk.split()) > self.max_chunk_size:
                if current_chunk and len(current_chunk.split()) >= self.min_chunk_size:
                    chunks.append(
                        _make_chunk(current_chunk.strip(), file_name, chunk_index, 0, total_pages)
                    )
                    chunk_index += 1
                    overlap = chunk_sentences[-2:] if len(chunk_sentences) > 1 else []
                    current_chunk = " ".join(overlap)
                    chunk_sentences = list(overlap)
            current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
            chunk_sentences.append(sentence)

        if current_chunk and len(current_chunk.split()) >= self.min_chunk_size:
            chunks.append(
                _make_chunk(current_chunk.strip(), file_name, chunk_index, 0, total_pages)
            )

        total = len(chunks)
        for c in chunks:
            c.metadata.total_chunks = total
        return chunks


def _make_chunk(
    content: str, file_name: str, chunk_index: int, total_chunks: int, total_pages: int
) -> DocumentChunk:
    safe_total = max(total_chunks, 1)
    page_number = max(1, int((chunk_index / safe_total) * total_pages) + 1)
    if safe_total <= 3:
        position = "unknown"
    elif chunk_index < safe_total * 0.33:
        position = "beginning"
    elif chunk_index > safe_total * 0.66:
        position = "end"
    else:
        position = "middle"

    return DocumentChunk(
        content=content,
        metadata=ChunkMetadata(
            page_number=min(page_number, total_pages) if total_pages else 1,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            file_name=file_name,
            chunk_position=position,
        ),
    )


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def parse_file(file_path: str) -> Document:
    """Convenience function to parse a file."""
    ext = Path(file_path).suffix.lower().lstrip(".")
    if ext not in ("pdf", "docx"):
        raise ValueError(f"Unsupported file type: {ext}")
    return DocumentParser().parse_document(file_path, ext)
