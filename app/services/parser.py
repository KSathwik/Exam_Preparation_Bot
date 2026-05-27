"""Document parsing module for PDF and DOCX files."""

import pdfplumber
from docx import Document as DocxDocument
from pathlib import Path
from typing import List, Tuple, Optional
from loguru import logger
from .models import DocumentChunk, ChunkMetadata, Document
from config.settings import settings
import re
from datetime import datetime


class DocumentParser:
    """Parse documents and extract text with metadata."""
    
    def __init__(self, max_chunk_size: int = None, chunk_overlap: int = None):
        """Initialize parser with chunking settings."""
        self.max_chunk_size = max_chunk_size or settings.max_chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_size = settings.min_chunk_size
        
    def parse_document(self, file_path: str, file_type: str) -> Document:
        """Parse a document and return chunks with metadata."""
        
        if file_type.lower() == "pdf":
            return self._parse_pdf(file_path)
        elif file_type.lower() == "docx":
            return self._parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _parse_pdf(self, file_path: str) -> Document:
        """Parse PDF file and extract text with page metadata."""
        logger.info(f"Parsing PDF: {file_path}")
        
        chunks: List[DocumentChunk] = []
        file_size_bytes = Path(file_path).stat().st_size
        file_name = Path(file_path).name
        
        try:
            with pdfplumber.open(file_path) as pdf:
                full_text = ""
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text from page
                    text = page.extract_text() or ""
                    
                    # Add page break marker for structure preservation
                    if full_text:
                        full_text += f"\n\n--- Page {page_num} ---\n\n"
                    
                    full_text += text
                
                # Create chunks with metadata
                chunks = self._create_chunks(
                    full_text, 
                    file_name=file_name,
                    total_pages=len(pdf.pages)
                )
                
                logger.info(f"Extracted {len(chunks)} chunks from PDF: {file_name}")
                
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            raise
        
        return Document(
            file_name=file_name,
            file_type="pdf",
            file_size_bytes=file_size_bytes,
            total_chunks=len(chunks),
            chunks=chunks,
            upload_timestamp=datetime.now().isoformat()
        )
    
    def _parse_docx(self, file_path: str) -> Document:
        """Parse DOCX file and extract text with section metadata."""
        logger.info(f"Parsing DOCX: {file_path}")
        
        chunks: List[DocumentChunk] = []
        file_size_bytes = Path(file_path).stat().st_size
        file_name = Path(file_path).name
        
        try:
            doc = DocxDocument(file_path)
            full_text = ""
            section_map = {}  # Track section titles
            
            for para_idx, para in enumerate(doc.paragraphs):
                text = para.text.strip()
                
                if not text:
                    continue
                
                # Detect heading levels
                style = para.style.name if para.style else "Normal"
                section_title = None
                section_level = None
                
                if "Heading" in style:
                    section_title = text
                    section_level = int(re.findall(r'\d+', style)[0]) if re.findall(r'\d+', style) else 1
                    section_map[section_level] = text
                
                # Add to full text
                full_text += text + "\n"
            
            # Create chunks
            chunks = self._create_chunks(
                full_text,
                file_name=file_name,
                total_pages=len(doc.paragraphs)
            )
            
            logger.info(f"Extracted {len(chunks)} chunks from DOCX: {file_name}")
            
        except Exception as e:
            logger.error(f"Error parsing DOCX {file_path}: {e}")
            raise
        
        return Document(
            file_name=file_name,
            file_type="docx",
            file_size_bytes=file_size_bytes,
            total_chunks=len(chunks),
            chunks=chunks,
            upload_timestamp=datetime.now().isoformat()
        )
    
    def _create_chunks(
        self, 
        text: str, 
        file_name: str,
        total_pages: int
    ) -> List[DocumentChunk]:
        """
        Create overlapping chunks from text.
        
        Uses smart chunking: splits on sentences when possible.
        """
        chunks = []
        sentences = self._split_sentences(text)
        
        current_chunk = ""
        chunk_sentences = []
        chunk_index = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed max size
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk.split()) > self.max_chunk_size:
                # Save current chunk if it has content
                if current_chunk and len(current_chunk.split()) >= self.min_chunk_size:
                    chunks.append(
                        self._create_chunk_object(
                            current_chunk.strip(),
                            file_name,
                            chunk_index,
                            len(chunks) + 1,  # Will update total later
                            total_pages
                        )
                    )
                    chunk_index += 1
                    
                    # Add overlap: include last few sentences
                    overlap_sentences = chunk_sentences[-2:] if len(chunk_sentences) > 1 else []
                    current_chunk = " ".join(overlap_sentences)
                    chunk_sentences = overlap_sentences
            
            # Add sentence to current chunk
            current_chunk += " " + sentence if current_chunk else sentence
            chunk_sentences.append(sentence)
        
        # Add final chunk
        if current_chunk and len(current_chunk.split()) >= self.min_chunk_size:
            chunks.append(
                self._create_chunk_object(
                    current_chunk.strip(),
                    file_name,
                    chunk_index,
                    len(chunks) + 1,
                    total_pages
                )
            )
        
        # Update total chunks in metadata
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.metadata.total_chunks = total_chunks
        
        return chunks
    
    def _create_chunk_object(
        self,
        content: str,
        file_name: str,
        chunk_index: int,
        total_chunks: int,
        total_pages: int
    ) -> DocumentChunk:
        """Create a DocumentChunk object with metadata."""
        
        # Estimate page number based on chunk position
        page_number = max(1, int((chunk_index / total_chunks) * total_pages) + 1)
        
        # Determine chunk position
        if total_chunks <= 3:
            position = "unknown"
        elif chunk_index < total_chunks * 0.33:
            position = "beginning"
        elif chunk_index > total_chunks * 0.66:
            position = "end"
        else:
            position = "middle"
        
        metadata = ChunkMetadata(
            page_number=min(page_number, total_pages),
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            file_name=file_name,
            chunk_position=position
        )
        
        return DocumentChunk(content=content, metadata=metadata)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving structure."""
        
        # Simple sentence splitter using regex
        # Handles: period, exclamation, question mark
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences


class ChunkProcessor:
    """Process and refine chunks for better retrieval."""
    
    @staticmethod
    def clean_chunk(chunk: DocumentChunk) -> DocumentChunk:
        """Clean chunk text (remove extra whitespace, normalize)."""
        content = chunk.content
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        # Remove page break markers if any
        content = content.replace("--- Page", "").replace("---", "")
        
        chunk.content = content
        return chunk
    
    @staticmethod
    def enrich_chunk_metadata(
        chunk: DocumentChunk,
        section_title: Optional[str] = None,
        section_level: Optional[int] = None
    ) -> DocumentChunk:
        """Add additional metadata to chunk."""
        if section_title:
            chunk.metadata.section_title = section_title
        if section_level:
            chunk.metadata.section_level = section_level
        
        return chunk


def parse_file(file_path: str) -> Document:
    """Convenience function to parse a file."""
    parser = DocumentParser()
    
    file_extension = Path(file_path).suffix.lower().lstrip('.')
    if file_extension not in ["pdf", "docx"]:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    return parser.parse_document(file_path, file_extension)
