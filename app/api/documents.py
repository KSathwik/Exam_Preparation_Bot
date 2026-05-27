"""Document management API endpoints"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from loguru import logger
import shutil
from pathlib import Path
import uuid

from app.core.config import settings
from app.models.schemas import DocumentUploadResponse, DocumentListResponse, DocumentInfo
from app.services.parser import DocumentParser
from app.services.vector_store import vector_store_manager
from datetime import datetime

router = APIRouter()

# Ensure upload directory exists
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF or DOCX)
    
    - **file**: PDF or DOCX file to upload
    """
    logger.info(f"Upload request for file: {file.filename}")
    
    # Validate file type
    file_extension = Path(file.filename).suffix.lower().lstrip('.')
    if file_extension not in ["pdf", "docx"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_extension}. Use PDF or DOCX."
        )
    
    # Check file size
    file_size_mb = 0
    try:
        # Save file temporarily
        document_id = str(uuid.uuid4())
        temp_path = Path(settings.UPLOAD_DIR) / f"{document_id}_{file.filename}"
        
        with open(temp_path, "wb") as buffer:
            contents = await file.read()
            file_size_mb = len(contents) / (1024 * 1024)
            
            if file_size_mb > settings.MAX_FILE_SIZE_MB:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
                )
            
            buffer.write(contents)
        
        # Parse document
        logger.info(f"Parsing document: {file.filename}")
        parser = DocumentParser()
        document = parser.parse_document(str(temp_path), file_extension)
        
        # Add to vector store
        logger.info(f"Adding document to vector store: {document.file_name}")
        vector_store_manager.add_document(document)
        
        logger.info(f"Document uploaded successfully: {document_id}")
        
        return DocumentUploadResponse(
            success=True,
            file_name=document.file_name,
            file_type=document.file_type,
            total_chunks=document.total_chunks,
            file_size_mb=file_size_mb,
            processing_time_seconds=0.0,  # Would need to track actual time
            document_id=document_id,
            message=f"Successfully uploaded {document.file_name} with {document.total_chunks} chunks"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )


@router.get("/list", response_model=DocumentListResponse)
async def list_documents():
    """
    List all uploaded documents
    """
    logger.info("Listing documents")
    
    try:
        # For now, return empty list
        # In production, this would query database
        documents = []
        
        return DocumentListResponse(
            success=True,
            total_documents=len(documents),
            documents=documents
        )
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document by ID
    
    - **document_id**: ID of document to delete
    """
    logger.info(f"Delete request for document: {document_id}")
    
    try:
        # In production, remove from database and vector store
        return {
            "success": True,
            "message": f"Document {document_id} deleted successfully"
        }
    
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{document_id}/stats")
async def get_document_stats(document_id: str):
    """
    Get statistics for a specific document
    
    - **document_id**: ID of document
    """
    logger.info(f"Getting stats for document: {document_id}")
    
    try:
        # In production, query database
        return {
            "success": True,
            "document_id": document_id,
            "total_chunks": 0,
            "total_queries": 0
        }
    
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
