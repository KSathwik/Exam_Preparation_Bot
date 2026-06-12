"""Document management API endpoints."""

from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from loguru import logger
from pathlib import Path
import uuid
from datetime import datetime
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.dependencies import get_bot
from app.models.schemas import DocumentUploadResponse
from app.models.db_models import DocumentRecord
from app.services.parser import DocumentParser

router = APIRouter()

Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

_ALLOWED = set(settings.allowed_file_types.split(","))


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    import time as _time
    start = _time.time()
    logger.info(f"[UPLOAD] Received file: {file.filename}  content_type={file.content_type}")

    ext = Path(file.filename or "").suffix.lower().lstrip(".")
    if ext not in _ALLOWED:
        logger.warning(f"[UPLOAD] Rejected: unsupported type '{ext}'  allowed={_ALLOWED}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(_ALLOWED)}",
        )

    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    logger.info(f"[UPLOAD] File size: {file_size_mb:.2f} MB  max_allowed={settings.max_file_size_mb} MB")
    if file_size_mb > settings.max_file_size_mb:
        logger.warning(f"[UPLOAD] Rejected: file too large ({file_size_mb:.1f} MB)")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({file_size_mb:.1f} MB). Max: {settings.max_file_size_mb} MB",
        )

    document_id = str(uuid.uuid4())
    save_path = Path(settings.upload_dir) / f"{document_id}_{file.filename}"
    save_path.write_bytes(contents)
    logger.debug(f"[UPLOAD] Saved to: {save_path}")

    try:
        t0 = _time.time()
        parser = DocumentParser()
        document = parser.parse_document(str(save_path), ext)
        logger.info(f"[UPLOAD] Parsed: chunks={document.total_chunks}  parse_time={_time.time()-t0:.2f}s")

        t0 = _time.time()
        bot = get_bot()
        bot.vector_store_manager.add_document(document)
        logger.info(f"[UPLOAD] Embedded & indexed: embed_time={_time.time()-t0:.2f}s")

        record = DocumentRecord(
            id=document_id,
            file_name=document.file_name,
            file_type=document.file_type,
            file_size_mb=file_size_mb,
            total_chunks=document.total_chunks,
            upload_path=str(save_path),
        )
        db.add(record)
        db.commit()
        logger.info(f"[UPLOAD] DB record saved: id={document_id}")

        total_time = _time.time() - start
        logger.info(f"[UPLOAD] SUCCESS: {document.file_name}  chunks={document.total_chunks}  total_time={total_time:.2f}s")

        return DocumentUploadResponse(
            success=True,
            file_name=document.file_name,
            file_type=document.file_type,
            total_chunks=document.total_chunks,
            file_size_mb=round(file_size_mb, 2),
            processing_time_seconds=round(total_time, 2),
            document_id=document_id,
            message=f"Successfully uploaded {document.file_name} with {document.total_chunks} chunks",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[UPLOAD] FAILED: {file.filename}  error={type(e).__name__}: {e}")
        logger.exception("[UPLOAD] Full traceback:")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_documents(db: Session = Depends(get_db)):
    docs = db.query(DocumentRecord).order_by(DocumentRecord.created_at.desc()).all()
    return {
        "success": True,
        "total_documents": len(docs),
        "documents": [
            {
                "document_id": d.id,
                "file_name": d.file_name,
                "file_type": d.file_type,
                "total_chunks": d.total_chunks,
                "file_size_mb": d.file_size_mb,
                "upload_timestamp": d.created_at.isoformat(),
            }
            for d in docs
        ],
    }


@router.delete("/{document_id}")
async def delete_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    db.delete(doc)
    db.commit()
    return {"success": True, "message": f"Document {document_id} deleted"}


@router.get("/stats")
async def get_document_stats():
    bot = get_bot()
    stats = bot.get_stats()
    return {
        "vector_store": stats["vector_store"],
        "chat_history_length": stats["chat_history_length"],
        "model": stats["model"],
        "embedding_model": stats["embedding_model"],
        "timestamp": datetime.now().isoformat(),
    }
