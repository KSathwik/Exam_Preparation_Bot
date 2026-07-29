"""Document management API endpoints."""

import time as _time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.dependencies import get_bot
from app.core.security import require_api_key
from app.models.db_models import DocumentRecord
from app.models.schemas import DocumentUploadResponse
from app.services.parser import DocumentParser

router = APIRouter(dependencies=[Depends(require_api_key)])

Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

_ALLOWED = set(settings.allowed_file_types.split(","))
_READ_CHUNK_BYTES = 1024 * 1024  # stream the upload in 1 MB chunks


async def _read_upload_within_limit(file: UploadFile) -> bytes:
    """Read the upload body in bounded chunks, aborting as soon as the size
    limit is exceeded instead of buffering an unbounded body into memory
    first — a large upload would otherwise be fully read (and held in RAM)
    before the size check ever ran."""
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"File too large. Max: {settings.max_file_size_mb} MB",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _safe_upload_path(raw_filename: str | None, document_id: str) -> Path:
    """Collapse the client-supplied filename to its bare basename so it cannot
    escape the upload directory via path traversal (e.g. ``../../evil.pdf`` or
    an absolute path). The extension is still whatever the client sent —
    callers must validate it separately."""
    basename = Path(raw_filename or "upload").name
    if not basename or basename in {".", ".."}:
        basename = "upload"
    return Path(settings.upload_dir) / f"{document_id}_{basename}"


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    start = _time.time()
    logger.info(f"[UPLOAD] Received file: {file.filename}  content_type={file.content_type}")

    ext = Path(file.filename or "").suffix.lower().lstrip(".")
    if ext not in _ALLOWED:
        logger.warning(f"[UPLOAD] Rejected: unsupported type '{ext}'  allowed={_ALLOWED}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(_ALLOWED)}",
        )

    contents = await _read_upload_within_limit(file)
    file_size_mb = len(contents) / (1024 * 1024)
    logger.info(f"[UPLOAD] File size: {file_size_mb:.2f} MB  max_allowed={settings.max_file_size_mb} MB")

    document_id = str(uuid.uuid4())
    save_path = _safe_upload_path(file.filename, document_id)
    save_path.write_bytes(contents)
    logger.debug(f"[UPLOAD] Saved to: {save_path}")

    try:
        t0 = _time.time()
        parser = DocumentParser()
        document = await run_in_threadpool(parser.parse_document, str(save_path), ext)
        logger.info(f"[UPLOAD] Parsed: chunks={document.total_chunks}  parse_time={_time.time()-t0:.2f}s")

        t0 = _time.time()
        bot = get_bot()
        await run_in_threadpool(bot.vector_store_manager.add_document, document, document_id)
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
        logger.info(
            f"[UPLOAD] SUCCESS: {document.file_name}  chunks={document.total_chunks}  total_time={total_time:.2f}s"
        )

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
        save_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        save_path.unlink(missing_ok=True)
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

    bot = get_bot()
    removed_vectors = await run_in_threadpool(bot.vector_store_manager.remove_document, document_id)

    file_removed = False
    if doc.upload_path:
        upload_path = Path(doc.upload_path)
        file_removed = upload_path.exists()
        upload_path.unlink(missing_ok=True)

    db.delete(doc)
    db.commit()
    logger.info(
        f"[DELETE] document_id={document_id}  removed_vectors={removed_vectors}  file_removed={file_removed}"
    )
    return {
        "success": True,
        "message": f"Document {document_id} deleted",
        "removed_vectors": removed_vectors,
    }


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
