"""Conversation (chat history) management endpoints.

Backs the frontend's per-browser conversation sidebar. A conversation is a
``ChatSession`` row, scoped by an anonymous client-generated ``device_id``
(not a real user — see ``ChatSession.device_id``'s comment in db_models.py).
``session_id`` (the conversation id) is itself the access capability once
known, same pattern ``app/api/documents.py``'s delete-by-id already uses —
there is no per-user ownership model in this app today.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from sqlalchemy.orm import Session

from app.api.documents import delete_document_record
from app.core.database import get_db
from app.core.dependencies import get_bot
from app.core.security import require_api_key
from app.models.db_models import ChatMessageRecord, ChatSession, ConversationMemory, DocumentRecord
from app.models.schemas import (
    ChatMessageOut,
    ConversationDeleteResponse,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationRenameRequest,
    ConversationRenameResponse,
    ConversationSummaryOut,
)

router = APIRouter(dependencies=[Depends(require_api_key)])


def _to_summary(session_row: ChatSession) -> ConversationSummaryOut:
    return ConversationSummaryOut(
        session_id=session_row.id,  # type: ignore[arg-type]
        title=session_row.title,  # type: ignore[arg-type]
        created_at=session_row.created_at,  # type: ignore[arg-type]
        updated_at=session_row.updated_at,  # type: ignore[arg-type]
        message_count=session_row.turn_count * 2,  # type: ignore[arg-type]
    )


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    device_id: str = Query(..., min_length=1),
    db: Session = Depends(get_db),
):
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.device_id == device_id)
        # Most-recently-active first, like ChatGPT/Claude — last_activity_at
        # only moves on a real turn (see ChatSession's comment), never on a
        # rename, so renaming a conversation never reorders the sidebar.
        .order_by(ChatSession.last_activity_at.desc())
        .all()
    )
    return ConversationListResponse(
        success=True,
        total=len(sessions),
        conversations=[_to_summary(s) for s in sessions],
    )


@router.get("/{session_id}", response_model=ConversationDetailResponse)
async def get_conversation(session_id: str, db: Session = Depends(get_db)):
    session_row = db.get(ChatSession, session_id)
    if session_row is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = (
        db.query(ChatMessageRecord).filter_by(session_id=session_id).order_by(ChatMessageRecord.id).all()
    )
    return ConversationDetailResponse(
        success=True,
        session_id=session_row.id,  # type: ignore[arg-type]
        title=session_row.title,  # type: ignore[arg-type]
        created_at=session_row.created_at,  # type: ignore[arg-type]
        updated_at=session_row.updated_at,  # type: ignore[arg-type]
        messages=[
            ChatMessageOut(
                role=m.role,  # type: ignore[arg-type]
                content=m.content,  # type: ignore[arg-type]
                timestamp=m.created_at.isoformat() if m.created_at else "",
                intent=m.intent,  # type: ignore[arg-type]
                format_type=m.format_type,  # type: ignore[arg-type]
            )
            for m in messages
        ],
    )


@router.patch("/{session_id}", response_model=ConversationRenameResponse)
async def rename_conversation(
    session_id: str, payload: ConversationRenameRequest, db: Session = Depends(get_db)
):
    session_row = db.get(ChatSession, session_id)
    if session_row is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Deliberately does not touch last_activity_at — a rename is metadata,
    # not activity, and must never reorder the sidebar (see
    # ChatSession.last_activity_at / list_conversations). updated_at still
    # advances automatically via the column's own onupdate.
    session_row.title = payload.title  # type: ignore[assignment]
    db.commit()
    db.refresh(session_row)
    return ConversationRenameResponse(success=True, conversation=_to_summary(session_row))


@router.delete("/{session_id}", response_model=ConversationDeleteResponse)
async def delete_conversation(session_id: str, db: Session = Depends(get_db)):
    session_row = db.get(ChatSession, session_id)
    if session_row is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Clean up any embedded FAISS memory vectors for this conversation first —
    # each embedded ConversationMemory row's id is already its FAISS
    # document_id (see MemoryAgent._maybe_summarize / VectorStoreManager.add_memory),
    # so this reuses the existing remove_document primitive rather than
    # leaving orphaned vectors that would keep surfacing in future
    # semantic-memory search.
    embedded_memories = db.query(ConversationMemory).filter_by(session_id=session_id, embedded=True).all()
    documents = db.query(DocumentRecord).filter_by(session_id=session_id).all()
    if embedded_memories or documents:
        bot = get_bot()
        for memory_row in embedded_memories:
            await run_in_threadpool(bot.vector_store_manager.remove_document, memory_row.id)  # type: ignore[arg-type]
        # Each chat is its own isolated notebook — deleting it deletes the
        # document(s) that belonged only to it (vectors, uploaded file, and
        # DB row), same cleanup as DELETE /documents/{document_id}.
        for doc in documents:
            await delete_document_record(db, bot, doc)

    # ORM-level delete (not a bulk .filter().delete()) — required for the
    # cascade="all, delete-orphan" relationships on messages/memories to fire.
    db.delete(session_row)
    db.commit()
    logger.info(f"[CONVERSATIONS] Deleted session_id={session_id}")
    return ConversationDeleteResponse(
        success=True, message=f"Conversation {session_id} deleted", session_id=session_id
    )
