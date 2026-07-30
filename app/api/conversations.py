"""Conversation (chat history) management endpoints.

Backs the frontend's per-browser conversation sidebar. A conversation is a
``ChatSession`` row, scoped by an anonymous client-generated ``device_id``
(not a real user — see ``ChatSession.device_id``'s comment in db_models.py).
``session_id`` (the conversation id) is itself the access capability once
known, same pattern ``app/api/documents.py``'s delete-by-id already uses —
there is no per-user ownership model in this app today.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import get_bot
from app.core.security import require_api_key
from app.models.db_models import ChatMessageRecord, ChatSession, ConversationMemory
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
        session_id=session_row.id,
        title=session_row.title,
        created_at=session_row.created_at,
        updated_at=session_row.updated_at,
        message_count=session_row.turn_count * 2,
    )


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    device_id: str = Query(..., min_length=1),
    db: Session = Depends(get_db),
):
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.device_id == device_id)
        .order_by(ChatSession.updated_at.desc())
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
        session_id=session_row.id,
        title=session_row.title,
        created_at=session_row.created_at,
        updated_at=session_row.updated_at,
        messages=[
            ChatMessageOut(
                role=m.role,
                content=m.content,
                timestamp=m.created_at.isoformat() if m.created_at else "",
                intent=m.intent,
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

    session_row.title = payload.title
    session_row.updated_at = datetime.now()
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
    if embedded_memories:
        bot = get_bot()
        for memory_row in embedded_memories:
            await run_in_threadpool(bot.vector_store_manager.remove_document, memory_row.id)

    # ORM-level delete (not a bulk .filter().delete()) — required for the
    # cascade="all, delete-orphan" relationships on messages/memories to fire.
    db.delete(session_row)
    db.commit()
    logger.info(f"[CONVERSATIONS] Deleted session_id={session_id}")
    return ConversationDeleteResponse(
        success=True, message=f"Conversation {session_id} deleted", session_id=session_id
    )
