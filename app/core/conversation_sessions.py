"""Shared get-or-create for ``ChatSession`` rows.

Both turn-persistence (``MemoryAgent``) and document upload
(``app.api.documents``) need a ``ChatSession`` row to exist before writing a
child row that references it (``chat_messages``/``conversation_memories`` via
a real FK, ``documents.session_id`` at the application layer — see
``db_models.py``). Centralized here so there's one get-or-create instead of
two copies of the same logic.
"""

from typing import Optional

from sqlalchemy.orm import Session

from app.models.db_models import ChatSession


def get_or_create_chat_session(
    db: Session,
    session_id: str,
    device_id: Optional[str] = None,
    title: Optional[str] = None,
) -> ChatSession:
    session_row = db.get(ChatSession, session_id)
    if session_row is None:
        session_row = ChatSession(id=session_id, device_id=device_id, title=title)
        db.add(session_row)
        db.flush()
    elif device_id and not session_row.device_id:
        session_row.device_id = device_id  # opportunistic backfill for older/partial rows
    return session_row
