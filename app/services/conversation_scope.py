"""Resolves the authoritative document scope for a conversation.

Client-sent ``document_ids`` (from the frontend's transient, per-tab
``state.documentIds`` — see ``frontend/js/state.js``) is never trusted as the
sole source of truth: it resets on every page reload and conversation switch,
which is exactly how one conversation's documents used to leak into another's
answers. The database is authoritative; a client-sent value can only narrow
within it, never expand beyond it.
"""

from typing import Callable, List, Optional

from app.models.db_models import DocumentRecord


def resolve_document_scope(
    db_session_factory: Callable,
    session_id: Optional[str],
    client_ids: Optional[List[str]],
) -> Optional[List[str]]:
    """Return the document ids retrieval should be scoped to.

    No ``session_id`` -> there is no conversation to scope by (legacy/no-
    session caller) -> ``client_ids`` is returned unchanged. Given a
    ``session_id``, the conversation's own document set (possibly empty) is
    authoritative; a non-empty ``client_ids`` narrows within it (e.g. a
    suggestion-chip click scoping to one specific document) but can never
    reach outside it.
    """
    if not session_id:
        return client_ids

    db = db_session_factory()
    try:
        session_scoped = [d.id for d in db.query(DocumentRecord).filter_by(session_id=session_id).all()]
    finally:
        db.close()

    if not client_ids:
        return session_scoped

    allowed = set(session_scoped)
    narrowed = [doc_id for doc_id in client_ids if doc_id in allowed]
    return narrowed if narrowed else session_scoped
