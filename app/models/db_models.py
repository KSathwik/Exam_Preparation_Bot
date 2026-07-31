"""SQLAlchemy ORM models for persistent storage."""

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    """Tracks every uploaded document."""

    __tablename__ = "documents"

    id = Column(String(36), primary_key=True)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False)
    file_size_mb = Column(Float, nullable=False)
    total_chunks = Column(Integer, nullable=False, default=0)
    upload_path = Column(String(512), nullable=True)
    # Which conversation this document was uploaded into — the authoritative
    # scope retrieval uses to keep one conversation's documents from leaking
    # into another's answers (see AdaptiveRetriever.retrieve /
    # resolve_document_scope). Nullable: documents uploaded before this
    # column existed have no known conversation and are only reachable via an
    # explicit global search. Deliberately not a ForeignKey: ChatSession
    # already has a document_id -> documents.id FK (see below), and a real FK
    # back from documents -> chat_sessions would form the same
    # mutually-dependent-tables cycle SQLite cannot create/migrate cleanly
    # that last_summarized_message_id avoids below — enforced at the
    # application layer only (see get_or_create_chat_session).
    session_id = Column(String(36), nullable=True, index=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)

    queries = relationship("QueryRecord", back_populates="document", cascade="all, delete-orphan")


class QueryRecord(Base):
    """Stores every question asked against a document."""

    __tablename__ = "queries"

    id = Column(String(36), primary_key=True)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=True)
    query_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=True)
    intent = Column(String(32), nullable=True)
    intent_confidence = Column(Float, nullable=True)
    overall_confidence = Column(Float, nullable=True)
    hallucination_risk = Column(String(16), nullable=True)
    response_time_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)

    document = relationship("DocumentRecord", back_populates="queries")


class ChatSession(Base):
    """Persistent chat session — backs the Memory Agent's short-term/episodic tiers."""

    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=True)
    # Nullable, forward-compat only: today's single shared APP_API_KEY model has
    # no concept of a logged-in user; this makes the schema ready for real
    # multi-user auth later without another migration (see architecture plan).
    user_id = Column(String(36), nullable=True)
    # Anonymous per-browser identity (crypto.randomUUID(), persisted client-side
    # in localStorage) used to scope the chat-history sidebar to one browser —
    # deliberately a separate column from user_id, which is reserved for a real
    # authenticated user later; conflating the two would make user_id ambiguous
    # once real auth lands.
    device_id = Column(String(36), nullable=True, index=True)
    # Derived once from the first user message (truncated, no LLM call) when
    # this session is created; user-renameable afterward via PATCH.
    title = Column(String(255), nullable=True)
    # Running counters, updated alongside message inserts, so the
    # summarization-trigger check (every N turns / token threshold) is an O(1)
    # column read instead of a COUNT(*) over chat_messages.
    turn_count = Column(Integer, nullable=False, default=0)
    # Deliberately not a ForeignKey: chat_messages.session_id already points
    # back at this table, and a real FK constraint here would form a circular
    # dependency between chat_sessions/chat_messages that SQLite (the dev DB)
    # cannot create or migrate cleanly. This logically references
    # chat_messages.id but is enforced at the application layer only.
    last_summarized_message_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    messages = relationship("ChatMessageRecord", back_populates="session", cascade="all, delete-orphan")
    memories = relationship("ConversationMemory", back_populates="session", cascade="all, delete-orphan")


class ChatMessageRecord(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String(16), nullable=False)
    content = Column(Text, nullable=False)
    intent = Column(String(32), nullable=True)
    # The presentation format (e.g. "key_points", "greeting", "mcq" — see
    # response_formats.py) — persisted so history replay can restore
    # format-driven UI behavior (e.g. suppressing badges on a replayed
    # greeting) rather than only ever having it live during the original turn.
    format_type = Column(String(32), nullable=True)
    # Running total (not per-message length) so the token-threshold half of
    # the dual summarization trigger is also an O(1) read.
    token_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    session = relationship("ChatSession", back_populates="messages")


class ConversationMemory(Base):
    """Long-term/semantic memory tier: an LLM-generated summary of a slice of
    a conversation, embedded into the shared FAISS index (content_type
    "memory") for later similarity retrieval — see architecture plan."""

    __tablename__ = "conversation_memories"

    id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    # Denormalized alongside session.user_id: same forward-compat rationale,
    # and lets memory lookups filter by user without a join once real
    # multi-user auth lands.
    user_id = Column(String(36), nullable=True)
    summary_text = Column(Text, nullable=False)
    covers_from_message_id = Column(Integer, ForeignKey("chat_messages.id"), nullable=False)
    covers_to_message_id = Column(Integer, ForeignKey("chat_messages.id"), nullable=False)
    # Defensive flag: a memory row is written before the embedding call, so a
    # crash between the two never loses the summary — an unembedded row is
    # simply invisible to semantic search and can be retried.
    embedded = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=_utcnow, nullable=False)

    session = relationship("ChatSession", back_populates="memories")
