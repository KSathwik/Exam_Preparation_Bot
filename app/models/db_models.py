"""SQLAlchemy ORM models for persistent storage."""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship, DeclarativeBase


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
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

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
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("DocumentRecord", back_populates="queries")


class ChatSession(Base):
    """Optional: persistent chat sessions."""
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship("ChatMessageRecord", back_populates="session", cascade="all, delete-orphan")


class ChatMessageRecord(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(16), nullable=False)
    content = Column(Text, nullable=False)
    intent = Column(String(32), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")
