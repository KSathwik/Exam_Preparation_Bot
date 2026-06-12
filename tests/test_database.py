"""Tests for SQLAlchemy models and database setup."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.db_models import Base, DocumentRecord, QueryRecord


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_create_tables(db_session):
    """Tables should be created without errors."""
    assert db_session.query(DocumentRecord).count() == 0


def test_insert_document(db_session):
    doc = DocumentRecord(
        id="doc-1",
        file_name="test.pdf",
        file_type="pdf",
        file_size_mb=1.5,
        total_chunks=10,
    )
    db_session.add(doc)
    db_session.commit()

    fetched = db_session.query(DocumentRecord).filter_by(id="doc-1").first()
    assert fetched is not None
    assert fetched.file_name == "test.pdf"
    assert fetched.total_chunks == 10


def test_insert_query_record(db_session):
    doc = DocumentRecord(
        id="doc-2", file_name="a.pdf", file_type="pdf", file_size_mb=1.0, total_chunks=5
    )
    db_session.add(doc)
    db_session.commit()

    query = QueryRecord(
        id="q-1",
        document_id="doc-2",
        query_text="What is X?",
        answer_text="X is ...",
        intent="definition",
        overall_confidence=0.85,
    )
    db_session.add(query)
    db_session.commit()

    assert db_session.query(QueryRecord).count() == 1
    assert db_session.query(QueryRecord).first().intent == "definition"


def test_cascade_delete(db_session):
    doc = DocumentRecord(
        id="doc-3", file_name="b.pdf", file_type="pdf", file_size_mb=2.0, total_chunks=3
    )
    db_session.add(doc)
    db_session.flush()

    q = QueryRecord(id="q-2", document_id="doc-3", query_text="Test?")
    db_session.add(q)
    db_session.commit()

    db_session.delete(doc)
    db_session.commit()
    assert db_session.query(QueryRecord).count() == 0
