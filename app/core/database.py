"""Database initialization and session management."""

from pathlib import Path

from loguru import logger
from sqlalchemy import create_engine
from typing import Iterator
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.models.db_models import Base

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables and runtime directories."""
    logger.info("Initializing database...")
    Base.metadata.create_all(bind=engine)

    for d in [settings.upload_dir, settings.cache_dir, settings.faiss_index_path, "./logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    logger.info("Database initialized successfully")


def get_db() -> Iterator[Session]:
    """FastAPI dependency that yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
