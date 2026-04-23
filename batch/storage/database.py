from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


settings = get_settings()


class Base(DeclarativeBase):
    """
    Base déclarative SQLAlchemy pour tous les modèles ORM.
    """
    pass


engine = create_engine(
    settings.database_url,
    echo=settings.is_development and settings.app_debug,
    future=True,
    pool_pre_ping=True,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_recycle=settings.db_pool_recycle,
)


SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    future=True,
    class_=Session,
)


def get_db_session() -> Generator[Session, None, None]:
    """
    Fournit une session SQLAlchemy utilisable dans FastAPI (dependency injection).
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()