from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config import get_settings
from batch.storage.database import Base, SessionLocal, engine


logger = logging.getLogger(__name__)
settings = get_settings()


def build_doc_key(content: str, metadata: dict[str, Any]) -> str:
    """
    Construit une clé unique stable pour un document.
    """
    key_payload = {
        "content": content,
        "source_file": metadata.get("source_file"),
        "chunk_index": metadata.get("chunk_index"),
        "source_type": metadata.get("source_type"),
        "category": metadata.get("category"),
        "document_id": metadata.get("document_id"),
        "faq_id": metadata.get("faq_id"),
    }

    raw = json.dumps(key_payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DocumentEmbedding(Base):
    """
    Table principale qui stocke :
    - le texte
    - les métadonnées
    - l'embedding vectoriel
    """

    __tablename__ = "document_embeddings"
    __table_args__ = {"schema": "rag"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.embedding_dimension),
        nullable=False,
    )


def create_tables() -> None:
    """
    Crée les extensions utiles et les tables SQLAlchemy si elles n'existent pas.
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS rag"))

    Base.metadata.create_all(bind=engine)
    logger.info("Tables créées ou déjà existantes.")


def validate_document_for_storage(doc: dict[str, Any]) -> dict[str, Any]:
    """
    Valide et normalise un document avant insertion/upsert.
    Format attendu :
    {
        "content": "...",
        "metadata": {...},
        "embedding": [...]
    }
    """
    if not isinstance(doc, dict):
        raise ValueError("Chaque document doit être un dictionnaire.")

    content = doc.get("content", "")
    metadata = doc.get("metadata", {})
    embedding = doc.get("embedding")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Le champ 'content' doit être une chaîne non vide.")

    if not isinstance(metadata, dict):
        raise ValueError("Le champ 'metadata' doit être un dictionnaire.")

    if not isinstance(embedding, list) or not embedding:
        raise ValueError("Le champ 'embedding' doit être une liste non vide.")

    if not all(isinstance(value, (int, float)) for value in embedding):
        raise ValueError("Tous les éléments de l'embedding doivent être numériques.")

    return {
        "content": content.strip(),
        "metadata": metadata,
        "embedding": embedding,
    }


def prepare_rows_for_upsert(
    documents_with_embeddings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Prépare les lignes pour l'upsert SQL.
    """
    rows: list[dict[str, Any]] = []

    for doc in documents_with_embeddings:
        validated_doc = validate_document_for_storage(doc)

        doc_key = build_doc_key(
            content=validated_doc["content"],
            metadata=validated_doc["metadata"],
        )

        rows.append(
            {
                "doc_key": doc_key,
                "content": validated_doc["content"],
                "metadata_json": validated_doc["metadata"],
                "embedding": validated_doc["embedding"],
            }
        )

    return rows


def upsert_documents(documents_with_embeddings: list[dict[str, Any]]) -> None:
    """
    Insert ou met à jour les documents selon doc_key.
    """
    if not documents_with_embeddings:
        logger.warning("Aucun document à insérer ou mettre à jour.")
        return

    rows = prepare_rows_for_upsert(documents_with_embeddings)

    with SessionLocal() as session:
        try:
            stmt = insert(DocumentEmbedding).values(rows)

            stmt = stmt.on_conflict_do_update(
                index_elements=["doc_key"],
                set_={
                    "content": stmt.excluded.content,
                    "metadata_json": stmt.excluded.metadata_json,
                    "embedding": stmt.excluded.embedding,
                },
            )

            session.execute(stmt)
            session.commit()

            logger.info("%s documents upsertés en base.", len(rows))

        except Exception as exc:
            session.rollback()
            logger.exception("Erreur lors du upsert en base : %s", exc)
            raise