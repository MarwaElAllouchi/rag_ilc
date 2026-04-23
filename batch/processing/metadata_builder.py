from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class MetadataBuilder:
    """
    Standardise et complète les métadonnées des documents RAG
    afin de garantir une structure homogène entre les différentes sources.
    """

    DEFAULT_METADATA = {
        "language": "fr",
        "source_type": "document",
        "category": "general",
        "source_file": "unknown",
    }

    @staticmethod
    def _clean_str(value: Any, default: str = "") -> str:
        """
        Nettoie une valeur texte simple.
        """
        if value is None:
            return default

        text = str(value).strip()
        if not text:
            return default

        return text

    @classmethod
    def _normalize_metadata(
        cls,
        metadata: dict[str, Any],
        source_file: str | None = None,
        language: str = "fr",
        default_source_type: str | None = None,
        default_category: str | None = None,
    ) -> dict[str, Any]:
        """
        Normalise et complète un dictionnaire de métadonnées.
        """
        normalized = cls.DEFAULT_METADATA.copy()
        normalized.update(metadata)

        normalized["language"] = cls._clean_str(
            language or normalized.get("language"),
            default=cls.DEFAULT_METADATA["language"],
        )

        normalized["source_type"] = cls._clean_str(
            normalized.get("source_type"),
            default=default_source_type or cls.DEFAULT_METADATA["source_type"],
        ).lower()

        normalized["category"] = cls._clean_str(
            normalized.get("category"),
            default=default_category or cls.DEFAULT_METADATA["category"],
        ).lower()

        if source_file:
            normalized["source_file"] = Path(source_file).name
        else:
            normalized["source_file"] = cls._clean_str(
                normalized.get("source_file"),
                default=cls.DEFAULT_METADATA["source_file"],
            )

        if "document_id" in normalized:
            normalized["document_id"] = cls._clean_str(normalized.get("document_id"))

        if "faq_id" in normalized:
            normalized["faq_id"] = cls._clean_str(normalized.get("faq_id"))

        return normalized

    @staticmethod
    def _ensure_document_id(
        metadata: dict[str, Any],
        fallback_index: int,
    ) -> dict[str, Any]:
        """
        Ajoute un document_id si absent.
        """
        if not metadata.get("document_id"):
            source_file = metadata.get("source_file", "unknown")
            stem = Path(source_file).stem if source_file else "document"
            metadata["document_id"] = f"{stem}_{fallback_index + 1}"

        return metadata

    @classmethod
    def enrich_documents(
        cls,
        documents: list[dict[str, Any]],
        source_file: str | None = None,
        language: str = "fr",
        default_source_type: str | None = None,
        default_category: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Harmonise les métadonnées des documents :
        - conserve les métadonnées existantes
        - ajoute les valeurs manquantes
        - ignore les documents invalides ou sans contenu
        - garantit une structure homogène
        """
        enriched_docs: list[dict[str, Any]] = []
        skipped_count = 0

        for idx, doc in enumerate(documents):
            if not isinstance(doc, dict):
                skipped_count += 1
                continue

            content = str(doc.get("content", "")).strip()
            metadata = doc.get("metadata", {})

            if not content:
                skipped_count += 1
                continue

            if not isinstance(metadata, dict):
                skipped_count += 1
                continue

            normalized_metadata = cls._normalize_metadata(
                metadata=metadata.copy(),
                source_file=source_file,
                language=language,
                default_source_type=default_source_type,
                default_category=default_category,
            )

            normalized_metadata = cls._ensure_document_id(
                metadata=normalized_metadata,
                fallback_index=idx,
            )

            enriched_docs.append(
                {
                    "content": content,
                    "metadata": normalized_metadata,
                }
            )

        logger.info(
            "MetadataBuilder : %s documents enrichis, %s ignorés",
            len(enriched_docs),
            skipped_count,
        )
        return enriched_docs