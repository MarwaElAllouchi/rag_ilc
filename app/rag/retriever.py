from __future__ import annotations
import unicodedata
import logging
import re
from typing import Any

from sqlalchemy import or_, select

from app.core.config import get_settings
from batch.embeddings.embedder import (
    EmbeddingServiceUnavailableError,
    MistralEmbedder,
)
from batch.storage.database import SessionLocal
from batch.storage.pgvector_store import DocumentEmbedding


logger = logging.getLogger(__name__)


class Retriever:
    """
    Recherche hybride :
    1. récupération vectorielle initiale via pgvector
    2. reranking générique local avec score enrichi
    """

    FAQ_PRIORITY_BONUS = 0.03
    EXACT_QUESTION_BONUS = 0.08
    HIGH_OVERLAP_BONUS = 0.06
    MEDIUM_OVERLAP_BONUS = 0.04
    LOW_OVERLAP_BONUS = 0.02

    CATEGORY_MATCH_BONUS = 0.02

    FAQ_SOURCE_TYPE = "faq"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedder = MistralEmbedder()

    @staticmethod
    def _clean_query(query: str) -> str:
        if not isinstance(query, str):
            return ""
        return query.strip()

    @staticmethod
    def _normalize_values(values: list[str] | None) -> list[str]:
        if not values:
            return []

        cleaned: list[str] = []
        for value in values:
            if not isinstance(value, str):
                continue
            normalized = value.strip().lower()
            if normalized and normalized not in cleaned:
                cleaned.append(normalized)
        return cleaned
    @staticmethod
    def _normalize_text(value: str) -> str:
        if not isinstance(value, str):
            return ""

        text = value.lower().strip()
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    @classmethod
    def _tokenize(cls, value: str) -> list[str]:
        text = cls._normalize_text(value)
        if not text:
            return []
        return text.split()

    @classmethod
    def _question_overlap_bonus(cls, query: str, content: str) -> float:
        """
        Bonus si le contenu ressemble fortement à la formulation de la question.
        """
        normalized_query = cls._normalize_text(query)
        normalized_content = cls._normalize_text(content)

        if not normalized_query or not normalized_content:
            return 0.0

        if normalized_query in normalized_content:
            return cls.EXACT_QUESTION_BONUS

        query_words = set(cls._tokenize(normalized_query))
        content_words = set(cls._tokenize(normalized_content))

        if not query_words or not content_words:
            return 0.0

        overlap_ratio = len(query_words & content_words) / len(query_words)

        if overlap_ratio >= 0.8:
            return cls.HIGH_OVERLAP_BONUS
        if overlap_ratio >= 0.6:
            return cls.MEDIUM_OVERLAP_BONUS
        if overlap_ratio >= 0.4:
            return cls.LOW_OVERLAP_BONUS

        return 0.0

    @classmethod
    def _keyword_overlap_bonus(cls, query: str, content: str) -> float:
        """
        Bonus lexical générique :
        plus il y a de mots significatifs de la question dans le document,
        plus le document remonte.
        """
        query_words = [w for w in cls._tokenize(query) if len(w) >= 4]
        normalized_content = cls._normalize_text(content)

        if not query_words or not normalized_content:
            return 0.0

        overlap_count = sum(1 for word in query_words if word in normalized_content)

        if overlap_count >= 5:
            return 0.08
        if overlap_count >= 4:
            return 0.06
        if overlap_count >= 3:
            return 0.04
        if overlap_count >= 2:
            return 0.02

        return 0.0

    @classmethod
    def _infer_query_categories(cls, query: str) -> set[str]:
        """
        Inférence légère et générique de catégories possibles
        à partir du vocabulaire de la question.
        """
        q = cls._normalize_text(query)

        inferred: set[str] = set()

        if any(word in q for word in ["tarif", "prix", "cout", "payer", "paiement"]):
            inferred.add("tarif")
            inferred.add("paiement")

        if any(word in q for word in ["inscription", "inscrire", "dossier", "documents", "myscol"]):
            inferred.add("inscription")

        if any(word in q for word in ["niveau", "cycle", "classe", "test de niveau", "année", "annee"]):
            inferred.add("niveau")

        if any(word in q for word in ["reglement", "autorisé", "interdit", "sortie", "absence", "sécurité"]):
            inferred.add("reglement")

        return inferred

    @classmethod
    def _category_bonus(cls, query: str, metadata: dict[str, Any]) -> float:
        """
        Bonus léger si la catégorie du document semble cohérente
        avec les mots de la question.
        """
        doc_category = str(metadata.get("category", "")).strip().lower()
        if not doc_category:
            return 0.0

        inferred_categories = cls._infer_query_categories(query)
        if doc_category in inferred_categories:
            return cls.CATEGORY_MATCH_BONUS

        return 0.0

    def _apply_generic_reranking(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Reranking générique sans règle métier ultra-spécifique.
        """
        reranked_documents: list[dict[str, Any]] = []

        for doc in documents:
            metadata = doc.get("metadata", {}) or {}
            source_type = str(metadata.get("source_type", "")).strip().lower()

            distance = float(doc.get("distance", 999.0))
            content = str(doc.get("content", ""))

            bonus = 0.0

            if source_type == self.FAQ_SOURCE_TYPE:
                bonus += self.FAQ_PRIORITY_BONUS

            bonus += self._question_overlap_bonus(query=query, content=content)
            bonus += self._keyword_overlap_bonus(query=query, content=content)
            bonus += self._category_bonus(query=query, metadata=metadata)

            adjusted_distance = max(0.0, distance - bonus)

            enriched_doc = dict(doc)
            enriched_doc["rerank_bonus"] = bonus
            enriched_doc["adjusted_distance"] = adjusted_distance
            reranked_documents.append(enriched_doc)

        reranked_documents.sort(
            key=lambda doc: (
                doc.get("adjusted_distance", 999.0),
                doc.get("distance", 999.0),
            )
        )

        return reranked_documents

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        categories: list[str] | None = None,
        source_types: list[str] | None = None,
        max_distance: float | None = None,
    ) -> list[dict[str, Any]]:
        cleaned_query = self._clean_query(query)
        if not cleaned_query:
            logger.warning("Requête vide reçue dans Retriever.")
            return []

        k = top_k if top_k is not None else self.settings.top_k
        if k <= 0:
            raise ValueError("top_k doit être strictement positif")

        normalized_categories = self._normalize_values(categories)
        normalized_source_types = self._normalize_values(source_types)

        logger.info(
            "Recherche sémantique lancée (top_k=%s, categories=%s, source_types=%s, max_distance=%s)",
            k,
            normalized_categories or None,
            normalized_source_types or None,
            max_distance,
        )

        try:
            query_embedding = self.embedder.embed_single_text(cleaned_query)
        except EmbeddingServiceUnavailableError:
            logger.warning("Service d'embedding indisponible pour la requête.")
            raise
        except Exception:
            logger.exception("Erreur lors de la vectorisation de la requête utilisateur")
            raise

        # On récupère un peu plus de candidats pour laisser le reranker travailler.
        initial_k = max(k * 2, 8)

        with SessionLocal() as session:
            distance_expr = DocumentEmbedding.embedding.cosine_distance(query_embedding)

            stmt = select(
                DocumentEmbedding,
                distance_expr.label("distance"),
            )

            if normalized_categories:
                stmt = stmt.where(
                    or_(
                        *[
                            DocumentEmbedding.metadata_json["category"].as_string() == category
                            for category in normalized_categories
                        ]
                    )
                )
                logger.info(
                    "Filtre métier appliqué sur les catégories : %s",
                    normalized_categories,
                )

            if normalized_source_types:
                stmt = stmt.where(
                    or_(
                        *[
                            DocumentEmbedding.metadata_json["source_type"].as_string() == source_type
                            for source_type in normalized_source_types
                        ]
                    )
                )
                logger.info(
                    "Filtre appliqué sur les types de source : %s",
                    normalized_source_types,
                )

            if max_distance is not None:
                stmt = stmt.where(distance_expr <= max_distance)

            stmt = stmt.order_by(distance_expr).limit(initial_k)

            results = session.execute(stmt).all()

            documents = [
                {
                    "id": row.DocumentEmbedding.id,
                    "doc_key": row.DocumentEmbedding.doc_key,
                    "content": row.DocumentEmbedding.content,
                    "metadata": row.DocumentEmbedding.metadata_json,
                    "distance": float(row.distance),
                }
                for row in results
            ]

        documents = self._apply_generic_reranking(
            query=cleaned_query,
            documents=documents,
        )

        documents = documents[:k]

        logger.info(
            "Retriever : %s documents récupérés pour la requête",
            len(documents),
        )
        return documents