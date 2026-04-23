from __future__ import annotations

import logging
from typing import Any

from app.core.config import get_settings
from app.rag.generator import (
    GenerationServiceUnavailableError,
    ResponseGenerator,
)
from app.rag.prompt_builder import PromptBuilder
from app.rag.query_analyzer import QueryAnalyzer
from app.rag.retriever import Retriever
from batch.embeddings.embedder import EmbeddingServiceUnavailableError


logger = logging.getLogger(__name__)


class RagPipeline:
    """
    VERSION PROD STABLE :
    - retrieval simple
    - anti-hallucination (seuil)
    - fallback sécurisé
    - FAQ shortcut
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.retriever = Retriever()
        self.generator = ResponseGenerator()

    @staticmethod
    def _clean_question(user_question: str) -> str:
        if not isinstance(user_question, str):
            return ""
        return user_question.strip()

    @staticmethod
    def _build_response(
        answer: str,
        documents: list[dict[str, Any]] | None = None,
        intent: str = "general",
    ) -> dict[str, Any]:
        return {
            "answer": answer,
            "documents": documents or [],
            "intent": intent,
        }

    @staticmethod
    def _build_no_document_answer(intent: str) -> str:
        return "Je n’ai pas cette information pour le moment."

    @staticmethod
    def _extract_faq_answer_from_content(content: str) -> str | None:
        if not isinstance(content, str):
            return None

        marker = "Réponse :"
        if marker not in content:
            return None

        return content.split(marker, 1)[1].strip() or None

    @staticmethod
    def _should_short_circuit_with_faq(documents: list[dict[str, Any]]) -> bool:
        if not documents:
            return False

        top1 = documents[0]
        metadata = top1.get("metadata", {}) or {}

        if str(metadata.get("source_type", "")).lower() != "faq":
            return False

        top1_dist = float(top1.get("adjusted_distance") or 999.0)

        # seuil strict FAQ
        if top1_dist > 0.20:
            return False

        if len(documents) == 1:
            return True

        top2_dist = float(documents[1].get("adjusted_distance") or 999.0)

        return (top2_dist - top1_dist) >= 0.08

    def run(
        self,
        user_question: str,
        top_k: int | None = None,
        intent: str | None = None,
    ) -> dict[str, Any]:

        cleaned_question = self._clean_question(user_question)

        if not cleaned_question:
            return self._build_response(
                answer="Pouvez-vous préciser votre question, s’il vous plaît ?",
                intent="general",
            )

        logger.info("Pipeline RAG : %s", cleaned_question)

        effective_intent = intent or QueryAnalyzer.detect_intent(cleaned_question)

        try:
            documents = self.retriever.retrieve(
                query=cleaned_question,
                top_k=top_k or self.settings.top_k,
            )
        except EmbeddingServiceUnavailableError:
            return self._build_response(
                answer="Le service de recherche est indisponible.",
                intent=effective_intent,
            )
        except Exception:
            logger.exception("Erreur retrieval")
            return self._build_response(
                answer="Une erreur technique est survenue.",
                intent=effective_intent,
            )

        # ❌ aucun document
        if not documents:
            return self._build_response(
                answer=self._build_no_document_answer(effective_intent),
                intent=effective_intent,
            )

        # 🔥 ANTI-HALLUCINATION (CRITIQUE)
        best_distance = float(documents[0].get("adjusted_distance") or 999.0)
        threshold = self.settings.max_retrieval_distance

        logger.info("Distance=%.3f | threshold=%.2f", best_distance, threshold)

        if best_distance > threshold:
            logger.warning("Fallback : documents non pertinents")
            return self._build_response(
                answer="Je n’ai pas cette information pour le moment.",
                intent=effective_intent,
            )

        # ⚡ FAQ direct
        if self._should_short_circuit_with_faq(documents):
            faq_answer = self._extract_faq_answer_from_content(
                documents[0].get("content", "")
            )
            if faq_answer:
                return self._build_response(
                    answer=faq_answer,
                    documents=[documents[0]],
                    intent=effective_intent,
                )

        # 🧠 génération
        context_docs = documents[:5]

        messages = PromptBuilder.build_prompt(
            user_question=cleaned_question,
            documents=context_docs,
        )

        try:
            answer = self.generator.generate(messages)
        except GenerationServiceUnavailableError:
            return self._build_response(
                answer="Le service de réponse est indisponible.",
                documents=context_docs,
                intent=effective_intent,
            )
        except Exception:
            logger.exception("Erreur génération")
            return self._build_response(
                answer="Une erreur est survenue lors de la réponse.",
                documents=context_docs,
                intent=effective_intent,
            )

        return self._build_response(
            answer=answer,
            documents=context_docs,
            intent=effective_intent,
        )