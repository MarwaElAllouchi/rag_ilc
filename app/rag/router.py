from __future__ import annotations

import logging
from typing import Any

from app.core.config import get_settings
from app.rag.business_rules import BusinessRulesEngine
from app.rag.pipeline import RagPipeline
from app.rag.query_analyzer import QueryAnalyzer
from app.rag.small_talk_detector import SmallTalkDetector


logger = logging.getLogger(__name__)


class RagRouter:
    """
    Route les requêtes utilisateur vers :
    - small talk local (salutation, merci, au revoir, etc.)
    - moteur métier tarif / niveau
    - mode hybride inscription
    - RAG général pour le reste
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.pipeline = RagPipeline()
        self.business_engine = BusinessRulesEngine()
        self.small_talk_detector = SmallTalkDetector()

    @staticmethod
    def _clean_question(question: str) -> str:
        """
        Nettoie la question utilisateur.
        """
        if not isinstance(question, str):
            return ""
        return question.strip()

    @staticmethod
    def _build_response(
        answer: str,
        documents: list[dict[str, Any]] | None = None,
        intent: str = "general",
        route: str = "unknown",
    ) -> dict[str, Any]:
        """
        Construit une réponse homogène du router.
        """
        return {
            "answer": answer,
            "documents": documents or [],
            "intent": intent,
            "route": route,
        }

    def route(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        """
        Route une question utilisateur vers le bon moteur.
        """
        cleaned_question = self._clean_question(question)
        if not cleaned_question:
            logger.warning("Router : question vide reçue.")
            return self._build_response(
                answer=(
                    "Je n’ai pas bien compris votre demande. "
                    "Pouvez-vous reformuler votre question, par exemple sur les tarifs, "
                    "les niveaux, l’inscription ou les informations générales de l’institut ?"
                ),
                intent="empty",
                route="empty_input",
            )

        effective_top_k = top_k if top_k is not None else self.settings.top_k
        if effective_top_k <= 0:
            raise ValueError("top_k doit être strictement positif")

        # 1. Small talk local
        small_talk = self.small_talk_detector.detect(cleaned_question)

        if small_talk.kind == "greeting":
            return self._build_response(
                answer=(
                    "Bonjour 👋 Je suis l’assistant virtuel de l’Institut ILC. "
                    "Je peux vous aider concernant les niveaux, les tarifs, l’inscription "
                    "et les informations générales de l’institut. Comment puis-je vous aider ?"
                ),
                intent="greeting",
                route="small_talk_greeting",
            )

        if small_talk.kind == "gratitude":
            return self._build_response(
                answer="Avec plaisir 😊 N’hésitez pas si vous avez d’autres questions.",
                intent="gratitude",
                route="small_talk_gratitude",
            )

        if small_talk.kind == "farewell":
            return self._build_response(
                answer="Au revoir 👋 N’hésitez pas à revenir si vous avez d’autres questions.",
                intent="farewell",
                route="small_talk_farewell",
            )

        if small_talk.kind == "acknowledgment":
            return self._build_response(
                answer="Très bien 😊 Je reste à votre disposition si vous avez besoin d’autres informations.",
                intent="acknowledgment",
                route="small_talk_acknowledgment",
            )

        if small_talk.kind == "praise":
            return self._build_response(
                answer="Merci beaucoup 😊 Je reste à votre disposition si vous avez besoin d’autres informations.",
                intent="praise",
                route="small_talk_praise",
            )

        # 2. Si la salutation était suivie d'une vraie question, on traite la question nettoyée
        if getattr(small_talk, "cleaned_text", ""):
            cleaned_question = self._clean_question(small_talk.cleaned_text)

        if not cleaned_question:
            logger.warning("Router : question vide après nettoyage small talk.")
            return self._build_response(
                answer=(
                    "Je suis à votre disposition. "
                    "Vous pouvez par exemple me poser une question sur les tarifs, "
                    "les niveaux ou l’inscription."
                ),
                intent="empty_after_small_talk_cleaning",
                route="empty_after_small_talk_cleaning",
            )

        # 3. Détection d'intention métier
        intent = QueryAnalyzer.detect_intent(cleaned_question)
        logger.info("Router → intention détectée : %s", intent)

        if intent == "tarif":
            return self._handle_tarif(cleaned_question)

        if intent == "niveau":
            return self._handle_niveau(cleaned_question, effective_top_k)

        if intent == "inscription":
            return self._handle_inscription(cleaned_question, effective_top_k)

        return self._handle_general(cleaned_question, effective_top_k)

    def _handle_tarif(self, question: str) -> dict[str, Any]:
        logger.info("Router → moteur hybride TARIF")

        # 1. question globale sur tous les tarifs
        if self.business_engine.is_tarif_global_question(question):
            answer = self.business_engine.build_tarif_answer()
            return self._build_response(
                answer=answer,
                intent="tarif",
                route="tarif_business_engine_global",
            )

        # 2. question sur une formule précise
        specific_answer = self.business_engine.build_tarif_specific_answer(question)
        if specific_answer:
            return self._build_response(
                answer=specific_answer,
                intent="tarif",
                route="tarif_business_engine_specific",
            )

        # 3. si la question parle d’un niveau, on évite toute invention
        if QueryAnalyzer.contains_niveau_pattern(question):
            return self._build_response(
                answer=(
                    "Je n’ai pas de tarif spécifique par niveau pour le moment. "
                    "Les tarifs disponibles sont définis par formule. "
                    "Souhaitez-vous que je vous présente les formules actuellement disponibles ?"
                ),
                intent="tarif",
                route="tarif_business_engine_no_level_pricing",
            )

        # 4. fallback cadré sans RAG libre
        return self._build_response(
            answer=(
                "Je n’ai pas trouvé de formule tarifaire correspondant précisément à votre demande. "
                "Je peux toutefois vous présenter les tarifs actuellement disponibles si vous le souhaitez."
            ),
            intent="tarif",
            route="tarif_business_engine_fallback",
        )

    def _handle_niveau(self, question: str, top_k: int) -> dict[str, Any]:
        logger.info("Router → moteur hybride NIVEAU")

        # 1. année de naissance
        birth_year_answer = self.business_engine.build_niveau_birth_year_answer(question)
        if birth_year_answer:
            return self._build_response(
                answer=birth_year_answer,
                intent="niveau",
                route="niveau_business_engine_birth_year",
            )

        # 2. niveau explicite
        niveau_answer = self.business_engine.build_niveau_specific_answer(question)
        if niveau_answer:
            return self._build_response(
                answer=niveau_answer,
                intent="niveau",
                route="niveau_business_engine_direct",
            )

        # 3. cycle explicite
        cycle_answer = self.business_engine.build_cycle_answer(question)
        if cycle_answer:
            return self._build_response(
                answer=cycle_answer,
                intent="niveau",
                route="cycle_business_engine_direct",
            )

        # 4. fallback RAG
        result = self.pipeline.run(
            user_question=question,
            top_k=top_k,
            intent="niveau",
        )

        return self._build_response(
            answer=result["answer"],
            documents=result.get("documents", []),
            intent="niveau",
            route="niveau_rag_detail",
        )

    def _handle_inscription(self, question: str, top_k: int) -> dict[str, Any]:
        logger.info("Router → mode hybride INSCRIPTION")

        result = self.pipeline.run(
            user_question=question,
            top_k=top_k,
            intent="inscription",
        )

        return self._build_response(
            answer=result["answer"],
            documents=result.get("documents", []),
            intent="inscription",
            route="inscription_hybrid",
        )

    def _handle_general(self, question: str, top_k: int) -> dict[str, Any]:
        logger.info("Router → mode RAG GENERAL")

        result = self.pipeline.run(
            user_question=question,
            top_k=top_k,
            intent="general",
        )

        return self._build_response(
            answer=result["answer"],
            documents=result.get("documents", []),
            intent="general",
            route="rag_general",
        )