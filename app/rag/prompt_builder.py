from __future__ import annotations

import logging
import re
from typing import Any

from app.core.config import get_settings


logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Construit le prompt envoyé au LLM à partir :
    - de la question utilisateur
    - des documents récupérés par le retriever
    """

    @staticmethod
    def _clean_text(value: Any) -> str:
        """
        Nettoie légèrement un texte pour le prompt.
        """
        if value is None:
            return ""

        text = str(value).strip()
        if not text:
            return ""

        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def build_context(
        cls,
        documents: list[dict[str, Any]],
        max_documents: int | None = None,
    ) -> str:
        """
        Construit le bloc de contexte à partir des documents retrouvés.
        """
        if not documents:
            return ""

        settings = get_settings()
        effective_max_documents = (
            max_documents if max_documents is not None else settings.max_context_chunks
        )

        selected_documents = documents[:effective_max_documents]
        context_parts: list[str] = []

        for idx, doc in enumerate(selected_documents, start=1):
            content = cls._clean_text(doc.get("content", ""))
            metadata = doc.get("metadata", {}) or {}

            if not content:
                continue

            source_file = cls._clean_text(metadata.get("source_file", "inconnu"))
            category = cls._clean_text(metadata.get("category", "general"))
            source_type = cls._clean_text(metadata.get("source_type", "document"))

            context_parts.append(
                "\n".join(
                    [
                        f"[Extrait {idx}]",
                        f"Type: {source_type}",
                        f"Catégorie: {category}",
                        f"Fichier: {source_file}",
                        f"Texte: {content}",
                    ]
                )
            )

        context = "\n\n".join(context_parts)

        logger.info(
            "Contexte construit : %s documents inclus",
            len(context_parts),
        )
        return context

    @staticmethod
    def build_system_prompt() -> str:
        return (
            "Vous êtes l'assistant virtuel de l'Institut Langues et Cultures.\n"
            "Vous répondez aux parents, élèves et visiteurs de manière claire, polie, naturelle et professionnelle.\n"
            "\n"
            "Règles obligatoires :\n"
            "- Répondez uniquement à partir des textes fournis.\n"
            "- N'inventez jamais d'information.\n"
            "- N'utilisez pas vos connaissances générales.\n"
            "- Ne déduisez jamais une information absente.\n"
            "- Si aucun texte ne permet de répondre, répondez exactement : Je n'ai pas cette information pour le moment.\n"
            "\n"
            "Utilisation des textes :\n"
            "- Si une information partielle ou indirecte existe, utilisez-la pour construire une réponse fidèle.\n"
            "- Reformulez les informations pour répondre clairement à la question.\n"
            "- Si plusieurs extraits sont pertinents, combinez-les.\n"
            "- Ne répondez pas 'Je n'ai pas cette information' si un élément de réponse existe réellement dans les textes.\n"
            "\n"
            "Cas spécifiques :\n"
            "- Pour une question fermée (oui/non), utilisez les règles présentes pour répondre fidèlement.\n"
            "- Si le texte indique une autorisation, répondez oui avec les conditions.\n"
            "- Si le texte indique une interdiction, répondez non clairement.\n"
            "- Pour toute règle ou réglementation, mentionnez d'abord la règle générale, puis les exceptions s'il y en a.\n"
            "\n"
            "Style attendu :\n"
            "- Ton poli, professionnel et rassurant.\n"
            "- Réponse naturelle, directe et concise.\n"
            "- Pas de répétition.\n"
            "- Ne mentionnez jamais les mots 'documents', 'sources' ou 'contexte'.\n"
            "\n"
            "Important final :\n"
            "- Répondez comme un humain de l'institut.\n"
            "- Ne dites 'Je n'ai pas cette information pour le moment' que si absolument aucun élément utile n'existe dans les textes.\n"
        )

    @classmethod
    def build_user_prompt(
        cls,
        user_question: str,
        context: str,
    ) -> str:
        cleaned_question = cls._clean_text(user_question)

        if context:
            return (
                f"Question utilisateur :\n{cleaned_question}\n\n"
                f"Textes disponibles :\n{context}\n\n"
                "Analysez les textes et donnez la meilleure réponse possible, de façon naturelle, directe et fidèle aux informations présentes."
            )

        return (
            f"Question utilisateur :\n{cleaned_question}\n\n"
            "Aucune information exploitable n'est disponible.\n"
            "Répondez exactement : Je n'ai pas cette information pour le moment."
        )

    @classmethod
    def build_prompt(
        cls,
        user_question: str,
        documents: list[dict[str, Any]],
        max_documents: int | None = None,
    ) -> list[dict[str, str]]:
        """
        Construit les messages pour le modèle de chat.
        """
        context = cls.build_context(
            documents=documents,
            max_documents=max_documents,
        )

        system_prompt = cls.build_system_prompt()
        user_prompt = cls.build_user_prompt(
            user_question=user_question,
            context=context,
        )

        logger.info(
            "Prompt construit avec contexte=%s",
            bool(context),
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]