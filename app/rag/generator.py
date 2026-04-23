from __future__ import annotations

import logging
import time
from typing import Any

from mistralai import Mistral

from app.core.config import get_settings


logger = logging.getLogger(__name__)


class GenerationServiceUnavailableError(Exception):
    """
    Erreur levée quand le service de génération est temporairement indisponible.
    """
    pass


class GenerationConfigurationError(Exception):
    """
    Erreur levée quand la configuration de génération est invalide.
    """
    pass


class ResponseGenerator:
    """
    Génère une réponse finale à partir des messages construits
    par le PromptBuilder.
    """

    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY_SECONDS = 1.0

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    ) -> None:
        settings = get_settings()

        self.api_key = settings.mistral_api_key
        self.model = settings.mistral_chat_model
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        self._validate_configuration()
        self.client = Mistral(api_key=self.api_key)

    def _validate_configuration(self) -> None:
        """
        Vérifie la configuration minimale du générateur.
        """
        if not self.api_key or not self.api_key.strip():
            raise GenerationConfigurationError(
                "MISTRAL_API_KEY est vide. Vérifie le fichier .env."
            )

        if not self.model or not self.model.strip():
            raise GenerationConfigurationError(
                "MISTRAL_CHAT_MODEL est vide ou invalide."
            )

        if self.max_retries < 0:
            raise GenerationConfigurationError("max_retries ne peut pas être négatif.")

        if self.retry_delay_seconds < 0:
            raise GenerationConfigurationError("retry_delay_seconds ne peut pas être négatif.")

    @staticmethod
    def _validate_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        Vérifie que les messages sont exploitables par le modèle.
        """
        if not isinstance(messages, list):
            raise ValueError("messages doit être une liste.")

        validated_messages: list[dict[str, str]] = []

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Chaque message doit être un dictionnaire.")

            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()

            if not role or not content:
                continue

            validated_messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return validated_messages

    @staticmethod
    def _is_temporary_unavailable_error(error_message: str) -> bool:
        """
        Détecte les erreurs temporaires connues.
        """
        lowered = error_message.lower()
        return (
            "429" in lowered
            or "rate limit" in lowered
            or "service_tier_capacity_exceeded" in lowered
            or "temporarily unavailable" in lowered
            or "timeout" in lowered
        )

    @staticmethod
    def _extract_answer(response: Any) -> str:
        """
        Extrait et nettoie le texte de réponse depuis la réponse Mistral.
        """
        if not response or not getattr(response, "choices", None):
            raise ValueError("Réponse invalide du modèle : choices absents.")

        first_choice = response.choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", None)

        if not content:
            raise ValueError("Réponse vide du modèle.")

        answer = str(content).strip()

        if not answer:
            raise ValueError("Réponse vide après nettoyage.")

        return answer

    def generate(self, messages: list[dict[str, str]]) -> str:
        """
        Génère une réponse à partir d'une liste de messages chat.
        """
        validated_messages = self._validate_messages(messages)

        if not validated_messages:
            logger.warning("Aucun message exploitable fourni au générateur.")
            return ""

        logger.info("Génération de la réponse avec le modèle %s", self.model)

        delay = self.retry_delay_seconds

        for attempt in range(1, self.max_retries + 2):
            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=validated_messages,
                )

                answer = self._extract_answer(response)

                logger.info("Réponse générée avec succès.")
                return answer

            except Exception as exc:
                error_message = str(exc)

                if self._is_temporary_unavailable_error(error_message):
                    logger.warning(
                        "Tentative %s/%s - service de génération temporairement indisponible, retry dans %ss",
                        attempt,
                        self.max_retries + 1,
                        delay,
                    )

                    if attempt <= self.max_retries:
                        time.sleep(delay)
                        delay *= 2
                        continue

                    logger.error("Échec après %s tentatives de génération", self.max_retries + 1)
                    raise GenerationServiceUnavailableError(
                        "Service de génération indisponible après plusieurs tentatives."
                    ) from exc

                logger.exception("Erreur inattendue lors de la génération")
                raise

        raise GenerationServiceUnavailableError(
            "Service de génération indisponible après plusieurs tentatives."
        )