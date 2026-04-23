from __future__ import annotations

import logging
import random
import time
from typing import Any

from mistralai import Mistral

from app.core.config import get_settings


logger = logging.getLogger(__name__)


class EmbeddingServiceUnavailableError(Exception):
    """
    Erreur levée quand le service d'embedding est temporairement indisponible.
    """
    pass


class EmbeddingConfigurationError(Exception):
    """
    Erreur levée quand la configuration du service d'embedding est invalide.
    """
    pass


class MistralEmbedder:
    """
    Génère des embeddings avec l'API Mistral.

    Fonctionnalités :
    - validation de la configuration
    - nettoyage des textes
    - batching
    - validation du nombre de vecteurs retournés
    - retry avec backoff exponentiel sur indisponibilité temporaire
    """

    DEFAULT_BATCH_SIZE = 32
    DEFAULT_MAX_RETRIES = 4
    DEFAULT_RETRY_DELAY_SECONDS = 3.0
    DEFAULT_MAX_RETRY_DELAY_SECONDS = 30.0

    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
        max_retry_delay_seconds: float = DEFAULT_MAX_RETRY_DELAY_SECONDS,
    ) -> None:
        settings = get_settings()

        self.api_key = settings.mistral_api_key
        self.model_name = settings.mistral_embed_model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.max_retry_delay_seconds = max_retry_delay_seconds

        self._validate_configuration()
        self.client = Mistral(api_key=self.api_key)

    def _validate_configuration(self) -> None:
        """
        Vérifie que la configuration minimale est disponible.
        """
        if not self.api_key or not self.api_key.strip():
            raise EmbeddingConfigurationError(
                "MISTRAL_API_KEY est manquante ou vide."
            )

        if not self.model_name or not self.model_name.strip():
            raise EmbeddingConfigurationError(
                "MISTRAL_EMBED_MODEL est manquant ou vide."
            )

        if self.batch_size <= 0:
            raise EmbeddingConfigurationError("batch_size doit être strictement positif.")

        if self.max_retries < 0:
            raise EmbeddingConfigurationError("max_retries ne peut pas être négatif.")

        if self.retry_delay_seconds < 0:
            raise EmbeddingConfigurationError("retry_delay_seconds ne peut pas être négatif.")

        if self.max_retry_delay_seconds <= 0:
            raise EmbeddingConfigurationError(
                "max_retry_delay_seconds doit être strictement positif."
            )

    @staticmethod
    def _clean_text(text: Any) -> str:
        """
        Nettoie un texte avant vectorisation.
        """
        if text is None:
            return ""

        return str(text).strip()

    def _clean_texts(self, texts: list[str]) -> list[str]:
        """
        Nettoie une liste de textes et rejette les textes vides.
        """
        cleaned_texts = [self._clean_text(text) for text in texts]
        valid_texts = [text for text in cleaned_texts if text]

        ignored_count = len(texts) - len(valid_texts)
        if ignored_count > 0:
            logger.warning(
                "%s textes vides ou invalides ont été ignorés avant embedding",
                ignored_count,
            )

        return valid_texts

    @staticmethod
    def _chunk_list(items: list[str], chunk_size: int) -> list[list[str]]:
        """
        Découpe une liste en sous-listes.
        """
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

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
            or "connection reset" in lowered
            or "server error" in lowered
        )

    def _compute_retry_delay(self, attempt: int) -> float:
        """
        Calcule un délai d'attente avec backoff exponentiel + jitter.
        """
        base_delay = self.retry_delay_seconds * (2 ** (attempt - 1))
        capped_delay = min(base_delay, self.max_retry_delay_seconds)

        # jitter léger pour éviter de retaper exactement au même moment
        jitter = random.uniform(0, 1)
        return capped_delay + jitter

    def _request_embeddings_batch(self, texts_batch: list[str]) -> list[list[float]]:
        """
        Appelle l'API Mistral pour un batch unique avec retry exponentiel.
        """
        if not texts_batch:
            return []

        last_exception: Exception | None = None
        total_attempts = self.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    inputs=texts_batch,
                )

                if not hasattr(response, "data") or response.data is None:
                    raise ValueError("Réponse embeddings invalide : champ 'data' absent.")

                embeddings = [item.embedding for item in response.data]

                if len(embeddings) != len(texts_batch):
                    raise ValueError(
                        f"Nombre d'embeddings incohérent : {len(embeddings)} "
                        f"pour {len(texts_batch)} textes"
                    )

                return embeddings

            except Exception as exc:
                last_exception = exc
                error_message = str(exc)

                if self._is_temporary_unavailable_error(error_message):
                    logger.warning(
                        "Service d'embedding temporairement indisponible "
                        "(tentative %s/%s) : %s",
                        attempt,
                        total_attempts,
                        error_message,
                    )

                    if attempt < total_attempts:
                        wait_time = self._compute_retry_delay(attempt)
                        logger.info(
                            "Nouvelle tentative dans %.1f secondes...",
                            wait_time,
                        )
                        time.sleep(wait_time)
                        continue

                    raise EmbeddingServiceUnavailableError(
                        "Le service d'embedding est temporairement indisponible."
                    ) from exc

                logger.exception(
                    "Erreur inattendue lors de la génération des embeddings"
                )
                raise

        if last_exception:
            raise last_exception

        raise RuntimeError("Erreur inattendue dans _request_embeddings_batch")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Génère les embeddings pour une liste de textes.
        """
        if not texts:
            logger.warning("Aucun texte fourni pour la génération d'embeddings.")
            return []

        cleaned_texts = self._clean_texts(texts)

        if not cleaned_texts:
            logger.warning("Aucun texte exploitable après nettoyage.")
            return []

        logger.info(
            "Génération des embeddings pour %s textes (batch_size=%s)",
            len(cleaned_texts),
            self.batch_size,
        )

        all_embeddings: list[list[float]] = []
        text_batches = self._chunk_list(cleaned_texts, self.batch_size)

        for batch_index, texts_batch in enumerate(text_batches, start=1):
            logger.info(
                "Traitement du batch embedding %s/%s (%s textes)",
                batch_index,
                len(text_batches),
                len(texts_batch),
            )

            batch_embeddings = self._request_embeddings_batch(texts_batch)
            all_embeddings.extend(batch_embeddings)

        if len(all_embeddings) != len(cleaned_texts):
            raise ValueError(
                f"Nombre total d'embeddings incohérent : {len(all_embeddings)} "
                f"pour {len(cleaned_texts)} textes"
            )

        logger.info(
            "Embeddings générés avec succès : %s vecteurs",
            len(all_embeddings),
        )
        return all_embeddings

    def embed_single_text(self, text: str) -> list[float]:
        """
        Génère l'embedding pour un seul texte.
        """
        cleaned_text = self._clean_text(text)

        if not cleaned_text:
            raise ValueError("Le texte à vectoriser ne peut pas être vide.")

        embeddings = self.embed_texts([cleaned_text])

        if not embeddings:
            raise ValueError("Aucun embedding généré pour le texte fourni.")

        return embeddings[0]