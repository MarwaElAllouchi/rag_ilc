from __future__ import annotations

import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings


logger = logging.getLogger(__name__)


class TextChunker:
    """
    Découpe un texte long en chunks pour le RAG.
    Utilisé pour les documents TXT / DOCX / PDF.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ".", " ", ""]

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ) -> None:
        settings = get_settings()

        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        )
        self.separators = separators or self.DEFAULT_SEPARATORS

        self._validate_config()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def _validate_config(self) -> None:
        """
        Valide les paramètres de chunking.
        """
        if self.chunk_size <= 0:
            raise ValueError("chunk_size doit être strictement positif")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap ne peut pas être négatif")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap doit être strictement inférieur à chunk_size")

    def split_text(self, text: str) -> List[str]:
        """
        Découpe un texte en liste de chunks nettoyés.
        """
        if not isinstance(text, str) or not text.strip():
            logger.warning("Texte vide ou invalide reçu pour découpage.")
            return []

        raw_chunks = self.splitter.split_text(text)

        chunks = [chunk.strip() for chunk in raw_chunks if isinstance(chunk, str) and chunk.strip()]

        logger.info(
            "Découpage terminé : %s chunks générés (chunk_size=%s, overlap=%s)",
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )

        return chunks