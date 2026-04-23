from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from batch.ingestion.file_loader import FileLoader
from batch.processing.chunker import TextChunker


logger = logging.getLogger(__name__)


class DocumentTransformer:
    """
    Transforme un document long (TXT / DOCX / PDF)
    en chunks prêts pour le RAG, avec métadonnées standardisées.

    Version compatible avec l'architecture actuelle :
    - même interface
    - même structure de sortie
    - amélioration légère du découpage des documents longs
    """

    MIN_PARAGRAPH_LENGTH = 3
    SECTION_TITLE_MAX_WORDS = 12

    @staticmethod
    def transform_document(
        file_path: Path | str,
        source_type: str = "document",
        category: str = "general",
        language: str = "fr",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Charge un document, le découpe en chunks
        et ajoute des métadonnées standard.

        Args:
            file_path: chemin du document source
            source_type: type métier de la source (document, faq, reglement, niveau, etc.)
            category: catégorie métier/logique
            language: langue du document
            chunk_size: taille cible des chunks
            chunk_overlap: chevauchement entre chunks

        Returns:
            Liste de documents chunkés avec contenu et métadonnées.
        """
        file_path = Path(file_path)
        DocumentTransformer._validate_inputs(
            file_path=file_path,
            source_type=source_type,
            category=category,
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        loader = FileLoader(file_path)
        content = loader.load()
        content = DocumentTransformer._clean_content(content)

        if not content:
            logger.warning("Document ignoré car vide après nettoyage : %s", file_path)
            return []

        prepared_content = DocumentTransformer._prepare_content_for_chunking(content)

        if not prepared_content:
            logger.warning("Document ignoré car vide après préparation : %s", file_path)
            return []

        chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = chunker.split_text(prepared_content)

        if not chunks:
            logger.warning("Aucun chunk généré pour le document : %s", file_path)
            return []

        total_chunks = len(chunks)
        document_id = DocumentTransformer._build_document_id(file_path)

        documents: list[dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            section_title = DocumentTransformer._extract_section_title_from_chunk(chunk)

            documents.append(
                {
                    "content": chunk,
                    "metadata": {
                        "document_id": document_id,
                        "source_type": source_type,
                        "category": category,
                        "language": language,
                        "source_file": file_path.name,
                        "source_path": str(file_path),
                        "document_title": file_path.stem,
                        "chunk_index": idx,
                        "chunk_size": len(chunk),
                        "total_chunks": total_chunks,
                        "is_chunked": total_chunks > 1,
                        "section_title": section_title,
                    },
                }
            )

        logger.info(
            "Document transformé : %s chunks générés pour %s",
            len(documents),
            file_path.name,
        )

        return documents

    @staticmethod
    def _validate_inputs(
        file_path: Path,
        source_type: str,
        category: str,
        language: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        """
        Valide les entrées principales.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Document introuvable : {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Le chemin n'est pas un fichier valide : {file_path}")

        if not source_type.strip():
            raise ValueError("source_type ne peut pas être vide")

        if not category.strip():
            raise ValueError("category ne peut pas être vide")

        if not language.strip():
            raise ValueError("language ne peut pas être vide")

        if chunk_size <= 0:
            raise ValueError("chunk_size doit être strictement positif")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap ne peut pas être négatif")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap doit être inférieur à chunk_size")

    @staticmethod
    def _clean_content(content: str) -> str:
        """
        Nettoie le contenu brut du document.
        """
        if not isinstance(content, str):
            return ""

        content = content.strip()
        return content

    @staticmethod
    def _build_document_id(file_path: Path) -> str:
        """
        Construit un identifiant simple et stable de document.
        """
        return file_path.stem.lower().replace(" ", "_")

    @staticmethod
    def _clean_paragraphs(content: str) -> list[str]:
        """
        Transforme le contenu en paragraphes propres.
        """
        raw_parts = re.split(r"\n+", content)

        paragraphs: list[str] = []
        for part in raw_parts:
            cleaned = re.sub(r"\s+", " ", part).strip()
            if len(cleaned) >= DocumentTransformer.MIN_PARAGRAPH_LENGTH:
                paragraphs.append(cleaned)

        return paragraphs

    @staticmethod
    def _is_probable_section_title(paragraph: str) -> bool:
        """
        Détecte légèrement les titres/sections probables.
        Méthode volontairement simple pour rester compatible.
        """
        text = paragraph.strip()
        if not text:
            return False

        words = text.split()
        if len(words) > DocumentTransformer.SECTION_TITLE_MAX_WORDS:
            return False

        if re.match(r"^\d+\s*[-.)]\s*", text):
            return True

        if text.endswith(":"):
            return True

        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if uppercase_ratio > 0.6 and len(words) <= 8:
                return True

        if len(words) <= 6 and text[:1].isupper():
            return True

        return False

    @staticmethod
    def _prepare_content_for_chunking(content: str) -> str:
        """
        Prépare le contenu pour un chunking plus robuste :
        - nettoyage par paragraphes
        - conservation légère des titres/sections
        - séparation claire entre blocs
        """
        paragraphs = DocumentTransformer._clean_paragraphs(content)
        if not paragraphs:
            return ""

        prepared_parts: list[str] = []

        current_section_title: str | None = None

        for paragraph in paragraphs:
            if DocumentTransformer._is_probable_section_title(paragraph):
                current_section_title = paragraph.strip(" :")
                prepared_parts.append(f"Section : {current_section_title}")
            else:
                prepared_parts.append(paragraph)

        return "\n\n".join(prepared_parts).strip()

    @staticmethod
    def _extract_section_title_from_chunk(chunk: str) -> str | None:
        """
        Essaie d'extraire un titre de section depuis le chunk préparé.
        """
        if not isinstance(chunk, str) or not chunk.strip():
            return None

        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if not lines:
            return None

        first_line = lines[0]
        if first_line.lower().startswith("section :"):
            title = first_line.split(":", 1)[1].strip()
            return title or None

        return None