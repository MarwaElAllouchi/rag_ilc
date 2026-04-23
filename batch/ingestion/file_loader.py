from __future__ import annotations

from pathlib import Path
import logging
import re
from typing import Optional

from docx import Document
from pypdf import PdfReader


logger = logging.getLogger(__name__)


class FileLoader:
    """
    Charge les fichiers texte non structurés.

    Formats supportés :
    - .txt
    - .pdf
    - .docx
    """

    SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}

    def __init__(
        self,
        file_path: Path | str,
        txt_encoding: str = "utf-8",
        txt_fallback_encoding: Optional[str] = "latin-1",
    ) -> None:
        self.file_path = Path(file_path)
        self.txt_encoding = txt_encoding
        self.txt_fallback_encoding = txt_fallback_encoding

    def load(self) -> str:
        """
        Charge le contenu texte d'un fichier selon son extension.
        """
        self._validate_file()

        suffix = self.file_path.suffix.lower()
        logger.info("Chargement du fichier non structuré : %s", self.file_path)

        try:
            if suffix == ".txt":
                content = self._load_txt()
            elif suffix == ".pdf":
                content = self._load_pdf()
            elif suffix == ".docx":
                content = self._load_docx()
            else:
                raise ValueError(f"Extension non gérée : {suffix}")
        except Exception as exc:
            logger.exception("Erreur pendant le chargement du fichier : %s", self.file_path)
            raise ValueError(
                f"Impossible de charger correctement le fichier : {self.file_path}"
            ) from exc

        content = self.clean_text(content)

        if not content:
            logger.warning("Le fichier a été chargé mais le contenu extrait est vide : %s", self.file_path)

        logger.info(
            "Fichier texte chargé avec succès : %s caractères extraits depuis %s",
            len(content),
            self.file_path.name,
        )

        return content

    def _validate_file(self) -> None:
        """
        Vérifie l'existence du fichier et son extension.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {self.file_path}")

        if not self.file_path.is_file():
            raise ValueError(f"Le chemin fourni n'est pas un fichier : {self.file_path}")

        suffix = self.file_path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Format non supporté : {suffix}. "
                f"Formats acceptés : {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

    def _load_txt(self) -> str:
        """
        Lit un fichier texte brut avec fallback d'encodage.
        """
        try:
            return self.file_path.read_text(encoding=self.txt_encoding)
        except UnicodeDecodeError:
            if not self.txt_fallback_encoding:
                raise

            logger.warning(
                "Échec lecture TXT en %s, tentative avec encodage de secours %s pour %s",
                self.txt_encoding,
                self.txt_fallback_encoding,
                self.file_path.name,
            )
            return self.file_path.read_text(encoding=self.txt_fallback_encoding)

    def _load_pdf(self) -> str:
        """
        Extrait le texte d'un PDF.
        """
        reader = PdfReader(str(self.file_path))
        pages_text: list[str] = []

        logger.info("Extraction PDF : %s pages détectées dans %s", len(reader.pages), self.file_path.name)

        for page_index, page in enumerate(reader.pages):
            try:
                extracted = page.extract_text() or ""
                extracted = extracted.strip()
                if extracted:
                    pages_text.append(extracted)
                else:
                    logger.debug(
                        "Aucun texte extrait de la page %s du PDF %s",
                        page_index,
                        self.file_path.name,
                    )
            except Exception as exc:
                logger.warning(
                    "Erreur d'extraction sur la page %s du PDF %s : %s",
                    page_index,
                    self.file_path.name,
                    exc,
                )

        return "\n".join(pages_text)

    def _load_docx(self) -> str:
        """
        Extrait le texte d'un fichier DOCX.
        """
        doc = Document(str(self.file_path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(paragraphs)

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Nettoie légèrement le texte :
        - remplace certains caractères parasites
        - normalise les espaces
        - supprime les lignes vides répétées
        """
        if not isinstance(text, str):
            return ""

        replacements = {
            "\uf0b7": "-",
            "\u2022": "-",
            "\xa0": " ",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # trim ligne par ligne
        lines = [line.strip() for line in text.splitlines()]

        # suppression lignes vides
        non_empty_lines = [line for line in lines if line]

        cleaned = "\n".join(non_empty_lines)

        # réduire espaces multiples internes
        cleaned = re.sub(r"[ \t]+", " ", cleaned)

        return cleaned.strip()