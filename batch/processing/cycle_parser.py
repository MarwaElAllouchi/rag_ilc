from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from batch.ingestion.file_loader import FileLoader


logger = logging.getLogger(__name__)


class CycleParser:
    """
    Parse les descriptions de cycles depuis un document texte/DOCX/PDF.

    Exemple attendu dans le document :
    - CYCLE I : ...
    - CYCLE II : ...
    - CYCLE III : ...
    """

    CYCLE_PATTERN = re.compile(
        r"(CYCLE\s+[IVX]+)\s*[:\-–]?\s*(.+?)(?=(CYCLE\s+[IVX]+)\s*[:\-–]?|$)",
        re.IGNORECASE | re.DOTALL,
    )

    EXPLICIT_LEVEL_PATTERNS = [
        r"\bJE\b",
        r"\bBARA[ÉE]M\b",
        r"\bCP1\b",
        r"\bCP2\b",
        r"\bNDA\b",
        r"\bN[1-7][AB]?\b",
    ]

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Nettoie légèrement un bloc de texte.
        """
        if not isinstance(text, str):
            return ""

        text = text.strip()
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text

    @staticmethod
    def _normalize_level(level: str) -> str:
        """
        Normalise un niveau extrait.
        """
        normalized = level.strip().upper()

        replacements = {
            "BARAEM": "BARAÉM",
        }

        return replacements.get(normalized, normalized)

    @classmethod
    def _extract_related_levels(cls, header_and_body: str) -> list[str]:
        """
        Extrait les niveaux mentionnés dans un bloc de cycle
        et normalise les formes :
        - Niveau 1 -> N1
        - Niveau 3A -> N3A
        - Niveau 7 -> N7
        """
        found_levels: list[str] = []

        text = header_and_body.upper()

        # Cas explicites : JE, CP1, N3A, etc.
        for pattern in cls.EXPLICIT_LEVEL_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                normalized = cls._normalize_level(match)
                if normalized not in found_levels:
                    found_levels.append(normalized)

        # Cas du type "NIVEAU 1", "NIVEAU 3A", "NIVEAU 7"
        niveau_matches = re.findall(r"\bNIVEAU\s+([1-7])\s*([AB]?)\b", text)
        for number, suffix in niveau_matches:
            normalized = cls._normalize_level(f"N{number}{suffix}")
            if normalized not in found_levels:
                found_levels.append(normalized)

        return found_levels

    @staticmethod
    def _deduplicate_cycles(cycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Supprime les doublons de cycles à partir du nom du cycle.
        """
        seen: set[str] = set()
        deduplicated: list[dict[str, Any]] = []

        for cycle in cycles:
            cycle_name = cycle.get("cycle_name", "")
            if cycle_name in seen:
                continue

            seen.add(cycle_name)
            deduplicated.append(cycle)

        return deduplicated

    @classmethod
    def parse_cycles(cls, file_path: Path | str) -> list[dict[str, Any]]:
        """
        Extrait les cycles depuis un document source.

        Retourne une liste de dictionnaires :
        [
          {
            "cycle_name": "CYCLE I",
            "related_levels": ["JE", "BARAÉM"],
            "description": "...",
            "source_file": "niveaux.docx"
          }
        ]
        """
        file_path = Path(file_path)

        loader = FileLoader(file_path)
        content = loader.load()
        content = cls._clean_text(content)

        if not content:
            logger.warning("CycleParser : contenu vide pour %s", file_path)
            return []

        matches = cls.CYCLE_PATTERN.findall(content)

        if not matches:
            logger.warning(
                "CycleParser : aucun cycle détecté dans %s",
                file_path.name,
            )
            return []

        cycles: list[dict[str, Any]] = []

        for match in matches:
            cycle_name = cls._clean_text(match[0]).upper()
            body = cls._clean_text(match[1])

            if not cycle_name or not body:
                logger.debug(
                    "Cycle ignoré car incomplet dans %s : cycle_name=%s",
                    file_path.name,
                    cycle_name,
                )
                continue

            related_levels = cls._extract_related_levels(f"{cycle_name}\n{body}")

            cycles.append(
                {
                    "cycle_name": cycle_name,
                    "related_levels": related_levels,
                    "description": body,
                    "source_file": file_path.name,
                }
            )

        cycles = cls._deduplicate_cycles(cycles)

        logger.info(
            "CycleParser : %s cycles extraits depuis %s",
            len(cycles),
            file_path.name,
        )
        return cycles