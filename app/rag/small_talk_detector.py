from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class SmallTalkResult:
    kind: str | None
    cleaned_text: str


class SmallTalkDetector:
    """
    Détecteur local de small talk pour la production.
    Aucun appel externe.
    """

    GREETING_PATTERNS = [
        r"^(bonjour|salut|coucou|hello|bonsoir|cc|slt|hi)\b",
    ]

    GRATITUDE_PATTERNS = [
        r"\bmerci\b",
        r"\bmerci beaucoup\b",
        r"\bmerci bien\b",
        r"\bje vous remercie\b",
        r"\bthx\b",
        r"\bthanks\b",
    ]

    FAREWELL_PATTERNS = [
        r"\bau revoir\b",
        r"\baurevoir\b",
        r"\ba bientot\b",
        r"\bbye\b",
        r"\bbonne journee\b",
        r"\bbonne soiree\b",
        r"\ba plus\b",
    ]

    ACK_PATTERNS = [
        r"^(ok|okay|oui ok|d accord|dac|ca marche|tres bien|parfait|nickel)$",
    ]

    PRAISE_PATTERNS = [
        r"\bbravo\b",
        r"\bsuper\b",
        r"\bgenial\b",
        r"\bexcellent\b",
        r"\btop\b",
    ]

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalise légèrement le texte :
        - minuscule
        - suppression accents
        - suppression ponctuation
        - espaces normalisés
        """
        if not isinstance(text, str):
            return ""

        text = text.lower().strip()
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _matches_any(text: str, patterns: list[str]) -> bool:
        """
        Vérifie si le texte match au moins un pattern.
        """
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    @classmethod
    def strip_greeting(cls, text: str) -> str:
        """
        Supprime une salutation en début de texte si présente.
        """
        normalized = cls.normalize(text)

        for pattern in cls.GREETING_PATTERNS:
            cleaned = re.sub(pattern, "", normalized, flags=re.IGNORECASE).strip()
            if cleaned != normalized:
                return cleaned

        return normalized

    @classmethod
    def detect(cls, text: str) -> SmallTalkResult:
        """
        Détecte un éventuel small talk.

        Cas possibles :
        - greeting
        - gratitude
        - farewell
        - acknowledgment
        - praise
        - None si vraie question ou autre contenu
        """
        normalized = cls.normalize(text)

        if not normalized:
            return SmallTalkResult(kind=None, cleaned_text="")

        # 1. salutation pure ou salutation + vraie question
        stripped = cls.strip_greeting(normalized)
        if stripped != normalized:
            if not stripped:
                return SmallTalkResult(kind="greeting", cleaned_text="")
            return SmallTalkResult(kind=None, cleaned_text=stripped)

        # 2. gratitude pure
        if cls._matches_any(normalized, cls.GRATITUDE_PATTERNS):
            return SmallTalkResult(kind="gratitude", cleaned_text="")

        # 3. au revoir / clôture
        if cls._matches_any(normalized, cls.FAREWELL_PATTERNS):
            return SmallTalkResult(kind="farewell", cleaned_text="")

        # 4. accusé de réception / validation courte
        if cls._matches_any(normalized, cls.ACK_PATTERNS):
            return SmallTalkResult(kind="acknowledgment", cleaned_text="")

        # 5. compliment court
        if cls._matches_any(normalized, cls.PRAISE_PATTERNS):
            return SmallTalkResult(kind="praise", cleaned_text="")

        return SmallTalkResult(kind=None, cleaned_text=normalized)