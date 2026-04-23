from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.core.config import get_settings


logger = logging.getLogger(__name__)


class BusinessRulesEngine:
    """
    Moteur de règles métier pour les cas structurés :
    - tarifs
    - niveaux
    - cycles
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.business_data_path = settings.business_data_path

        self._tarifs_cache: list[dict[str, str]] | None = None
        self._cycles_niveaux_cache: dict[str, Any] | None = None

    @staticmethod
    def _clean_text(value: Any) -> str:
        """
        Nettoie une valeur texte simple.
        """
        if value is None:
            return ""

        text = str(value).strip()
        if not text:
            return ""

        return text

    @staticmethod
    def _normalize_text(value: Any) -> str:
        """
        Normalise un texte pour les comparaisons simples.
        """
        return BusinessRulesEngine._clean_text(value).lower()

    def _load_json(self, file_name: str) -> Any:
        """
        Charge un fichier JSON métier.
        """
        file_path = self.business_data_path / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"Fichier business introuvable : {file_path}")

        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def get_all_tarifs(self, force_reload: bool = False) -> list[dict[str, str]]:
        """
        Charge les tarifs depuis le JSON métier, avec cache mémoire simple.
        """
        if self._tarifs_cache is None or force_reload:
            data = self._load_json("formules_tarifs.json")
            if not isinstance(data, list):
                raise ValueError("formules_tarifs.json doit contenir une liste.")
            self._tarifs_cache = data

        logger.info(
            "BusinessRulesEngine : %s tarifs chargés depuis JSON",
            len(self._tarifs_cache),
        )
        return self._tarifs_cache

    def get_cycles_niveaux_data(self, force_reload: bool = False) -> dict[str, Any]:
        """
        Charge les niveaux/cycles depuis le JSON métier, avec cache mémoire simple.
        """
        if self._cycles_niveaux_cache is None or force_reload:
            data = self._load_json("cycles_niveaux.json")
            if not isinstance(data, dict):
                raise ValueError("cycles_niveaux.json doit contenir un objet JSON.")
            self._cycles_niveaux_cache = data

        logger.info(
            "BusinessRulesEngine : %s niveaux et %s cycles chargés depuis JSON",
            len(self._cycles_niveaux_cache.get("niveaux_par_annee", [])),
            len(self._cycles_niveaux_cache.get("cycles", [])),
        )
        return self._cycles_niveaux_cache

    @staticmethod
    def _sort_tarifs(tarifs: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        Trie les tarifs par prix croissant si possible.
        """
        def extract_price(item: dict[str, str]) -> float:
            raw_price = str(item.get("tarif", "")).replace(",", ".").strip()
            raw_price = re.sub(r"[^\d.]", "", raw_price)

            try:
                return float(raw_price)
            except Exception:
                return 999999.0

        return sorted(tarifs, key=extract_price)

    @staticmethod
    def _extract_birth_year(question: str) -> str | None:
        """
        Extrait une année de naissance de type 20xx.
        """
        match = re.search(r"\b(20\d{2})\b", question)
        return match.group(1) if match else None

    @staticmethod
    def _extract_level_name(question: str) -> str | None:
        """
        Détecte un niveau explicite dans la question.
        """
        q = question.lower()

        patterns = [
            r"\bnda\b",
            r"\bn[1-7][ab]?\b",
            r"\bcp1\b",
            r"\bcp2\b",
            r"\bbaream\b",
            r"\bbaraem\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, q)
            if match:
                value = match.group(0).upper()
                if value == "BARAEM":
                    return "BARAÉM"
                return value

        # Cas particulier du niveau JE :
        if re.search(r"\bniveau\s+je\b", q):
            return "JE"

        if re.search(r"\bJE\b", question):
            return "JE"

        return None

    @staticmethod
    def _extract_cycle_name(question: str) -> str | None:
        """
        Détecte un cycle cité dans la question :
        - cycle i
        - cycle ii
        - cycle iii
        - cycle iv
        """
        match = re.search(r"\bcycle\s+(i|ii|iii|iv|v|vi)\b", question.lower())
        if not match:
            return None

        roman = match.group(1).upper()
        return f"CYCLE {roman}"

    @staticmethod
    def _normalize_level_name(value: str) -> str:
        """
        Normalise un nom de niveau.
        """
        normalized = value.strip().upper()
        if normalized == "BARAEM":
            return "BARAÉM"
        return normalized

    def _find_niveau_by_birth_year(self, birth_year: str) -> dict[str, str] | None:
        """
        Trouve un niveau à partir d'une année de naissance.
        Gère aussi les formats du type '2014/2013'.
        """
        data = self.get_cycles_niveaux_data()
        niveaux = data.get("niveaux_par_annee", [])

        for item in niveaux:
            annee_value = self._clean_text(item.get("annee", ""))
            if not annee_value:
                continue

            if birth_year == annee_value:
                return item

            split_values = [x.strip() for x in annee_value.split("/") if x.strip()]
            if birth_year in split_values:
                return item

        return None

    def _find_niveau_by_name(self, level_name: str) -> dict[str, str] | None:
        """
        Trouve un niveau à partir de son nom explicite.
        """
        data = self.get_cycles_niveaux_data()
        niveaux = data.get("niveaux_par_annee", [])

        normalized_target = self._normalize_level_name(level_name)

        for item in niveaux:
            item_level = self._normalize_level_name(item.get("niveau", ""))
            if item_level == normalized_target:
                return item

        return None

    def _find_cycle_by_name(self, cycle_name: str) -> dict[str, Any] | None:
        """
        Trouve un cycle à partir de son nom.
        """
        data = self.get_cycles_niveaux_data()
        cycles = data.get("cycles", [])

        target = self._normalize_text(cycle_name)

        for item in cycles:
            item_name = self._normalize_text(item.get("cycle_name", ""))
            if item_name == target:
                return item

        return None

    def build_tarif_answer(self) -> str:
        """
        Construit une réponse globale sur les tarifs.
        """
        tarifs = self.get_all_tarifs()
        tarifs = self._sort_tarifs(tarifs)

        if not tarifs:
            return "Je suis désolé, je ne dispose pas actuellement des informations tarifaires."

        lines = [
            "Voici les tarifs actuellement disponibles à l’Institut Langues et Cultures :",
            "",
        ]

        for item in tarifs:
            formule = self._clean_text(item.get("formule", ""))
            duree = self._clean_text(item.get("duree", ""))
            tarif = self._clean_text(item.get("tarif", ""))

            lines.append(
                f"- **{formule}** : {duree} → **{tarif} €**"
            )

        lines.append("")
        lines.append(
            "N’hésitez pas à préciser la formule qui vous intéresse si vous souhaitez davantage de détails."
        )

        return "\n".join(lines)

    def build_tarif_specific_answer(self, question: str) -> str | None:
        """
        Construit une réponse ciblée sur une formule tarifaire précise.
        """
        tarifs = self.get_all_tarifs()
        if not tarifs:
            return None

        q = self._normalize_text(question)
        matches: list[dict[str, str]] = []

        for item in tarifs:
            formule = self._normalize_text(item.get("formule", ""))
            if not formule:
                continue

            # match exact de formule ou match partiel par tokens
            if formule in q or any(token and token in q for token in formule.split()):
                matches.append(item)

        unique_matches: list[dict[str, str]] = []
        seen: set[str] = set()

        for item in matches:
            key = self._normalize_text(item.get("formule", ""))
            if key and key not in seen:
                seen.add(key)
                unique_matches.append(item)

        if not unique_matches:
            return None

        if len(unique_matches) == 1:
            item = unique_matches[0]
            return (
                f"Voici l’information disponible pour la formule **{item.get('formule', '')}** : "
                f"durée **{item.get('duree', '')}**, tarif **{item.get('tarif', '')} €**."
            )

        lines = ["Voici les formules correspondant à votre demande :", ""]
        for item in self._sort_tarifs(unique_matches):
            lines.append(
                f"- **{item.get('formule', '')}** : {item.get('duree', '')} → **{item.get('tarif', '')} €**"
            )

        return "\n".join(lines)

    def build_niveau_birth_year_answer(self, question: str) -> str | None:
        """
        Répond à partir de l'année de naissance.
        """
        birth_year = self._extract_birth_year(question)
        if not birth_year:
            return None

        niveau_item = self._find_niveau_by_birth_year(birth_year)

        if not niveau_item:
            return (
                f"Je suis désolé, je n’ai pas trouvé de niveau correspondant à l’année de naissance **{birth_year}** dans les informations disponibles."
            )

        niveau = self._clean_text(niveau_item.get("niveau", ""))
        cycle_name = self._clean_text(niveau_item.get("cycle_name", ""))

        if cycle_name:
            return (
                f"D’après les informations disponibles, un élève né en **{birth_year}** "
                f"correspond au **niveau {niveau}**, rattaché au **{cycle_name}**."
            )

        return (
            f"D’après les informations disponibles, un élève né en **{birth_year}** correspond au **niveau {niveau}**."
        )

    def build_niveau_specific_answer(self, question: str) -> str | None:
        """
        Retourne une réponse directe sur un niveau explicite :
        - année de naissance
        - cycle associé
        - description du cycle
        """
        level_name = self._extract_level_name(question)
        if not level_name:
            return None

        niveau_item = self._find_niveau_by_name(level_name)
        if not niveau_item:
            return None

        niveau = self._clean_text(niveau_item.get("niveau", ""))
        annee = self._clean_text(niveau_item.get("annee", ""))
        cycle_name = self._clean_text(niveau_item.get("cycle_name", ""))

        cycle_item = self._find_cycle_by_name(cycle_name) if cycle_name else None
        cycle_description = self._clean_text(cycle_item.get("description", "")) if cycle_item else ""

        lines = [
            f"D’après les informations disponibles, le **niveau {niveau}** correspond aux élèves nés en **{annee}**."
        ]

        if cycle_name:
            lines.append(f"Ce niveau est rattaché au **{cycle_name}**.")

        if cycle_description:
            lines.append("")
            lines.append(f"**Description du {cycle_name} :**")
            lines.append(cycle_description)

        return "\n".join(lines)

    def build_cycle_answer(self, question: str) -> str | None:
        """
        Retourne directement la description d’un cycle explicite cité dans la question.
        """
        cycle_name = self._extract_cycle_name(question)
        if not cycle_name:
            return None

        cycle_item = self._find_cycle_by_name(cycle_name)
        if not cycle_item:
            return None

        description = self._clean_text(cycle_item.get("description", ""))
        related_levels = cycle_item.get("related_levels", [])

        lines = [f"Voici les informations disponibles pour le **{cycle_name}** :"]

        if related_levels:
            lines.append("")
            lines.append(f"**Niveaux associés :** {', '.join(related_levels)}")

        if description:
            lines.append("")
            lines.append("**Description :**")
            lines.append(description)

        return "\n".join(lines)

    @staticmethod
    def is_tarif_global_question(question: str) -> bool:
        """
        Détecte les formulations globales de demande de tarifs.
        """
        q = question.lower()

        global_patterns = [
            "quels sont les tarifs",
            "quels sont vos tarifs",
            "donne moi les tarifs",
            "donnez moi les tarifs",
            "donnez-moi les tarifs",
            "liste des tarifs",
            "tous les tarifs",
            "les tarifs",
            "combien coûtent les cours",
            "combien coutent les cours",
        ]

        # cas ultra courts
        short_forms = {"prix", "tarif", "tarifs"}

        if q.strip() in short_forms:
            return True

        return any(pattern in q for pattern in global_patterns)