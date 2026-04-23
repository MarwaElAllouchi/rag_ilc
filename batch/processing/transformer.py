from __future__ import annotations

import logging
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


class DataTransformer:
    """
    Transforme des données tabulaires génériques en documents textuels
    exploitables par le pipeline RAG.

    Ce transformer est volontairement générique :
    - il ne gère pas la FAQ hybride métier/RAG
    - il ne gère pas les règles métier déterministes
    - il convertit des lignes tabulaires en documents RAG standardisés
    """

    @staticmethod
    def _clean_value(value: Any) -> str:
        """
        Nettoie une valeur provenant d'un DataFrame :
        - retourne "" si la valeur est vide / NaN / None
        - sinon retourne une chaîne propre
        """
        if pd.isna(value):
            return ""

        text = str(value).strip()

        if text.lower() in {"", "nan", "none", "null"}:
            return ""

        return text

    @classmethod
    def _clean_row(cls, row: pd.Series, columns: list[str]) -> dict[str, str]:
        """
        Convertit une ligne pandas en dictionnaire nettoyé.
        """
        return {
            col: cls._clean_value(row.get(col, ""))
            for col in columns
        }

    @staticmethod
    def _resolve_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
        """
        Retourne le premier alias trouvé dans les colonnes du DataFrame.
        """
        df_columns = set(df.columns)
        for alias in aliases:
            if alias in df_columns:
                return alias
        return None

    @classmethod
    def _resolve_columns(
        cls,
        df: pd.DataFrame,
        aliases_mapping: dict[str, list[str]],
        required_fields: set[str],
    ) -> dict[str, str]:
        """
        Résout les colonnes réelles à partir d'un mapping d'alias.
        """
        resolved: dict[str, str] = {}
        missing: list[str] = []

        for logical_name, aliases in aliases_mapping.items():
            found = cls._resolve_column(df, aliases)
            if found:
                resolved[logical_name] = found

        for field in required_fields:
            if field not in resolved:
                missing.append(field)

        if missing:
            raise ValueError(
                f"Colonnes obligatoires manquantes : {sorted(missing)}. "
                f"Colonnes disponibles : {sorted(df.columns.tolist())}"
            )

        return resolved

    @staticmethod
    def _build_content_from_columns(
        row_data: dict[str, str],
        content_columns: list[str],
        labels: dict[str, str] | None = None,
    ) -> str:
        """
        Construit le contenu textuel à partir d'une liste de colonnes.

        Exemple :
        labels = {"titre": "Titre", "description": "Description"}
        """
        parts: list[str] = []

        for col in content_columns:
            value = row_data.get(col, "")
            if not value:
                continue

            label = labels.get(col, col) if labels else col
            parts.append(f"{label} : {value}")

        return "\n".join(parts).strip()

    @staticmethod
    def _deduplicate_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Supprime les doublons simples sur la base du contenu + catégorie + source_file.
        """
        seen: set[tuple[str, str, str]] = set()
        deduplicated: list[dict[str, Any]] = []

        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            category = metadata.get("category", "")
            source_file = metadata.get("source_file", "")

            key = (content, category, source_file)

            if key in seen:
                continue

            seen.add(key)
            deduplicated.append(doc)

        return deduplicated

    @classmethod
    def transform_rows_to_documents(
        cls,
        df: pd.DataFrame,
        content_columns: list[str],
        source_type: str = "table",
        category_column: str | None = None,
        default_category: str = "general",
        language: str = "fr",
        source_file: str = "unknown",
        labels: dict[str, str] | None = None,
        extra_metadata_columns: list[str] | None = None,
        document_id_prefix: str = "doc",
    ) -> list[dict[str, Any]]:
        """
        Transforme des lignes tabulaires en documents RAG génériques.

        Args:
            df: DataFrame source
            content_columns: colonnes à inclure dans le contenu final
            source_type: type logique de la source
            category_column: colonne à utiliser comme catégorie si elle existe
            default_category: catégorie par défaut
            language: langue du contenu
            source_file: nom logique du fichier source
            labels: labels d'affichage pour les colonnes dans le contenu
            extra_metadata_columns: colonnes à injecter dans les métadonnées
            document_id_prefix: préfixe des identifiants générés

        Returns:
            Liste de documents RAG standardisés.
        """
        if df.empty:
            logger.warning("DataTransformer : DataFrame vide reçu.")
            return []

        if not content_columns:
            raise ValueError("content_columns ne peut pas être vide")

        missing_content_columns = [col for col in content_columns if col not in df.columns]
        if missing_content_columns:
            raise ValueError(
                f"Colonnes de contenu introuvables : {sorted(missing_content_columns)}. "
                f"Colonnes disponibles : {sorted(df.columns.tolist())}"
            )

        if category_column and category_column not in df.columns:
            raise ValueError(
                f"category_column introuvable : {category_column}. "
                f"Colonnes disponibles : {sorted(df.columns.tolist())}"
            )

        extra_metadata_columns = extra_metadata_columns or []
        for col in extra_metadata_columns:
            if col not in df.columns:
                raise ValueError(
                    f"Colonne de métadonnée introuvable : {col}. "
                    f"Colonnes disponibles : {sorted(df.columns.tolist())}"
                )

        documents: list[dict[str, Any]] = []
        skipped_rows = 0

        for idx, row in df.iterrows():
            row_data = cls._clean_row(row, list(df.columns))

            content = cls._build_content_from_columns(
                row_data=row_data,
                content_columns=content_columns,
                labels=labels,
            )

            if not content:
                skipped_rows += 1
                continue

            category = default_category
            if category_column:
                category = row_data.get(category_column, "") or default_category

            metadata = {
                "document_id": f"{document_id_prefix}_{idx + 1}",
                "source_type": source_type,
                "category": category,
                "language": language,
                "source_file": source_file,
                "row_index": idx,
            }

            for col in extra_metadata_columns:
                value = row_data.get(col, "")
                if value:
                    metadata[col] = value

            documents.append(
                {
                    "content": content,
                    "metadata": metadata,
                }
            )

        documents = cls._deduplicate_documents(documents)

        logger.info(
            "DataTransformer : %s documents générés, %s lignes ignorées",
            len(documents),
            skipped_rows,
        )

        return documents

    @classmethod
    def transform_simple_faq_for_rag(
        cls,
        df: pd.DataFrame,
        source_file: str = "faq.xlsx",
        language: str = "fr",
    ) -> list[dict[str, Any]]:
        """
        Version simple orientée RAG uniquement.
        À utiliser seulement si on veut transformer une FAQ en documents RAG
        sans la logique hybride métier/RAG.

        En pratique, pour ton projet ILC, on préférera faq_transformer.py.
        """
        aliases_mapping = {
            "question": ["question", "questions"],
            "reponse": ["reponse", "réponse", "answer"],
            "categorie": ["categorie", "catégorie", "category", "categorie_optionnel"],
        }

        resolved = cls._resolve_columns(
            df=df,
            aliases_mapping=aliases_mapping,
            required_fields={"question", "reponse"},
        )

        question_col = resolved["question"]
        reponse_col = resolved["reponse"]
        categorie_col = resolved.get("categorie")

        documents: list[dict[str, Any]] = []
        skipped_rows = 0

        for idx, row in df.iterrows():
            row_data = cls._clean_row(row, list(df.columns))

            question = row_data.get(question_col, "")
            reponse = row_data.get(reponse_col, "")

            if not question or not reponse:
                skipped_rows += 1
                continue

            categorie = row_data.get(categorie_col, "") if categorie_col else ""
            if not categorie:
                categorie = "general"

            content = f"Question : {question}\nRéponse : {reponse}"

            documents.append(
                {
                    "content": content,
                    "metadata": {
                        "document_id": f"faq_{idx + 1}",
                        "source_type": "faq",
                        "category": categorie,
                        "language": language,
                        "source_file": source_file,
                        "row_index": idx,
                    },
                }
            )

        documents = cls._deduplicate_documents(documents)

        logger.info(
            "DataTransformer.transform_simple_faq_for_rag : %s documents générés, %s lignes ignorées",
            len(documents),
            skipped_rows,
        )

        return documents