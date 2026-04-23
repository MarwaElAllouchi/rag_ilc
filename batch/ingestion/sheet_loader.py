from __future__ import annotations

from pathlib import Path
import logging
import re
import unicodedata
from typing import Optional

import pandas as pd


logger = logging.getLogger(__name__)


class SheetLoader:
    """
    Charge les fichiers tabulaires issus de Google Sheets / Excel.

    Formats supportés :
    - .csv
    - .xlsx

    Fonctionnalités :
    - lecture configurable
    - normalisation optionnelle des colonnes
    - suppression des lignes vides
    - nettoyage des cellules textuelles
    """

    SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}

    def __init__(
        self,
        file_path: Path | str,
        sheet_name: Optional[str] = None,
        csv_separator: str = ",",
        csv_encoding: str = "utf-8",
        normalize_columns: bool = True,
        drop_empty_rows: bool = True,
        strip_string_values: bool = True,
    ) -> None:
        self.file_path = Path(file_path)
        self.sheet_name = sheet_name
        self.csv_separator = csv_separator
        self.csv_encoding = csv_encoding
        self.auto_normalize_columns = normalize_columns
        self.auto_drop_empty_rows = drop_empty_rows
        self.strip_string_values = strip_string_values

    def load(self) -> pd.DataFrame:
        """
        Charge un fichier tabulaire en DataFrame selon son extension.
        """
        self._validate_file()

        suffix = self.file_path.suffix.lower()
        logger.info("Chargement du fichier tabulaire : %s", self.file_path)

        try:
            if suffix == ".csv":
                df = pd.read_csv(
                    self.file_path,
                    sep=self.csv_separator,
                    encoding=self.csv_encoding,
                )
            elif suffix == ".xlsx":
                df = pd.read_excel(
                    self.file_path,
                    sheet_name=self.sheet_name if self.sheet_name else 0,
                )
            else:
                raise ValueError(
                    f"Format non supporté : {suffix}. "
                    f"Formats acceptés : {sorted(self.SUPPORTED_EXTENSIONS)}"
                )

        except Exception as exc:
            logger.exception("Échec du chargement du fichier : %s", self.file_path)
            raise ValueError(
                f"Impossible de charger le fichier tabulaire : {self.file_path}"
            ) from exc

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Le chargement de {self.file_path} n'a pas renvoyé un DataFrame."
            )

        if self.strip_string_values:
            df = self.clean_string_values(df)

        if self.auto_normalize_columns:
            df = self.normalize_dataframe_columns(df)

        if self.auto_drop_empty_rows:
            df = self.clean_empty_rows(df)

        logger.info(
            "Fichier chargé avec succès : %s lignes, %s colonnes",
            df.shape[0],
            df.shape[1],
        )

        return df

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

    @staticmethod
    def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise les noms de colonnes :
        - minuscule
        - suppression accents
        - espaces -> underscore
        - suppression caractères spéciaux
        - gestion des doublons
        """
        df = df.copy()

        normalized_columns = [SheetLoader.normalize_column_name(col) for col in df.columns]
        normalized_columns = SheetLoader.make_columns_unique(normalized_columns)

        df.columns = normalized_columns
        return df

    @staticmethod
    def normalize_column_name(col: object) -> str:
        """
        Normalise un nom de colonne.
        """
        col = str(col).strip().lower()
        col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("utf-8")
        col = col.replace(" ", "_")
        col = re.sub(r"[^a-z0-9_]", "", col)
        col = re.sub(r"_+", "_", col).strip("_")

        return col or "unnamed_column"

    @staticmethod
    def make_columns_unique(columns: list[str]) -> list[str]:
        """
        Rend les noms de colonnes uniques après normalisation.
        Exemple :
        ['question', 'question'] -> ['question', 'question_2']
        """
        seen: dict[str, int] = {}
        unique_columns: list[str] = []

        for col in columns:
            if col not in seen:
                seen[col] = 1
                unique_columns.append(col)
            else:
                seen[col] += 1
                unique_columns.append(f"{col}_{seen[col]}")

        return unique_columns

    @staticmethod
    def clean_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
        """
        Supprime les lignes entièrement vides, y compris celles
        contenant uniquement des espaces dans les colonnes texte.
        """
        df = df.copy()

        text_columns = df.select_dtypes(include=["object", "string"]).columns
        for col in text_columns:
            df[col] = df[col].apply(
                lambda value: value.strip() if isinstance(value, str) else value
            )

        df = df.replace(r"^\s*$", pd.NA, regex=True)
        df = df.dropna(how="all").reset_index(drop=True)
        return df

    @staticmethod
    def clean_string_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les valeurs textuelles :
        - trim espaces début/fin
        - conserve les NaN
        """
        df = df.copy()

        text_columns = df.select_dtypes(include=["object", "string"]).columns
        for col in text_columns:
            df[col] = df[col].apply(
                lambda value: value.strip() if isinstance(value, str) else value
            )

        return df