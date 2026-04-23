from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from batch.ingestion.sheet_loader import SheetLoader
from batch.processing.faq_transformer import FaqTransformer


logger = logging.getLogger(__name__)


class FaqBatchRunner:
    """
    Orchestration batch pour transformer une FAQ tabulaire
    en sorties hybrides :
    - business
    - rag
    - rejected
    - stats
    """

    def __init__(
        self,
        input_path: str | Path,
        business_output_path: str | Path,
        rag_output_path: str | Path,
        rejected_output_path: str | Path,
        stats_output_path: str | Path,
        sheet_name: str | None = None,
        language: str = "fr",
    ) -> None:
        self.input_path = Path(input_path)
        self.business_output_path = Path(business_output_path)
        self.rag_output_path = Path(rag_output_path)
        self.rejected_output_path = Path(rejected_output_path)
        self.stats_output_path = Path(stats_output_path)
        self.sheet_name = sheet_name
        self.language = language

    def run(self) -> dict[str, Any]:
        """
        Exécute le pipeline FAQ complet.
        """
        logger.info("Démarrage du batch FAQ : %s", self.input_path)

        df = self._load_input_dataframe()

        result = FaqTransformer.transform(
            df=df,
            source_file=self.input_path.name,
            language=self.language,
        )

        self._ensure_parent_dirs_exist()

        self._write_json(self.business_output_path, result["business_documents"])
        self._write_json(self.rag_output_path, result["rag_documents"])
        self._write_json(self.rejected_output_path, result["rejected_rows"])
        self._write_json(self.stats_output_path, result["stats"])

        logger.info(
            "Batch FAQ terminé : %s business, %s rag, %s rejected",
            result["stats"]["business_count"],
            result["stats"]["rag_count"],
            result["stats"]["rejected_count"],
        )

        return result

    def _load_input_dataframe(self):
        """
        Charge et prépare le DataFrame source.
        """
        loader = SheetLoader(
            file_path=self.input_path,
            sheet_name=self.sheet_name,
            normalize_columns=True,
            drop_empty_rows=True,
            strip_string_values=True,
        )
        return loader.load()

    def _ensure_parent_dirs_exist(self) -> None:
        """
        Crée les dossiers parents des fichiers de sortie si nécessaire.
        """
        output_paths = [
            self.business_output_path,
            self.rag_output_path,
            self.rejected_output_path,
            self.stats_output_path,
        ]

        for path in output_paths:
            path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_json(output_path: Path, data: Any) -> None:
        """
        Écrit un fichier JSON avec indentation UTF-8.
        """
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Fichier JSON écrit : %s", output_path)


def run() -> dict[str, Any]:
    """
    Point d'entrée standard du batch FAQ.
    """
    runner = FaqBatchRunner(
        input_path="data/raw/google_sheets/faq.xlsx",
        business_output_path="data/business/faq_structured.json",
        rag_output_path="data/rag/faq_documents.json",
        rejected_output_path="data/quality/faq_rejected_rows.json",
        stats_output_path="data/quality/faq_stats.json",
        sheet_name=None,  # ou "faq" si ton fichier a un onglet spécifique
        language="fr",
    )
    return runner.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run()