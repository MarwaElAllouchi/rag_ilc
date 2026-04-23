from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging_config import setup_logging

from batch.run_business_batch import BusinessBatchRunner
from batch.run_faq_batch import FaqBatchRunner
from batch.run_indexing import IndexingBatchRunner
from batch.run_rag_batch import RagBatchRunner


logger = logging.getLogger(__name__)


class PipelineConfigLoader:
    """
    Charge la configuration JSON du pipeline.
    """

    @staticmethod
    def load(config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"Fichier de configuration introuvable : {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        if not isinstance(config, dict):
            raise ValueError("Le fichier de configuration doit contenir un objet JSON.")

        return config


class PipelineRunner:
    """
    Orchestrateur global du pipeline batch.

    Responsabilités :
    - charger la configuration
    - exécuter les sous-pipelines activés
    - lancer l'indexation finale
    - produire un rapport global
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.settings = get_settings()
        self.config = PipelineConfigLoader.load(self.config_path)

    def run(self) -> dict[str, Any]:
        logger.info("Démarrage du pipeline global")

        report: dict[str, Any] = {
            "config_path": str(self.config_path),
            "faq": None,
            "business": None,
            "rag": None,
            "indexing": None,
            "status": "success",
        }

        try:
            report["faq"] = self._run_faq_pipeline_if_enabled()
            report["business"] = self._run_business_pipeline_if_enabled()
            report["rag"] = self._run_rag_pipeline_if_enabled()
            report["indexing"] = self._run_indexing_if_enabled()

        except Exception as exc:
            logger.exception("Échec du pipeline global")
            report["status"] = "failed"
            report["error"] = str(exc)
            raise

        logger.info("Pipeline global terminé avec succès")
        return report

    def _run_faq_pipeline_if_enabled(self) -> dict[str, Any] | None:
        """
        Exécute le pipeline FAQ hybride si activé dans la config.
        """
        hybrid_config = self.config.get("hybrid", {})
        sheet_sources = hybrid_config.get("sheet_sources", [])

        if not sheet_sources:
            logger.info("Aucune source FAQ hybride configurée.")
            return None

        faq_source = self._find_first_enabled_source(sheet_sources)
        if not faq_source:
            logger.info("Aucune source FAQ hybride activée.")
            return None

        if faq_source.get("transformer") != "faq_hybrid":
            logger.info("Aucune source faq_hybrid activée trouvée.")
            return None

        outputs = faq_source.get("outputs", {})

        required_output_keys = {"business_file", "rag_file", "quality_file", "stats_file"}
        missing_keys = required_output_keys - set(outputs.keys())
        if missing_keys:
            raise ValueError(
                f"Clés outputs manquantes pour la FAQ hybride : {sorted(missing_keys)}"
            )

        runner = FaqBatchRunner(
            input_path=self.settings.raw_data_path / "google_sheets" / faq_source["file_name"],
            business_output_path=self.settings.business_data_path / outputs["business_file"],
            rag_output_path=self.settings.rag_data_path / outputs["rag_file"],
            rejected_output_path=self.settings.quality_data_path / outputs["quality_file"],
            stats_output_path=self.settings.quality_data_path / outputs["stats_file"],
            sheet_name=faq_source.get("sheet_name"),
            language=faq_source.get("language", "fr"),
        )

        result = runner.run()
        logger.info("Pipeline FAQ exécuté avec succès")
        return result["stats"]

    def _run_business_pipeline_if_enabled(self) -> dict[str, Any] | None:
        """
        Exécute le pipeline business si la section business est présente.
        """
        business_config = self.config.get("business", {})
        if not business_config:
            logger.info("Aucune section business configurée.")
            return None

        runner = BusinessBatchRunner(
            raw_sheets_dir=self.settings.raw_data_path / "google_sheets",
            raw_docs_dir=self.settings.raw_data_path / "google_drive_docs",
            business_dir=self.settings.business_data_path,
        )

        result = runner.run()
        logger.info("Pipeline business exécuté avec succès")
        return result["stats"]

    def _run_rag_pipeline_if_enabled(self) -> dict[str, Any] | None:
        """
        Exécute le pipeline RAG documents si activé dans la config.
        """
        rag_config = self.config.get("rag", {})
        document_sources = rag_config.get("document_sources", [])

        if not document_sources:
            logger.info("Aucune source documentaire RAG configurée.")
            return None

        rag_source = self._find_first_enabled_source(document_sources)
        if not rag_source:
            logger.info("Aucune source documentaire RAG activée.")
            return None

        if rag_source.get("transformer") != "document":
            logger.info("Aucune source documentaire avec transformer=document activée trouvée.")
            return None

        output_file = rag_source.get("output_file")
        if not output_file:
            raise ValueError("output_file manquant dans la configuration rag.document_sources")

        raw_docs_dir = self.settings.raw_data_path / rag_source["directory"]
        rag_output_path = self.settings.rag_data_path / output_file

        runner = RagBatchRunner(
            raw_docs_dir=raw_docs_dir,
            rag_output_path=rag_output_path,
        )

        result = runner.run()
        logger.info("Pipeline RAG documents exécuté avec succès")
        return result["stats"]

    def _run_indexing_if_enabled(self) -> dict[str, Any] | None:
        """
        Exécute l'indexation multi-sources si activée.
        """
        indexing_config = self.config.get("indexing", {})
        if not indexing_config.get("enabled", False):
            logger.info("Indexation désactivée dans la configuration.")
            return None

        input_files = indexing_config.get("input_files", [])
        if not input_files:
            logger.info("Aucun fichier d'entrée configuré pour l'indexation.")
            return None

        resolved_input_files = [
            self.settings.rag_data_path / file_name
            for file_name in input_files
        ]

        runner = IndexingBatchRunner(input_files=resolved_input_files)
        result = runner.run()

        logger.info("Pipeline d'indexation exécuté avec succès")
        return result["stats"]

    @staticmethod
    def _find_first_enabled_source(sources: list[dict[str, Any]]) -> dict[str, Any] | None:
        """
        Retourne la première source activée.
        """
        for source in sources:
            if source.get("enabled", False):
                return source
        return None


def main() -> dict[str, Any]:
    setup_logging("INFO")
    settings = get_settings()

    runner = PipelineRunner(
        config_path=settings.config_path / "pipeline_sources.json",
    )
    return runner.run()


if __name__ == "__main__":
    main()