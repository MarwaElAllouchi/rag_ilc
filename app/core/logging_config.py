import logging
import sys


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure un logging simple, propre et réutilisable pour tout le projet.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )