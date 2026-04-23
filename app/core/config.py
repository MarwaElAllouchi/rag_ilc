from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """
    Configuration centralisée de l'application et des pipelines batch.
    """

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    app_name: str = Field(default="rag_ilc", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="127.0.0.1", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # -------------------------------------------------------------------------
    # LLM / Embeddings
    # -------------------------------------------------------------------------
    mistral_api_key: str = Field(default="", alias="MISTRAL_API_KEY")
    mistral_chat_model: str = Field(default="mistral-large-latest", alias="MISTRAL_CHAT_MODEL")
    mistral_embed_model: str = Field(default="mistral-embed", alias="MISTRAL_EMBED_MODEL")
    embedding_dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")

    # -------------------------------------------------------------------------
    # Base de données
    # -------------------------------------------------------------------------
    db_host: str = Field(default="db", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_name: str = Field(default="rag_db", alias="DB_NAME")
    db_user: str = Field(default="postgres", alias="DB_USER")
    db_password: str = Field(default="", alias="DB_PASSWORD")
    db_echo: bool = Field(default=False, alias="DB_ECHO")
    db_pool_size: int = Field(default=5, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=10, alias="DB_MAX_OVERFLOW")
    db_pool_recycle: int = Field(default=1800, alias="DB_POOL_RECYCLE")

    # -------------------------------------------------------------------------
    # Répertoires
    # -------------------------------------------------------------------------
    data_dir: str = Field(default="data", alias="DATA_DIR")
    raw_data_dir: str = Field(default="data/raw", alias="RAW_DATA_DIR")
    processed_data_dir: str = Field(default="data/processed", alias="PROCESSED_DATA_DIR")
    export_data_dir: str = Field(default="data/exports", alias="EXPORT_DATA_DIR")
    rag_data_dir: str = Field(default="data/rag", alias="RAG_DATA_DIR")
    business_data_dir: str = Field(default="data/business", alias="BUSINESS_DATA_DIR")
    quality_data_dir: str = Field(default="data/quality", alias="QUALITY_DATA_DIR")
    config_dir: str = Field(default="config", alias="CONFIG_DIR")

    # -------------------------------------------------------------------------
    # RAG
    # -------------------------------------------------------------------------
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=5, alias="TOP_K")
    max_context_chunks: int = Field(default=5, alias="MAX_CONTEXT_CHUNKS")
    max_retrieval_distance: float = Field(default=0.28, alias="MAX_RETRIEVAL_DISTANCE")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------
    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, value: str) -> str:
        allowed = {"development", "test", "production"}
        value = value.strip().lower()
        if value not in allowed:
            raise ValueError(f"APP_ENV invalide : {value}. Valeurs autorisées : {sorted(allowed)}")
        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        value = value.strip().upper()
        if value not in allowed:
            raise ValueError(f"LOG_LEVEL invalide : {value}. Valeurs autorisées : {sorted(allowed)}")
        return value

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("CHUNK_SIZE doit être strictement positif")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError("CHUNK_OVERLAP ne peut pas être négatif")
        return value

    @field_validator("top_k", "max_context_chunks")
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Cette valeur doit être strictement positive")
        return value

    # -------------------------------------------------------------------------
    # Computed properties
    # -------------------------------------------------------------------------
    @property
    def database_url(self) -> URL:
        return URL.create(
            drivername="postgresql+psycopg2",
            username=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
        )

    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    @property
    def data_path(self) -> Path:
        return BASE_DIR / self.data_dir

    @property
    def raw_data_path(self) -> Path:
        return BASE_DIR / self.raw_data_dir

    @property
    def processed_data_path(self) -> Path:
        return BASE_DIR / self.processed_data_dir

    @property
    def export_data_path(self) -> Path:
        return BASE_DIR / self.export_data_dir

    @property
    def rag_data_path(self) -> Path:
        return BASE_DIR / self.rag_data_dir

    @property
    def business_data_path(self) -> Path:
        return BASE_DIR / self.business_data_dir

    @property
    def quality_data_path(self) -> Path:
        return BASE_DIR / self.quality_data_dir

    @property
    def config_path(self) -> Path:
        return BASE_DIR / self.config_dir

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"

    @property
    def rag_input_files_for_indexing(self) -> list[Path]:
        return [
            self.rag_data_path / "transformed_documents.json",
            self.rag_data_path / "faq_documents.json",
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()