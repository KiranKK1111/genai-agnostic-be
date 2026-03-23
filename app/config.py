"""All application settings loaded from environment variables via pydantic-settings."""
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    # ── PostgreSQL ─────────────────────────────────────────
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "genai_dashboard"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_SCHEMA: str = "public"
    APP_SCHEMA: str = "genai_app"

    # ── LLM / Embeddings ──────────────────────────────────
    AI_FACTORY_API: str = "http://localhost:11434/v1"
    AI_FACTORY_MODEL: str = "llama3.1:8b"
    EMBEDDINGS_API: str = "http://localhost:11434/api/embeddings"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_DIMENSIONS: int = 768

    # ── Auth ───────────────────────────────────────────────
    AUTH_ENABLED: bool = False
    JWT_SECRET: str = "local-dev-secret-change-in-production"
    JWT_EXPIRE_MIN: int = 1440

    # ── Session ────────────────────────────────────────────
    SESSION_TTL_HOURS: int = 24

    # ── SQL Safety ─────────────────────────────────────────
    SQL_TIMEOUT_SEC: int = 8
    RECORD_WARN_THRESHOLD: int = 5000

    # ── File Upload ────────────────────────────────────────
    UPLOAD_DIR: str = "./uploads"
    ALLOWED_FILE_TYPES: str = "txt,json,xlsx,pdf,csv,docx"
    MAX_UPLOAD_STORAGE_MB: int = 5000
    EXPORT_FILE_TTL_HOURS: int = 1

    # ── RAG ────────────────────────────────────────────────
    CHUNK_SIZE: int = 1600
    CHUNK_OVERLAP: int = 320

    # ── Input Quality ──────────────────────────────────────
    GIBBERISH_THRESHOLD: float = 0.4
    GIBBERISH_HARD_THRESHOLD: float = 0.7
    MAX_INPUT_LENGTH: int = 10000

    # ── FAISS Vector Search ─────────────────────────────────
    FAISS_IVF_THRESHOLD: int = 10000      # Switch from Flat to IVF above this many vectors
    FAISS_IVF_NCENTROIDS: int = 100       # Number of IVF centroids (clusters)
    FAISS_IVF_NPROBE: int = 10            # Number of clusters to search at query time
    RETRIEVAL_CACHE_TTL_MIN: int = 30     # Cache TTL in minutes (invalidated on schema rebuild)

    # ── Schema ─────────────────────────────────────────────
    SCHEMA_WATCH_INTERVAL_MIN: int = 15

    # ── Pool ───────────────────────────────────────────────
    DB_POOL_MIN: int = 1
    DB_POOL_MAX: int = 5

    # ── CORS ───────────────────────────────────────────────
    CORS_ORIGINS: str = "*"

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def allowed_file_types_list(self) -> list[str]:
        return [t.strip() for t in self.ALLOWED_FILE_TYPES.split(",")]

    @property
    def cors_origins_list(self) -> list[str]:
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
