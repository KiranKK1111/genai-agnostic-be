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

    # ── LLM ──────────────────────────────────────────────
    AI_FACTORY_API: str = "http://localhost:11434/v1"
    AI_FACTORY_TOKEN: str = ""
    AI_FACTORY_MODEL: str = "llama3.1:8b"
    LLM_MAX_TOKENS: int = 2000
    LLM_DEFAULT_TEMPERATURE: float = 0.7
    LLM_JSON_TEMPERATURE: float = 0.1
    LLM_SQL_TEMPERATURE: float = 0.1
    LLM_REQUEST_TIMEOUT: int = 60
    LLM_STREAM_TIMEOUT: int = 120

    # ── Embeddings ───────────────────────────────────────
    EMBEDDING_MODEL: str = "local-sentence-encoder"
    EMBEDDING_DIMENSIONS: int = 384
    EMBEDDING_QUERY_IDF_BOOST: float = 1.3    # Asymmetric: boost rare terms for queries
    EMBEDDING_DOC_IDF_DAMPEN: float = 0.85    # Asymmetric: dampen for documents
    EMBED_CACHE_L1_MAX_SIZE: int = 1000       # In-memory embedding cache entries

    # ── Auth ───────────────────────────────────────────────
    AUTH_ENABLED: bool = False
    JWT_SECRET: str = "local-dev-secret-change-in-production"
    JWT_EXPIRE_MIN: int = 1440

    # ── Session ────────────────────────────────────────────
    SESSION_TTL_HOURS: int = 24
    HISTORY_COMPRESS_THRESHOLD: int = 10   # Compress history when turns exceed this

    # ── SQL Safety ─────────────────────────────────────────
    SQL_TIMEOUT_SEC: int = 8
    RECORD_WARN_THRESHOLD: int = 5000
    MAX_RESULT_ROWS: int = 10000
    MAX_POPULATE_ROWS: int = 100000  # Max rows for "populate all" (browser rendering limit)

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

    # ── FAISS Vector Search ────────────────────────────────
    FAISS_IVF_THRESHOLD: int = 10000
    FAISS_IVF_NCENTROIDS: int = 100
    FAISS_IVF_NPROBE: int = 10
    FAISS_MAX_CHUNKS: int = 50000
    FAISS_VALUE_SIMILARITY: float = 0.80      # Min similarity for FAISS value grounding
    RETRIEVAL_CACHE_TTL_MIN: int = 30

    # ── BM25 Search ────────────────────────────────────────
    BM25_K1: float = 1.5                      # Term frequency saturation
    BM25_B: float = 0.75                      # Document length normalization

    # ── Hybrid Search ──────────────────────────────────────
    HYBRID_DENSE_WEIGHT: float = 0.6          # RRF weight for dense (FAISS) results
    HYBRID_RETRIEVAL_K: int = 20              # Candidates per source before fusion

    # ── Similarity & Matching ──────────────────────────────
    AMBIGUITY_GAP: float = 0.12              # Top-2 score gap below which = ambiguous
    MIN_SIMILARITY: float = 0.50             # Min similarity to consider a result relevant
    FUZZY_MATCH_THRESHOLD: int = 75          # Levenshtein score for fuzzy value matching
    VALUE_FUZZY_MIN_LENGTH_RATIO: float = 0.4  # Min length ratio for fuzzy matching

    # ── Schema Inspection ──────────────────────────────────
    SCHEMA_WATCH_INTERVAL_MIN: int = 15
    SCHEMA_SAMPLE_ROWS: int = 5              # Sample rows per table for value grounding
    SCHEMA_LLM_CONCURRENCY: int = 10         # Max parallel LLM calls during inspection
    SCHEMA_MAX_DISTINCT_VALUES: int = 200    # Max distinct values per column to index

    # ── Neural Training (Stages 9-12) ────────────────────
    NEURAL_TRAIN_EPOCHS: int = 50           # Max contrastive training epochs (early stopping)
    NEURAL_TRAIN_LR: float = 0.0003         # Adam learning rate (lower = more stable)
    NEURAL_TRAIN_BATCH_SIZE: int = 16       # Training batch size
    NEURAL_TRAIN_PAIRS: int = 50            # LLM-generated training pairs

    # ── Drift Monitoring ───────────────────────────────────
    DRIFT_ALERT_THRESHOLD: float = 0.15      # Similarity distribution shift threshold
    DRIFT_CENTROID_K: int = 10               # Number of centroids for drift detection

    # ── Database Pool ──────────────────────────────────────
    DB_POOL_MIN: int = 1
    DB_POOL_MAX: int = 5
    DB_COMMAND_TIMEOUT: int = 60             # asyncpg command timeout (seconds)

    # ── API Limits ─────────────────────────────────────────
    CHAT_HISTORY_LIMIT: int = 50             # Max sessions returned in list
    DEFAULT_SEARCH_LIMIT: int = 20           # Default search result limit
    AUDIT_LOG_LIMIT: int = 50                # Default audit log entries

    # ── CORS ───────────────────────────────────────────────
    CORS_ORIGINS: str = "*"

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def ollama_base_url(self) -> str:
        """Base Ollama URL (without /v1) for health checks."""
        api = self.AI_FACTORY_API
        return api.rsplit("/v1", 1)[0] if "/v1" in api else api

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
