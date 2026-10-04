from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    SECRET_KEY: str = "changeme"

    # CORS — comma-separated origins. In prod set to your real domain.
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost", "http://localhost:80"]

    # API key auth — set a strong random value in prod. Leave empty to disable (dev only).
    API_KEY: str = ""

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/faithtrace"

    # Vector DB
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "faithtrace_chunks"

    # LLM
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Vision page parsing (text_table_vision, opt-in per PDF upload). Each page
    # sent is one paid vision call; pages past VISION_MAX_PAGES are not sent.
    VISION_MODEL: str = "gpt-4o"
    VISION_MAX_PAGES: int = 20

    # Chunking strategies text is indexed with at ingest (comma-separated:
    # fixed_size, recursive, semantic, structure_aware). Every strategy listed
    # is embedded and stored, so embedding cost grows with each one; semantic
    # also calls the embeddings API while chunking, so it is off by default.
    # Runs whose chunking_strategy is not indexed are failed (reindex after
    # changing this).
    INGEST_CHUNKING_STRATEGIES: str = "fixed_size,recursive,structure_aware"

    # Cross-encoder used by the hybrid_reranker retrieval strategy (downloaded from
    # Hugging Face on first use, then cached per process)
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # OpenAI spend guard — max LLM cost per pipeline run (USD). Past it the run's
    # remaining queries are skipped ("budget exceeded") and the run is failed.
    # <= 0 disables the limit.
    MAX_COST_PER_RUN_USD: float = 5.0

    # Celery / Redis
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # Storage
    OBJECT_STORAGE_PATH: str = "./storage"

    # File upload limit (MB)
    MAX_UPLOAD_SIZE_MB: int = 50

    # ML classifier
    ML_CLASSIFIER_PATH: str = "/app/storage/ml_models/xgb_classifier.pkl"

    # Error tracking (optional — set Sentry DSN to enable)
    SENTRY_DSN: Optional[str] = None

    # LangSmith (optional tracing)
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "faithtrace"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
