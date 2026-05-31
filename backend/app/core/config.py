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

    # OpenAI spend guard — max cost per experiment run (USD)
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
