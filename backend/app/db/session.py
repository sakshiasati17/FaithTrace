"""
Database session management.

Provides both async (FastAPI) and sync (Celery) session factories.
"""

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings

# ─── Async engine + session (for FastAPI) ─────────────────────────────────────

async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncSession:
    """FastAPI dependency: yields an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ─── Sync engine + session (for Celery workers) ───────────────────────────────

_sync_db_url = settings.DATABASE_URL.replace(
    "postgresql+asyncpg://", "postgresql+psycopg2://"
).replace("postgresql+asyncpg:", "postgresql:")

# Fallback: if the URL uses asyncpg scheme, replace it
if "+asyncpg" in settings.DATABASE_URL:
    _sync_db_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2")
else:
    _sync_db_url = settings.DATABASE_URL

sync_engine = create_engine(
    _sync_db_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def get_sync_db() -> Session:
    """Celery helper: returns a new sync DB session (caller must close)."""
    return SyncSessionLocal()
