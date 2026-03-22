"""
FastAPI dependency injection helpers.
"""

from app.db.session import get_db, AsyncSession
from qdrant_client import AsyncQdrantClient
from app.core.config import settings

__all__ = ["get_db", "get_qdrant_client"]


async def get_qdrant_client() -> AsyncQdrantClient:
    """FastAPI dependency: yields a configured Qdrant async client."""
    client = AsyncQdrantClient(url=settings.QDRANT_URL)
    try:
        yield client
    finally:
        await client.close()
