"""
Qdrant vector index operations.

Handles collection initialization, chunk upserting, and deletion.
Dates stored as Unix epoch integers for Qdrant range filter support.
"""

import uuid
from datetime import datetime
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


def _get_client() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_URL)


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )


def ensure_collection() -> None:
    """Create the Qdrant collection if it doesn't exist."""
    client = _get_client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )


def _dt_to_epoch(dt_str: Optional[str]) -> Optional[int]:
    """Convert ISO date string to Unix epoch integer for Qdrant filtering."""
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str)
        return int(dt.timestamp())
    except (ValueError, TypeError):
        return None


def upsert_chunks(chunks: list[dict], doc_id: str) -> None:
    """
    Embed and upsert chunks into Qdrant.

    Each chunk dict must have: content, chunk_type, page, table_id, metadata.
    Additional fields from document metadata (doc_version, effective_from,
    effective_to, filename) should also be present.
    """
    if not chunks:
        return

    client = _get_client()
    embeddings = _get_embeddings()

    ensure_collection()

    texts = [c["content"] for c in chunks]
    vectors = embeddings.embed_documents(texts)

    points = []
    for chunk, vector in zip(chunks, vectors):
        payload = {
            "doc_id": doc_id,
            "doc_version": chunk.get("doc_version", "v1"),
            "chunk_type": chunk.get("chunk_type", "text"),
            "page": chunk.get("page"),
            "table_id": chunk.get("table_id"),
            "filename": chunk.get("filename", ""),
            "content": chunk["content"],
            # Store dates as epoch integers for range filtering
            "effective_from": _dt_to_epoch(chunk.get("effective_from")),
            "effective_to": _dt_to_epoch(chunk.get("effective_to")),
        }
        # Merge any additional metadata
        if "metadata" in chunk and isinstance(chunk["metadata"], dict):
            for k, v in chunk["metadata"].items():
                if k not in payload:
                    payload[k] = v

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
        )

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points[i : i + batch_size],
        )


def delete_doc_chunks(doc_id: str) -> None:
    """Remove all chunks belonging to a document from Qdrant."""
    client = _get_client()
    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
    )


def fetch_all_chunks(doc_ids: Optional[list[str]] = None) -> list[dict]:
    """
    Fetch all chunk payloads (for BM25 in-memory indexing).
    Optionally filter by doc_ids.
    """
    client = _get_client()

    scroll_filter = None
    if doc_ids:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        scroll_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))]
        )

    results = []
    offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=scroll_filter,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for r in records:
            results.append(r.payload)
        if next_offset is None:
            break
        offset = next_offset

    return results
