"""
Corpus ingestion and management endpoints.

Handles upload, parsing, versioning, and indexing of enterprise documents
including PDFs, DOCX, HTML, CSV, and XLSX files.
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import Document
from app.schemas.corpus import DocumentResponse
from app.core.config import settings

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".csv", ".html", ".htm"}


def _detect_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    mapping = {
        ".pdf": "pdf", ".docx": "docx", ".doc": "docx",
        ".xlsx": "xlsx", ".csv": "csv", ".html": "html", ".htm": "html",
    }
    return mapping.get(ext, "unknown")


def _default_strategy(file_type: str) -> str:
    if file_type in ("xlsx", "csv"):
        return "spreadsheet_aware"
    return "text_table"


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: Annotated[UploadFile, File()],
    version_label: Annotated[str, Form()] = "v1",
    effective_from: Annotated[str, Form()] = "",
    effective_to: Annotated[str, Form()] = "",
    db: AsyncSession = Depends(get_db),
):
    """Upload and ingest a single enterprise document."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    document_id = str(uuid.uuid4())
    file_type = _detect_file_type(file.filename or "")
    strategy = _default_strategy(file_type)

    # Save file to storage
    storage_root = Path(settings.OBJECT_STORAGE_PATH)
    doc_dir = storage_root / document_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    file_path = doc_dir / (file.filename or "document")

    contents = await file.read()
    with open(str(file_path), "wb") as f:
        f.write(contents)

    # Parse optional date fields
    eff_from = None
    eff_to = None
    if effective_from:
        try:
            eff_from = datetime.fromisoformat(effective_from)
        except ValueError:
            pass
    if effective_to:
        try:
            eff_to = datetime.fromisoformat(effective_to)
        except ValueError:
            pass

    doc = Document(
        id=document_id,
        filename=file.filename or "document",
        file_type=file_type,
        version_label=version_label,
        effective_from=eff_from,
        effective_to=eff_to,
        storage_path=str(file_path),
        parse_status="pending",
        index_status="pending",
        doc_metadata={"original_filename": file.filename, "strategy": strategy},
    )
    db.add(doc)
    await db.commit()  # commit before dispatching so the worker can read the row

    # Enqueue ingestion task
    from app.workers.tasks import ingest_document
    ingest_document.delay(document_id, strategy)

    return DocumentResponse.model_validate(doc)


@router.get("/", response_model=list[DocumentResponse])
async def list_documents(db: AsyncSession = Depends(get_db)):
    """List all ingested documents with version metadata."""
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    docs = result.scalars().all()
    return [DocumentResponse.model_validate(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieve document metadata, version history, and chunk index status."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse.model_validate(doc)


@router.post("/{document_id}/reindex", response_model=DocumentResponse)
async def reindex_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Re-ingest an already-uploaded document, deleting old chunks and re-indexing from disk."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = Path(doc.storage_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Original file no longer on disk")

    # Delete existing Qdrant chunks for this document
    try:
        from app.services.ingestion.indexer import delete_doc_chunks
        delete_doc_chunks(document_id)
    except Exception:
        pass

    strategy = _default_strategy(doc.file_type)
    doc.parse_status = "pending"
    doc.index_status = "pending"
    doc.doc_metadata = {**doc.doc_metadata, "strategy": strategy}
    await db.commit()

    from app.workers.tasks import ingest_document
    ingest_document.delay(document_id, strategy)

    return DocumentResponse.model_validate(doc)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str, db: AsyncSession = Depends(get_db)):
    """Remove a document and its associated chunks from the index."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove from Qdrant
    try:
        from app.services.ingestion.indexer import delete_doc_chunks
        delete_doc_chunks(document_id)
    except Exception:
        pass  # Non-fatal: proceed with DB deletion

    # Remove stored file
    try:
        import shutil
        doc_dir = Path(doc.storage_path).parent
        if doc_dir.exists():
            shutil.rmtree(str(doc_dir))
    except Exception:
        pass

    await db.delete(doc)
    await db.commit()
