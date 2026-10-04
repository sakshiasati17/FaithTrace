"""
Corpus ingestion and management endpoints.

Handles upload, parsing, versioning, and indexing of enterprise documents
including PDFs, DOCX, HTML, CSV, and XLSX files.
"""

import logging
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

logger = logging.getLogger(__name__)

router = APIRouter()

# Parsing strategy that sends rendered PDF pages to a vision model (paid, opt-in).
VISION_STRATEGY = "text_table_vision"

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


def _select_strategy(file_type: str, enable_vision: bool = False) -> str:
    """
    Parsing strategy for an upload or reindex.

    Vision parsing is opt-in and only applies to PDFs; for every other case
    (and when not requested) this is exactly _default_strategy.
    """
    if enable_vision and file_type == "pdf":
        return VISION_STRATEGY
    return _default_strategy(file_type)


def _require_vision_key(strategy: str) -> None:
    """Refuse vision parsing up front when no OpenAI key is configured."""
    if strategy == VISION_STRATEGY and not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vision parsing needs OPENAI_API_KEY to be set on the server. "
                   "Set it, or upload without enable_vision.",
        )


def _strategy_metadata(file_type: str, strategy: str, enable_vision: bool) -> dict:
    """doc_metadata entries describing how the document is parsed."""
    meta: dict = {"strategy": strategy, "vision_requested": enable_vision}
    if enable_vision and strategy != VISION_STRATEGY:
        meta["vision_ignored"] = f"vision parsing only applies to PDFs, not {file_type}"
    return meta


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: Annotated[UploadFile, File()],
    version_label: Annotated[str, Form()] = "v1",
    effective_from: Annotated[str, Form()] = "",
    effective_to: Annotated[str, Form()] = "",
    enable_vision: Annotated[bool, Form()] = False,
    db: AsyncSession = Depends(get_db),
):
    """
    Upload and ingest a single enterprise document.

    enable_vision (PDF only) parses with text_table_vision: up to
    VISION_MAX_PAGES pages are sent to VISION_MODEL. It costs money, so it is
    off by default and rejected with 400 when OPENAI_API_KEY is not set.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    document_id = str(uuid.uuid4())
    file_type = _detect_file_type(file.filename or "")
    strategy = _select_strategy(file_type, enable_vision)
    _require_vision_key(strategy)
    if enable_vision and strategy != VISION_STRATEGY:
        logger.info("enable_vision ignored for %s: vision parsing only applies to PDFs", file.filename)

    # Save file to storage
    storage_root = Path(settings.OBJECT_STORAGE_PATH)
    doc_dir = storage_root / document_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    file_path = doc_dir / (file.filename or "document")

    contents = await file.read()

    # Enforce upload size limit
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({len(contents) // (1024*1024)} MB). Max allowed: {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

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
        doc_metadata={"original_filename": file.filename, **_strategy_metadata(file_type, strategy, enable_vision)},
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
    enable_vision: Annotated[bool, Form()] = False,
    db: AsyncSession = Depends(get_db),
):
    """
    Re-ingest an already-uploaded document, deleting old chunks and re-indexing from disk.

    Vision parsing is opt-in on every reindex (enable_vision, PDF only), like upload.
    """
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = Path(doc.storage_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Original file no longer on disk")

    strategy = _select_strategy(doc.file_type, enable_vision)
    _require_vision_key(strategy)

    # Delete existing Qdrant chunks for this document
    try:
        from app.services.ingestion.indexer import delete_doc_chunks
        delete_doc_chunks(document_id)
    except Exception as exc:
        logger.warning("Could not delete old chunks of %s before reindex: %s", document_id, exc)

    doc.parse_status = "pending"
    doc.index_status = "pending"
    old_meta = {k: v for k, v in (doc.doc_metadata or {}).items() if k not in ("vision", "vision_ignored")}
    doc.doc_metadata = {**old_meta, **_strategy_metadata(doc.file_type, strategy, enable_vision)}
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
    except Exception as exc:
        # Non-fatal: proceed with DB deletion
        logger.warning("Could not delete chunks of %s from Qdrant: %s", document_id, exc)

    # Remove stored file
    try:
        import shutil
        doc_dir = Path(doc.storage_path).parent
        if doc_dir.exists():
            shutil.rmtree(str(doc_dir))
    except Exception as exc:
        logger.warning("Could not remove stored file of %s: %s", document_id, exc)

    await db.delete(doc)
    await db.commit()
