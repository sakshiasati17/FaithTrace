"""
Corpus ingestion and management endpoints.

Handles upload, parsing, versioning, and indexing of enterprise documents
including PDFs, DOCX, HTML, CSV, and XLSX files.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Annotated

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: Annotated[UploadFile, File()],
    version_label: Annotated[str, Form()] = "v1",
    effective_from: Annotated[str, Form()] = "",
    effective_to: Annotated[str, Form()] = "",
):
    """Upload and ingest a single enterprise document."""
    raise NotImplementedError


@router.get("/")
async def list_documents():
    """List all ingested documents with version metadata."""
    raise NotImplementedError


@router.get("/{document_id}")
async def get_document(document_id: str):
    """Retrieve document metadata, version history, and chunk index status."""
    raise NotImplementedError


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Remove a document and its associated chunks from the index."""
    raise NotImplementedError
