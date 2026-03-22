"""Pydantic schemas for corpus endpoints."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    version_label: str
    effective_from: Optional[datetime] = None
    effective_to: Optional[datetime] = None
    storage_path: str
    parse_status: str
    index_status: str
    created_at: datetime
    doc_metadata: dict = {}

    class Config:
        from_attributes = True
