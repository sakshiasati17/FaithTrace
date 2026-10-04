"""Pydantic schemas for corpus endpoints."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, model_validator


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    version_label: str
    effective_from: Optional[datetime] = None
    effective_to: Optional[datetime] = None
    parse_status: str
    index_status: str
    created_at: datetime
    doc_metadata: dict = {}
    # Parsing strategy used for ingestion (text_only, text_table,
    # text_table_vision, spreadsheet_aware); copied from doc_metadata["strategy"].
    parsing_strategy: Optional[str] = None

    @model_validator(mode="after")
    def _fill_parsing_strategy(self):
        if self.parsing_strategy is None:
            self.parsing_strategy = (self.doc_metadata or {}).get("strategy")
        return self

    class Config:
        from_attributes = True
