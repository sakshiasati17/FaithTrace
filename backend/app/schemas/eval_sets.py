"""Pydantic schemas for eval set endpoints."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class EvalSetSummary(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    source: str  # "upload" | "builtin"
    filename: Optional[str] = None
    path: Optional[str] = None  # built-in sets only: "eval_sets/<file>.json"
    item_count: int
    created_at: Optional[datetime] = None
    is_default: bool = False

    class Config:
        from_attributes = True


class EvalSetDetail(EvalSetSummary):
    items: list[dict] = []


class EvalSetUploadResponse(EvalSetSummary):
    warnings: list[str] = []
