"""
Eval set endpoints.

Upload, list, inspect and delete question sets. Built-in sets are the JSON
files in the repo's eval_sets/ folder; they are listed with source="builtin"
and an id of "builtin:<filename>".
"""

import logging
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, EvalSet, Experiment
from app.db.session import get_db
from app.schemas.eval_sets import EvalSetDetail, EvalSetSummary, EvalSetUploadResponse
from app.services.evaluation import eval_sets as es

logger = logging.getLogger(__name__)

router = APIRouter()


def _summary(row: EvalSet) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description or "",
        "source": row.source,
        "filename": row.filename,
        "path": None,
        "item_count": row.item_count,
        "created_at": row.created_at,
        "is_default": False,
    }


@router.post("/", response_model=EvalSetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_eval_set(
    file: Annotated[UploadFile, File()],
    name: Annotated[str, Form()] = "",
    description: Annotated[str, Form()] = "",
    db: AsyncSession = Depends(get_db),
):
    """Upload a JSON or CSV eval set. Returns 422 with per-row errors if invalid."""
    # Read one byte past the limit so oversized files are rejected without reading them whole.
    content = await file.read(es.MAX_EVAL_SET_BYTES + 1)
    try:
        raw_items = es.parse_upload(file.filename or "", content)
    except es.EvalSetError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(exc), "errors": [
                {"row": None, "id": None, "field": None, "message": str(exc)}
            ]},
        )

    result = es.validate_items(raw_items)
    if not result.ok:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": f"{len(result.errors)} validation error(s)", "errors": result.errors},
        )

    filenames = set((await db.execute(select(Document.filename).distinct())).scalars().all())
    missing = es.unmatched_source_docs(result.items, filenames)
    warnings = [f"source_docs not found in uploaded documents: {doc}" for doc in missing]

    row = EvalSet(
        id=str(uuid.uuid4()),
        name=name.strip() or (file.filename or "uploaded eval set"),
        description=description,
        source="upload",
        filename=file.filename,
        items=result.items,
        item_count=len(result.items),
        created_at=datetime.utcnow(),
    )
    db.add(row)
    await db.commit()
    return {**_summary(row), "warnings": warnings}


@router.get("/", response_model=list[EvalSetSummary])
async def list_eval_sets(db: AsyncSession = Depends(get_db)):
    """List built-in eval set files followed by uploaded sets (newest first)."""
    rows = (await db.execute(select(EvalSet).order_by(EvalSet.created_at.desc()))).scalars().all()
    return es.list_builtin() + [_summary(r) for r in rows]


@router.get("/{eval_set_id}", response_model=EvalSetDetail)
async def get_eval_set(eval_set_id: str, db: AsyncSession = Depends(get_db)):
    """Return one eval set with its items."""
    if eval_set_id.startswith(es.BUILTIN_ID_PREFIX):
        filename = eval_set_id[len(es.BUILTIN_ID_PREFIX):]
        try:
            items = es.load_builtin(filename)
        except es.EvalSetError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        for entry in es.list_builtin():
            if entry["filename"] == filename:
                return {**entry, "items": items}
        raise HTTPException(status_code=404, detail="Eval set not found")

    row = await db.get(EvalSet, eval_set_id)
    if not row:
        raise HTTPException(status_code=404, detail="Eval set not found")
    return {**_summary(row), "items": row.items or []}


@router.delete("/{eval_set_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_eval_set(eval_set_id: str, db: AsyncSession = Depends(get_db)):
    """Delete an uploaded eval set. 409 if an experiment uses it; built-ins cannot be deleted."""
    if eval_set_id.startswith(es.BUILTIN_ID_PREFIX):
        raise HTTPException(status_code=400, detail="Built-in eval sets cannot be deleted")

    row = await db.get(EvalSet, eval_set_id)
    if not row:
        raise HTTPException(status_code=404, detail="Eval set not found")

    in_use = (await db.execute(
        select(func.count()).select_from(Experiment).where(Experiment.eval_set_id == eval_set_id)
    )).scalar_one()
    if in_use:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Eval set is used by {in_use} experiment(s) and cannot be deleted",
        )

    await db.delete(row)
    await db.commit()
