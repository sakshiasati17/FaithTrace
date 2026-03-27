"""
Human feedback endpoints.

Allows users to rate individual query answers (thumbs up / thumbs down)
and optionally provide a correct failure label. Feedback is stored and
used to improve XGBoost classifier training.

Route ordering: specific paths (/run/...) must come BEFORE generic (/{id})
so FastAPI doesn't swallow them as path parameters.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional

from app.db.session import get_db
from app.db.models import QueryResult, QueryFeedback

router = APIRouter()


class FeedbackPayload(BaseModel):
    rating: str                          # "positive" | "negative"
    correct_label: Optional[str] = None  # optional override e.g. "LOW_RECALL_RETRIEVAL"
    comment: Optional[str] = None


# ── Specific routes first (before generic /{query_result_id}) ──────────────────

@router.get("/run/{run_id}/summary")
async def get_run_feedback_summary(
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Aggregate feedback stats for all queries in a run."""
    qrs_result = await db.execute(
        select(QueryResult.id).where(QueryResult.run_id == run_id)
    )
    qr_ids = [r[0] for r in qrs_result.all()]
    if not qr_ids:
        return {"positive": 0, "negative": 0, "total": 0, "coverage": 0.0}

    fb_result = await db.execute(
        select(QueryFeedback).where(QueryFeedback.query_result_id.in_(qr_ids))
    )
    feedbacks = fb_result.scalars().all()

    positive = sum(1 for f in feedbacks if f.rating == "positive")
    negative = sum(1 for f in feedbacks if f.rating == "negative")
    total_rated = len(feedbacks)

    return {
        "positive": positive,
        "negative": negative,
        "total": total_rated,
        "coverage": round(total_rated / len(qr_ids), 3),
    }


# ── Generic routes last ────────────────────────────────────────────────────────

@router.post("/{query_result_id}")
async def submit_feedback(
    query_result_id: str,
    payload: FeedbackPayload,
    db: AsyncSession = Depends(get_db),
):
    """Submit thumbs-up / thumbs-down feedback for a query result."""
    if payload.rating not in ("positive", "negative"):
        raise HTTPException(status_code=422, detail="rating must be 'positive' or 'negative'")

    qr = await db.get(QueryResult, query_result_id)
    if not qr:
        raise HTTPException(status_code=404, detail="Query result not found")

    existing = await db.execute(
        select(QueryFeedback).where(QueryFeedback.query_result_id == query_result_id)
    )
    existing_fb = existing.scalar_one_or_none()

    if existing_fb:
        existing_fb.rating = payload.rating
        existing_fb.correct_label = payload.correct_label
        existing_fb.comment = payload.comment
        fb = existing_fb
    else:
        fb = QueryFeedback(
            query_result_id=query_result_id,
            rating=payload.rating,
            correct_label=payload.correct_label,
            comment=payload.comment,
        )
        db.add(fb)

    await db.commit()
    await db.refresh(fb)

    return {
        "id": fb.id,
        "query_result_id": fb.query_result_id,
        "rating": fb.rating,
        "correct_label": fb.correct_label,
        "comment": fb.comment,
        "created_at": fb.created_at,
    }


@router.get("/{query_result_id}")
async def get_feedback(
    query_result_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get existing feedback for a query result (null if none submitted)."""
    result = await db.execute(
        select(QueryFeedback).where(QueryFeedback.query_result_id == query_result_id)
    )
    fb = result.scalar_one_or_none()
    if not fb:
        return None
    return {
        "id": fb.id,
        "rating": fb.rating,
        "correct_label": fb.correct_label,
        "comment": fb.comment,
    }
