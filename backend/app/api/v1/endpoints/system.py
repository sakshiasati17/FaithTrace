"""
System status endpoint.

Reports live per-component health (database, Qdrant, Redis broker, Celery
workers, ML classifier) so the UI can show real status instead of hard-coded
indicators. Every check is bounded by a timeout and never raises: a failing
component is reported as {"ok": false, "detail": "..."}.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter

from app import __version__
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

CHECK_TIMEOUT_S = 2.0

# Blocking client calls run here rather than in the loop's default executor,
# so a hung check never delays the response or the loop's executor shutdown.
_CHECK_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="status-check")


# ─── Individual checks ────────────────────────────────────────────────────────
# Each returns a dict of extra fields on success and raises on failure;
# _run_check turns that into the {"ok": ..., "detail": ...} shape.

async def _check_database() -> dict:
    from sqlalchemy import text
    from app.db.session import async_engine

    async with async_engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return {"detail": "SELECT 1 succeeded"}


def _check_qdrant() -> dict:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.QDRANT_URL, timeout=int(CHECK_TIMEOUT_S))
    names = [c.name for c in client.get_collections().collections]
    return {
        "detail": f"{len(names)} collection(s)",
        "collection_exists": settings.QDRANT_COLLECTION in names,
    }


def _check_redis() -> dict:
    import redis

    client = redis.Redis.from_url(
        settings.CELERY_BROKER_URL,
        socket_timeout=CHECK_TIMEOUT_S,
        socket_connect_timeout=CHECK_TIMEOUT_S,
    )
    try:
        client.ping()
    finally:
        client.close()
    return {"detail": "PING succeeded"}


def _check_workers() -> dict:
    from app.workers.tasks import celery_app

    replies = celery_app.control.inspect(timeout=CHECK_TIMEOUT_S).ping() or {}
    if not replies:
        raise RuntimeError("no Celery workers responded to ping")
    return {"detail": f"{len(replies)} worker(s) responding", "workers": sorted(replies)}


def _check_ml_classifier() -> dict:
    from app.services.diagnostics.ml_classifier import is_trained

    trained = is_trained()
    return {
        "trained": trained,
        "classifier_type": "xgboost" if trained else "heuristic",
        "detail": "XGBoost model trained" if trained else "not trained; using heuristic rules",
    }


async def _run_check(name: str, check: Callable[[], Any], timeout: float) -> dict:
    """Run one check with a timeout. Never raises."""
    start = time.perf_counter()
    try:
        if asyncio.iscoroutinefunction(check):
            extra = await asyncio.wait_for(check(), timeout=timeout)
        else:
            loop = asyncio.get_running_loop()
            extra = await asyncio.wait_for(
                loop.run_in_executor(_CHECK_EXECUTOR, check), timeout=timeout
            )
        result = {"ok": True, **extra}
    except asyncio.TimeoutError:
        logger.warning("System status check %s timed out after %.1fs", name, timeout)
        result = {"ok": False, "detail": f"timed out after {timeout:.1f}s"}
    except Exception as exc:
        logger.warning("System status check %s failed: %s", name, exc)
        result = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"[:300]}
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return result


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@router.get("/status")
async def get_system_status():
    """
    Live per-component status. Always returns 200 with `ok: false` entries
    for failing components; the top-level `ok` is true only if all are ok.
    """
    database, qdrant, redis_status = await asyncio.gather(
        _run_check("database", _check_database, CHECK_TIMEOUT_S),
        _run_check("qdrant", _check_qdrant, CHECK_TIMEOUT_S),
        _run_check("redis", _check_redis, CHECK_TIMEOUT_S),
    )

    # Celery's inspect() blocks on broker reconnects when Redis is down,
    # so only ping workers once the broker is reachable.
    if redis_status["ok"]:
        workers = await _run_check("workers", _check_workers, CHECK_TIMEOUT_S + 1.0)
    else:
        workers = {"ok": False, "detail": "skipped: broker (redis) unreachable", "latency_ms": 0.0}

    ml_classifier = await _run_check("ml_classifier", _check_ml_classifier, CHECK_TIMEOUT_S)

    components = {
        "database": database,
        "qdrant": qdrant,
        "redis": redis_status,
        "workers": workers,
        "ml_classifier": ml_classifier,
    }
    return {
        "ok": all(c["ok"] for c in components.values()),
        "version": __version__,
        "env": settings.APP_ENV,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "components": components,
    }
