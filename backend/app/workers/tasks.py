"""
Celery task definitions.

Async workers for ingestion, pipeline runs, evaluation, and diagnostics.
"""

from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "faithtrace",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.task_routes = {
    "app.workers.tasks.ingest_document": {"queue": "ingestion"},
    "app.workers.tasks.run_experiment": {"queue": "experiments"},
    "app.workers.tasks.evaluate_run": {"queue": "evaluation"},
    "app.workers.tasks.diagnose_run": {"queue": "diagnostics"},
}


@celery_app.task(name="app.workers.tasks.ingest_document", bind=True, max_retries=3)
def ingest_document(self, document_id: str, strategy: str):
    """Parse, chunk, embed, and index a document."""
    raise NotImplementedError


@celery_app.task(name="app.workers.tasks.run_experiment", bind=True, max_retries=1)
def run_experiment(self, experiment_id: str):
    """Execute all pipeline runs for an experiment."""
    raise NotImplementedError


@celery_app.task(name="app.workers.tasks.evaluate_run", bind=True, max_retries=2)
def evaluate_run(self, run_id: str):
    """Compute metrics for a completed pipeline run."""
    raise NotImplementedError


@celery_app.task(name="app.workers.tasks.diagnose_run", bind=True, max_retries=2)
def diagnose_run(self, run_id: str):
    """Run root-cause diagnostics on a completed, evaluated run."""
    raise NotImplementedError
