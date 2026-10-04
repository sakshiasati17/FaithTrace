"""
Celery task definitions.

Async workers for ingestion, pipeline runs, evaluation, and diagnostics.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

from celery import Celery
from app.core.config import settings

logger = logging.getLogger(__name__)

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
    "app.workers.tasks.train_failure_classifier": {"queue": "diagnostics"},
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_eval_set(eval_set_path: str) -> list[dict]:
    """Load a built-in eval set file. Only files inside eval_sets/ are allowed."""
    from app.services.evaluation import eval_sets as es
    try:
        return es.load_builtin(eval_set_path)
    except (es.EvalSetError, OSError, json.JSONDecodeError) as exc:
        logger.error("Could not load eval set %r: %s", eval_set_path, exc)
        return []


def _str_or_none(value) -> str | None:
    return value if isinstance(value, str) and value else None


def load_eval_set_for(db, experiment=None, eval_set_path: str | None = None) -> list[dict]:
    """
    The eval set an experiment runs against, in order of preference:
      1. the uploaded set (experiment.eval_set_id) from the eval_sets table;
      2. the experiment's stored built-in path (experiment.eval_set_path);
      3. an eval_set_path passed to the task (legacy callers / pre-005 experiments);
      4. DEFAULT_EVAL_SET_PATH.
    Paths are restricted to eval_sets/. Returns [] (and logs) when nothing loads.
    """
    from app.db.models import EvalSet
    from app.services.evaluation.eval_sets import DEFAULT_EVAL_SET_PATH

    eval_set_id = _str_or_none(getattr(experiment, "eval_set_id", None))
    if eval_set_id:
        row = db.get(EvalSet, eval_set_id)
        if row is None:
            logger.error("Eval set %s (experiment %s) not found",
                         eval_set_id, getattr(experiment, "id", None))
            return []
        return list(row.items or [])

    path = (
        _str_or_none(getattr(experiment, "eval_set_path", None))
        or _str_or_none(eval_set_path)
        or DEFAULT_EVAL_SET_PATH
    )
    return _load_eval_set(path)


def _describe_eval_set(experiment, eval_set_path: str | None) -> str:
    eval_set_id = _str_or_none(getattr(experiment, "eval_set_id", None))
    if eval_set_id:
        return f"eval set {eval_set_id}"
    return _str_or_none(getattr(experiment, "eval_set_path", None)) or eval_set_path or "default eval set"


def _is_errored(qr) -> bool:
    """True when a stored query result recorded a pipeline error (not an answer)."""
    return qr.status == "error"


def _run_should_fail(error_count: int, total: int) -> bool:
    """A run is failed when more than half of its queries errored."""
    return total > 0 and error_count * 2 > total


# ─── Experiment lifecycle ─────────────────────────────────────────────────────
#
# pending -> running (generating) -> evaluating -> diagnosing -> done, or failed.
# Runs keep pending/running/done/failed; a done run is finished once its
# diagnosed_at is set. The experiment status is recomputed from its runs by
# refresh_experiment_status, which every stage calls after committing its own
# run's progress.

_POST_GENERATION_STATUSES = ("evaluating", "diagnosing")


def derive_experiment_status(runs) -> str | None:
    """
    Experiment status implied by its runs once generation has finished, or
    None while any run is still pending/running (generating).

    Failed runs are ignored; with no run left the experiment is failed.
    """
    if any(r.status in ("pending", "running") for r in runs):
        return None
    active = [r for r in runs if r.status == "done"]
    if not active:
        return "failed"
    if all(r.diagnosed_at is not None for r in active):
        return "done"
    if all(r.evaluated_at is not None for r in active):
        return "diagnosing"
    return "evaluating"


def refresh_experiment_status(db, experiment_id: str, generation_finished: bool = False) -> str | None:
    """
    Recompute an experiment's status from its runs and commit it.

    The experiment row is locked (SELECT ... FOR UPDATE) before its runs are
    read, so when two workers finish the last runs at the same time they
    update the status one after the other, and the second one sees the
    first's committed run progress: the experiment moves to done exactly once
    and never back. Callers must commit their own run's progress first.

    Only experiments past generation (evaluating/diagnosing) are updated;
    run_experiment passes generation_finished=True to move one out of
    running. Returns the resulting status (None if the experiment is gone).
    """
    from app.db.models import Experiment, Run
    from sqlalchemy import select

    experiment = db.execute(
        select(Experiment)
        .where(Experiment.id == experiment_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()
    if experiment is None:
        db.rollback()
        logger.error("Experiment %s not found while refreshing its status", experiment_id)
        return None

    current = experiment.status
    if current not in _POST_GENERATION_STATUSES and not (
        generation_finished and current == "running"
    ):
        db.commit()  # release the row lock
        return current

    runs = db.execute(
        select(Run)
        .where(Run.experiment_id == experiment_id)
        .execution_options(populate_existing=True)
    ).scalars().all()
    new = derive_experiment_status(runs)
    if new is None and generation_finished:
        # A run is still pending/running after run_experiment processed all of
        # them: nothing will move it on, so treat the experiment as evaluating.
        logger.error("Experiment %s: runs still generating after run_experiment finished", experiment_id)
        new = "evaluating"
    if new is not None and new != current:
        experiment.status = new
        if new in ("done", "failed"):
            experiment.completed_at = datetime.utcnow()
        logger.info("Experiment %s: %s -> %s", experiment_id, current, new)
        if new == "failed":
            logger.error("Experiment %s failed: no run finished successfully", experiment_id)
    db.commit()
    return experiment.status


def _fail_run(db, run_id: str, reason: str) -> None:
    """
    Mark a run failed (it will not be evaluated/diagnosed) and update its
    experiment, so an in-progress experiment can still finish. Runs of an
    experiment that already finished (e.g. a manual re-evaluation failing) keep
    their status.
    """
    from app.db.models import Run

    db.rollback()
    run = db.get(Run, run_id)
    if run is None:
        logger.error("Run %s not found while marking it failed: %s", run_id, reason)
        return
    experiment = run.experiment
    if experiment is not None and experiment.status in ("done", "failed"):
        logger.error(
            "Run %s: %s (experiment %s already %s; run status left as %s)",
            run_id, reason, run.experiment_id, experiment.status, run.status,
        )
        return
    logger.error("Run %s marked failed: %s", run_id, reason)
    run.status = "failed"
    db.commit()
    refresh_experiment_status(db, run.experiment_id)


def _retries_exhausted(task) -> bool:
    return task.request.retries >= task.max_retries


def _to_runner_result(qr):
    """Rebuild a runner QueryResult dataclass from a stored row."""
    from app.services.experiment.runner import QueryResult
    return QueryResult(
        query_id=qr.query_id,
        question=qr.question,
        generated_answer=qr.generated_answer,
        retrieved_chunks=qr.retrieved_chunks or [],
        latency_ms=qr.latency_ms,
        input_tokens=qr.input_tokens,
        output_tokens=qr.output_tokens,
        cost_usd=qr.cost_usd,
    )


# ─── Task: ingest_document ────────────────────────────────────────────────────

@celery_app.task(name="app.workers.tasks.ingest_document", bind=True, max_retries=3)
def ingest_document(self, document_id: str, strategy: str):
    """Parse, chunk, embed, and index a document."""
    from app.db.session import get_sync_db
    from app.db.models import Document
    from app.services.ingestion import parser as parser_mod
    from app.services.ingestion import chunker as chunker_mod
    from app.services.ingestion import indexer as indexer_mod

    db = get_sync_db()
    try:
        doc = db.get(Document, document_id)
        if not doc:
            return {"error": f"Document {document_id} not found"}

        doc.parse_status = "running"
        db.commit()

        # 1. Ensure Qdrant collection exists
        indexer_mod.ensure_collection()

        # 2. Parse the document
        file_path = Path(doc.storage_path)
        raw_chunks = parser_mod.parse_document(file_path, strategy)

        # 3. Chunk text-type chunks; pass through table/spreadsheet chunks as-is
        all_chunks = []
        for raw in raw_chunks:
            chunk_type = raw.get("chunk_type", "text")
            content = raw.get("content", "")

            if not content.strip():
                continue

            if chunk_type == "text":
                # Apply chunking strategy to text chunks
                chunk_strategy = "recursive"
                sub_chunks = chunker_mod.chunk(content, strategy=chunk_strategy)
                for sc in sub_chunks:
                    enriched = {
                        "content": sc["content"],
                        "chunk_type": "text",
                        "page": raw.get("page"),
                        "table_id": None,
                        "doc_version": doc.version_label,
                        "effective_from": doc.effective_from.isoformat() if doc.effective_from else None,
                        "effective_to": doc.effective_to.isoformat() if doc.effective_to else None,
                        "filename": doc.filename,
                        "metadata": {**raw.get("metadata", {}), **sc.get("metadata", {})},
                    }
                    all_chunks.append(enriched)
            else:
                # Table, spreadsheet_cell, image chunks are atomic
                enriched = {
                    "content": content,
                    "chunk_type": chunk_type,
                    "page": raw.get("page"),
                    "table_id": raw.get("table_id"),
                    "doc_version": doc.version_label,
                    "effective_from": doc.effective_from.isoformat() if doc.effective_from else None,
                    "effective_to": doc.effective_to.isoformat() if doc.effective_to else None,
                    "filename": doc.filename,
                    "metadata": raw.get("metadata", {}),
                }
                all_chunks.append(enriched)

        # 4. Upsert to Qdrant
        if all_chunks:
            indexer_mod.upsert_chunks(all_chunks, doc_id=document_id)

        doc.parse_status = "done"
        doc.index_status = "done"
        doc.doc_metadata = {
            **(doc.doc_metadata or {}),
            "chunks_indexed": len(all_chunks),
            "strategy": strategy,
        }
        db.commit()
        return {"document_id": document_id, "chunks_indexed": len(all_chunks)}

    except Exception as exc:
        try:
            doc = db.get(Document, document_id)
            if doc:
                doc.parse_status = "failed"
                doc.index_status = "failed"
                db.commit()
        except Exception:
            pass
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
    finally:
        db.close()


# ─── Task: run_experiment ─────────────────────────────────────────────────────

@celery_app.task(name="app.workers.tasks.run_experiment", bind=True, max_retries=1)
def run_experiment(self, experiment_id: str, eval_set_path: str | None = None):
    """Execute all pipeline runs for an experiment."""
    from app.db.session import get_sync_db
    from app.db.models import Experiment, Run, QueryResult as QueryResultModel
    from app.services.experiment.runner import BUDGET_EXCEEDED, PipelineConfig, run_pipeline
    from dataclasses import fields

    db = get_sync_db()
    try:
        experiment = db.get(Experiment, experiment_id)
        if not experiment:
            return {"error": f"Experiment {experiment_id} not found"}

        experiment.status = "running"
        db.commit()

        eval_set = load_eval_set_for(db, experiment, eval_set_path)
        if not eval_set:
            experiment.status = "failed"
            db.commit()
            return {"error": f"Eval set is empty or not found: {_describe_eval_set(experiment, eval_set_path)}"}

        # Fetch all pending/queued runs
        from sqlalchemy import select
        runs = db.execute(
            select(Run).where(Run.experiment_id == experiment_id)
        ).scalars().all()

        run_errors: dict[str, int] = {}
        for run in runs:
            try:
                run.status = "running"
                db.commit()

                # Deserialize config
                config_dict = run.config
                config = PipelineConfig(**{
                    k: config_dict[k]
                    for k in config_dict
                    if k in {f.name for f in fields(PipelineConfig)}
                })

                # Execute pipeline (stops issuing queries past the cost limit)
                results = run_pipeline(
                    config,
                    eval_set,
                    max_cost_usd=settings.MAX_COST_PER_RUN_USD,
                    document_ids=experiment.document_ids,
                )

                # Persist QueryResult rows
                for result in results:
                    qr = QueryResultModel(
                        id=str(uuid.uuid4()),
                        run_id=run.id,
                        query_id=result.query_id,
                        question=result.question,
                        generated_answer=result.generated_answer,
                        retrieved_chunks=result.retrieved_chunks,
                        latency_ms=result.latency_ms,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        cost_usd=result.cost_usd,
                        status=result.status,
                        error_message=result.error_message,
                    )
                    db.add(qr)

                error_count = sum(1 for r in results if r.status == "error")
                run_errors[run.id] = error_count
                if error_count:
                    logger.warning(
                        "Run %s: %d of %d queries errored", run.id, error_count, len(results)
                    )

                run.completed_at = datetime.utcnow()
                skipped = sum(1 for r in results if r.error_message == BUDGET_EXCEEDED)
                if skipped:
                    run.status = "failed"
                    db.commit()
                    logger.error(
                        "Run %s marked failed: cost limit MAX_COST_PER_RUN_USD=$%.4f exceeded "
                        "($%.4f spent); %d of %d queries not run; skipping evaluation",
                        run.id, settings.MAX_COST_PER_RUN_USD,
                        sum(r.cost_usd for r in results), skipped, len(results),
                    )
                    continue
                if _run_should_fail(error_count, len(results)):
                    run.status = "failed"
                    db.commit()
                    logger.error(
                        "Run %s marked failed: %d of %d queries errored; skipping evaluation",
                        run.id, error_count, len(results),
                    )
                    continue

                run.status = "done"
                db.commit()

                # Trigger evaluation for this run
                evaluate_run.delay(run.id, eval_set_path)

            except Exception as run_exc:
                logger.exception("Run %s failed: %s", run.id, run_exc)
                run.status = "failed"
                db.commit()
                # Continue with other runs

        # Generation finished for every run: evaluating (or failed when no run
        # succeeded). Evaluations queued above may already have finished.
        final_status = refresh_experiment_status(db, experiment_id, generation_finished=True)

        return {
            "experiment_id": experiment_id,
            "status": final_status,
            "runs_processed": len(runs),
            "query_errors_by_run": run_errors,
        }

    except Exception as exc:
        try:
            experiment = db.get(Experiment, experiment_id)
            if experiment:
                experiment.status = "failed"
                db.commit()
        except Exception:
            pass
        # Rate limit errors need longer recovery time (60s base, doubles each retry)
        try:
            from openai import RateLimitError
            if isinstance(exc, RateLimitError):
                raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        except ImportError:
            pass
        raise self.retry(exc=exc, countdown=5)
    finally:
        db.close()


# ─── Task: evaluate_run ───────────────────────────────────────────────────────

@celery_app.task(name="app.workers.tasks.evaluate_run", bind=True, max_retries=2)
def evaluate_run(self, run_id: str, eval_set_path: str | None = None):
    """Compute metrics for a completed pipeline run."""
    from app.db.session import get_sync_db
    from app.db.models import Run, QueryResult as QueryResultModel, RunMetrics as RunMetricsModel
    from app.services.evaluation.metrics import compute_metrics
    from sqlalchemy import select

    db = get_sync_db()
    try:
        run = db.get(Run, run_id)
        if not run:
            return {"error": f"Run {run_id} not found"}

        eval_set = load_eval_set_for(db, run.experiment, eval_set_path)
        if not eval_set:
            error = f"Eval set not found or empty: {_describe_eval_set(run.experiment, eval_set_path)}"
            _fail_run(db, run_id, f"evaluation: {error}")
            return {"error": error}

        # Load query results from DB
        qr_rows = db.execute(
            select(QueryResultModel).where(QueryResultModel.run_id == run_id)
        ).scalars().all()

        if not qr_rows:
            _fail_run(db, run_id, "evaluation: no query results")
            return {"error": "No query results found for this run"}

        # Errored queries have no answer to score: leave them out of metrics.
        ok_rows = [qr for qr in qr_rows if not _is_errored(qr)]
        error_count = len(qr_rows) - len(ok_rows)
        if error_count:
            logger.warning(
                "Run %s: excluding %d errored of %d queries from metrics",
                run_id, error_count, len(qr_rows),
            )

        if not ok_rows:
            logger.error("Run %s: every query errored; no metrics written", run_id)
            run.evaluated_at = datetime.utcnow()
            db.commit()
            # Still diagnose so errored rows are marked as skipped.
            diagnose_run.delay(run_id, eval_set_path)
            refresh_experiment_status(db, run.experiment_id)
            return {
                "run_id": run_id,
                "faithfulness": None,
                "queries_evaluated": 0,
                "queries_errored": error_count,
            }

        # Reconstruct QueryResult dataclasses
        results = [_to_runner_result(qr) for qr in ok_rows]

        # Compute metrics
        metrics = compute_metrics(results, eval_set)

        # Check for existing metrics row
        existing = db.execute(
            select(RunMetricsModel).where(RunMetricsModel.run_id == run_id)
        ).scalar_one_or_none()

        if existing:
            rm = existing
        else:
            rm = RunMetricsModel(id=str(uuid.uuid4()), run_id=run_id)
            db.add(rm)

        rm.answer_correctness = metrics.answer_correctness
        rm.faithfulness = metrics.faithfulness
        rm.context_precision = metrics.context_precision
        rm.context_recall = metrics.context_recall
        rm.answer_relevance = metrics.answer_relevance
        rm.latency_p50_ms = metrics.latency_p50_ms
        rm.latency_p95_ms = metrics.latency_p95_ms
        rm.avg_token_usage = metrics.avg_token_usage
        rm.avg_cost_usd = metrics.avg_cost_usd
        rm.freshness_validity = metrics.freshness_validity
        rm.temporal_citation_accuracy = metrics.temporal_citation_accuracy
        rm.multimodal_grounding_rate = metrics.multimodal_grounding_rate
        # root_cause_diagnostic_accuracy is written by diagnose_run, after diagnoses exist.
        run.evaluated_at = datetime.utcnow()
        db.commit()

        # Enqueue diagnostics
        diagnose_run.delay(run_id, eval_set_path)
        refresh_experiment_status(db, run.experiment_id)

        return {
            "run_id": run_id,
            "faithfulness": metrics.faithfulness,
            "queries_evaluated": len(ok_rows),
            "queries_errored": error_count,
        }

    except Exception as exc:
        if _retries_exhausted(self):
            _give_up_on_run(db, run_id, "evaluation", exc)
            raise
        # Rate limit errors need longer recovery time (60s base, doubles each retry)
        try:
            from openai import RateLimitError
            if isinstance(exc, RateLimitError):
                raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        except ImportError:
            pass
        raise self.retry(exc=exc, countdown=10)
    finally:
        db.close()


def _give_up_on_run(db, run_id: str, stage: str, exc: Exception) -> None:
    """Out of retries: fail the run so its experiment can still finish."""
    logger.exception("Run %s: %s failed after all retries", run_id, stage)
    try:
        _fail_run(db, run_id, f"{stage} failed after retries: {type(exc).__name__}: {exc}")
    except Exception:
        logger.exception("Run %s: could not mark run failed after %s error", run_id, stage)


# ─── Task: diagnose_run ───────────────────────────────────────────────────────

@celery_app.task(name="app.workers.tasks.diagnose_run", bind=True, max_retries=2)
def diagnose_run(self, run_id: str, eval_set_path: str | None = None):
    """Run root-cause diagnostics on a completed, evaluated run."""
    from app.db.session import get_sync_db
    from app.db.models import Run, QueryResult as QueryResultModel, RunMetrics as RunMetricsModel
    from app.services.diagnostics.classifier import (
        compute_diagnostic_accuracy,
        diagnose_run as classifier_diagnose_run,
    )
    from app.services.evaluation.ragas_runner import run_ragas_evaluation
    from sqlalchemy import select

    db = get_sync_db()
    try:
        run = db.get(Run, run_id)
        if not run:
            return {"error": f"Run {run_id} not found"}

        eval_set = load_eval_set_for(db, run.experiment, eval_set_path)
        if not eval_set:
            error = f"Eval set not found or empty: {_describe_eval_set(run.experiment, eval_set_path)}"
            _fail_run(db, run_id, f"diagnosis: {error}")
            return {"error": error}

        # Load query results (deterministic order)
        qr_rows = db.execute(
            select(QueryResultModel)
            .where(QueryResultModel.run_id == run_id)
            .order_by(QueryResultModel.query_id, QueryResultModel.id)
        ).scalars().all()

        if not qr_rows:
            _fail_run(db, run_id, "diagnosis: no query results")
            return {"error": "No query results found"}

        # Errored queries have no answer to diagnose: mark them skipped.
        ok_rows = []
        error_count = 0
        for qr in qr_rows:
            if _is_errored(qr):
                error_count += 1
                qr.failure_category = None
                qr.diagnosis_evidence = {"skipped": "query errored"}
            else:
                ok_rows.append(qr)
        if error_count:
            logger.warning(
                "Run %s: skipping diagnosis for %d errored of %d queries",
                run_id, error_count, len(qr_rows),
            )

        results = [_to_runner_result(qr) for qr in ok_rows]

        diagnoses = []
        if results:
            # Get per-query Ragas scores for diagnostics
            try:
                per_query_scores = run_ragas_evaluation(results, eval_set)
            except Exception as exc:
                logger.warning(
                    "Ragas scoring failed for run %s; diagnosing without per-query metrics: %s",
                    run_id, exc,
                )
                per_query_scores = [{}] * len(results)

            # Run diagnostics
            diagnoses = classifier_diagnose_run(results, eval_set, per_query_scores)

        # Update query result rows with diagnosis
        qr_by_id = {qr.query_id: qr for qr in ok_rows}
        for diagnosis in diagnoses:
            qr = qr_by_id.get(diagnosis.query_id)
            if qr:
                qr.failure_category = diagnosis.primary_failure.value
                qr.diagnosis_evidence = {
                    **diagnosis.evidence,
                    "confidence": diagnosis.confidence,
                    "secondary_failures": [f.value for f in diagnosis.secondary_failures],
                }
        db.commit()

        # Root-cause diagnostic accuracy: stored diagnoses vs ground-truth
        # failure_type labels (None when no item is labelled).
        diag_accuracy = compute_diagnostic_accuracy(
            {qr.query_id: qr.failure_category for qr in ok_rows},
            eval_set,
        )
        rm = db.execute(
            select(RunMetricsModel).where(RunMetricsModel.run_id == run_id)
        ).scalar_one_or_none()
        if rm is None:
            # evaluate_run creates the metrics row before enqueueing this task.
            logger.warning(
                "No RunMetrics row for run %s; root_cause_diagnostic_accuracy not stored", run_id
            )
        else:
            rm.root_cause_diagnostic_accuracy = diag_accuracy

        run.diagnosed_at = datetime.utcnow()
        db.commit()
        # The last run to finish diagnosis moves the experiment to done.
        refresh_experiment_status(db, run.experiment_id)

        return {
            "run_id": run_id,
            "diagnoses_written": len(diagnoses),
            "queries_errored": error_count,
            "root_cause_diagnostic_accuracy": diag_accuracy,
        }

    except Exception as exc:
        if _retries_exhausted(self):
            _give_up_on_run(db, run_id, "diagnosis", exc)
            raise
        raise self.retry(exc=exc, countdown=10)
    finally:
        db.close()


# ─── Task: train_failure_classifier ──────────────────────────────────────────

@celery_app.task(name="app.workers.tasks.train_failure_classifier", bind=True, max_retries=1)
def train_failure_classifier(self, experiment_id: str, eval_set_path: str | None = None):
    """
    Train the XGBoost failure classifier on all labeled query results from
    a completed experiment.

    Collects (features, label) pairs from every QueryResult that has a
    non-null failure_category, then trains and persists the model.
    """
    from app.db.session import get_sync_db
    from app.db.models import Experiment, Run, QueryResult as QueryResultModel, QueryFeedback
    from app.services.diagnostics.ml_classifier import train as ml_train
    from app.services.diagnostics.classifier import index_eval_set
    from sqlalchemy import select

    db = get_sync_db()
    try:
        # The experiment's own eval set unless an explicit path was given.
        experiment = db.get(Experiment, experiment_id)
        if _str_or_none(eval_set_path):
            eval_set = _load_eval_set(eval_set_path)
        else:
            eval_set = load_eval_set_for(db, experiment)
        # Eval sets key items by "id" ("query_id" accepted as a fallback).
        eval_by_id = index_eval_set(eval_set)

        # Collect all labeled QueryResults for this experiment
        run_rows = db.execute(
            select(Run).where(Run.experiment_id == experiment_id)
        ).scalars().all()

        if not run_rows:
            return {"error": f"No runs found for experiment {experiment_id}"}

        run_ids = [r.id for r in run_rows]
        qr_rows = db.execute(
            select(QueryResultModel)
            .where(
                QueryResultModel.run_id.in_(run_ids),
                QueryResultModel.failure_category.isnot(None),
            )
            .order_by(QueryResultModel.run_id, QueryResultModel.query_id, QueryResultModel.id)
        ).scalars().all()

        if not qr_rows:
            return {
                "error": "No labeled query results found. "
                         "Run diagnose_run first, or add failure_type labels to your eval set."
            }

        # Load all human feedback for these query results — feedback labels override classifier labels
        qr_ids = [qr.id for qr in qr_rows]
        fb_rows = db.execute(
            select(QueryFeedback).where(QueryFeedback.query_result_id.in_(qr_ids))
        ).scalars().all()
        # Map: query_result.id → feedback
        feedback_by_qr_id = {fb.query_result_id: fb for fb in fb_rows}
        feedback_overrides = sum(1 for fb in fb_rows if fb.correct_label)
        positive_feedback = sum(1 for fb in fb_rows if fb.rating == "positive")

        all_metrics, all_chunks, all_eval_items, all_labels = [], [], [], []
        for qr in qr_rows:
            evidence = qr.diagnosis_evidence or {}
            metrics = {
                "faithfulness":       evidence.get("faithfulness", 0.0),
                "context_recall":     evidence.get("context_recall", 0.0),
                "context_precision":  evidence.get("context_precision", 0.0),
                "answer_correctness": evidence.get("answer_correctness", 0.0),
                "answer_relevance":   evidence.get("answer_relevance", 0.0),
                "latency_ms":         qr.latency_ms or 0.0,
                "cost_usd":           qr.cost_usd or 0.0,
            }
            eval_item = eval_by_id.get(qr.query_id, {})

            # Determine label: human feedback > heuristic/ML classifier label
            fb = feedback_by_qr_id.get(qr.id)
            if fb and fb.rating == "positive":
                # User confirmed answer was correct → NO_FAILURE
                label = "NO_FAILURE"
            elif fb and fb.correct_label:
                # User provided correct failure label → use it
                label = fb.correct_label
            else:
                # Fall back to classifier-assigned label
                label = qr.failure_category

            all_metrics.append(metrics)
            all_chunks.append(qr.retrieved_chunks or [])
            all_eval_items.append(eval_item)
            all_labels.append(label)

        summary = ml_train(all_metrics, all_chunks, all_eval_items, all_labels)

        # Reload model into memory for immediate use
        from app.services.diagnostics.ml_classifier import reload_model
        reload_model()

        return {
            "experiment_id": experiment_id,
            "samples_used": len(all_labels),
            "feedback_overrides": feedback_overrides,
            "positive_feedback_used": positive_feedback,
            **summary,
        }

    except Exception as exc:
        raise self.retry(exc=exc, countdown=10)
    finally:
        db.close()

