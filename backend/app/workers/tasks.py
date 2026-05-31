"""
Celery task definitions.

Async workers for ingestion, pipeline runs, evaluation, and diagnostics.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

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
    "app.workers.tasks.train_failure_classifier": {"queue": "diagnostics"},
    "app.workers.tasks.run_optimizer_agent": {"queue": "experiments"},
    "app.workers.tasks.train_pytorch_failure_classifier": {"queue": "diagnostics"},
    "app.workers.tasks.run_inference_benchmark": {"queue": "experiments"},
    "app.workers.tasks.train_embedding_finetuner": {"queue": "diagnostics"},
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_eval_set(eval_set_path: str) -> list[dict]:
    """Load eval set from JSON file. Tries multiple path resolutions."""
    paths_to_try = [
        Path(eval_set_path),
        Path("/app") / eval_set_path,
        Path(__file__).parent.parent.parent.parent / eval_set_path,
    ]
    for p in paths_to_try:
        if p.exists():
            with open(str(p)) as f:
                return json.load(f)
    # Return empty eval set as fallback
    return []


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
def run_experiment(self, experiment_id: str, eval_set_path: str = "eval_sets/sample_eval_set.json"):
    """Execute all pipeline runs for an experiment."""
    from app.db.session import get_sync_db
    from app.db.models import Experiment, Run, QueryResult as QueryResultModel
    from app.services.experiment.runner import PipelineConfig, run_pipeline
    from dataclasses import fields

    db = get_sync_db()
    try:
        experiment = db.get(Experiment, experiment_id)
        if not experiment:
            return {"error": f"Experiment {experiment_id} not found"}

        experiment.status = "running"
        db.commit()

        eval_set = _load_eval_set(eval_set_path)
        if not eval_set:
            experiment.status = "failed"
            db.commit()
            return {"error": "Eval set is empty or not found"}

        # Fetch all pending/queued runs
        from sqlalchemy import select
        runs = db.execute(
            select(Run).where(Run.experiment_id == experiment_id)
        ).scalars().all()

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

                # Execute pipeline
                results = run_pipeline(config, eval_set)

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
                    )
                    db.add(qr)

                run.status = "done"
                run.completed_at = datetime.utcnow()
                db.commit()

                # Trigger evaluation for this run
                evaluate_run.delay(run.id, eval_set_path)

            except Exception as run_exc:
                run.status = "failed"
                db.commit()
                # Continue with other runs

        experiment.status = "done"
        experiment.completed_at = datetime.utcnow()
        db.commit()

        return {"experiment_id": experiment_id, "runs_processed": len(runs)}

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
def evaluate_run(self, run_id: str, eval_set_path: str = "eval_sets/sample_eval_set.json"):
    """Compute metrics for a completed pipeline run."""
    from app.db.session import get_sync_db
    from app.db.models import Run, QueryResult as QueryResultModel, RunMetrics as RunMetricsModel
    from app.services.experiment.runner import QueryResult
    from app.services.evaluation.metrics import compute_metrics
    from sqlalchemy import select

    db = get_sync_db()
    try:
        run = db.get(Run, run_id)
        if not run:
            return {"error": f"Run {run_id} not found"}

        eval_set = _load_eval_set(eval_set_path)
        if not eval_set:
            return {"error": f"Eval set not found or empty: {eval_set_path}"}

        # Load query results from DB
        qr_rows = db.execute(
            select(QueryResultModel).where(QueryResultModel.run_id == run_id)
        ).scalars().all()

        if not qr_rows:
            return {"error": "No query results found for this run"}

        # Reconstruct QueryResult dataclasses
        results = [
            QueryResult(
                query_id=qr.query_id,
                question=qr.question,
                generated_answer=qr.generated_answer,
                retrieved_chunks=qr.retrieved_chunks or [],
                latency_ms=qr.latency_ms,
                input_tokens=qr.input_tokens,
                output_tokens=qr.output_tokens,
                cost_usd=qr.cost_usd,
            )
            for qr in qr_rows
        ]

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
        rm.root_cause_diagnostic_accuracy = metrics.root_cause_diagnostic_accuracy
        db.commit()

        # Enqueue diagnostics
        diagnose_run.delay(run_id, eval_set_path)

        return {"run_id": run_id, "faithfulness": metrics.faithfulness}

    except Exception as exc:
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


# ─── Task: diagnose_run ───────────────────────────────────────────────────────

@celery_app.task(name="app.workers.tasks.diagnose_run", bind=True, max_retries=2)
def diagnose_run(self, run_id: str, eval_set_path: str = "eval_sets/sample_eval_set.json"):
    """Run root-cause diagnostics on a completed, evaluated run."""
    from app.db.session import get_sync_db
    from app.db.models import Run, QueryResult as QueryResultModel, RunMetrics as RunMetricsModel
    from app.services.experiment.runner import QueryResult
    from app.services.diagnostics.classifier import diagnose_run as classifier_diagnose_run
    from app.services.evaluation.ragas_runner import run_ragas_evaluation
    from sqlalchemy import select

    db = get_sync_db()
    try:
        run = db.get(Run, run_id)
        if not run:
            return {"error": f"Run {run_id} not found"}

        eval_set = _load_eval_set(eval_set_path)
        if not eval_set:
            return {"error": f"Eval set not found or empty: {eval_set_path}"}

        # Load query results
        qr_rows = db.execute(
            select(QueryResultModel).where(QueryResultModel.run_id == run_id)
        ).scalars().all()

        if not qr_rows:
            return {"error": "No query results found"}

        results = [
            QueryResult(
                query_id=qr.query_id,
                question=qr.question,
                generated_answer=qr.generated_answer,
                retrieved_chunks=qr.retrieved_chunks or [],
                latency_ms=qr.latency_ms,
                input_tokens=qr.input_tokens,
                output_tokens=qr.output_tokens,
                cost_usd=qr.cost_usd,
            )
            for qr in qr_rows
        ]

        # Get per-query Ragas scores for diagnostics
        try:
            per_query_scores = run_ragas_evaluation(results, eval_set)
        except Exception:
            per_query_scores = [{}] * len(results)

        # Run diagnostics
        diagnoses = classifier_diagnose_run(results, eval_set, per_query_scores)

        # Update query result rows with diagnosis
        qr_by_id = {qr.query_id: qr for qr in qr_rows}
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

        return {"run_id": run_id, "diagnoses_written": len(diagnoses)}

    except Exception as exc:
        raise self.retry(exc=exc, countdown=10)
    finally:
        db.close()


# ─── Task: train_failure_classifier ──────────────────────────────────────────

@celery_app.task(name="app.workers.tasks.train_failure_classifier", bind=True, max_retries=1)
def train_failure_classifier(self, experiment_id: str, eval_set_path: str):
    """
    Train the XGBoost failure classifier on all labeled query results from
    a completed experiment.

    Collects (features, label) pairs from every QueryResult that has a
    non-null failure_category, then trains and persists the model.
    """
    from app.db.session import get_sync_db
    from app.db.models import Run, QueryResult as QueryResultModel, QueryFeedback
    from app.services.diagnostics.ml_classifier import train as ml_train
    from sqlalchemy import select

    db = get_sync_db()
    try:
        eval_set = _load_eval_set(eval_set_path)
        eval_by_id = {item["query_id"]: item for item in eval_set if "query_id" in item}

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


# ─── Task: run_optimizer_agent ────────────────────────────────────────────────

@celery_app.task(name="app.workers.tasks.run_optimizer_agent", bind=True, max_retries=1)
def run_optimizer_agent(self, job_id: str):
    """Orchestrate the autonomous optimizer loop."""
    from app.db.session import get_sync_db
    from app.db.models import OptimizerJob
    from app.services.experiment.optimizer import OptimizerGoal, run_optimizer_loop

    db = get_sync_db()
    try:
        job = db.get(OptimizerJob, job_id)
        if not job:
            return {"error": f"Optimizer job {job_id} not found"}

        job.status = "running"
        db.commit()

        # Reconstruct goal from persisted JSON
        goal_dict = job.goal or {}
        goal = OptimizerGoal(
            target_metric=goal_dict.get("target_metric", "faithfulness"),
            target_threshold=goal_dict.get("target_threshold", 0.85),
            max_iterations=goal_dict.get("max_iterations", 5),
            max_cost_usd=goal_dict.get("max_cost_usd", 0.50),
            eval_set_path=goal_dict.get("eval_set_path", "eval_sets/sample_eval_set.json"),
        )

        state = run_optimizer_loop(goal, job_id)

        return {
            "job_id": job_id,
            "status": state.status,
            "best_score": state.best_score,
            "iterations": state.iteration,
            "message": state.message,
        }

    except Exception as exc:
        try:
            job = db.get(OptimizerJob, job_id)
            if job:
                job.status = "failed"
                db.commit()
        except Exception:
            pass
        raise self.retry(exc=exc, countdown=30)
    finally:
        db.close()



# ─── Task: train_pytorch_failure_classifier ────────────────────────────────────

@celery_app.task(name="app.workers.tasks.train_pytorch_failure_classifier", bind=True, max_retries=1)
def train_pytorch_failure_classifier(self, experiment_id: str, output_dir: str = "checkpoints/failure_classifier", num_epochs: int = 30):
    """
    Train the PyTorch DistilBERT failure classifier on labeled query results
    from a completed experiment.

    Training data is sourced from QueryResult rows with non-null failure_category,
    which are written by the heuristic / XGBoost diagnostic agent during evaluation.
    After training, the checkpoint is saved to output_dir/best_model.pt.
    """
    from app.db.session import get_sync_db
    from app.db.models import QueryResult, Run
    from sqlalchemy import select

    db = get_sync_db()
    try:
        # Collect all query results for this experiment
        run_ids_result = db.execute(select(Run.id).where(Run.experiment_id == experiment_id))
        run_ids = [r[0] for r in run_ids_result.all()]

        if not run_ids:
            return {"error": "No runs found for experiment"}

        rows = db.execute(
            select(QueryResult)
            .where(QueryResult.run_id.in_(run_ids), QueryResult.failure_category.isnot(None))
        ).scalars().all()

        if len(rows) < 50:
            return {"error": f"Only {len(rows)} labeled samples — need at least 50 to train."}

        # Build records list expected by FailureDataset
        records = []
        for qr in rows:
            ev = qr.diagnosis_evidence or {}
            records.append({
                "query": qr.question,
                "retrieved_context": " ".join(
                    chunk.get("content", "") for chunk in (qr.retrieved_chunks or [])[:3]
                ),
                "generated_answer": qr.generated_answer,
                "faithfulness": ev.get("faithfulness", 0.5),
                "answer_relevance": ev.get("answer_relevance", 0.5),
                "context_recall": ev.get("context_recall", 0.5),
                "context_precision": ev.get("context_precision", 0.5),
                "correctness": ev.get("answer_correctness", 0.5),
                "failure_type": qr.failure_category,  # UPPERCASE from DB; dataset.py normalises it
            })

        from app.models.failure_classifier.train import train as pytorch_train
        pytorch_train(records, output_dir=output_dir, num_epochs=num_epochs)

        return {
            "status": "trained",
            "samples": len(records),
            "checkpoint": f"{output_dir}/best_model.pt",
            "experiment_id": experiment_id,
        }

    except Exception as exc:
        raise self.retry(exc=exc, countdown=30)
    finally:
        db.close()


# ─── Task: run_inference_benchmark ────────────────────────────────────────────

@celery_app.task(name="app.workers.tasks.run_inference_benchmark", bind=True, max_retries=0)
def run_inference_benchmark(self):
    """
    Run the full inference benchmark across PyTorch and any available
    TensorRT engines, save results to benchmarks/latest_report.json.
    """
    import torch
    from pathlib import Path
    from app.optimization.benchmark import InferenceBenchmark

    checkpoint_path = "checkpoints/failure_classifier/best_model.pt"
    if not Path(checkpoint_path).exists():
        return {"error": "No trained classifier found. Train first with /optimization/train/pytorch"}

    from app.models.failure_classifier import FailureClassifier
    model = FailureClassifier(num_classes=6)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    dummy_inputs = (
        torch.randint(0, 30522, (1, 512)),
        torch.ones(1, 512, dtype=torch.long),
        torch.rand(1, 5),
    )

    engine_dir = "models/tensorrt"
    engine_paths = {}
    for precision in ["fp32", "fp16", "int8"]:
        path = f"{engine_dir}/failure_classifier_{precision}.engine"
        if Path(path).exists():
            engine_paths[precision] = path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench = InferenceBenchmark(warmup_runs=20, benchmark_runs=200)

    report = bench.full_report(model, engine_paths, dummy_inputs)
    return {"status": "completed", "results_count": len(report["results"])}


# ─── Task: train_embedding_finetuner ──────────────────────────────────────────

@celery_app.task(name="app.workers.tasks.train_embedding_finetuner", bind=True, max_retries=1)
def train_embedding_finetuner(
    self,
    experiment_id: str,
    eval_set_path: str = "eval_sets/sample_eval_set.json",
    output_dir: str = "checkpoints/embedding_finetuner",
    num_epochs: int = 20,
    batch_size: int = 32,
):
    """
    Domain-adaptive embedding fine-tuning (Extension 4).

    Mines hard negatives from experiment query results, trains
    FineTunableEmbeddingModel with CombinedContrastiveLoss, and
    evaluates retrieval improvements before vs after.
    """
    try:
        from app.models.embedding_finetuner.integration import train_embedding_task
        return train_embedding_task(
            experiment_id=experiment_id,
            eval_set_path=eval_set_path,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
    except Exception as exc:
        raise self.retry(exc=exc, countdown=30)
