"""
Integration layer: Celery task, recommendation trigger, and API helpers.

The Celery task wraps the full pipeline:
  mine triplets → train → evaluate before/after → persist report

EmbeddingOptimizationRecommender checks whether the embedding model
should be retrained based on recent retrieval failure rates.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Celery task ───────────────────────────────────────────────────────────────

def get_celery_app():
    from app.workers.tasks import celery_app
    return celery_app


def train_embedding_task(
    experiment_id: str,
    eval_set_path: str = "eval_sets/sample_eval_set.json",
    output_dir: str = "checkpoints/embedding_finetuner",
    num_epochs: int = 20,
    batch_size: int = 32,
) -> dict[str, Any]:
    """
    End-to-end embedding fine-tuning pipeline (called by Celery worker).

    1. Load labeled query results for experiment_id
    2. Mine triplets with EvalDataMiner
    3. Train FineTunableEmbeddingModel with CombinedContrastiveLoss
    4. Evaluate before/after retrieval metrics
    5. Persist report to output_dir/eval_report.json
    """
    import json
    from app.db.session import get_sync_db
    from app.db.models import QueryResult, Run
    from sqlalchemy import select
    from .data_mining import EvalDataMiner
    from .train import train as embedding_train
    from .evaluate import EmbeddingEvaluator
    from .model import FineTunableEmbeddingModel
    import torch

    db = get_sync_db()
    try:
        # Load eval set
        eval_set: list[dict] = []
        paths_to_try = [
            Path(eval_set_path),
            Path("/app") / eval_set_path,
            Path(__file__).parents[4] / eval_set_path,
        ]
        for p in paths_to_try:
            if p.exists():
                with open(p) as f:
                    eval_set = json.load(f)
                break

        # Load query results
        run_ids_result = db.execute(select(Run.id).where(Run.experiment_id == experiment_id))
        run_ids = [r[0] for r in run_ids_result.all()]
        if not run_ids:
            return {"error": f"No runs for experiment {experiment_id}"}

        rows = db.execute(
            select(QueryResult)
            .where(QueryResult.run_id.in_(run_ids))
        ).scalars().all()

        query_results = [
            {
                "query_id": qr.query_id,
                "question": qr.question,
                "retrieved_chunks": qr.retrieved_chunks or [],
                "run_id": qr.run_id,
            }
            for qr in rows
        ]

        ragas_scores = [
            (qr.diagnosis_evidence or {})
            for qr in rows
        ]

        if not query_results:
            return {"error": "No query results to mine from"}

        # Mine triplets
        miner = EvalDataMiner()
        triplets = miner.mine_triplets(query_results, eval_set, ragas_scores)

        if len(triplets) < 10:
            return {
                "error": f"Too few training triplets ({len(triplets)}). "
                         "Run more experiments or lower EvalDataMiner thresholds."
            }

        logger.info("Mined %d training triplets for experiment %s", len(triplets), experiment_id)

        # Train
        train_summary = embedding_train(
            triplets_or_records=triplets,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

        if "error" in train_summary:
            return train_summary

        # Evaluate before/after
        checkpoint_path = Path(output_dir) / "best_model.pt"
        if checkpoint_path.exists() and eval_set:
            baseline = FineTunableEmbeddingModel()
            finetuned = FineTunableEmbeddingModel()
            ckpt = torch.load(str(checkpoint_path), map_location="cpu")
            finetuned.load_state_dict(ckpt["model_state_dict"])

            evaluator = EmbeddingEvaluator()
            report = evaluator.evaluate(baseline, finetuned, eval_set)
            report_dict = {
                "baseline": asdict(report.baseline),
                "finetuned": asdict(report.finetuned),
                "improvements": report.improvements,
                "embedding_shift": report.embedding_shift,
                "eval_samples": report.eval_samples,
            }

            with open(Path(output_dir) / "eval_report.json", "w") as f:
                json.dump(report_dict, f, indent=2)
        else:
            report_dict = {"note": "No eval set or checkpoint for before/after comparison"}

        return {
            "experiment_id": experiment_id,
            "triplets_mined": len(triplets),
            "training": train_summary,
            "evaluation": report_dict,
        }

    finally:
        db.close()


# ── Recommendation trigger ─────────────────────────────────────────────────────

class EmbeddingOptimizationRecommender:
    """
    Decides whether embedding fine-tuning is warranted based on
    recent run diagnostics. Triggers automatically when >30 % of
    queries fail due to retrieval (low recall, table miss, context pollution).
    """

    RETRIEVAL_FAILURE_CATEGORIES = {
        "LOW_RECALL_RETRIEVAL",
        "TABLE_RETRIEVAL_MISS",
        "IRRELEVANT_CONTEXT_POLLUTION",
        "CHUNKING_BOUNDARY_ERROR",
    }
    TRIGGER_THRESHOLD = 0.30

    def should_retrain(self, run_diagnostics: list[dict[str, Any]]) -> bool:
        if not run_diagnostics:
            return False
        retrieval_failures = sum(
            1 for d in run_diagnostics
            if d.get("failure_category") in self.RETRIEVAL_FAILURE_CATEGORIES
        )
        rate = retrieval_failures / len(run_diagnostics)
        if rate >= self.TRIGGER_THRESHOLD:
            logger.info(
                "Retrieval failure rate %.1f%% ≥ threshold %.0f%% — "
                "embedding fine-tuning recommended",
                rate * 100, self.TRIGGER_THRESHOLD * 100,
            )
            return True
        return False

    def recommendation_text(self, run_diagnostics: list[dict[str, Any]]) -> str:
        retrieval_failures = sum(
            1 for d in run_diagnostics
            if d.get("failure_category") in self.RETRIEVAL_FAILURE_CATEGORIES
        )
        rate = retrieval_failures / max(len(run_diagnostics), 1)
        return (
            f"{rate:.0%} of queries failed due to retrieval issues "
            f"({retrieval_failures}/{len(run_diagnostics)}). "
            "Domain-adaptive embedding fine-tuning is recommended: "
            "POST /optimization/embedding/train to start."
        )


# ── Voice command patterns ─────────────────────────────────────────────────────

EMBEDDING_VOICE_PATTERNS = [
    (
        r"(?:fine.?tune|train|adapt|retrain) (?:the )?embeddings?",
        "train_embeddings",
        "/optimization/embedding/train",
        "POST",
        lambda m: {},
    ),
    (
        r"(?:show|get) embedding (?:report|results|evaluation|metrics)",
        "get_embedding_report",
        "/optimization/embedding/report",
        "GET",
        lambda m: {},
    ),
    (
        r"(?:how (?:much|well) did|compare) (?:the )?embeddings? (?:improve|change)",
        "get_embedding_report",
        "/optimization/embedding/report",
        "GET",
        lambda m: {},
    ),
]
