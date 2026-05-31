"""
Autonomous Experiment Optimizer.

Iteratively searches for the best RAG pipeline configuration by:
1. Running a broad initial sweep (MVP matrix)
2. Analyzing top performers
3. Generating neighbor configs (vary one axis at a time)
4. Repeating until the target metric threshold is met or budget is exhausted.
"""

import time
import uuid
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)

# ─── Available axis values ────────────────────────────────────────────────────

RETRIEVAL_OPTIONS = ["vector_only", "bm25", "hybrid", "hybrid_reranker"]
CHUNKING_OPTIONS = ["fixed_size", "recursive", "semantic", "structure_aware"]
PARSING_OPTIONS = ["text_only", "text_table", "text_table_vision", "spreadsheet_aware"]
FRESHNESS_OPTIONS = ["none", "recency_biased", "effective_date_filter", "version_aware"]


@dataclass
class OptimizerGoal:
    """Defines what the optimizer is trying to achieve."""
    target_metric: str = "faithfulness"
    target_threshold: float = 0.85
    max_iterations: int = 5
    max_cost_usd: float = 0.50
    eval_set_path: str = "eval_sets/sample_eval_set.json"


@dataclass
class IterationResult:
    """Result of a single optimizer iteration."""
    iteration: int
    experiment_id: str
    configs_tested: int
    best_config: dict
    best_score: float
    total_cost_usd: float


@dataclass
class OptimizerState:
    """Tracks the full optimizer run state (persisted to DB as JSON)."""
    iteration: int = 0
    best_config: dict = field(default_factory=dict)
    best_score: float = 0.0
    total_cost_usd: float = 0.0
    status: str = "pending"  # pending | running | converged | budget_exceeded | max_iterations | failed
    history: list = field(default_factory=list)
    message: str = ""


def _generate_neighbors(top_configs: list[dict], n: int = 3) -> list[dict]:
    """
    Generate neighbor configs by varying one axis at a time from top performers.
    Deduplicates against the input configs.
    """
    from app.services.experiment.runner import PipelineConfig

    axes = {
        "retrieval_strategy": RETRIEVAL_OPTIONS,
        "chunking_strategy": CHUNKING_OPTIONS,
        "parsing_strategy": PARSING_OPTIONS,
        "freshness_policy": FRESHNESS_OPTIONS,
    }

    seen = {_config_key(c) for c in top_configs}
    neighbors = []

    for config in top_configs[:n]:
        for axis, options in axes.items():
            for option in options:
                if option == config.get(axis):
                    continue
                neighbor = {**config, axis: option}
                neighbor["reranker_enabled"] = (neighbor.get("retrieval_strategy") == "hybrid_reranker")
                key = _config_key(neighbor)
                if key not in seen:
                    seen.add(key)
                    neighbors.append(neighbor)

    return neighbors


def _config_key(config: dict) -> str:
    """Unique key for a pipeline config."""
    return "|".join([
        config.get("retrieval_strategy", ""),
        config.get("chunking_strategy", ""),
        config.get("parsing_strategy", ""),
        config.get("freshness_policy", ""),
    ])


def run_optimizer_loop(goal: OptimizerGoal, job_id: str) -> OptimizerState:
    """
    Main optimizer agent loop.

    Iteration 1: Broad sweep with MVP matrix
    Iteration 2+: Narrow search around top performers
    """
    from app.db.session import get_sync_db
    from app.db.models import Experiment, Run, RunMetrics, OptimizerJob
    from app.services.experiment.config_matrix import build_mvp_matrix
    from app.services.experiment.runner import PipelineConfig
    from app.workers.tasks import run_experiment as run_experiment_task
    from sqlalchemy import select, func
    from dataclasses import fields as dc_fields

    state = OptimizerState(status="running")

    db = get_sync_db()
    try:
        for iteration in range(1, goal.max_iterations + 1):
            state.iteration = iteration
            _persist_state(db, job_id, state)

            # ── Build configs for this iteration ──
            if iteration == 1:
                # Broad sweep: MVP matrix
                pipeline_configs = build_mvp_matrix()
                config_dicts = [asdict(c) for c in pipeline_configs]
            else:
                # Narrow: neighbors of top 3 from last iteration
                prev_results = state.history[-1] if state.history else {}
                top_configs = prev_results.get("top_configs", [])
                if not top_configs:
                    state.status = "max_iterations"
                    state.message = "No top configs to generate neighbors from."
                    break
                config_dicts = _generate_neighbors(top_configs, n=3)
                if not config_dicts:
                    state.status = "converged"
                    state.message = f"No new configs to explore. Best {goal.target_metric}: {state.best_score:.3f}"
                    break

            # ── Create experiment + runs ──
            experiment_id = str(uuid.uuid4())
            experiment = Experiment(
                id=experiment_id,
                name=f"optimizer-{job_id[:8]}-iter{iteration}",
                description=f"Optimizer iteration {iteration} targeting {goal.target_metric} >= {goal.target_threshold}",
                status="pending",
            )
            db.add(experiment)
            db.flush()

            run_ids = []
            for config_dict in config_dicts:
                # Ensure only valid PipelineConfig fields
                valid_keys = {f.name for f in dc_fields(PipelineConfig)}
                clean = {k: v for k, v in config_dict.items() if k in valid_keys}
                run = Run(
                    id=str(uuid.uuid4()),
                    experiment_id=experiment_id,
                    config=clean,
                    status="pending",
                )
                db.add(run)
                run_ids.append(run.id)
            db.commit()

            # ── Enqueue and wait ──
            run_experiment_task.delay(experiment_id, goal.eval_set_path)

            # Poll for completion (max 20 min per iteration)
            max_wait = 1200
            poll_interval = 10
            elapsed = 0
            while elapsed < max_wait:
                time.sleep(poll_interval)
                elapsed += poll_interval
                db.expire_all()
                exp = db.get(Experiment, experiment_id)
                if exp and exp.status == "failed":
                    break
                    
                # Ensure all runs have their metrics computed
                metrics_count = db.execute(
                    select(func.count(RunMetrics.id)).where(RunMetrics.run_id.in_(run_ids))
                ).scalar() or 0
                
                # Count fails (they won't get metrics)
                failed_count = db.execute(
                    select(func.count(Run.id)).where(Run.id.in_(run_ids), Run.status == "failed")
                ).scalar() or 0
                
                if (metrics_count + failed_count) >= len(run_ids):
                    logger.info("Iteration %d complete: %d metrics found.", iteration, metrics_count)
                    break

            # ── Collect results ──
            scored_runs = []
            for run_id in run_ids:
                metrics_row = db.execute(
                    select(RunMetrics).where(RunMetrics.run_id == run_id)
                ).scalar_one_or_none()
                run_row = db.get(Run, run_id)
                if metrics_row and run_row:
                    score = getattr(metrics_row, goal.target_metric, None)
                    cost = metrics_row.avg_cost_usd or 0.0
                    if score is not None:
                        scored_runs.append({
                            "run_id": run_id,
                            "config": run_row.config,
                            "score": float(score),
                            "cost": float(cost),
                        })

            if not scored_runs:
                logger.error("Optimizer iteration %d produced NO scored runs. Check experiment logs.", iteration)
                state.status = "failed"
                state.message = f"Iteration {iteration} failure: No valid metrics were computed for any run."
                state.history.append({
                    "iteration": iteration,
                    "experiment_id": experiment_id,
                    "configs_tested": len(config_dicts),
                    "best_score": 0.0,
                    "top_configs": [],
                })
                break

            # Sort by target metric descending
            scored_runs.sort(key=lambda r: r["score"], reverse=True)
            iter_best = scored_runs[0]

            # Update global best
            if iter_best["score"] > state.best_score:
                state.best_score = iter_best["score"]
                state.best_config = iter_best["config"]

            state.total_cost_usd += sum(r["cost"] for r in scored_runs)
            state.history.append({
                "iteration": iteration,
                "experiment_id": experiment_id,
                "configs_tested": len(config_dicts),
                "best_score": iter_best["score"],
                "best_config": iter_best["config"],
                "top_configs": [r["config"] for r in scored_runs[:3]],
            })

            _persist_state(db, job_id, state)
            logger.info(
                "Optimizer iter %d: best %s=%.3f (global best=%.3f)",
                iteration, goal.target_metric, iter_best["score"], state.best_score,
            )

            # ── Check stop conditions ──
            if state.best_score >= goal.target_threshold:
                state.status = "converged"
                state.message = f"Target reached! {goal.target_metric} = {state.best_score:.3f} >= {goal.target_threshold}"
                break

            if state.total_cost_usd >= goal.max_cost_usd:
                state.status = "budget_exceeded"
                state.message = f"Budget exhausted: ${state.total_cost_usd:.2f} >= ${goal.max_cost_usd:.2f}"
                break

        # If loop ended without converging
        if state.status == "running":
            state.status = "max_iterations"
            state.message = f"Completed {goal.max_iterations} iterations. Best {goal.target_metric}: {state.best_score:.3f}"

        _persist_state(db, job_id, state)
        return state

    except Exception as exc:
        state.status = "failed"
        state.message = str(exc)
        try:
            _persist_state(db, job_id, state)
        except Exception:
            pass
        raise
    finally:
        db.close()


def _persist_state(db, job_id: str, state: OptimizerState):
    """Update the optimizer job in the database."""
    from app.db.models import OptimizerJob

    job = db.get(OptimizerJob, job_id)
    if job:
        job.state = asdict(state)
        job.status = state.status
        if state.status not in ("pending", "running"):
            job.completed_at = datetime.utcnow()
        db.commit()
