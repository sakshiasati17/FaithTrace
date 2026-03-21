"""
Recommendation engine.

Selects the best pipeline configuration per objective based on aggregated
experiment metrics and failure diagnostics.
"""

from dataclasses import dataclass
from app.services.experiment.runner import PipelineConfig
from app.services.evaluation.metrics import RunMetrics


@dataclass
class Recommendation:
    objective: str
    best_config: PipelineConfig
    run_id: str
    score: float
    rationale: str


OBJECTIVES = [
    "best_overall",
    "lowest_cost",
    "best_latency",
    "best_faithfulness",
    "best_for_tables",
    "best_for_drift",
    "best_for_long_pdfs",
]


def recommend(
    runs: list[dict],  # list of {config, metrics, diagnostics}
    objectives: list[str] | None = None,
) -> list[Recommendation]:
    """
    For each objective, select and explain the best pipeline configuration.

    Args:
        runs: List of completed run dicts with config, metrics, and diagnostics
        objectives: Subset of OBJECTIVES to evaluate; defaults to all

    Returns:
        List of Recommendation objects, one per objective
    """
    targets = objectives or OBJECTIVES
    return [_recommend_for_objective(runs, obj) for obj in targets]


def _recommend_for_objective(runs: list[dict], objective: str) -> Recommendation:
    raise NotImplementedError
