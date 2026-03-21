"""
Metric computation module.

Computes standard Ragas metrics, operational metrics, and FaithTrace custom metrics
for a completed pipeline run.
"""

from dataclasses import dataclass
from app.services.experiment.runner import QueryResult


@dataclass
class RunMetrics:
    """Aggregated metrics for a single pipeline run."""
    # Standard RAG metrics (via Ragas)
    answer_correctness: float
    faithfulness: float
    context_precision: float
    context_recall: float
    answer_relevance: float

    # Operational metrics
    latency_p50_ms: float
    latency_p95_ms: float
    avg_token_usage: float
    avg_cost_usd: float

    # FaithTrace custom metrics
    freshness_validity: float          # fraction of answers using version-correct knowledge
    temporal_citation_accuracy: float  # fraction of citations that are time-correct
    multimodal_grounding_rate: float   # for table/chart questions, fraction with correct non-text grounding
    root_cause_diagnostic_accuracy: float  # accuracy of failure classifier vs ground truth labels


def compute_metrics(results: list[QueryResult], eval_set: list[dict]) -> RunMetrics:
    """
    Compute all metrics for a run.

    Args:
        results: Per-query results from the pipeline runner
        eval_set: Original evaluation set with ground truth, modality, and date fields

    Returns:
        RunMetrics dataclass
    """
    raise NotImplementedError


def compute_freshness_validity(results: list[QueryResult], eval_set: list[dict]) -> float:
    """
    Score the fraction of answers that drew from the document version
    valid at the query's effective date.
    """
    raise NotImplementedError


def compute_multimodal_grounding_rate(results: list[QueryResult], eval_set: list[dict]) -> float:
    """
    For questions labeled as table, chart, or spreadsheet modality,
    score the fraction where the retrieved context included the correct non-plain-text evidence.
    """
    raise NotImplementedError
