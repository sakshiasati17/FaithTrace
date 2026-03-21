"""
Root-cause failure classifier.

Classifies each failed query into one or more failure categories based on
retrieval context, metric scores, document metadata, and modality labels.
"""

from enum import Enum
from dataclasses import dataclass
from app.services.experiment.runner import QueryResult


class FailureCategory(str, Enum):
    STALE_ANSWER = "STALE_ANSWER"
    WRONG_VERSION = "WRONG_VERSION"
    TABLE_RETRIEVAL_MISS = "TABLE_RETRIEVAL_MISS"
    CHART_LAYOUT_BLINDNESS = "CHART_LAYOUT_BLINDNESS"
    CHUNKING_BOUNDARY_ERROR = "CHUNKING_BOUNDARY_ERROR"
    LOW_RECALL_RETRIEVAL = "LOW_RECALL_RETRIEVAL"
    IRRELEVANT_CONTEXT_POLLUTION = "IRRELEVANT_CONTEXT_POLLUTION"
    UNSUPPORTED_SYNTHESIS = "UNSUPPORTED_SYNTHESIS"
    NO_FAILURE = "NO_FAILURE"


@dataclass
class DiagnosisResult:
    query_id: str
    primary_failure: FailureCategory
    secondary_failures: list[FailureCategory]
    confidence: float
    evidence: dict   # supporting signals used to reach the diagnosis


def diagnose(result: QueryResult, eval_item: dict, metrics: dict) -> DiagnosisResult:
    """
    Classify the root cause of a failed or low-quality query result.

    Args:
        result: The pipeline's QueryResult for this query
        eval_item: The ground-truth evaluation item including modality, valid dates
        metrics: Per-query metric scores

    Returns:
        DiagnosisResult with primary and secondary failure categories
    """
    raise NotImplementedError


def diagnose_run(results: list[QueryResult], eval_set: list[dict], run_metrics: list[dict]) -> list[DiagnosisResult]:
    """Diagnose all queries in a run."""
    return [
        diagnose(result, eval_item, metric)
        for result, eval_item, metric in zip(results, eval_set, run_metrics)
    ]
