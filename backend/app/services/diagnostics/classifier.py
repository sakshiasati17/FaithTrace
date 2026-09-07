"""
Root-cause failure classifier.

Classifies each failed query into one or more failure categories based on
retrieval context, metric scores, document metadata, and modality labels.

Strategy:
  1. Try the trained XGBoost ML classifier (ml_classifier.py).
  2. If no model is trained yet, fall back to deterministic heuristic rules.
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from app.services.experiment.runner import QueryResult

logger = logging.getLogger(__name__)


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
    secondary_failures: list[FailureCategory] = field(default_factory=list)
    confidence: float = 0.7
    evidence: dict = field(default_factory=dict)


def _has_temporal_violation(result: QueryResult, eval_item: dict) -> bool:
    """Check if retrieved chunks contain temporally invalid content."""
    valid_from_str = eval_item.get("valid_from")
    if not valid_from_str:
        return False
    try:
        query_epoch = int(datetime.fromisoformat(valid_from_str).timestamp())
    except (ValueError, TypeError):
        return False

    for chunk in result.retrieved_chunks:
        eff_from = chunk.get("effective_from")
        if eff_from is None:
            continue
        eff_to = chunk.get("effective_to")
        from_ok = eff_from <= query_epoch
        to_ok = (eff_to is None) or (eff_to >= query_epoch)
        if not (from_ok and to_ok):
            return True
    return False


def _has_nontext_chunk(result: QueryResult) -> bool:
    return any(
        c.get("chunk_type", "text") in ("table", "spreadsheet_cell", "image")
        for c in result.retrieved_chunks
    )


def _has_vision_chunk(result: QueryResult) -> bool:
    """Check if any retrieved chunk came from GPT-4o vision (charts/diagrams)."""
    return any(
        c.get("chunk_type") == "image" or c.get("metadata", {}).get("source") == "gpt4o_vision"
        for c in result.retrieved_chunks
    )


def _has_version_mismatch(result: QueryResult, eval_item: dict) -> bool:
    """Check if retrieved chunks are from a different document version than expected."""
    expected_version = eval_item.get("expected_version")
    if not expected_version:
        return False
    for chunk in result.retrieved_chunks:
        chunk_version = chunk.get("doc_version")
        if chunk_version and chunk_version != expected_version:
            return True
    return False


def _heuristic_diagnose(result: QueryResult, eval_item: dict, metrics: dict) -> DiagnosisResult:
    """Deterministic heuristic classifier (fallback when ML model not trained)."""
    faithfulness = float(metrics.get("faithfulness") or 1.0)
    context_recall = float(metrics.get("context_recall") or 1.0)
    context_precision = float(metrics.get("context_precision") or 1.0)
    answer_correctness = float(metrics.get("answer_correctness") or 1.0)
    modality = eval_item.get("modality", "text")

    primary = FailureCategory.NO_FAILURE
    secondary = []
    evidence = {
        "faithfulness": faithfulness,
        "context_recall": context_recall,
        "context_precision": context_precision,
        "answer_correctness": answer_correctness,
        "modality": modality,
    }

    temporal_violation = _has_temporal_violation(result, eval_item)
    version_mismatch = _has_version_mismatch(result, eval_item)
    has_nontext = _has_nontext_chunk(result)
    has_vision = _has_vision_chunk(result)
    needs_nontext = modality in ("table", "chart", "spreadsheet", "mixed")
    needs_vision = modality == "chart"

    # Priority order: temporal > version > modality > recall > precision/synthesis

    if temporal_violation and eval_item.get("valid_from"):
        primary = FailureCategory.STALE_ANSWER
        evidence["temporal_violation"] = True

    elif version_mismatch:
        primary = FailureCategory.WRONG_VERSION
        evidence["version_mismatch"] = True

    elif needs_vision and not has_vision:
        primary = FailureCategory.CHART_LAYOUT_BLINDNESS
        evidence["no_vision_chunk_retrieved"] = True

    elif needs_nontext and not has_nontext:
        primary = FailureCategory.TABLE_RETRIEVAL_MISS
        evidence["no_nontext_chunk_retrieved"] = True

    elif context_recall < 0.3:
        primary = FailureCategory.LOW_RECALL_RETRIEVAL
        if faithfulness < 0.4:
            secondary.append(FailureCategory.UNSUPPORTED_SYNTHESIS)

    elif faithfulness < 0.4:
        primary = FailureCategory.UNSUPPORTED_SYNTHESIS
        if context_precision < 0.3:
            secondary.append(FailureCategory.IRRELEVANT_CONTEXT_POLLUTION)

    elif context_precision < 0.3 and faithfulness > 0.5:
        primary = FailureCategory.IRRELEVANT_CONTEXT_POLLUTION

    elif answer_correctness < 0.4 and context_recall > 0.5:
        # Content was retrieved but answer is wrong → likely boundary/synthesis issue
        primary = FailureCategory.CHUNKING_BOUNDARY_ERROR

    evidence["chunks_retrieved"] = len(result.retrieved_chunks)
    evidence["has_nontext_chunk"] = has_nontext
    evidence["has_vision_chunk"] = has_vision

    return DiagnosisResult(
        query_id=result.query_id,
        primary_failure=primary,
        secondary_failures=secondary,
        confidence=0.7,
        evidence=evidence,
    )


def diagnose(result: QueryResult, eval_item: dict, metrics: dict) -> DiagnosisResult:
    """
    Classify the root cause of a failed or low-quality query result.

    Tries the trained XGBoost ML classifier first; falls back to heuristics
    if no model is available.

    Args:
        result: The pipeline's QueryResult for this query
        eval_item: The ground-truth evaluation item (modality, valid dates)
        metrics: Per-query metric scores from Ragas

    Returns:
        DiagnosisResult with primary/secondary failure categories and confidence
    """
    # --- Try ML classifier first ---
    try:
        from app.services.diagnostics.ml_classifier import predict as ml_predict
        ml_result = ml_predict(metrics, result.retrieved_chunks, eval_item)
        if ml_result is not None:
            primary_failure, confidence = ml_result
            # Build lightweight evidence dict for ML path
            evidence = {
                "faithfulness": metrics.get("faithfulness"),
                "context_recall": metrics.get("context_recall"),
                "context_precision": metrics.get("context_precision"),
                "answer_correctness": metrics.get("answer_correctness"),
                "modality": eval_item.get("modality", "text"),
                "chunks_retrieved": len(result.retrieved_chunks),
                "classifier": "xgboost",
            }
            logger.debug(
                "ML classifier: query=%s → %s (conf=%.2f)",
                result.query_id, primary_failure, confidence,
            )
            return DiagnosisResult(
                query_id=result.query_id,
                primary_failure=primary_failure,
                secondary_failures=[],
                confidence=confidence,
                evidence=evidence,
            )
    except Exception as exc:
        logger.warning("ML classifier error, falling back to heuristics: %s", exc)

    # --- Fall back to heuristic classifier ---
    heuristic = _heuristic_diagnose(result, eval_item, metrics)
    heuristic.evidence["classifier"] = "heuristic"
    return heuristic


def diagnose_run(
    results: list[QueryResult],
    eval_set: list[dict],
    run_metrics: list[dict],
) -> list[DiagnosisResult]:
    """Diagnose all queries in a run."""
    return [
        diagnose(result, eval_item, metric)
        for result, eval_item, metric in zip(results, eval_set, run_metrics)
    ]
