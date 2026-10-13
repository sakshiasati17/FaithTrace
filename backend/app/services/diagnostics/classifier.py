"""
Root-cause failure classifier.

Classifies each failed query into one or more failure categories based on
retrieval context, metric scores, document metadata, and modality labels.

Strategy:
  1. Try the trained XGBoost ML classifier (ml_classifier.py).
  2. If no model is trained yet, fall back to deterministic heuristic rules.
"""

import logging
import math
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
    # None when the query could not be diagnosed (key metrics unscored and no
    # metric-free rule fired); evidence then has {"skipped": "not scored"}.
    primary_failure: FailureCategory | None
    secondary_failures: list[FailureCategory] = field(default_factory=list)
    confidence: float = 0.7
    evidence: dict = field(default_factory=dict)


# A temporal failure is only reported when the answer was not judged correct:
# answer_correctness below this (or unscored). Wrong-period chunks next to a
# correct answer are recorded as evidence, not as a failure.
TEMPORAL_ANSWER_CORRECT = 0.5

# Chunk position relative to the question date (eval item valid_from).
IN_PERIOD = "in_period"
EXPIRED = "expired"              # effective_to < question date: an older, superseded version
NOT_YET_VALID = "not_yet_valid"  # effective_from > question date: a newer version


def _chunk_period(chunk: dict, query_epoch: int) -> str | None:
    """IN_PERIOD / EXPIRED / NOT_YET_VALID, or None for an undated chunk."""
    eff_from = chunk.get("effective_from")
    if eff_from is None:
        return None
    if eff_from > query_epoch:
        return NOT_YET_VALID
    eff_to = chunk.get("effective_to")
    if eff_to is not None and eff_to < query_epoch:
        return EXPIRED
    return IN_PERIOD


def _temporal_evidence(result: QueryResult, eval_item: dict) -> dict | None:
    """
    Where the retrieved chunks sit relative to the question date, in rank order.

    Returns None when the item has no usable valid_from (no temporal rule).
    Otherwise: counts per period, whether the top-ranked chunk is out of
    period, the direction of the highest-ranked out-of-period chunk, and
    whether out-of-period evidence is dominant (the top chunk is out of period,
    or more than half of the dated chunks are).
    """
    valid_from_str = eval_item.get("valid_from")
    if not valid_from_str:
        return None
    try:
        query_epoch = int(datetime.fromisoformat(valid_from_str).timestamp())
    except (ValueError, TypeError):
        logger.warning(
            "Query %s: unparseable valid_from %r; temporal rule not checked",
            result.query_id, valid_from_str,
        )
        return None

    periods = [_chunk_period(c, query_epoch) for c in result.retrieved_chunks]
    counts = {p: periods.count(p) for p in (IN_PERIOD, EXPIRED, NOT_YET_VALID)}
    dated = sum(counts.values())
    out_of_period = counts[EXPIRED] + counts[NOT_YET_VALID]
    top_out = bool(periods) and periods[0] in (EXPIRED, NOT_YET_VALID)
    first_out = next((p for p in periods if p in (EXPIRED, NOT_YET_VALID)), None)
    return {
        "in_period_chunks": counts[IN_PERIOD],
        "expired_chunks": counts[EXPIRED],
        "not_yet_valid_chunks": counts[NOT_YET_VALID],
        "dated_chunks": dated,
        "out_of_period_chunks": out_of_period,
        "top_chunk_out_of_period": top_out,
        "out_of_period_direction": first_out,
        "dominant": top_out or (dated > 0 and out_of_period * 2 > dated),
    }


def _has_temporal_violation(result: QueryResult, eval_item: dict) -> bool:
    """Whether any retrieved chunk is outside the question's validity period."""
    temporal = _temporal_evidence(result, eval_item)
    return bool(temporal and temporal["out_of_period_chunks"])


def _has_nontext_chunk(result: QueryResult) -> bool:
    return any(
        c.get("chunk_type", "text") in ("table", "spreadsheet_cell", "image")
        for c in result.retrieved_chunks
    )


def is_vision_chunk(chunk: dict) -> bool:
    """
    Whether a chunk came from GPT-4o vision (charts/diagrams). Retrieved chunks
    are flat (metadata fields at the top level); a nested metadata.source is
    also accepted.
    """
    nested = chunk.get("metadata")
    return (
        chunk.get("chunk_type") == "image"
        or chunk.get("source") == "gpt4o_vision"
        or (isinstance(nested, dict) and nested.get("source") == "gpt4o_vision")
    )


def _has_vision_chunk(result: QueryResult) -> bool:
    """Check if any retrieved chunk came from GPT-4o vision (charts/diagrams)."""
    return any(is_vision_chunk(c) for c in result.retrieved_chunks)


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


# Ragas metrics the metric-based rules read. A query missing any of these
# cannot be labelled NO_FAILURE.
KEY_METRICS = ("faithfulness", "context_recall", "context_precision", "answer_correctness")


def _metric_or_none(metrics: dict, key: str) -> float | None:
    """A finite metric value, or None when it is missing/unscored (never a default)."""
    value = metrics.get(key)
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def missing_key_metrics(metrics: dict) -> list[str]:
    """The KEY_METRICS that have no usable score in ``metrics``."""
    return [k for k in KEY_METRICS if _metric_or_none(metrics, k) is None]


def _heuristic_diagnose(result: QueryResult, eval_item: dict, metrics: dict) -> DiagnosisResult:
    """
    Deterministic heuristic classifier (fallback when ML model not trained).

    Rules are checked in priority order. The temporal/version/modality rules
    do not need metrics. A metric rule whose metric is unscored cannot be
    decided, so checking stops there: lower-priority rules (and NO_FAILURE)
    would be guesses. If no rule fired, ``primary_failure`` is None and the
    evidence says ``{"skipped": "not scored"}``.
    """
    faithfulness = _metric_or_none(metrics, "faithfulness")
    context_recall = _metric_or_none(metrics, "context_recall")
    context_precision = _metric_or_none(metrics, "context_precision")
    answer_correctness = _metric_or_none(metrics, "answer_correctness")
    modality = eval_item.get("modality", "text")

    primary: FailureCategory | None = None
    secondary = []
    evidence = {
        "faithfulness": faithfulness,
        "context_recall": context_recall,
        "context_precision": context_precision,
        "answer_correctness": answer_correctness,
        "modality": modality,
    }

    temporal = _temporal_evidence(result, eval_item)
    if temporal is not None:
        evidence["temporal"] = temporal
    # Temporal failure: wrong-period evidence dominates the retrieved context
    # and the answer was not judged correct (unscored counts as not correct,
    # since this rule needs no metrics). The direction names it: an expired
    # version is a stale answer; a version not yet in effect is a wrong version.
    temporal_failure = None
    if temporal and temporal["dominant"] and (
        answer_correctness is None or answer_correctness < TEMPORAL_ANSWER_CORRECT
    ):
        temporal_failure = (
            FailureCategory.STALE_ANSWER
            if temporal["out_of_period_direction"] == EXPIRED
            else FailureCategory.WRONG_VERSION
        )
    version_mismatch = _has_version_mismatch(result, eval_item)
    has_nontext = _has_nontext_chunk(result)
    has_vision = _has_vision_chunk(result)
    needs_nontext = modality in ("table", "chart", "spreadsheet", "mixed")
    needs_vision = modality == "chart"

    # Priority order: temporal > version > modality > recall > precision/synthesis

    if temporal_failure is not None:
        primary = temporal_failure
        evidence["temporal_violation"] = True
        evidence["temporal_rule"] = (
            "top_chunk_out_of_period" if temporal["top_chunk_out_of_period"]
            else "majority_out_of_period"
        )
        if answer_correctness is None:
            evidence["answer_not_scored"] = True

    elif version_mismatch:
        primary = FailureCategory.WRONG_VERSION
        evidence["version_mismatch"] = True

    elif needs_vision and not has_vision:
        primary = FailureCategory.CHART_LAYOUT_BLINDNESS
        evidence["no_vision_chunk_retrieved"] = True

    elif needs_nontext and not has_nontext:
        primary = FailureCategory.TABLE_RETRIEVAL_MISS
        evidence["no_nontext_chunk_retrieved"] = True

    elif context_recall is None:
        pass  # undecidable

    elif context_recall < 0.3:
        primary = FailureCategory.LOW_RECALL_RETRIEVAL
        if faithfulness is not None and faithfulness < 0.4:
            secondary.append(FailureCategory.UNSUPPORTED_SYNTHESIS)

    elif faithfulness is None:
        pass  # undecidable

    elif faithfulness < 0.4:
        primary = FailureCategory.UNSUPPORTED_SYNTHESIS
        if context_precision is not None and context_precision < 0.3:
            secondary.append(FailureCategory.IRRELEVANT_CONTEXT_POLLUTION)

    elif context_precision is None:
        pass  # undecidable

    elif context_precision < 0.3 and faithfulness > 0.5:
        primary = FailureCategory.IRRELEVANT_CONTEXT_POLLUTION

    elif answer_correctness is None:
        pass  # undecidable

    elif answer_correctness < 0.4 and context_recall > 0.5:
        # Content was retrieved but answer is wrong → likely boundary/synthesis issue
        primary = FailureCategory.CHUNKING_BOUNDARY_ERROR

    else:
        primary = FailureCategory.NO_FAILURE

    evidence["chunks_retrieved"] = len(result.retrieved_chunks)
    evidence["has_nontext_chunk"] = has_nontext
    evidence["has_vision_chunk"] = has_vision
    missing = missing_key_metrics(metrics)
    if missing:
        evidence["missing_metrics"] = missing
    if primary is None:
        evidence["skipped"] = "not scored"

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
    # The ML features treat a missing metric as 0.0, so only use it when the
    # key metrics were actually scored.
    missing = missing_key_metrics(metrics)
    if missing:
        logger.info(
            "Query %s: metrics not scored (%s); using metric-free rules only",
            result.query_id, ", ".join(missing),
        )
    else:
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


def eval_item_id(item: dict) -> str | None:
    """Return an eval item's identifier. Eval sets use "id"; "query_id" is accepted as a fallback."""
    return item.get("id") or item.get("query_id")


def index_eval_set(eval_set: list[dict]) -> dict[str, dict]:
    """Map eval item id -> eval item. Items without an id are skipped."""
    return {
        item_id: item
        for item in eval_set
        if (item_id := eval_item_id(item))
    }


def diagnose_run(
    results: list[QueryResult],
    eval_set: list[dict],
    run_metrics: list[dict],
) -> list[DiagnosisResult]:
    """
    Diagnose all queries in a run.

    Each result is matched to its eval item by ``result.query_id`` == eval item id
    (a missing item yields an empty dict). ``run_metrics`` is aligned with
    ``results`` by position (it is produced from ``results`` in order); missing
    entries are treated as empty metrics.
    """
    eval_by_id = index_eval_set(eval_set)
    diagnoses = []
    for i, result in enumerate(results):
        eval_item = eval_by_id.get(result.query_id, {})
        metric = run_metrics[i] if i < len(run_metrics) else {}
        diagnoses.append(diagnose(result, eval_item, metric or {}))
    return diagnoses


def compute_diagnostic_accuracy(
    predictions: dict[str, str | None],
    eval_set: list[dict],
) -> float | None:
    """
    Root-cause diagnostic accuracy: the fraction of labelled eval items whose
    predicted failure category matches the ground-truth ``failure_type``.

    Args:
        predictions: query_id -> predicted failure category (e.g. the stored
            ``QueryResult.failure_category``)
        eval_set: eval items; only those with a ``failure_type`` label count

    Returns:
        Accuracy in [0, 1], or None when no predicted query has a label.
    """
    eval_by_id = index_eval_set(eval_set)
    labelled = 0
    correct = 0
    for query_id, predicted in predictions.items():
        gt_label = eval_by_id.get(query_id, {}).get("failure_type")
        if not gt_label:
            continue
        labelled += 1
        if predicted == gt_label:
            correct += 1
    if labelled == 0:
        return None
    return correct / labelled


# STALE_ANSWER (expired version) and WRONG_VERSION (version not yet in effect)
# are two directions of one failure: evidence from the wrong document version.
TEMPORAL_FAMILY = frozenset({FailureCategory.STALE_ANSWER.value, FailureCategory.WRONG_VERSION.value})


def _family(label: str | None) -> str | None:
    return "TEMPORAL" if label in TEMPORAL_FAMILY else label


def compute_temporal_family_diagnostic_accuracy(
    predictions: dict[str, str | None],
    eval_set: list[dict],
) -> float | None:
    """
    Like compute_diagnostic_accuracy, over the same labelled items, but
    STALE_ANSWER and WRONG_VERSION count as one category. The strict
    per-label accuracy is unchanged; this is reported alongside it.
    """
    eval_by_id = index_eval_set(eval_set)
    labelled = 0
    correct = 0
    for query_id, predicted in predictions.items():
        gt_label = eval_by_id.get(query_id, {}).get("failure_type")
        if not gt_label:
            continue
        labelled += 1
        if _family(predicted) == _family(gt_label):
            correct += 1
    if labelled == 0:
        return None
    return correct / labelled
