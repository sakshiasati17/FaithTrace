"""
Metric computation module.

Computes standard Ragas metrics, operational metrics, and FaithTrace custom metrics
for a completed pipeline run.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from app.services.experiment.runner import QueryResult

logger = logging.getLogger(__name__)


@dataclass
class RunMetrics:
    """
    Aggregated metrics for a single pipeline run.

    A metric that could not be computed (Ragas failed, or no eval item the
    metric applies to) is None, never 0.0 or 1.0.
    """
    # Standard RAG metrics (via Ragas)
    answer_correctness: float | None
    faithfulness: float | None
    context_precision: float | None
    context_recall: float | None
    answer_relevance: float | None

    # Operational metrics
    latency_p50_ms: float
    latency_p95_ms: float
    avg_token_usage: float
    avg_cost_usd: float

    # FaithTrace custom metrics (None when no eval item applies)
    freshness_validity: float | None          # fraction of answers using version-correct knowledge
    temporal_citation_accuracy: float | None  # fraction of citations that are time-correct
    multimodal_grounding_rate: float | None   # for table/chart questions, fraction with correct non-text grounding
    # Accuracy of stored failure diagnoses vs ground-truth failure_type labels.
    # Computed by the diagnose_run task after diagnoses are written; None here
    # (and None when no eval items are labelled).
    root_cause_diagnostic_accuracy: float | None = None
    # Number of queries with a valid score, per Ragas metric. Not stored in
    # the DB; returned by evaluate_run and logged.
    scored_counts: dict[str, int] = field(default_factory=dict)
    # Per-query Ragas score dicts, in the order of the results passed in.
    # Not stored on RunMetrics; evaluate_run saves them on each query result.
    per_query_scores: list[dict] = field(default_factory=list)


# RunMetrics field -> key in the per-query Ragas score dicts.
RAGAS_FIELDS = {
    "faithfulness": "faithfulness",
    "context_precision": "context_precision",
    "context_recall": "context_recall",
    "answer_relevance": "answer_relevancy",
    "answer_correctness": "answer_correctness",
}


def _valid(value) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(scores: list) -> float | None:
    """Mean over valid (non-None, finite) scores; None when there are none."""
    valid = [float(s) for s in scores if _valid(s)]
    return float(np.mean(valid)) if valid else None


def compute_metrics(results: list[QueryResult], eval_set: list[dict]) -> RunMetrics:
    """
    Compute all metrics for a run.

    Args:
        results: Per-query results from the pipeline runner
        eval_set: Original evaluation set with ground truth, modality, and date fields

    Returns:
        RunMetrics dataclass
    """
    from app.services.evaluation.ragas_runner import run_ragas_evaluation

    if not results:
        # Nothing was answered, so nothing can be scored.
        return RunMetrics(
            answer_correctness=None, faithfulness=None, context_precision=None,
            context_recall=None, answer_relevance=None, latency_p50_ms=None,
            latency_p95_ms=None, avg_token_usage=None, avg_cost_usd=None,
            freshness_validity=None, temporal_citation_accuracy=None,
            multimodal_grounding_rate=None, root_cause_diagnostic_accuracy=None,
            scored_counts={name: 0 for name in RAGAS_FIELDS},
        )

    # Run Ragas evaluation
    per_query_scores = run_ragas_evaluation(results, eval_set)

    # Aggregate Ragas scores over the queries that were actually scored.
    ragas_agg = {}
    scored_counts = {}
    for name, key in RAGAS_FIELDS.items():
        values = [s.get(key) for s in per_query_scores]
        ragas_agg[name] = _mean(values)
        scored_counts[name] = sum(1 for v in values if _valid(v))
    unscored = {n: len(results) - c for n, c in scored_counts.items() if c < len(results)}
    if unscored:
        logger.warning(
            "Ragas metrics missing for some of %d queries (unscored per metric: %s); "
            "averages use scored queries only",
            len(results), unscored,
        )

    # Operational metrics
    latencies = [r.latency_ms for r in results]
    token_usages = [r.input_tokens + r.output_tokens for r in results]
    costs = [r.cost_usd for r in results]

    # Custom metrics
    freshness = compute_freshness_validity(results, eval_set)
    temporal_accuracy = _compute_temporal_citation_accuracy(results, eval_set)
    multimodal = compute_multimodal_grounding_rate(results, eval_set)

    return RunMetrics(
        answer_correctness=ragas_agg["answer_correctness"],
        faithfulness=ragas_agg["faithfulness"],
        context_precision=ragas_agg["context_precision"],
        context_recall=ragas_agg["context_recall"],
        answer_relevance=ragas_agg["answer_relevance"],
        latency_p50_ms=float(np.percentile(latencies, 50)) if latencies else 0.0,
        latency_p95_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
        avg_token_usage=float(np.mean(token_usages)) if token_usages else 0.0,
        avg_cost_usd=float(np.mean(costs)) if costs else 0.0,
        freshness_validity=freshness,
        temporal_citation_accuracy=temporal_accuracy,
        multimodal_grounding_rate=multimodal,
        root_cause_diagnostic_accuracy=None,
        scored_counts=scored_counts,
        per_query_scores=per_query_scores,
    )


def compute_freshness_validity(results: list[QueryResult], eval_set: list[dict]) -> float | None:
    """
    Score the fraction of answers that drew from the document version
    valid at the query's effective date.

    None when no result has a temporal eval item (``valid_from``): the
    metric does not apply, which is not the same as a perfect score.
    """
    eval_by_id = {item.get("id", ""): item for item in eval_set}
    temporal_items = [
        (r, eval_by_id.get(r.query_id, {}))
        for r in results
        if eval_by_id.get(r.query_id, {}).get("valid_from")
    ]

    if not temporal_items:
        return None

    correct = 0
    scored = 0
    for result, eval_item in temporal_items:
        valid_from_str = eval_item.get("valid_from")

        try:
            query_date_epoch = int(datetime.fromisoformat(valid_from_str).timestamp())
        except (ValueError, TypeError):
            # An unparseable date cannot be scored: leave it out rather
            # than count it as a stale answer.
            logger.warning(
                "freshness_validity: query %s has unparseable valid_from %r; not scored",
                result.query_id, valid_from_str,
            )
            continue
        scored += 1

        # Check if at least one retrieved chunk is temporally valid
        for chunk in result.retrieved_chunks:
            eff_from = chunk.get("effective_from")
            eff_to = chunk.get("effective_to")
            if eff_from is None:
                continue
            from_ok = eff_from <= query_date_epoch
            to_ok = (eff_to is None) or (eff_to >= query_date_epoch)
            if from_ok and to_ok:
                correct += 1
                break

    return correct / scored if scored else None


def _compute_temporal_citation_accuracy(results: list[QueryResult], eval_set: list[dict]) -> float | None:
    """
    Per-citation temporal accuracy: fraction of individual chunks (across all
    temporal queries) that are time-correct.

    None when there is no dated chunk retrieved for a temporal query.
    """
    eval_by_id = {item.get("id", ""): item for item in eval_set}
    total_citations = 0
    correct_citations = 0

    for result in results:
        eval_item = eval_by_id.get(result.query_id, {})
        valid_from_str = eval_item.get("valid_from")
        if not valid_from_str:
            continue

        try:
            query_date_epoch = int(datetime.fromisoformat(valid_from_str).timestamp())
        except (ValueError, TypeError):
            logger.warning(
                "temporal_citation_accuracy: query %s has unparseable valid_from %r; not scored",
                result.query_id, valid_from_str,
            )
            continue

        for chunk in result.retrieved_chunks:
            eff_from = chunk.get("effective_from")
            if eff_from is None:
                continue
            total_citations += 1
            eff_to = chunk.get("effective_to")
            from_ok = eff_from <= query_date_epoch
            to_ok = (eff_to is None) or (eff_to >= query_date_epoch)
            if from_ok and to_ok:
                correct_citations += 1

    return correct_citations / total_citations if total_citations > 0 else None


def compute_multimodal_grounding_rate(results: list[QueryResult], eval_set: list[dict]) -> float | None:
    """
    For questions labeled as table, chart, or spreadsheet modality,
    score the fraction where the retrieved context included the correct non-plain-text evidence.

    None when no result has a multimodal eval item.
    """
    eval_by_id = {item.get("id", ""): item for item in eval_set}
    multimodal_items = [
        (r, eval_by_id.get(r.query_id, {}))
        for r in results
        if eval_by_id.get(r.query_id, {}).get("modality") in ("table", "chart", "spreadsheet", "mixed")
    ]

    if not multimodal_items:
        return None

    correct = 0
    for result, eval_item in multimodal_items:
        source_docs = set(eval_item.get("source_docs", []))
        for chunk in result.retrieved_chunks:
            chunk_type = chunk.get("chunk_type", "text")
            chunk_filename = chunk.get("filename", "")
            is_nontext = chunk_type in ("table", "spreadsheet_cell", "image")
            in_source = not source_docs or chunk_filename in source_docs
            if is_nontext and in_source:
                correct += 1
                break

    return correct / len(multimodal_items)
