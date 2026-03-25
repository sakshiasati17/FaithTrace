"""
Metric computation module.

Computes standard Ragas metrics, operational metrics, and FaithTrace custom metrics
for a completed pipeline run.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

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
    from app.services.evaluation.ragas_runner import run_ragas_evaluation

    if not results:
        return RunMetrics(
            answer_correctness=0.0, faithfulness=0.0, context_precision=0.0,
            context_recall=0.0, answer_relevance=0.0, latency_p50_ms=0.0,
            latency_p95_ms=0.0, avg_token_usage=0.0, avg_cost_usd=0.0,
            freshness_validity=1.0, temporal_citation_accuracy=1.0,
            multimodal_grounding_rate=1.0, root_cause_diagnostic_accuracy=0.0,
        )

    # Run Ragas evaluation
    per_query_scores = run_ragas_evaluation(results, eval_set)

    # Aggregate Ragas scores
    def _mean(scores: list[float]) -> float:
        import math
        valid = [s for s in scores if s is not None and math.isfinite(float(s))]
        return float(np.mean(valid)) if valid else 0.0

    ragas_agg = {
        "faithfulness": _mean([s.get("faithfulness", 0.0) for s in per_query_scores]),
        "context_precision": _mean([s.get("context_precision", 0.0) for s in per_query_scores]),
        "context_recall": _mean([s.get("context_recall", 0.0) for s in per_query_scores]),
        "answer_relevance": _mean([s.get("answer_relevancy", 0.0) for s in per_query_scores]),
        "answer_correctness": _mean([s.get("answer_correctness", 0.0) for s in per_query_scores]),
    }

    # Operational metrics
    latencies = [r.latency_ms for r in results]
    token_usages = [r.input_tokens + r.output_tokens for r in results]
    costs = [r.cost_usd for r in results]

    # Custom metrics
    freshness = compute_freshness_validity(results, eval_set)
    temporal_accuracy = _compute_temporal_citation_accuracy(results, eval_set)
    multimodal = compute_multimodal_grounding_rate(results, eval_set)
    diag_accuracy = _compute_diagnostic_accuracy(results, eval_set)

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
        root_cause_diagnostic_accuracy=diag_accuracy,
    )


def compute_freshness_validity(results: list[QueryResult], eval_set: list[dict]) -> float:
    """
    Score the fraction of answers that drew from the document version
    valid at the query's effective date.
    """
    eval_by_id = {item.get("id", ""): item for item in eval_set}
    temporal_items = [
        (r, eval_by_id.get(r.query_id, {}))
        for r in results
        if eval_by_id.get(r.query_id, {}).get("valid_from")
    ]

    if not temporal_items:
        return 1.0

    correct = 0
    for result, eval_item in temporal_items:
        valid_from_str = eval_item.get("valid_from")
        valid_to_str = eval_item.get("valid_to")

        try:
            query_date_epoch = int(datetime.fromisoformat(valid_from_str).timestamp())
        except (ValueError, TypeError):
            continue

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

    return correct / len(temporal_items)


def _compute_temporal_citation_accuracy(results: list[QueryResult], eval_set: list[dict]) -> float:
    """
    Per-citation temporal accuracy: fraction of individual chunks (across all
    temporal queries) that are time-correct.
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

    return correct_citations / total_citations if total_citations > 0 else 1.0


def compute_multimodal_grounding_rate(results: list[QueryResult], eval_set: list[dict]) -> float:
    """
    For questions labeled as table, chart, or spreadsheet modality,
    score the fraction where the retrieved context included the correct non-plain-text evidence.
    """
    eval_by_id = {item.get("id", ""): item for item in eval_set}
    multimodal_items = [
        (r, eval_by_id.get(r.query_id, {}))
        for r in results
        if eval_by_id.get(r.query_id, {}).get("modality") in ("table", "chart", "spreadsheet", "mixed")
    ]

    if not multimodal_items:
        return 1.0

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


def _compute_diagnostic_accuracy(results: list[QueryResult], eval_set: list[dict]) -> float:
    """
    Compare the heuristic classifier's predicted failure_type against
    the ground-truth failure_type in the eval set.

    Returns the fraction of labelled items where the prediction matches.
    Items without a ground-truth failure_type label are skipped.
    """
    from app.services.diagnostics.classifier import diagnose, FailureCategory

    eval_by_id = {item.get("id", ""): item for item in eval_set}

    # Map ground-truth labels to FailureCategory enum values
    _label_map = {
        "STALE_ANSWER": FailureCategory.STALE_ANSWER,
        "WRONG_VERSION": FailureCategory.WRONG_VERSION,
        "TABLE_RETRIEVAL_MISS": FailureCategory.TABLE_RETRIEVAL_MISS,
        "CHART_LAYOUT_BLINDNESS": FailureCategory.CHART_LAYOUT_BLINDNESS,
        "CHUNKING_BOUNDARY_ERROR": FailureCategory.CHUNKING_BOUNDARY_ERROR,
        "LOW_RECALL_RETRIEVAL": FailureCategory.LOW_RECALL_RETRIEVAL,
        "IRRELEVANT_CONTEXT_POLLUTION": FailureCategory.IRRELEVANT_CONTEXT_POLLUTION,
        "UNSUPPORTED_SYNTHESIS": FailureCategory.UNSUPPORTED_SYNTHESIS,
        "NO_FAILURE": FailureCategory.NO_FAILURE,
        None: None,
    }

    labelled = []
    for result in results:
        eval_item = eval_by_id.get(result.query_id, {})
        gt_label = eval_item.get("failure_type")
        if gt_label is None:
            continue
        expected = _label_map.get(gt_label)
        if expected is None:
            continue
        labelled.append((result, eval_item, expected))

    if not labelled:
        return 1.0  # No labelled items to evaluate — return perfect by default

    correct = 0
    for result, eval_item, expected in labelled:
        # Run the heuristic classifier on this result with zero metrics
        # (metrics are not available at this stage; use empty dict)
        diagnosis = diagnose(result, eval_item, {})
        if diagnosis.primary_failure == expected:
            correct += 1

    return correct / len(labelled)
