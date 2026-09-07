"""
Regression test suite.

Validates that core pipeline components produce consistent outputs against
a frozen golden eval set. Catches metric regressions, schema changes, and
broken pipeline stages without requiring live services.
"""

import json
from pathlib import Path
from dataclasses import asdict

import pytest


class TestGoldenEvalSet:
    """Ensure the frozen eval set hasn't been tampered with."""

    def test_golden_set_exists(self):
        path = Path(__file__).parent.parent.parent / "eval_sets" / "golden_regression_set.json"
        assert path.exists(), "Golden regression eval set is missing"

    def test_golden_set_has_expected_queries(self, sample_eval_set):
        assert len(sample_eval_set) == 5, (
            f"Golden set should have 5 queries, found {len(sample_eval_set)}"
        )

    def test_golden_set_required_fields(self, sample_eval_set):
        required = {"id", "question", "ground_truth", "modality"}
        for item in sample_eval_set:
            missing = required - set(item.keys())
            assert not missing, f"Query {item.get('id')} missing fields: {missing}"

    def test_golden_set_has_temporal_queries(self, sample_eval_set):
        temporal = [q for q in sample_eval_set if q.get("valid_from")]
        assert len(temporal) >= 2, "Golden set must include temporal queries"

    def test_golden_set_has_table_queries(self, sample_eval_set):
        table = [q for q in sample_eval_set if q.get("modality") in ("table", "spreadsheet")]
        assert len(table) >= 1, "Golden set must include table/spreadsheet queries"

    def test_golden_set_has_failure_labels(self, sample_eval_set):
        labeled = [q for q in sample_eval_set if q.get("failure_type")]
        assert len(labeled) >= 2, "Golden set must include ground-truth failure labels"


class TestConfigMatrix:
    """Ensure the config matrix produces the expected number of combinations."""

    def test_full_matrix_is_256(self):
        from app.services.experiment.config_matrix import build_matrix
        matrix = build_matrix()
        assert len(matrix) == 256, f"Full matrix should be 256, got {len(matrix)}"

    def test_mvp_matrix_is_24(self):
        from app.services.experiment.config_matrix import build_mvp_matrix
        matrix = build_mvp_matrix()
        assert len(matrix) == 24, f"MVP matrix should be 24, got {len(matrix)}"

    def test_all_configs_have_required_fields(self):
        from app.services.experiment.config_matrix import build_matrix
        matrix = build_matrix()
        for config in matrix:
            assert config.retrieval_strategy
            assert config.chunking_strategy
            assert config.parsing_strategy
            assert config.freshness_policy
            assert config.top_k == 5

    def test_reranker_flag_matches_strategy(self):
        from app.services.experiment.config_matrix import build_matrix
        matrix = build_matrix()
        for config in matrix:
            if config.retrieval_strategy == "hybrid_reranker":
                assert config.reranker_enabled is True
            else:
                assert config.reranker_enabled is False


class TestBaselineComparison:
    """Validate the baseline comparison module."""

    def test_baseline_config_is_naive(self):
        from app.services.evaluation.baseline import BASELINE_CONFIG
        assert BASELINE_CONFIG.retrieval_strategy == "vector_only"
        assert BASELINE_CONFIG.chunking_strategy == "fixed_size"
        assert BASELINE_CONFIG.parsing_strategy == "text_only"
        assert BASELINE_CONFIG.freshness_policy == "none"
        assert BASELINE_CONFIG.reranker_enabled is False

    def test_is_baseline_config(self):
        from app.services.evaluation.baseline import is_baseline_config
        assert is_baseline_config({
            "retrieval_strategy": "vector_only",
            "chunking_strategy": "fixed_size",
            "parsing_strategy": "text_only",
            "freshness_policy": "none",
        })
        assert not is_baseline_config({
            "retrieval_strategy": "hybrid",
            "chunking_strategy": "fixed_size",
            "parsing_strategy": "text_only",
            "freshness_policy": "none",
        })

    def test_compute_deltas_improvement(self):
        from app.services.evaluation.baseline import compute_deltas
        baseline = {"faithfulness": 0.5, "context_recall": 0.4, "latency_p50_ms": 1000.0,
                     "answer_correctness": 0.5, "context_precision": 0.5,
                     "answer_relevance": 0.5, "avg_cost_usd": 0.01,
                     "freshness_validity": 0.5, "multimodal_grounding_rate": 0.5}
        comparison = {"faithfulness": 0.888, "context_recall": 0.9, "latency_p50_ms": 734.0,
                       "answer_correctness": 0.7, "context_precision": 0.6,
                       "answer_relevance": 0.7, "avg_cost_usd": 0.001,
                       "freshness_validity": 0.8, "multimodal_grounding_rate": 1.0}
        deltas = compute_deltas(baseline, comparison)

        faith_delta = next(d for d in deltas if d.metric == "faithfulness")
        assert faith_delta.improved is True
        assert faith_delta.absolute_delta == pytest.approx(0.388, abs=0.001)

        latency_delta = next(d for d in deltas if d.metric == "latency_p50_ms")
        assert latency_delta.improved is True  # lower is better

        cost_delta = next(d for d in deltas if d.metric == "avg_cost_usd")
        assert cost_delta.improved is True  # lower is better

    def test_compare_run_to_baseline_no_baseline(self):
        from app.services.evaluation.baseline import compare_run_to_baseline
        runs = [{"run_id": "r1", "config": {"retrieval_strategy": "hybrid"},
                 "metrics": {"faithfulness": 0.8}}]
        result = compare_run_to_baseline(runs)
        assert "error" in result

    def test_compare_run_to_baseline_success(self):
        from app.services.evaluation.baseline import compare_run_to_baseline
        runs = [
            {"run_id": "baseline", "config": {
                "retrieval_strategy": "vector_only", "chunking_strategy": "fixed_size",
                "parsing_strategy": "text_only", "freshness_policy": "none"},
             "metrics": {"faithfulness": 0.5, "answer_correctness": 0.4,
                         "context_recall": 0.4, "context_precision": 0.4,
                         "answer_relevance": 0.4, "latency_p50_ms": 1000,
                         "avg_cost_usd": 0.01, "freshness_validity": 0.5,
                         "multimodal_grounding_rate": 0.5}},
            {"run_id": "hybrid", "config": {
                "retrieval_strategy": "hybrid", "chunking_strategy": "recursive",
                "parsing_strategy": "text_table", "freshness_policy": "version_aware"},
             "metrics": {"faithfulness": 0.888, "answer_correctness": 0.7,
                         "context_recall": 0.9, "context_precision": 0.6,
                         "answer_relevance": 0.7, "latency_p50_ms": 734,
                         "avg_cost_usd": 0.001, "freshness_validity": 0.8,
                         "multimodal_grounding_rate": 1.0}},
        ]
        result = compare_run_to_baseline(runs)
        assert "error" not in result
        assert result["baseline"]["run_id"] == "baseline"
        assert result["comparison"]["run_id"] == "hybrid"
        assert len(result["deltas"]) == 9
        assert result["summary"]["improvements"] > 0


class TestDiagnosticsRegression:
    """Ensure the heuristic classifier produces stable outputs."""

    def test_failure_categories_count(self):
        from app.services.diagnostics.classifier import FailureCategory
        categories = [fc for fc in FailureCategory]
        assert len(categories) == 9

    def test_heuristic_low_faithfulness(self):
        from app.services.diagnostics.classifier import _heuristic_diagnose, FailureCategory
        from app.services.experiment.runner import QueryResult
        result = QueryResult(
            query_id="test", question="test?", generated_answer="wrong",
            retrieved_chunks=[], latency_ms=100, input_tokens=10,
            output_tokens=10, cost_usd=0.001,
        )
        metrics = {"faithfulness": 0.1, "context_recall": 0.8}
        diagnosis = _heuristic_diagnose(result, {}, metrics)
        assert diagnosis.primary_failure == FailureCategory.UNSUPPORTED_SYNTHESIS

    def test_heuristic_low_recall(self):
        from app.services.diagnostics.classifier import _heuristic_diagnose, FailureCategory
        from app.services.experiment.runner import QueryResult
        result = QueryResult(
            query_id="test", question="test?", generated_answer="answer",
            retrieved_chunks=[], latency_ms=100, input_tokens=10,
            output_tokens=10, cost_usd=0.001,
        )
        metrics = {"faithfulness": 0.8, "context_recall": 0.1}
        diagnosis = _heuristic_diagnose(result, {}, metrics)
        assert diagnosis.primary_failure == FailureCategory.LOW_RECALL_RETRIEVAL


class TestRecommendationEngine:
    """Ensure recommendation engine produces stable results."""

    def test_recommend_empty_runs(self):
        from app.services.recommendation.engine import recommend
        assert recommend([]) == []

    def test_recommend_all_objectives(self):
        from app.services.recommendation.engine import recommend, OBJECTIVES
        runs = [
            {"run_id": "r1", "config": {"parsing_strategy": "text_table",
             "freshness_policy": "version_aware"},
             "metrics": {"faithfulness": 0.888, "answer_correctness": 0.7,
                         "context_recall": 0.9, "latency_p50_ms": 734,
                         "avg_cost_usd": 0.001, "freshness_validity": 0.8,
                         "multimodal_grounding_rate": 1.0}},
        ]
        recs = recommend(runs)
        assert len(recs) == len(OBJECTIVES)

    def test_best_overall_weights(self):
        from app.services.recommendation.engine import recommend
        runs = [
            {"run_id": "cheap", "config": {},
             "metrics": {"faithfulness": 0.5, "answer_correctness": 0.5,
                         "context_recall": 0.5, "avg_cost_usd": 0.0001}},
            {"run_id": "faithful", "config": {},
             "metrics": {"faithfulness": 0.95, "answer_correctness": 0.8,
                         "context_recall": 0.9, "avg_cost_usd": 0.01}},
        ]
        recs = recommend(runs, objectives=["best_overall"])
        assert recs[0].run_id == "faithful"
