"""
Silent failures in scoring, diagnosis and recommendations.

Unknown is not zero and not perfect:
  - a Ragas metric that could not be computed (NaN/None/missing, or the whole
    evaluation failed) is None, and averages use valid values only
  - custom metrics with no applicable eval item are None, not 1.0
  - the classifier does not label an unscored query NO_FAILURE
  - the recommendation engine skips runs missing the objective's metric and
    reports objectives it cannot score instead of dropping them
"""

import logging
import math
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.services.experiment.runner import QueryResult


METRIC_KEYS = ["faithfulness", "context_precision", "context_recall",
               "answer_relevancy", "answer_correctness"]

FULL_SCORES = {
    "faithfulness": 0.9, "context_recall": 0.9, "context_precision": 0.9,
    "answer_correctness": 0.9, "answer_relevancy": 0.9,
}


def _result(query_id="q1", chunks=None, latency=100.0, cost=0.001) -> QueryResult:
    return QueryResult(
        query_id=query_id,
        question=f"question for {query_id}",
        generated_answer="answer",
        retrieved_chunks=chunks if chunks is not None else [{"content": "plain text", "chunk_type": "text"}],
        latency_ms=latency,
        input_tokens=10,
        output_tokens=5,
        cost_usd=cost,
    )


@pytest.fixture(autouse=True)
def _heuristic_only():
    """Force the heuristic classifier so results don't depend on a trained model on disk."""
    with patch("app.services.diagnostics.ml_classifier.predict", return_value=None):
        yield


# ─── Ragas runner ─────────────────────────────────────────────────────────────

class _FakeRagasResult:
    def __init__(self, rows):
        self._rows = rows

    def to_pandas(self):
        import pandas as pd
        return pd.DataFrame(self._rows)


def _run_ragas(evaluate_mock, n=2):
    from app.services.evaluation.ragas_runner import run_ragas_evaluation

    results = [_result(f"q{i}") for i in range(n)]
    eval_set = [{"id": f"q{i}", "ground_truth": "gt"} for i in range(n)]
    with patch("ragas.evaluate", evaluate_mock), \
         patch("langchain_openai.ChatOpenAI"), \
         patch("langchain_openai.OpenAIEmbeddings"):
        return run_ragas_evaluation(results, eval_set)


class TestSanitizeFloat:

    @pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf"), "x", object()])
    def test_unknown_values_become_none(self, value):
        from app.services.evaluation.ragas_runner import _sanitize_float
        assert _sanitize_float(value) is None

    @pytest.mark.parametrize("value,expected", [(0.0, 0.0), (0, 0.0), (0.42, 0.42), ("0.5", 0.5)])
    def test_real_values_kept(self, value, expected):
        from app.services.evaluation.ragas_runner import _sanitize_float
        assert _sanitize_float(value) == expected


class TestRunRagasEvaluation:

    def test_nan_none_and_missing_scores_become_none(self):
        rows = [
            {"faithfulness": float("nan"), "context_precision": None, "context_recall": 0.0,
             "answer_relevancy": 0.7},  # answer_correctness missing
            dict(FULL_SCORES),
        ]
        scores = _run_ragas(MagicMock(return_value=_FakeRagasResult(rows)))

        assert scores[0] == {
            "faithfulness": None, "context_precision": None, "context_recall": 0.0,
            "answer_relevancy": 0.7, "answer_correctness": None,
        }
        assert scores[1] == {k: 0.9 for k in METRIC_KEYS}

    def test_whole_evaluation_failure_returns_none_and_logs_error(self, caplog):
        boom = MagicMock(side_effect=ValueError("bad dataset"))
        with caplog.at_level(logging.ERROR, logger="app.services.evaluation.ragas_runner"):
            scores = _run_ragas(boom, n=3)

        assert scores == [{k: None for k in METRIC_KEYS}] * 3
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert errors and "bad dataset" in errors[0].getMessage()
        assert errors[0].exc_info is not None

    def test_rate_limit_error_is_reraised(self):
        from openai import RateLimitError

        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        exc = RateLimitError("slow down", response=httpx.Response(429, request=request), body=None)
        with pytest.raises(RateLimitError):
            _run_ragas(MagicMock(side_effect=exc))

    def test_row_count_mismatch_is_unscored(self, caplog):
        rows = [dict(FULL_SCORES)]  # one row for two queries
        with caplog.at_level(logging.ERROR, logger="app.services.evaluation.ragas_runner"):
            scores = _run_ragas(MagicMock(return_value=_FakeRagasResult(rows)), n=2)
        assert scores == [{k: None for k in METRIC_KEYS}] * 2
        assert any(r.levelno == logging.ERROR for r in caplog.records)


# ─── Metric aggregation ───────────────────────────────────────────────────────

class TestMean:

    def test_mixed_values_average_valid_only(self):
        from app.services.evaluation.metrics import _mean
        assert _mean([1.0, None, float("nan"), 0.0, float("inf")]) == pytest.approx(0.5)

    def test_zero_is_a_real_value(self):
        from app.services.evaluation.metrics import _mean
        assert _mean([0.0, 0.0]) == 0.0

    @pytest.mark.parametrize("values", [[], [None], [None, float("nan")]])
    def test_no_valid_values_is_none(self, values):
        from app.services.evaluation.metrics import _mean
        assert _mean(values) is None


class TestComputeMetrics:

    def test_ragas_failure_gives_none_but_operational_metrics(self):
        from app.services.evaluation.metrics import compute_metrics

        results = [_result("q1", latency=100.0, cost=0.002), _result("q2", latency=300.0, cost=0.004)]
        unscored = [{k: None for k in METRIC_KEYS}] * 2
        with patch("app.services.evaluation.ragas_runner.run_ragas_evaluation", return_value=unscored):
            m = compute_metrics(results, [{"id": "q1"}, {"id": "q2"}])

        for name in ("faithfulness", "context_precision", "context_recall",
                     "answer_relevance", "answer_correctness"):
            assert getattr(m, name) is None, name
        assert m.scored_counts == {
            "faithfulness": 0, "context_precision": 0, "context_recall": 0,
            "answer_relevance": 0, "answer_correctness": 0,
        }
        assert m.latency_p50_ms == pytest.approx(200.0)
        assert m.avg_cost_usd == pytest.approx(0.003)
        assert m.avg_token_usage == pytest.approx(15.0)

    def test_partial_scores_average_scored_rows_and_count_them(self):
        from app.services.evaluation.metrics import compute_metrics

        per_query = [
            dict(FULL_SCORES, faithfulness=None),
            dict(FULL_SCORES, faithfulness=0.0),
            dict(FULL_SCORES, faithfulness=0.5),
        ]
        results = [_result(f"q{i}") for i in range(3)]
        with patch("app.services.evaluation.ragas_runner.run_ragas_evaluation", return_value=per_query):
            m = compute_metrics(results, [])

        assert m.faithfulness == pytest.approx(0.25)
        assert m.scored_counts["faithfulness"] == 2
        assert m.scored_counts["context_recall"] == 3

    def test_no_results_is_all_none(self):
        from app.services.evaluation.metrics import compute_metrics

        m = compute_metrics([], [])
        for name in ("faithfulness", "answer_correctness", "freshness_validity",
                     "temporal_citation_accuracy", "multimodal_grounding_rate", "avg_cost_usd"):
            assert getattr(m, name) is None, name

    def test_custom_metrics_none_when_not_applicable(self):
        from app.services.evaluation.metrics import compute_metrics

        results = [_result("q1")]
        with patch("app.services.evaluation.ragas_runner.run_ragas_evaluation", return_value=[FULL_SCORES]):
            m = compute_metrics(results, [{"id": "q1", "modality": "text"}])

        assert m.freshness_validity is None
        assert m.temporal_citation_accuracy is None
        assert m.multimodal_grounding_rate is None
        assert m.faithfulness == pytest.approx(0.9)


class TestCustomMetrics:

    def test_freshness_none_without_temporal_items(self):
        from app.services.evaluation.metrics import compute_freshness_validity
        assert compute_freshness_validity([_result("q1")], [{"id": "q1"}]) is None

    def test_freshness_scored_with_temporal_items(self):
        from datetime import datetime
        from app.services.evaluation.metrics import compute_freshness_validity

        epoch = int(datetime.fromisoformat("2024-01-01").timestamp())
        good = _result("q1", chunks=[{"effective_from": epoch - 10, "effective_to": None}])
        stale = _result("q2", chunks=[{"effective_from": epoch - 100, "effective_to": epoch - 50}])
        eval_set = [{"id": "q1", "valid_from": "2024-01-01"}, {"id": "q2", "valid_from": "2024-01-01"}]
        assert compute_freshness_validity([good, stale], eval_set) == pytest.approx(0.5)

    def test_unparseable_date_is_not_counted_as_stale(self, caplog):
        from datetime import datetime
        from app.services.evaluation.metrics import compute_freshness_validity

        epoch = int(datetime.fromisoformat("2024-01-01").timestamp())
        good = _result("q1", chunks=[{"effective_from": epoch - 10, "effective_to": None}])
        bad_date = _result("q2", chunks=[{"effective_from": epoch - 10, "effective_to": None}])
        eval_set = [{"id": "q1", "valid_from": "2024-01-01"}, {"id": "q2", "valid_from": "not a date"}]
        with caplog.at_level(logging.WARNING, logger="app.services.evaluation.metrics"):
            assert compute_freshness_validity([good, bad_date], eval_set) == 1.0
            assert compute_freshness_validity([bad_date], eval_set) is None
        assert any("not a date" in r.getMessage() for r in caplog.records)

    def test_temporal_citation_accuracy_none_without_dated_chunks(self):
        from app.services.evaluation.metrics import _compute_temporal_citation_accuracy
        assert _compute_temporal_citation_accuracy([_result("q1")], [{"id": "q1"}]) is None
        # Temporal item, but no retrieved chunk carries a date.
        assert _compute_temporal_citation_accuracy(
            [_result("q1")], [{"id": "q1", "valid_from": "2024-01-01"}]
        ) is None

    def test_multimodal_none_without_multimodal_items(self):
        from app.services.evaluation.metrics import compute_multimodal_grounding_rate
        assert compute_multimodal_grounding_rate([_result("q1")], [{"id": "q1", "modality": "text"}]) is None

    def test_multimodal_zero_is_real(self):
        from app.services.evaluation.metrics import compute_multimodal_grounding_rate
        assert compute_multimodal_grounding_rate([_result("q1")], [{"id": "q1", "modality": "table"}]) == 0.0


# ─── Classifier ───────────────────────────────────────────────────────────────

class TestClassifierMissingMetrics:

    def test_missing_metrics_never_no_failure(self):
        from app.services.diagnostics.classifier import diagnose

        for metrics in ({}, {k: None for k in METRIC_KEYS}, {k: float("nan") for k in METRIC_KEYS}):
            d = diagnose(_result("q1"), {"id": "q1", "modality": "text"}, metrics)
            assert d.primary_failure is None, metrics
            assert d.evidence["skipped"] == "not scored"
            assert set(d.evidence["missing_metrics"]) == {
                "faithfulness", "context_recall", "context_precision", "answer_correctness",
            }
            # Unknown metrics are recorded as None, not as a perfect 1.0.
            assert d.evidence["faithfulness"] is None

    def test_temporal_rule_still_fires_without_metrics(self):
        from datetime import datetime
        from app.services.diagnostics.classifier import FailureCategory, diagnose

        epoch = int(datetime.fromisoformat("2024-01-01").timestamp())
        stale = _result("q1", chunks=[{"effective_from": epoch - 100, "effective_to": epoch - 50}])
        d = diagnose(stale, {"id": "q1", "valid_from": "2024-01-01"}, {})
        assert d.primary_failure == FailureCategory.STALE_ANSWER
        assert "skipped" not in d.evidence

    def test_modality_rule_still_fires_without_metrics(self):
        from app.services.diagnostics.classifier import FailureCategory, diagnose

        d = diagnose(_result("q1"), {"id": "q1", "modality": "table"}, {})
        assert d.primary_failure == FailureCategory.TABLE_RETRIEVAL_MISS

    def test_partially_scored_stops_at_undecidable_rule(self):
        from app.services.diagnostics.classifier import FailureCategory, diagnose

        # Recall is known and low: LOW_RECALL_RETRIEVAL is decidable.
        d = diagnose(_result("q1"), {}, {"context_recall": 0.1})
        assert d.primary_failure == FailureCategory.LOW_RECALL_RETRIEVAL
        # Recall fine, faithfulness unknown: cannot rule out synthesis failure.
        d = diagnose(_result("q1"), {}, {"context_recall": 0.9, "context_precision": 0.9,
                                         "answer_correctness": 0.9})
        assert d.primary_failure is None

    def test_full_good_scores_are_no_failure(self):
        from app.services.diagnostics.classifier import FailureCategory, diagnose

        d = diagnose(_result("q1"), {}, FULL_SCORES)
        assert d.primary_failure == FailureCategory.NO_FAILURE
        assert "missing_metrics" not in d.evidence

    def test_ml_classifier_not_used_when_unscored(self):
        from app.services.diagnostics.classifier import diagnose

        with patch("app.services.diagnostics.ml_classifier.predict") as predict:
            diagnose(_result("q1"), {}, {})
        predict.assert_not_called()


# ─── diagnose_run task ────────────────────────────────────────────────────────

def _qr_row(query_id, status="ok"):
    row = MagicMock()
    row.id = f"row_{query_id}"
    row.query_id = query_id
    row.question = f"question for {query_id}"
    row.generated_answer = "answer"
    row.retrieved_chunks = [{"content": "plain text", "chunk_type": "text"}]
    row.latency_ms = 100.0
    row.input_tokens = 10
    row.output_tokens = 5
    row.cost_usd = 0.0001
    row.failure_category = None
    row.diagnosis_evidence = {}
    row.status = status
    row.error_message = None
    return row


def _scalars_all(rows):
    res = MagicMock()
    res.scalars.return_value.all.return_value = rows
    return res


def _scalar_one_or_none(obj):
    res = MagicMock()
    res.scalar_one_or_none.return_value = obj
    return res


class TestDiagnoseRunTaskUnscored:

    def _run(self, rows, eval_set, per_query_scores, rm):
        from app.workers import tasks

        db = MagicMock()
        db.get.return_value = MagicMock()
        db.execute.side_effect = [_scalars_all(rows), _scalar_one_or_none(rm)]
        with patch("app.db.session.get_sync_db", return_value=db), \
             patch.object(tasks, "_load_eval_set", return_value=eval_set), \
             patch.object(tasks, "refresh_experiment_status", return_value=None), \
             patch("app.services.evaluation.ragas_runner.run_ragas_evaluation",
                   return_value=per_query_scores):
            return tasks.diagnose_run.run("run_1", "eval.json")

    def test_unscored_query_left_undiagnosed_and_out_of_accuracy(self):
        eval_set = [
            {"id": "q1", "modality": "text", "failure_type": "NO_FAILURE"},
            {"id": "q2", "modality": "text", "failure_type": "NO_FAILURE"},
        ]
        scored, unscored = _qr_row("q1"), _qr_row("q2")
        unscored.failure_category = "NO_FAILURE"  # stale value from an earlier diagnosis
        rm = MagicMock()

        out = self._run([scored, unscored], eval_set, [FULL_SCORES, {k: None for k in METRIC_KEYS}], rm)

        assert scored.failure_category == "NO_FAILURE"
        assert unscored.failure_category is None
        assert unscored.diagnosis_evidence["skipped"] == "not scored"
        assert out["queries_undiagnosed"] == 1
        assert out["diagnoses_written"] == 2
        # Only q1 counts: an undiagnosed query is neither right nor wrong.
        assert out["root_cause_diagnostic_accuracy"] == 1.0
        assert rm.root_cause_diagnostic_accuracy == 1.0

    def test_all_unscored_accuracy_none(self):
        eval_set = [{"id": "q1", "modality": "text", "failure_type": "NO_FAILURE"}]
        rm = MagicMock()
        out = self._run([_qr_row("q1")], eval_set, [{k: None for k in METRIC_KEYS}], rm)
        assert out["root_cause_diagnostic_accuracy"] is None
        assert rm.root_cause_diagnostic_accuracy is None


class TestTrainSkipsUnscoredLabels:

    def test_rows_labelled_without_metrics_are_not_training_samples(self):
        from app.workers import tasks

        scored = _qr_row("q1")
        scored.failure_category = "NO_FAILURE"
        scored.diagnosis_evidence = {"faithfulness": 0.9, "context_recall": 0.9,
                                     "context_precision": 0.9, "answer_correctness": 0.9}
        rule_only = _qr_row("q2")
        rule_only.failure_category = "STALE_ANSWER"
        rule_only.diagnosis_evidence = {"faithfulness": None, "missing_metrics": ["faithfulness"]}

        db = MagicMock()
        db.execute.side_effect = [
            _scalars_all([MagicMock(id="run_1")]),
            _scalars_all([scored, rule_only]),
            _scalars_all([]),
        ]
        ml_train = MagicMock(return_value={"trained": True})
        with patch("app.db.session.get_sync_db", return_value=db), \
             patch.object(tasks, "_load_eval_set", return_value=[{"id": "q1"}, {"id": "q2"}]), \
             patch("app.services.diagnostics.ml_classifier.train", ml_train), \
             patch("app.services.diagnostics.ml_classifier.reload_model"):
            out = tasks.train_failure_classifier.run("exp_1", "eval.json")

        assert out["samples_used"] == 1
        assert out["skipped_unscored"] == 1
        assert ml_train.call_args.args[3] == ["NO_FAILURE"]


class TestRunExperimentFailureHandlerLogs:

    def test_status_update_failure_is_logged(self, caplog):
        from app.workers import tasks

        db = MagicMock()
        db.get.side_effect = RuntimeError("db down")

        task = tasks.run_experiment
        with patch("app.db.session.get_sync_db", return_value=db), \
             patch.object(task, "retry", side_effect=RuntimeError("retry")), \
             caplog.at_level(logging.ERROR, logger="app.workers.tasks"):
            with pytest.raises(RuntimeError):
                task.run("exp_1")

        assert any("could not mark experiment failed" in r.getMessage() for r in caplog.records)


# ─── Recommendation engine ────────────────────────────────────────────────────

def _run(run_id, **metrics):
    base = {
        "faithfulness": 0.5, "answer_correctness": 0.5, "context_recall": 0.5,
        "context_precision": 0.5, "answer_relevance": 0.5, "latency_p50_ms": 1000.0,
        "latency_p95_ms": 2000.0, "avg_cost_usd": 0.01, "freshness_validity": 0.5,
        "multimodal_grounding_rate": 0.5,
    }
    base.update(metrics)
    return {"run_id": run_id, "config": {"parsing_strategy": "text_only"}, "metrics": base}


def _by_objective(recs):
    return {r.objective: r for r in recs}


class TestRecommendationEngine:

    def test_zero_cost_run_wins_lowest_cost(self):
        from app.services.recommendation.engine import recommend

        recs = _by_objective(recommend([_run("paid", avg_cost_usd=0.01), _run("free", avg_cost_usd=0.0)]))
        assert recs["lowest_cost"].run_id == "free"
        assert recs["lowest_cost"].score == 0.0
        assert recs["lowest_cost"].status == "ok"

    def test_zero_latency_run_wins_best_latency(self):
        from app.services.recommendation.engine import recommend

        recs = _by_objective(recommend([_run("slow", latency_p50_ms=500.0), _run("instant", latency_p50_ms=0.0)]))
        assert recs["best_latency"].run_id == "instant"

    def test_none_metric_runs_are_skipped(self):
        from app.services.recommendation.engine import recommend

        runs = [
            _run("unscored", faithfulness=None, avg_cost_usd=None),
            _run("scored", faithfulness=0.2, avg_cost_usd=0.05, latency_p50_ms=2000.0),
        ]
        recs = _by_objective(recommend(runs))
        # The unscored run is not "cheapest" or "most faithful" by default.
        assert recs["lowest_cost"].run_id == "scored"
        assert recs["best_faithfulness"].run_id == "scored"
        assert recs["best_overall"].run_id == "scored"
        # Objectives it does have metrics for still consider it.
        assert recs["best_latency"].run_id == "unscored"

    def test_no_eligible_runs_is_reported(self, caplog):
        from app.services.recommendation.engine import OBJECTIVES, recommend

        runs = [_run("a", freshness_validity=None), _run("b", freshness_validity=None)]
        with caplog.at_level(logging.WARNING, logger="app.services.recommendation.engine"):
            recs = recommend(runs)

        assert [r.objective for r in recs] == OBJECTIVES  # nothing dropped
        drift = _by_objective(recs)["best_for_drift"]
        assert drift.status == "no_eligible_runs"
        assert drift.score is None
        assert drift.run_id == ""
        assert "freshness_validity" in drift.rationale
        assert any("best_for_drift" in r.getMessage() for r in caplog.records)

    def test_exceptions_are_logged_and_reported(self, caplog):
        from app.services.recommendation import engine

        with patch.object(engine, "_recommend_for_objective", side_effect=ZeroDivisionError("boom")), \
             caplog.at_level(logging.ERROR, logger="app.services.recommendation.engine"):
            recs = engine.recommend([_run("a")], objectives=["best_overall"])

        assert len(recs) == 1
        assert recs[0].status == "error"
        assert recs[0].score is None
        assert "boom" in recs[0].rationale
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert errors and errors[0].exc_info is not None

    def test_safe_metric_keeps_zero(self):
        from app.services.recommendation.engine import _safe_metric

        assert _safe_metric({"metrics": {"avg_cost_usd": 0.0}}, "avg_cost_usd") == 0.0
        assert _safe_metric({"metrics": {"avg_cost_usd": None}}, "avg_cost_usd") is None
        assert _safe_metric({"metrics": {}}, "avg_cost_usd") is None
        assert _safe_metric({"metrics": {"x": float("nan")}}, "x") is None

    def test_best_overall_with_all_zero_costs(self):
        from app.services.recommendation.engine import recommend

        recs = _by_objective(recommend([_run("a", avg_cost_usd=0.0, faithfulness=0.9),
                                        _run("b", avg_cost_usd=0.0)]))
        assert recs["best_overall"].run_id == "a"
        assert math.isfinite(recs["best_overall"].score)


# ─── Baseline comparison ──────────────────────────────────────────────────────

class TestBaselineDeltas:

    def test_missing_metric_has_no_delta(self):
        from app.services.evaluation.baseline import compute_deltas

        deltas = {d.metric: d for d in compute_deltas(
            {"faithfulness": 0.5, "freshness_validity": None},
            {"faithfulness": 0.0, "freshness_validity": 0.9},
        )}
        assert deltas["faithfulness"].improved is False
        assert deltas["faithfulness"].comparison_value == 0.0
        assert deltas["freshness_validity"].improved is None
        assert deltas["freshness_validity"].absolute_delta is None
        assert deltas["freshness_validity"].comparison_value == 0.9


# ─── Diagnostics summary endpoint ─────────────────────────────────────────────

class TestFailureSummary:

    def _summary(self, category_rows):
        import asyncio
        from unittest.mock import AsyncMock
        from app.api.v1.endpoints.diagnostics import get_failure_summary

        def result(**attrs):
            r = MagicMock()
            for k, v in attrs.items():
                getattr(r, k).return_value = v
            return r

        db = MagicMock()
        db.execute = AsyncMock(side_effect=[
            result(scalar_one_or_none=MagicMock()),  # experiment
            result(all=[("run_1",)]),                # run ids
            result(all=category_rows),               # category counts (errored excluded)
            result(scalar=8),                        # total queries
            result(scalar=1),                        # errored queries
        ])
        return asyncio.run(get_failure_summary("exp_1", db))

    def test_null_category_is_undiagnosed_not_no_failure(self):
        out = self._summary([(None, 3), ("NO_FAILURE", 2), ("STALE_ANSWER", 2)])

        # Previously {None or "NO_FAILURE": 3} and ("NO_FAILURE", 2) collided:
        # one count overwrote the other.
        assert out["failure_counts"] == {"NO_FAILURE": 2, "STALE_ANSWER": 2}
        assert out["undiagnosed_queries"] == 3
        assert out["errored_queries"] == 1
        assert out["total_queries"] == 8

    def test_only_undiagnosed(self):
        out = self._summary([(None, 4)])
        assert out["failure_counts"] == {}
        assert "NO_FAILURE" not in out["failure_counts"]
        assert out["undiagnosed_queries"] == 4


class TestBaselineOverallScore:

    def test_missing_metrics_do_not_count_as_zero(self):
        from app.services.evaluation.baseline import compare_run_to_baseline

        baseline_cfg = {"retrieval_strategy": "vector_only", "chunking_strategy": "recursive",
                        "parsing_strategy": "text_only", "freshness_policy": "none"}
        runs = [
            {"run_id": "base", "config": baseline_cfg,
             "metrics": {"faithfulness": 0.5, "answer_correctness": 0.5, "context_recall": 0.5}},
            # Unscored run: with "x or 0" its score would be 0 and it could still be picked.
            {"run_id": "unscored", "config": {"retrieval_strategy": "hybrid"},
             "metrics": {"faithfulness": None, "answer_correctness": None, "context_recall": None,
                         "avg_cost_usd": 0.01}},
            {"run_id": "scored", "config": {"retrieval_strategy": "bm25"},
             "metrics": {"faithfulness": 0.1, "answer_correctness": 0.1, "context_recall": 0.1}},
        ]
        out = compare_run_to_baseline(runs)
        assert out["comparison"]["run_id"] == "scored"

    def test_no_scored_comparison_run_is_an_error(self):
        from app.services.evaluation.baseline import compare_run_to_baseline

        baseline_cfg = {"retrieval_strategy": "vector_only", "chunking_strategy": "recursive",
                        "parsing_strategy": "text_only", "freshness_policy": "none"}
        runs = [
            {"run_id": "base", "config": baseline_cfg, "metrics": {"faithfulness": 0.5}},
            {"run_id": "unscored", "config": {"retrieval_strategy": "hybrid"},
             "metrics": {"faithfulness": None, "avg_cost_usd": 0.01}},
        ]
        assert "error" in compare_run_to_baseline(runs)
