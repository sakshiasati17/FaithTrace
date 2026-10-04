"""
Query error handling tests (Known Issue #3).

Covers:
  - run_pipeline records a per-query error status instead of an "Error: ..." answer
  - evaluate_run / diagnose_run exclude errored rows from metrics and diagnosis
  - run_experiment marks a run failed when more than half its queries errored
  - an all-errored run does not crash evaluation or diagnosis
"""

from unittest.mock import MagicMock, patch

import pytest

from app.services.experiment.runner import PipelineConfig, QueryResult


EVAL_SET = [
    {"id": "q1", "question": "Q1?", "ground_truth": "a", "modality": "text"},
    {"id": "q2", "question": "Q2?", "ground_truth": "b", "modality": "text"},
    {"id": "q3", "question": "Q3?", "ground_truth": "c", "modality": "text"},
]

GOOD_METRICS = {
    "faithfulness": 0.9, "context_recall": 0.9, "context_precision": 0.9,
    "answer_correctness": 0.9, "answer_relevancy": 0.9,
}


def _config(**overrides) -> PipelineConfig:
    base = dict(
        retrieval_strategy="vector_only",
        chunking_strategy="recursive",
        parsing_strategy="text_only",
        freshness_policy="none",
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        top_k=3,
    )
    base.update(overrides)
    return PipelineConfig(**base)


@pytest.fixture(autouse=True)
def _heuristic_only():
    """Force the heuristic classifier so results don't depend on a trained model on disk."""
    with patch("app.services.diagnostics.ml_classifier.predict", return_value=None):
        yield


@pytest.fixture(autouse=True)
def _no_status_refresh():
    """The DB here is a mock fed a fixed query sequence; the experiment status
    refresh is covered against a real DB in test_experiment_lifecycle.py."""
    with patch("app.workers.tasks.refresh_experiment_status", return_value=None):
        yield


# ─── Runner ───────────────────────────────────────────────────────────────────

class TestRunnerRecordsErrors:

    def test_dataclass_defaults_keep_existing_callers_working(self):
        r = QueryResult("q", "Q?", "A", [], 1.0, 0, 0, 0.0)
        assert r.status == "ok"
        assert r.error_message is None

    def test_retriever_exception_is_recorded_not_answered(self, caplog):
        from app.services.experiment import runner

        with patch("langchain_openai.ChatOpenAI"), \
             patch.object(runner, "_build_vector_retriever",
                          side_effect=ConnectionError("qdrant unreachable")):
            results = runner.run_pipeline(_config(), EVAL_SET[:2])

        assert len(results) == 2
        for r in results:
            assert r.status == "error"
            assert r.error_message == "ConnectionError: qdrant unreachable"
            assert r.generated_answer == ""
            assert not r.generated_answer.startswith("Error:")
            assert r.retrieved_chunks == []
            assert r.input_tokens == 0 and r.output_tokens == 0
        assert "Query q1 failed" in caplog.text

    def test_llm_exception_is_recorded(self):
        from app.services.experiment import runner

        retriever = MagicMock()
        retriever.invoke.return_value = []
        llm = MagicMock()
        llm.invoke.side_effect = ValueError("bad request")

        with patch("langchain_openai.ChatOpenAI", return_value=llm), \
             patch.object(runner, "_build_vector_retriever", return_value=retriever):
            [r] = runner.run_pipeline(_config(), EVAL_SET[:1])

        assert r.status == "error"
        assert r.error_message == "ValueError: bad request"
        assert r.generated_answer == ""

    def test_successful_query_has_ok_status(self):
        from app.services.experiment import runner

        doc = MagicMock(page_content="context", metadata={"chunk_type": "text"})
        retriever = MagicMock()
        retriever.invoke.return_value = [doc]
        response = MagicMock(content="the answer", usage_metadata={"input_tokens": 7, "output_tokens": 3})
        llm = MagicMock()
        llm.invoke.return_value = response

        with patch("langchain_openai.ChatOpenAI", return_value=llm), \
             patch.object(runner, "_build_vector_retriever", return_value=retriever):
            [r] = runner.run_pipeline(_config(), EVAL_SET[:1])

        assert r.status == "ok"
        assert r.error_message is None
        assert r.generated_answer == "the answer"
        assert r.retrieved_chunks[0]["content"] == "context"


# ─── Celery tasks (DB mocked) ─────────────────────────────────────────────────

def _qr_row(query_id: str, status: str = "ok", error_message=None):
    row = MagicMock()
    row.id = f"row_{query_id}"
    row.query_id = query_id
    row.question = f"question for {query_id}"
    row.generated_answer = "" if status == "error" else "answer"
    row.retrieved_chunks = [] if status == "error" else [{"content": "plain text", "chunk_type": "text"}]
    row.latency_ms = 100.0
    row.input_tokens = 10
    row.output_tokens = 5
    row.cost_usd = 0.0001
    row.failure_category = None
    row.diagnosis_evidence = {}
    row.status = status
    row.error_message = error_message
    return row


def _scalars_all(rows):
    res = MagicMock()
    res.scalars.return_value.all.return_value = rows
    return res


def _scalar_one_or_none(obj):
    res = MagicMock()
    res.scalar_one_or_none.return_value = obj
    return res


class TestEvaluateRunExcludesErrors:

    def _run(self, rows, metrics_row):
        from app.workers import tasks

        db = MagicMock()
        db.get.return_value = MagicMock()  # the Run
        db.execute.side_effect = [_scalars_all(rows), _scalar_one_or_none(metrics_row)]
        ragas = MagicMock(side_effect=lambda results, eval_set: [GOOD_METRICS] * len(results))

        with patch("app.db.session.get_sync_db", return_value=db), \
             patch.object(tasks, "_load_eval_set", return_value=EVAL_SET), \
             patch.object(tasks.diagnose_run, "delay") as diag_delay, \
             patch("app.services.evaluation.ragas_runner.run_ragas_evaluation", ragas):
            out = tasks.evaluate_run.run("run_1", "eval.json")
        return out, db, ragas, diag_delay

    def test_errored_rows_are_not_scored(self):
        rows = [_qr_row("q1"), _qr_row("q2", "error", "TimeoutError: x"), _qr_row("q3")]
        rm = MagicMock()

        out, _, ragas, diag_delay = self._run(rows, rm)

        scored = ragas.call_args.args[0]
        assert [r.query_id for r in scored] == ["q1", "q3"]
        assert rm.faithfulness == pytest.approx(0.9)
        assert out["queries_evaluated"] == 2
        assert out["queries_errored"] == 1
        diag_delay.assert_called_once_with("run_1", "eval.json")

    def test_all_errored_writes_no_metrics(self):
        rows = [_qr_row("q1", "error", "E: 1"), _qr_row("q2", "error", "E: 2")]

        out, db, ragas, diag_delay = self._run(rows, None)

        ragas.assert_not_called()
        db.add.assert_not_called()
        assert out["faithfulness"] is None
        assert out["queries_evaluated"] == 0
        assert out["queries_errored"] == 2
        # Diagnosis still runs so errored rows get marked as skipped.
        diag_delay.assert_called_once()


class TestDiagnoseRunExcludesErrors:

    def _run(self, rows, metrics_row):
        from app.workers import tasks

        db = MagicMock()
        db.get.return_value = MagicMock()  # the Run
        db.execute.side_effect = [_scalars_all(rows), _scalar_one_or_none(metrics_row)]
        ragas = MagicMock(side_effect=lambda results, eval_set: [GOOD_METRICS] * len(results))

        with patch("app.db.session.get_sync_db", return_value=db), \
             patch.object(tasks, "_load_eval_set", return_value=EVAL_SET), \
             patch("app.services.evaluation.ragas_runner.run_ragas_evaluation", ragas):
            out = tasks.diagnose_run.run("run_1", "eval.json")
        return out, ragas

    def test_errored_rows_are_skipped(self):
        ok1, err, ok2 = _qr_row("q1"), _qr_row("q2", "error", "TimeoutError: x"), _qr_row("q3")
        err.failure_category = "LOW_RECALL_RETRIEVAL"  # stale value from an earlier diagnosis
        rm = MagicMock()

        out, ragas = self._run([ok1, err, ok2], rm)

        assert [r.query_id for r in ragas.call_args.args[0]] == ["q1", "q3"]
        assert ok1.failure_category == "NO_FAILURE"
        assert ok2.failure_category == "NO_FAILURE"
        assert err.failure_category is None
        assert err.diagnosis_evidence == {"skipped": "query errored"}
        assert out["diagnoses_written"] == 2
        assert out["queries_errored"] == 1

    def test_errored_rows_do_not_count_toward_diagnostic_accuracy(self):
        from app.workers import tasks

        labelled = [dict(item, failure_type="NO_FAILURE") for item in EVAL_SET]
        rows = [_qr_row("q1"), _qr_row("q2", "error", "E: x")]
        rm = MagicMock()
        db = MagicMock()
        db.get.return_value = MagicMock()
        db.execute.side_effect = [_scalars_all(rows), _scalar_one_or_none(rm)]

        with patch("app.db.session.get_sync_db", return_value=db), \
             patch.object(tasks, "_load_eval_set", return_value=labelled), \
             patch("app.services.evaluation.ragas_runner.run_ragas_evaluation",
                   side_effect=lambda results, eval_set: [GOOD_METRICS] * len(results)):
            out = tasks.diagnose_run.run("run_1", "eval.json")

        # Only q1 counts; it is correctly diagnosed NO_FAILURE.
        assert out["root_cause_diagnostic_accuracy"] == 1.0
        assert rm.root_cause_diagnostic_accuracy == 1.0

    def test_all_errored_does_not_crash(self):
        rows = [_qr_row("q1", "error", "E: 1"), _qr_row("q2", "error", "E: 2")]
        rm = MagicMock()

        out, ragas = self._run(rows, rm)

        ragas.assert_not_called()
        assert out["diagnoses_written"] == 0
        assert out["queries_errored"] == 2
        assert out["root_cause_diagnostic_accuracy"] is None
        for row in rows:
            assert row.failure_category is None
            assert row.diagnosis_evidence == {"skipped": "query errored"}


class TestRunExperimentStatus:

    def _results(self, statuses):
        return [
            QueryResult(
                query_id=f"q{i}", question="Q?",
                generated_answer="" if s == "error" else "answer",
                retrieved_chunks=[], latency_ms=1.0, input_tokens=0, output_tokens=0,
                cost_usd=0.0, status=s,
                error_message="RuntimeError: boom" if s == "error" else None,
            )
            for i, s in enumerate(statuses)
        ]

    def _run(self, statuses):
        from app.workers import tasks

        run = MagicMock(id="run_1", status="pending",
                        config={"retrieval_strategy": "vector_only", "chunking_strategy": "recursive",
                                "parsing_strategy": "text_only", "freshness_policy": "none",
                                "embedding_model": "e", "llm_model": "gpt-4o-mini"})
        experiment = MagicMock()
        db = MagicMock()
        db.get.return_value = experiment
        db.execute.return_value = _scalars_all([run])

        with patch("app.db.session.get_sync_db", return_value=db), \
             patch.object(tasks, "_load_eval_set", return_value=EVAL_SET), \
             patch("app.services.experiment.runner.run_pipeline",
                   return_value=self._results(statuses)), \
             patch("app.services.experiment.runner.chunking_strategy_not_indexed",
                   return_value=None), \
             patch.object(tasks.evaluate_run, "delay") as eval_delay:
            out = tasks.run_experiment.run("exp_1", "eval.json")
        return out, run, db, eval_delay

    def test_status_and_error_are_persisted(self):
        out, run, db, _ = self._run(["ok", "error", "ok"])

        added = [c.args[0] for c in db.add.call_args_list]
        assert [qr.status for qr in added] == ["ok", "error", "ok"]
        assert added[1].error_message == "RuntimeError: boom"
        assert added[1].generated_answer == ""
        assert out["query_errors_by_run"] == {"run_1": 1}

    def test_minority_errors_keep_run_done_and_evaluate(self):
        _, run, _, eval_delay = self._run(["ok", "error", "ok"])

        assert run.status == "done"
        eval_delay.assert_called_once_with("run_1", "eval.json")

    def test_majority_errors_mark_run_failed(self):
        out, run, _, eval_delay = self._run(["error", "error", "ok"])

        assert run.status == "failed"
        eval_delay.assert_not_called()
        assert out["query_errors_by_run"] == {"run_1": 2}

    def test_exactly_half_errored_is_not_failed(self):
        from app.workers.tasks import _run_should_fail

        assert _run_should_fail(2, 4) is False
        assert _run_should_fail(3, 4) is True
        assert _run_should_fail(0, 0) is False

    def test_all_errored_run_fails_without_crashing(self):
        out, run, _, eval_delay = self._run(["error", "error", "error"])

        assert run.status == "failed"
        eval_delay.assert_not_called()
        assert out["runs_processed"] == 1
