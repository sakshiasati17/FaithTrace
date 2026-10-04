"""
Diagnosis correctness tests (Known Issue #2).

Covers:
  - train_failure_classifier looks eval items up by "id" (not "query_id")
  - diagnose_run matches results to eval items by query_id, not list position
  - root-cause diagnostic accuracy is the correct fraction over labelled items,
    and None when nothing is labelled
"""

from unittest.mock import MagicMock, patch

import pytest

from app.services.experiment.runner import QueryResult


# A small eval set where the heuristic diagnosis depends on the eval item's
# modality, so pairing a result with the wrong item changes the outcome.
EVAL_SET = [
    {"id": "q_chart", "question": "Chart q", "ground_truth": "a", "modality": "chart",
     "failure_type": "CHART_LAYOUT_BLINDNESS"},
    {"id": "q_table", "question": "Table q", "ground_truth": "b", "modality": "table",
     "failure_type": "TABLE_RETRIEVAL_MISS"},
    {"id": "q_text", "question": "Text q", "ground_truth": "c", "modality": "text",
     "failure_type": "STALE_ANSWER"},  # deliberately wrong label: heuristic says NO_FAILURE
]

GOOD_METRICS = {
    "faithfulness": 0.9, "context_recall": 0.9, "context_precision": 0.9,
    "answer_correctness": 0.9, "answer_relevancy": 0.9,
}

EXPECTED = {
    "q_chart": "CHART_LAYOUT_BLINDNESS",
    "q_table": "TABLE_RETRIEVAL_MISS",
    "q_text": "NO_FAILURE",
}


def _result(query_id: str) -> QueryResult:
    return QueryResult(
        query_id=query_id,
        question=f"question for {query_id}",
        generated_answer="answer",
        retrieved_chunks=[{"content": "plain text", "chunk_type": "text"}],
        latency_ms=100.0,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0001,
    )


@pytest.fixture(autouse=True)
def _heuristic_only():
    """Force the heuristic classifier so results don't depend on a trained model on disk."""
    with patch("app.services.diagnostics.ml_classifier.predict", return_value=None):
        yield


# ─── Classifier: id-based matching ────────────────────────────────────────────

class TestClassifierDiagnoseRun:

    def test_matches_eval_items_by_id_when_results_are_shuffled(self):
        from app.services.diagnostics.classifier import diagnose_run

        # Reverse order relative to EVAL_SET: positional pairing would mismatch.
        results = [_result("q_text"), _result("q_table"), _result("q_chart")]
        diagnoses = diagnose_run(results, EVAL_SET, [GOOD_METRICS] * len(results))

        assert [d.query_id for d in diagnoses] == ["q_text", "q_table", "q_chart"]
        for d in diagnoses:
            assert d.primary_failure.value == EXPECTED[d.query_id], d.query_id

    def test_missing_eval_item_uses_empty_dict(self):
        from app.services.diagnostics.classifier import diagnose_run

        diagnoses = diagnose_run([_result("q_unknown")], EVAL_SET, [GOOD_METRICS])

        assert len(diagnoses) == 1
        assert diagnoses[0].primary_failure.value == "NO_FAILURE"
        assert diagnoses[0].evidence["modality"] == "text"

    def test_diagnoses_every_result_even_if_eval_set_is_shorter(self):
        from app.services.diagnostics.classifier import diagnose_run

        results = [_result("q_chart"), _result("q_extra")]
        diagnoses = diagnose_run(results, EVAL_SET[:1], [GOOD_METRICS])

        assert [d.query_id for d in diagnoses] == ["q_chart", "q_extra"]

    def test_query_id_key_accepted_as_fallback(self):
        from app.services.diagnostics.classifier import index_eval_set

        lookup = index_eval_set([{"query_id": "legacy", "modality": "table"}, {"id": "new"}])
        assert set(lookup) == {"legacy", "new"}


# ─── Diagnostic accuracy ──────────────────────────────────────────────────────

class TestDiagnosticAccuracy:

    def test_fraction_correct_over_labelled_items(self):
        from app.services.diagnostics.classifier import compute_diagnostic_accuracy

        predictions = {
            "q_chart": "CHART_LAYOUT_BLINDNESS",  # correct
            "q_table": "LOW_RECALL_RETRIEVAL",    # wrong
            "q_text": "STALE_ANSWER",             # correct
            "q_unlabelled": "NO_FAILURE",         # not in eval set: ignored
        }
        assert compute_diagnostic_accuracy(predictions, EVAL_SET) == pytest.approx(2 / 3)

    def test_unlabelled_items_are_skipped(self):
        from app.services.diagnostics.classifier import compute_diagnostic_accuracy

        eval_set = EVAL_SET + [{"id": "q_nolabel", "modality": "text"}]
        predictions = {"q_chart": "CHART_LAYOUT_BLINDNESS", "q_nolabel": "STALE_ANSWER"}
        assert compute_diagnostic_accuracy(predictions, eval_set) == 1.0

    def test_none_when_nothing_labelled(self):
        from app.services.diagnostics.classifier import compute_diagnostic_accuracy

        eval_set = [{"id": "a", "modality": "text"}, {"id": "b", "failure_type": None}]
        assert compute_diagnostic_accuracy({"a": "NO_FAILURE", "b": "STALE_ANSWER"}, eval_set) is None
        assert compute_diagnostic_accuracy({}, EVAL_SET) is None

    def test_compute_metrics_returns_none_for_diagnostic_accuracy(self):
        from app.services.evaluation.metrics import compute_metrics

        assert compute_metrics([], EVAL_SET).root_cause_diagnostic_accuracy is None

        per_query = [GOOD_METRICS] * 3
        with patch("app.services.evaluation.ragas_runner.run_ragas_evaluation", return_value=per_query):
            m = compute_metrics([_result("q_chart"), _result("q_table"), _result("q_text")], EVAL_SET)
        assert m.root_cause_diagnostic_accuracy is None
        assert m.faithfulness == pytest.approx(0.9)


# ─── Celery tasks (DB mocked) ─────────────────────────────────────────────────

def _qr_row(query_id: str, failure_category=None, row_id=None):
    row = MagicMock()
    row.id = row_id or f"row_{query_id}"
    row.query_id = query_id
    row.question = f"question for {query_id}"
    row.generated_answer = "answer"
    row.retrieved_chunks = [{"content": "plain text", "chunk_type": "text"}]
    row.latency_ms = 100.0
    row.input_tokens = 10
    row.output_tokens = 5
    row.cost_usd = 0.0001
    row.failure_category = failure_category
    row.diagnosis_evidence = {}
    return row


def _scalars_all(rows):
    res = MagicMock()
    res.scalars.return_value.all.return_value = rows
    return res


def _scalar_one_or_none(obj):
    res = MagicMock()
    res.scalar_one_or_none.return_value = obj
    return res


class TestDiagnoseRunTask:

    def _run(self, rows, eval_set, per_query_scores, run_metrics_row):
        from app.workers import tasks

        db = MagicMock()
        db.get.return_value = MagicMock()  # the Run
        db.execute.side_effect = [_scalars_all(rows), _scalar_one_or_none(run_metrics_row)]

        with patch("app.db.session.get_sync_db", return_value=db), \
             patch.object(tasks, "_load_eval_set", return_value=eval_set), \
             patch("app.services.evaluation.ragas_runner.run_ragas_evaluation",
                   return_value=per_query_scores):
            out = tasks.diagnose_run.run("run_1", "eval.json")
        return out, db

    def test_rows_are_matched_by_query_id_and_accuracy_stored(self):
        # DB returns rows in a different order from the eval set.
        rows = [_qr_row("q_table"), _qr_row("q_text"), _qr_row("q_chart")]
        rm = MagicMock()

        out, db = self._run(rows, EVAL_SET, [GOOD_METRICS] * 3, rm)

        for row in rows:
            assert row.failure_category == EXPECTED[row.query_id], row.query_id
        # chart + table correct, text labelled STALE_ANSWER but diagnosed NO_FAILURE
        assert rm.root_cause_diagnostic_accuracy == pytest.approx(2 / 3)
        assert out["root_cause_diagnostic_accuracy"] == pytest.approx(2 / 3)
        assert out["diagnoses_written"] == 3

    def test_accuracy_is_none_when_eval_set_unlabelled(self):
        unlabelled = [{k: v for k, v in item.items() if k != "failure_type"} for item in EVAL_SET]
        rows = [_qr_row("q_chart"), _qr_row("q_text")]
        rm = MagicMock()

        out, _ = self._run(rows, unlabelled, [GOOD_METRICS] * 2, rm)

        assert rm.root_cause_diagnostic_accuracy is None
        assert out["root_cause_diagnostic_accuracy"] is None

    def test_missing_metrics_row_is_not_created(self):
        rows = [_qr_row("q_chart")]

        out, db = self._run(rows, EVAL_SET, [GOOD_METRICS], None)

        db.add.assert_not_called()
        assert out["root_cause_diagnostic_accuracy"] == 1.0


class TestTrainFailureClassifierTask:

    def test_eval_items_are_looked_up_by_id(self):
        from app.workers import tasks

        eval_set = [
            {"id": "q_chart", "modality": "chart", "valid_from": "2024-01-01", "valid_to": None},
            {"id": "q_table", "modality": "table", "valid_from": "2023-01-01", "valid_to": "2023-12-31"},
        ]
        rows = [
            _qr_row("q_table", failure_category="TABLE_RETRIEVAL_MISS"),
            _qr_row("q_chart", failure_category="CHART_LAYOUT_BLINDNESS"),
            _qr_row("q_missing", failure_category="NO_FAILURE"),
        ]
        db = MagicMock()
        db.execute.side_effect = [
            _scalars_all([MagicMock(id="run_1")]),  # runs
            _scalars_all(rows),                       # labelled query results
            _scalars_all([]),                         # feedback
        ]
        ml_train = MagicMock(return_value={"trained": True})

        with patch("app.db.session.get_sync_db", return_value=db), \
             patch.object(tasks, "_load_eval_set", return_value=eval_set), \
             patch("app.services.diagnostics.ml_classifier.train", ml_train), \
             patch("app.services.diagnostics.ml_classifier.reload_model"):
            out = tasks.train_failure_classifier.run("exp_1", "eval.json")

        assert out["samples_used"] == 3
        _metrics, _chunks, eval_items, labels = ml_train.call_args.args
        assert eval_items[0] is eval_set[1]   # q_table
        assert eval_items[1] is eval_set[0]   # q_chart
        assert eval_items[2] == {}            # not in eval set
        assert eval_items[0]["modality"] == "table"
        assert eval_items[1]["valid_from"] == "2024-01-01"
        assert labels == ["TABLE_RETRIEVAL_MISS", "CHART_LAYOUT_BLINDNESS", "NO_FAILURE"]
