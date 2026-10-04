"""
Ragas scores are computed once per run.

evaluate_run saves each query's Ragas scores on its query result, and
diagnose_run reuses them instead of running a second (paid) Ragas evaluation
of the same answers. Diagnosis evidence also carries answer_relevance, which
classifier training reads.
"""

from unittest.mock import MagicMock, patch

from app.services.evaluation.metrics import RunMetrics


SCORES_Q1 = {"faithfulness": 0.9, "context_precision": 0.8, "context_recall": 0.7,
             "answer_relevancy": 0.6, "answer_correctness": 0.5}
SCORES_Q2 = {"faithfulness": 0.4, "context_precision": 0.3, "context_recall": 0.2,
             "answer_relevancy": 0.1, "answer_correctness": 0.0}


def _qr_row(query_id, evidence=None):
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
    row.diagnosis_evidence = evidence if evidence is not None else {}
    row.status = "ok"
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


def _run_metrics(per_query_scores):
    return RunMetrics(
        answer_correctness=0.25, faithfulness=0.65, context_precision=0.55,
        context_recall=0.45, answer_relevance=0.35, latency_p50_ms=100.0,
        latency_p95_ms=100.0, avg_token_usage=15.0, avg_cost_usd=0.0001,
        freshness_validity=None, temporal_citation_accuracy=None,
        multimodal_grounding_rate=None, root_cause_diagnostic_accuracy=None,
        per_query_scores=per_query_scores,
    )


EVAL_SET = [{"id": "q1", "modality": "text"}, {"id": "q2", "modality": "text"}]


class TestEvaluateRunSavesScores:

    def _run(self, rows, metrics):
        from app.workers import tasks

        db = MagicMock()
        db.get.return_value = MagicMock()
        db.execute.side_effect = [_scalars_all(rows), _scalar_one_or_none(MagicMock())]
        with patch("app.db.session.get_sync_db", return_value=db), \
             patch.object(tasks, "load_eval_set_for", return_value=EVAL_SET), \
             patch.object(tasks, "refresh_experiment_status", return_value=None), \
             patch.object(tasks.diagnose_run, "delay"), \
             patch("app.services.evaluation.metrics.compute_metrics", return_value=metrics):
            return tasks.evaluate_run.run("run_1", None)

    def test_per_query_scores_saved_on_each_row(self):
        q1, q2 = _qr_row("q1", {"keep": "me"}), _qr_row("q2")
        self._run([q1, q2], _run_metrics([SCORES_Q1, SCORES_Q2]))
        assert q1.diagnosis_evidence == {"keep": "me", "scores": SCORES_Q1}
        assert q2.diagnosis_evidence == {"scores": SCORES_Q2}

    def test_mismatched_score_rows_are_not_saved(self, caplog):
        q1, q2 = _qr_row("q1"), _qr_row("q2")
        self._run([q1, q2], _run_metrics([SCORES_Q1]))
        assert "scores" not in q1.diagnosis_evidence
        assert "scores" not in q2.diagnosis_evidence
        assert "not saving them" in caplog.text


class TestDiagnoseRunReusesScores:

    def _run(self, rows, ragas_return=None):
        from app.workers import tasks

        db = MagicMock()
        db.get.return_value = MagicMock()
        db.execute.side_effect = [_scalars_all(rows), _scalar_one_or_none(MagicMock())]
        ragas = MagicMock(return_value=ragas_return)
        with patch("app.db.session.get_sync_db", return_value=db), \
             patch.object(tasks, "load_eval_set_for", return_value=EVAL_SET), \
             patch.object(tasks, "refresh_experiment_status", return_value=None), \
             patch("app.services.evaluation.ragas_runner.run_ragas_evaluation", ragas):
            out = tasks.diagnose_run.run("run_1", None)
        return out, ragas

    def test_saved_scores_reused_without_second_ragas_call(self):
        q1 = _qr_row("q1", {"scores": SCORES_Q1})
        q2 = _qr_row("q2", {"scores": SCORES_Q2})
        out, ragas = self._run([q1, q2])
        ragas.assert_not_called()
        assert out["diagnoses_written"] == 2
        # Evidence keeps the scores and exposes answer_relevance for training.
        assert q1.diagnosis_evidence["scores"] == SCORES_Q1
        assert q1.diagnosis_evidence["answer_relevance"] == 0.6
        assert q2.diagnosis_evidence["answer_relevance"] == 0.1

    def test_missing_scores_trigger_one_rescore(self, caplog):
        q1 = _qr_row("q1", {"scores": SCORES_Q1})
        q2 = _qr_row("q2")  # evaluated before scores were saved
        out, ragas = self._run([q1, q2], ragas_return=[SCORES_Q1, SCORES_Q2])
        ragas.assert_called_once()
        assert "no saved Ragas scores" in caplog.text
        assert q2.diagnosis_evidence["answer_relevance"] == 0.1

    def test_rediagnosis_keeps_reusing_scores(self):
        # A second diagnose_run (e.g. re-diagnosis) still finds the scores,
        # because the diagnosis evidence it wrote keeps them.
        q1 = _qr_row("q1", {"scores": SCORES_Q1})
        q2 = _qr_row("q2", {"scores": SCORES_Q2})
        self._run([q1, q2])
        _, ragas = self._run([q1, q2])
        ragas.assert_not_called()
