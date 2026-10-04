"""
Experiment lifecycle and cost limit tests (Known Issue #9).

Covers:
  - experiment status: pending -> running -> evaluating -> diagnosing -> done, or failed
  - the last run to finish diagnosis moves the experiment to done exactly once,
    also when two workers finish at the same time (PostgreSQL, see below)
  - all runs failed -> experiment failed (not stuck in running/evaluating)
  - MAX_COST_PER_RUN_USD: queries past the limit are not issued, recorded as
    "budget exceeded" errors, and the run is failed; <= 0 disables the limit

The worker tasks run against a real SQLite database (tables from the ORM
models); OpenAI, Qdrant and Celery are mocked. The concurrency test also runs
against PostgreSQL when FAITHTRACE_TEST_PG_URL is set, e.g.
  FAITHTRACE_TEST_PG_URL=postgresql+psycopg2://user:pw@localhost:5432/faithtrace_test
"""

import logging
import os
import threading
import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Experiment, QueryResult as QueryResultModel, Run
from app.services.experiment.runner import BUDGET_EXCEEDED, PipelineConfig, QueryResult
from app.workers import tasks


EVAL_SET = [
    {"id": "q1", "question": "Q1?", "ground_truth": "a", "modality": "text"},
    {"id": "q2", "question": "Q2?", "ground_truth": "b", "modality": "text"},
    {"id": "q3", "question": "Q3?", "ground_truth": "c", "modality": "text"},
]

GOOD_METRICS = {
    "faithfulness": 0.9, "context_recall": 0.9, "context_precision": 0.9,
    "answer_correctness": 0.9, "answer_relevancy": 0.9,
}

CONFIG = {
    "retrieval_strategy": "vector_only", "chunking_strategy": "recursive",
    "parsing_strategy": "text_only", "freshness_policy": "none",
    "embedding_model": "text-embedding-3-small", "llm_model": "gpt-4o-mini", "top_k": 3,
}


def _result(query_id, status="ok", error_message=None, cost=0.001):
    return QueryResult(
        query_id=query_id, question=f"{query_id}?",
        generated_answer="" if status == "error" else "answer",
        retrieved_chunks=[] if status == "error" else [{"content": "ctx", "chunk_type": "text"}],
        latency_ms=1.0, input_tokens=0 if status == "error" else 10,
        output_tokens=0 if status == "error" else 5,
        cost_usd=0.0 if status == "error" else cost,
        status=status, error_message=error_message,
    )


ALL_OK = [_result("q1"), _result("q2"), _result("q3")]
ALL_ERRORED = [_result(q, "error", "ConnectionError: x") for q in ("q1", "q2", "q3")]


@pytest.fixture(autouse=True)
def _heuristic_only():
    """Force the heuristic classifier so results don't depend on a trained model on disk."""
    with patch("app.services.diagnostics.ml_classifier.predict", return_value=None):
        yield


# ─── Real database plumbing ──────────────────────────────────────────────────

def _make_session_factory(url):
    engine = create_engine(url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    # Same options as app.db.session.SyncSessionLocal
    return engine, sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


@pytest.fixture
def Session(tmp_path):
    engine, factory = _make_session_factory(f"sqlite:///{tmp_path / 'lifecycle.db'}")
    yield factory
    engine.dispose()


def _seed(Session, n_runs=2, status="pending", document_ids=None):
    with Session() as db:
        exp = Experiment(id=str(uuid.uuid4()), name="exp", description="", status=status,
                         eval_set_path="eval_sets/faithtrace_v1.json", document_ids=document_ids)
        db.add(exp)
        run_ids = []
        for _ in range(n_runs):
            run = Run(id=str(uuid.uuid4()), experiment_id=exp.id, config=dict(CONFIG), status="pending")
            db.add(run)
            run_ids.append(run.id)
        db.commit()
        return exp.id, run_ids


def _exp(Session, exp_id):
    with Session() as db:
        return db.get(Experiment, exp_id)


def _run_row(Session, run_id):
    with Session() as db:
        return db.get(Run, run_id)


class _Worker:
    """Runs the real Celery task bodies against the test database, capturing
    (instead of sending) the tasks they enqueue."""

    def __init__(self, Session):
        self.Session = Session
        self.evaluate_queue: list[str] = []
        self.diagnose_queue: list[str] = []

    def _patches(self):
        return [
            patch("app.db.session.get_sync_db", side_effect=lambda: self.Session()),
            patch.object(tasks, "_load_eval_set", return_value=EVAL_SET),
            patch.object(tasks.evaluate_run, "delay",
                         side_effect=lambda run_id, *a: self.evaluate_queue.append(run_id)),
            patch.object(tasks.diagnose_run, "delay",
                         side_effect=lambda run_id, *a: self.diagnose_queue.append(run_id)),
            patch("app.services.evaluation.ragas_runner.run_ragas_evaluation",
                  side_effect=lambda results, eval_set: [GOOD_METRICS] * len(results)),
            # No Qdrant in tests: every chunking strategy counts as indexed.
            patch("app.services.experiment.runner.chunking_strategy_not_indexed", return_value=None),
        ]

    def _call(self, fn, *args, pipeline=None):
        ps = self._patches()
        if pipeline is not None:
            ps.append(patch("app.services.experiment.runner.run_pipeline", pipeline))
        for p in ps:
            p.start()
        try:
            return fn(*args)
        finally:
            for p in reversed(ps):
                p.stop()

    def run_experiment(self, exp_id, results_by_call):
        pipeline = MagicMock(side_effect=list(results_by_call))
        out = self._call(tasks.run_experiment.run, exp_id, pipeline=pipeline)
        return out, pipeline

    def evaluate(self, run_id):
        return self._call(tasks.evaluate_run.run, run_id)

    def diagnose(self, run_id):
        return self._call(tasks.diagnose_run.run, run_id)


# ─── derive_experiment_status ────────────────────────────────────────────────

def _r(status, evaluated=False, diagnosed=False):
    now = datetime.utcnow()
    return SimpleNamespace(status=status, evaluated_at=now if evaluated else None,
                           diagnosed_at=now if diagnosed else None)


class TestDeriveExperimentStatus:

    def test_generating_while_any_run_pending_or_running(self):
        assert tasks.derive_experiment_status([_r("done", True, True), _r("running")]) is None
        assert tasks.derive_experiment_status([_r("pending")]) is None

    def test_evaluating_until_every_done_run_is_evaluated(self):
        assert tasks.derive_experiment_status([_r("done", True), _r("done")]) == "evaluating"

    def test_diagnosing_once_all_evaluated(self):
        assert tasks.derive_experiment_status([_r("done", True, True), _r("done", True)]) == "diagnosing"

    def test_done_when_every_non_failed_run_is_diagnosed(self):
        runs = [_r("done", True, True), _r("failed"), _r("done", True, True)]
        assert tasks.derive_experiment_status(runs) == "done"

    def test_failed_when_no_run_succeeded(self):
        assert tasks.derive_experiment_status([_r("failed"), _r("failed")]) == "failed"
        assert tasks.derive_experiment_status([]) == "failed"


# ─── Full lifecycle through the tasks ────────────────────────────────────────

class TestLifecycle:

    def test_status_moves_through_every_stage(self, Session):
        exp_id, (r1, r2) = _seed(Session)
        w = _Worker(Session)
        assert _exp(Session, exp_id).status == "pending"

        out, _ = w.run_experiment(exp_id, [ALL_OK, ALL_OK])
        assert out["status"] == "evaluating"
        exp = _exp(Session, exp_id)
        assert exp.status == "evaluating"   # generation done, evaluations queued
        assert exp.completed_at is None
        assert w.evaluate_queue == [r1, r2]

        w.evaluate(r1)
        assert _exp(Session, exp_id).status == "evaluating"
        w.evaluate(r2)
        assert _exp(Session, exp_id).status == "diagnosing"
        assert w.diagnose_queue == [r1, r2]
        assert _run_row(Session, r1).evaluated_at is not None

        w.diagnose(r1)
        assert _exp(Session, exp_id).status == "diagnosing"
        w.diagnose(r2)
        exp = _exp(Session, exp_id)
        assert exp.status == "done"
        assert exp.completed_at is not None
        assert _run_row(Session, r2).diagnosed_at is not None
        # Run statuses are unchanged by the lifecycle.
        assert {_run_row(Session, r).status for r in (r1, r2)} == {"done"}

    def test_experiment_is_running_while_generating(self, Session):
        exp_id, _ = _seed(Session, n_runs=1)
        seen = []

        def pipeline(*a, **kw):
            seen.append(_exp(Session, exp_id).status)
            return ALL_OK

        w = _Worker(Session)
        w._call(tasks.run_experiment.run, exp_id, pipeline=MagicMock(side_effect=pipeline))
        assert seen == ["running"]

    def test_runs_finishing_before_generation_ends_still_finish_experiment(self, Session):
        # Run 1 is evaluated and diagnosed while run 2 is still generating: the
        # experiment must not finish early, and must finish once run 2 is done.
        exp_id, (r1, r2) = _seed(Session)
        w = _Worker(Session)

        def pipeline(*a, **kw):
            if w.evaluate_queue == [r1] and not w.diagnose_queue:
                w.evaluate(r1)
                w.diagnose(r1)
                assert _exp(Session, exp_id).status == "running"
            return ALL_OK

        w._call(tasks.run_experiment.run, exp_id, pipeline=MagicMock(side_effect=pipeline))
        assert _exp(Session, exp_id).status == "evaluating"
        w.evaluate(r2)
        w.diagnose(r2)
        assert _exp(Session, exp_id).status == "done"

    def test_last_run_failing_after_others_finished_completes_experiment(self, Session):
        # Run 1 is fully diagnosed by other workers while run 2 generates; run 2
        # then fails. run_experiment's session still holds its own (stale) copy
        # of run 1, so it must re-read the runs to see that run 1 is diagnosed.
        exp_id, (r1, r2) = _seed(Session)
        w = _Worker(Session)

        def pipeline(*a, **kw):
            if not w.evaluate_queue:
                return ALL_OK
            w.evaluate(r1)
            w.diagnose(r1)
            return ALL_ERRORED

        w._call(tasks.run_experiment.run, exp_id, pipeline=MagicMock(side_effect=pipeline))
        assert _run_row(Session, r2).status == "failed"
        assert _exp(Session, exp_id).status == "done"

    def test_all_runs_failed_marks_experiment_failed(self, Session):
        exp_id, run_ids = _seed(Session)
        w = _Worker(Session)

        out, _ = w.run_experiment(exp_id, [ALL_ERRORED, ALL_ERRORED])

        exp = _exp(Session, exp_id)
        assert out["status"] == "failed"
        assert exp.status == "failed"
        assert exp.completed_at is not None
        assert w.evaluate_queue == []
        assert {_run_row(Session, r).status for r in run_ids} == {"failed"}

    def test_failed_run_does_not_block_done(self, Session):
        exp_id, (r1, r2) = _seed(Session)
        w = _Worker(Session)

        w.run_experiment(exp_id, [ALL_ERRORED, ALL_OK])
        assert w.evaluate_queue == [r2]
        w.evaluate(r2)
        w.diagnose(r2)

        assert _exp(Session, exp_id).status == "done"
        assert _run_row(Session, r1).status == "failed"

    def test_pipeline_exception_on_every_run_fails_experiment(self, Session):
        exp_id, _ = _seed(Session)
        w = _Worker(Session)
        w.run_experiment(exp_id, [RuntimeError("boom"), RuntimeError("boom")])
        assert _exp(Session, exp_id).status == "failed"

    def test_all_errored_evaluation_path_still_finishes(self, Session):
        # A run marked done whose rows all errored (e.g. re-run) is evaluated with
        # no metrics, then diagnosed; the experiment still reaches done.
        exp_id, (r1,) = _seed(Session, n_runs=1, status="evaluating")
        with Session() as db:
            db.get(Run, r1).status = "done"
            for r in ALL_ERRORED:
                db.add(QueryResultModel(id=str(uuid.uuid4()), run_id=r1, query_id=r.query_id,
                                        question=r.question, generated_answer="",
                                        status="error", error_message=r.error_message))
            db.commit()
        w = _Worker(Session)

        out = w.evaluate(r1)
        assert out["queries_evaluated"] == 0
        assert _exp(Session, exp_id).status == "diagnosing"
        w.diagnose(r1)
        assert _exp(Session, exp_id).status == "done"

    def test_evaluation_without_query_results_fails_run(self, Session):
        exp_id, (r1,) = _seed(Session, n_runs=1, status="evaluating")
        with Session() as db:
            db.get(Run, r1).status = "done"
            db.commit()

        out = _Worker(Session).evaluate(r1)

        assert out == {"error": "No query results found for this run"}
        assert _run_row(Session, r1).status == "failed"
        assert _exp(Session, exp_id).status == "failed"

    def test_diagnosis_out_of_retries_fails_run(self, Session):
        exp_id, (r1, r2) = _seed(Session)
        w = _Worker(Session)
        w.run_experiment(exp_id, [ALL_OK, ALL_OK])
        w.evaluate(r1)
        w.evaluate(r2)
        w.diagnose(r1)

        tasks.diagnose_run.push_request(retries=tasks.diagnose_run.max_retries)
        try:
            with patch("app.services.diagnostics.classifier.diagnose_run",
                       side_effect=RuntimeError("classifier down")):
                with pytest.raises(RuntimeError, match="classifier down"):
                    w.diagnose(r2)
        finally:
            tasks.diagnose_run.pop_request()

        assert _run_row(Session, r2).status == "failed"
        assert _exp(Session, exp_id).status == "done"   # the other run finished

    def test_refresh_leaves_terminal_and_generating_experiments_alone(self, Session):
        for status in ("pending", "running", "done", "failed"):
            exp_id, _ = _seed(Session, n_runs=1, status=status)
            with Session() as db:
                assert tasks.refresh_experiment_status(db, exp_id) == status
            assert _exp(Session, exp_id).status == status


# ─── "Am I the last one?" ────────────────────────────────────────────────────

def _finish_all_diagnoses(Session, exp_id, run_ids):
    with Session() as db:
        db.get(Experiment, exp_id).status = "diagnosing"
        for rid in run_ids:
            run = db.get(Run, rid)
            run.status = "done"
            run.evaluated_at = run.diagnosed_at = datetime.utcnow()
        db.commit()


class TestLastRunFinishes:

    def test_two_refreshes_set_done_once(self, Session, caplog):
        # Both workers committed their run's diagnosed_at; both ask "am I last?".
        exp_id, run_ids = _seed(Session)
        _finish_all_diagnoses(Session, exp_id, run_ids)

        caplog.set_level(logging.INFO, logger="app.workers.tasks")
        with Session() as a, Session() as b:
            assert tasks.refresh_experiment_status(a, exp_id) == "done"
            first_completed = _exp(Session, exp_id).completed_at
            assert tasks.refresh_experiment_status(b, exp_id) == "done"

        assert _exp(Session, exp_id).completed_at == first_completed
        assert caplog.text.count("-> done") == 1

    @pytest.mark.skipif(not os.environ.get("FAITHTRACE_TEST_PG_URL"),
                        reason="set FAITHTRACE_TEST_PG_URL to run against PostgreSQL")
    def test_concurrent_last_runs_on_postgres(self, caplog):
        # Two workers diagnose the last two runs at the same moment, each in its
        # own transaction. Without the row lock one of them can read the other's
        # run as unfinished and leave (or put back) the experiment in diagnosing.
        engine, PgSession = _make_session_factory(os.environ["FAITHTRACE_TEST_PG_URL"])
        caplog.set_level(logging.INFO, logger="app.workers.tasks")
        try:
            for _ in range(25):
                exp_id, run_ids = _seed(PgSession)
                _finish_all_diagnoses(PgSession, exp_id, [])
                with PgSession() as db:
                    for rid in run_ids:
                        run = db.get(Run, rid)
                        run.status = "done"
                        run.evaluated_at = datetime.utcnow()
                    db.commit()

                barrier = threading.Barrier(2)
                errors = []

                def worker(rid):
                    try:
                        with PgSession() as db:
                            db.get(Run, rid).diagnosed_at = datetime.utcnow()
                            db.commit()
                            barrier.wait()
                            tasks.refresh_experiment_status(db, exp_id)
                    except Exception as exc:  # surfaced below
                        errors.append(exc)

                caplog.clear()
                threads = [threading.Thread(target=worker, args=(rid,)) for rid in run_ids]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=30)

                assert not errors, errors
                exp = _exp(PgSession, exp_id)
                assert exp.status == "done"
                assert exp.completed_at is not None
                assert caplog.text.count("-> done") == 1
        finally:
            Base.metadata.drop_all(engine)
            engine.dispose()


# ─── Cost limit ──────────────────────────────────────────────────────────────

def _config():
    return PipelineConfig(**{k: v for k, v in CONFIG.items()})


def _pipeline_with_cost(max_cost, n=5, input_tokens=1_000_000, output_tokens=0):
    """Run the real run_pipeline; each query costs input_tokens at gpt-4o-mini rates
    ($0.15 per 1M input tokens)."""
    from app.services.experiment import runner

    eval_set = [{"id": f"q{i}", "question": f"Q{i}?"} for i in range(n)]
    retriever = MagicMock()
    retriever.invoke.return_value = []
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content="answer", usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens})
    with patch("langchain_openai.ChatOpenAI", return_value=llm), \
         patch.object(runner, "_build_vector_retriever", return_value=retriever):
        results = runner.run_pipeline(_config(), eval_set, max_cost_usd=max_cost)
    return results, llm


class TestCostLimit:

    def test_queries_stop_once_limit_exceeded(self, caplog):
        # $0.15 per query; limit $0.40 is exceeded by the 3rd query (0.45).
        results, llm = _pipeline_with_cost(0.40)

        assert llm.invoke.call_count == 3
        assert [r.status for r in results] == ["ok", "ok", "ok", "error", "error"]
        assert [r.error_message for r in results[3:]] == [BUDGET_EXCEEDED] * 2
        assert all(r.cost_usd == 0.0 and r.generated_answer == "" for r in results[3:])
        assert [r.query_id for r in results] == ["q0", "q1", "q2", "q3", "q4"]
        assert "Cost limit" in caplog.text

    def test_limit_reached_exactly_is_not_exceeded(self):
        results, llm = _pipeline_with_cost(0.15 * 5)
        assert llm.invoke.call_count == 5
        assert all(r.status == "ok" for r in results)

    @pytest.mark.parametrize("limit", [0, -1, None])
    def test_limit_disabled_when_not_positive(self, limit):
        results, llm = _pipeline_with_cost(limit, input_tokens=100_000_000)  # $15/query
        assert llm.invoke.call_count == 5
        assert all(r.status == "ok" for r in results)

    def test_existing_callers_have_no_limit(self):
        from app.services.experiment import runner
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content="a", usage_metadata={"input_tokens": 10**9, "output_tokens": 0})
        with patch("langchain_openai.ChatOpenAI", return_value=llm), \
             patch.object(runner, "_build_vector_retriever", return_value=MagicMock(invoke=MagicMock(return_value=[]))):
            results = runner.run_pipeline(_config(), EVAL_SET)
        assert llm.invoke.call_count == 3
        assert all(r.status == "ok" for r in results)

    def test_run_over_budget_is_failed_and_others_continue(self, Session, caplog):
        exp_id, (r1, r2) = _seed(Session)
        over_budget = [_result("q1", cost=6.0)] + [
            _result(q, "error", BUDGET_EXCEEDED) for q in ("q2", "q3")]
        w = _Worker(Session)

        with patch.object(tasks.settings, "MAX_COST_PER_RUN_USD", 5.0):
            _, pipeline = w.run_experiment(exp_id, [over_budget, ALL_OK])

        assert pipeline.call_args_list[0].kwargs["max_cost_usd"] == 5.0
        assert _run_row(Session, r1).status == "failed"
        assert _run_row(Session, r2).status == "done"
        assert w.evaluate_queue == [r2]
        assert "cost limit MAX_COST_PER_RUN_USD=$5.0000 exceeded" in caplog.text
        with Session() as db:
            rows = db.execute(select(QueryResultModel).where(QueryResultModel.run_id == r1)).scalars().all()
        assert sorted(r.error_message or "" for r in rows) == ["", BUDGET_EXCEEDED, BUDGET_EXCEEDED]

        w.evaluate(r2)
        w.diagnose(r2)
        assert _exp(Session, exp_id).status == "done"

    def test_run_experiment_passes_document_scope(self, Session):
        exp_id, _ = _seed(Session, n_runs=1, document_ids=["doc-a"])
        _, pipeline = _Worker(Session).run_experiment(exp_id, [ALL_OK])
        assert pipeline.call_args.kwargs["document_ids"] == ["doc-a"]

        exp_id, _ = _seed(Session, n_runs=1)
        _, pipeline = _Worker(Session).run_experiment(exp_id, [ALL_OK])
        assert pipeline.call_args.kwargs["document_ids"] is None


class TestReevaluationOfFinishedExperiment:

    def test_failed_reevaluation_keeps_run_of_finished_experiment(self, Session):
        # POST /evaluation/run/{id} on a done experiment: if that evaluation
        # cannot run, the run keeps its status (and its place on the leaderboard).
        exp_id, (r1,) = _seed(Session, n_runs=1, status="done")
        with Session() as db:
            db.get(Run, r1).status = "done"
            db.commit()

        out = _Worker(Session).evaluate(r1)   # no query results stored

        assert out == {"error": "No query results found for this run"}
        assert _run_row(Session, r1).status == "done"
        assert _exp(Session, exp_id).status == "done"
