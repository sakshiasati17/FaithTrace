"""
Re-diagnosis without re-scoring (feat/rediagnose-endpoint).

Covers:
  1. POST /diagnostics/run/{id}/rediagnose: 404 / 400 paths, enqueues
     diagnose_run (never evaluate_run) with the experiment's eval set, and
     reports how many ok queries diagnose_run would re-score
  2. POST /diagnostics/experiment/{id}/rediagnose: per-run results with skip
     reasons, enqueue errors reported per run
  3. scripts/rediagnose.py with urlopen patched (no network)

The DB session and Celery are mocked.
"""

import importlib.util
import io
import json
import urllib.error
from datetime import datetime
from email.message import Message
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.db.models import Experiment, Run
from app.db.session import get_db
from app.main import app


# ─── API plumbing ─────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    session = MagicMock()
    session.execute = AsyncMock()
    session.get = AsyncMock(return_value=None)
    return session


@pytest.fixture
def client(db):
    async def _override():
        yield db
    app.dependency_overrides[get_db] = _override
    with patch("app.main.settings.API_KEY", ""):
        yield TestClient(app)
    app.dependency_overrides.pop(get_db, None)


def _one(obj):
    res = MagicMock()
    res.scalar_one_or_none.return_value = obj
    return res


def _all_runs(runs):
    res = MagicMock()
    res.scalars.return_value.all.return_value = runs
    return res


def _rows(rows):
    """Result of the (run_id, status, diagnosis_evidence) coverage query."""
    res = MagicMock()
    res.all.return_value = rows
    return res


def _run(run_id="r1", status="done", evaluated=True):
    return Run(
        id=run_id, experiment_id="e1", config={}, status=status,
        evaluated_at=datetime(2026, 10, 1) if evaluated else None,
    )


SCORED = {"scores": {"answer_correctness": 0.8}, "confidence": 0.7}


# ─── Single run ───────────────────────────────────────────────────────────────

class TestRediagnoseRun:

    def _post(self, client, run_id="r1"):
        with patch("app.workers.tasks.diagnose_run.delay") as diag, \
                patch("app.workers.tasks.evaluate_run.delay") as evaluate:
            resp = client.post(f"/api/v1/diagnostics/run/{run_id}/rediagnose")
        return resp, diag, evaluate

    def test_missing_run_404(self, client, db):
        db.execute.return_value = _one(None)
        resp, diag, _ = self._post(client, "nope")
        assert resp.status_code == 404
        diag.assert_not_called()

    @pytest.mark.parametrize("status", ["pending", "running", "failed"])
    def test_run_not_done_400(self, client, db, status):
        db.execute.return_value = _one(_run(status=status))
        resp, diag, _ = self._post(client)
        assert resp.status_code == 400
        assert f"'{status}'" in resp.json()["detail"]
        diag.assert_not_called()

    def test_run_not_evaluated_400(self, client, db):
        db.execute.return_value = _one(_run(evaluated=False))
        resp, diag, _ = self._post(client)
        assert resp.status_code == 400
        assert "evaluated_at" in resp.json()["detail"]
        diag.assert_not_called()

    def test_run_without_query_results_400(self, client, db):
        db.execute.side_effect = [_one(_run()), _rows([])]
        resp, diag, _ = self._post(client)
        assert resp.status_code == 400
        assert "no query results" in resp.json()["detail"]
        diag.assert_not_called()

    def test_enqueues_diagnosis_only_with_experiments_path(self, client, db):
        db.execute.side_effect = [
            _one(_run()),
            _rows([("r1", "ok", SCORED), ("r1", None, SCORED), ("r1", "error", {"skipped": "query errored"})]),
        ]
        db.get.return_value = Experiment(id="e1", name="e", eval_set_path="eval_sets/golden_regression_set.json")

        resp, diag, evaluate = self._post(client)

        assert resp.status_code == 200, resp.text
        diag.assert_called_once_with("r1", "eval_sets/golden_regression_set.json")
        evaluate.assert_not_called()
        body = resp.json()
        assert body["run_id"] == "r1"
        assert body["status"] == "enqueued"
        assert body["eval_set_path"] == "eval_sets/golden_regression_set.json"
        assert body["eval_set_id"] is None
        # status None counts as ok (pre-003 rows); the errored row is not counted.
        assert body["ok_queries"] == 2
        assert body["queries_missing_scores"] == 0
        assert body["will_rescore_queries"] == 0

    def test_uploaded_eval_set_passes_no_path(self, client, db):
        db.execute.side_effect = [_one(_run()), _rows([("r1", "ok", SCORED)])]
        db.get.return_value = Experiment(id="e1", name="e", eval_set_id="u1")

        resp, diag, _ = self._post(client)

        # No path: the worker loads the experiment's uploaded set (eval_set_id u1).
        diag.assert_called_once_with("r1", None)
        assert resp.json()["eval_set_id"] == "u1"

    def test_reports_queries_that_would_be_rescored(self, client, db):
        db.execute.side_effect = [
            _one(_run()),
            _rows([
                ("r1", "ok", SCORED),
                ("r1", "ok", {"confidence": 0.7}),       # diagnosed before scores were stored
                ("r1", "ok", None),
                ("r1", "error", None),                   # errored: never scored, not counted
            ]),
        ]
        db.get.return_value = Experiment(id="e1", name="e")

        resp, diag, evaluate = self._post(client)

        assert resp.status_code == 200, resp.text
        diag.assert_called_once()
        evaluate.assert_not_called()
        body = resp.json()
        assert body["ok_queries"] == 3
        assert body["queries_missing_scores"] == 2
        # diagnose_run re-scores all ok queries when any lacks stored scores.
        assert body["will_rescore_queries"] == 3

    def test_enqueue_failure_is_503(self, client, db):
        db.execute.side_effect = [_one(_run()), _rows([("r1", "ok", SCORED)])]
        db.get.return_value = Experiment(id="e1", name="e")
        with patch("app.workers.tasks.diagnose_run.delay", side_effect=ConnectionError("broker down")):
            resp = client.post("/api/v1/diagnostics/run/r1/rediagnose")
        assert resp.status_code == 503
        assert "broker down" in resp.json()["detail"]


# ─── Experiment ───────────────────────────────────────────────────────────────

class TestRediagnoseExperiment:

    def test_missing_experiment_404(self, client, db):
        with patch("app.workers.tasks.diagnose_run.delay") as diag:
            resp = client.post("/api/v1/diagnostics/experiment/nope/rediagnose")
        assert resp.status_code == 404
        diag.assert_not_called()

    def test_per_run_results_and_skip_reasons(self, client, db):
        db.get.return_value = Experiment(id="e1", name="e", eval_set_path="eval_sets/sample_eval_set.json")
        runs = [
            _run("r-ok"),
            _run("r-running", status="running"),
            _run("r-unevaluated", evaluated=False),
            _run("r-empty"),
            _run("r-missing"),
        ]
        db.execute.side_effect = [
            _all_runs(runs),
            _rows([
                ("r-ok", "ok", SCORED),
                ("r-ok", "error", None),
                ("r-missing", "ok", SCORED),
                ("r-missing", "ok", {}),
            ]),
        ]

        with patch("app.workers.tasks.diagnose_run.delay") as diag, \
                patch("app.workers.tasks.evaluate_run.delay") as evaluate:
            resp = client.post("/api/v1/diagnostics/experiment/e1/rediagnose")

        assert resp.status_code == 200, resp.text
        assert [c.args for c in diag.call_args_list] == [
            ("r-ok", "eval_sets/sample_eval_set.json"),
            ("r-missing", "eval_sets/sample_eval_set.json"),
        ]
        evaluate.assert_not_called()

        body = resp.json()
        assert body["experiment_id"] == "e1"
        assert body["eval_set_path"] == "eval_sets/sample_eval_set.json"
        assert (body["enqueued"], body["skipped"], body["errors"]) == (2, 3, 0)
        assert body["will_rescore_queries"] == 2

        by_id = {r["run_id"]: r for r in body["runs"]}
        assert [r["run_id"] for r in body["runs"]] == [r.id for r in runs]
        assert by_id["r-ok"]["status"] == "enqueued"
        assert by_id["r-ok"]["will_rescore_queries"] == 0
        assert by_id["r-ok"]["ok_queries"] == 1
        assert by_id["r-running"] == {
            "run_id": "r-running", "run_status": "running", "status": "skipped",
            "reason": "run status is 'running', must be 'done'",
        }
        assert by_id["r-unevaluated"]["status"] == "skipped"
        assert "evaluated_at" in by_id["r-unevaluated"]["reason"]
        assert by_id["r-empty"]["status"] == "skipped"
        assert "no query results" in by_id["r-empty"]["reason"]
        assert by_id["r-missing"]["status"] == "enqueued"
        assert by_id["r-missing"]["queries_missing_scores"] == 1
        assert by_id["r-missing"]["will_rescore_queries"] == 2

    def test_no_eligible_runs_enqueues_nothing(self, client, db):
        db.get.return_value = Experiment(id="e1", name="e")
        db.execute.side_effect = [_all_runs([_run("r1", status="failed")])]

        with patch("app.workers.tasks.diagnose_run.delay") as diag:
            resp = client.post("/api/v1/diagnostics/experiment/e1/rediagnose")

        assert resp.status_code == 200, resp.text
        diag.assert_not_called()
        body = resp.json()
        assert (body["enqueued"], body["skipped"], body["errors"]) == (0, 1, 0)
        # Only the runs query ran: no coverage query for zero eligible runs.
        assert db.execute.await_count == 1

    def test_enqueue_failure_reported_per_run(self, client, db):
        db.get.return_value = Experiment(id="e1", name="e")
        db.execute.side_effect = [
            _all_runs([_run("r1"), _run("r2")]),
            _rows([("r1", "ok", SCORED), ("r2", "ok", SCORED)]),
        ]

        with patch("app.workers.tasks.diagnose_run.delay",
                   side_effect=[ConnectionError("broker down"), MagicMock()]) as diag:
            resp = client.post("/api/v1/diagnostics/experiment/e1/rediagnose")

        assert resp.status_code == 200, resp.text
        assert diag.call_count == 2
        body = resp.json()
        assert (body["enqueued"], body["skipped"], body["errors"]) == (1, 0, 1)
        assert body["runs"][0]["status"] == "error"
        assert "broker down" in body["runs"][0]["reason"]
        assert body["runs"][1]["status"] == "enqueued"


# ─── scripts/rediagnose.py ───────────────────────────────────────────────────

SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "rediagnose.py"


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("faithtrace_rediagnose_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _urlopen(outcomes):
    """urlopen stand-in returning (or raising) each outcome in turn and recording requests."""
    calls = []
    queue = list(outcomes)

    def fake(req, timeout=None):
        calls.append((req.get_method(), req.full_url, req.get_header("X-api-key")))
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)

    fake.calls = calls
    return fake


def _summary(experiment_id, runs, errors=0):
    return {
        "experiment_id": experiment_id,
        "eval_set_id": None,
        "eval_set_path": "eval_sets/faithtrace_v1.json",
        "enqueued": sum(1 for r in runs if r["status"] == "enqueued"),
        "skipped": sum(1 for r in runs if r["status"] == "skipped"),
        "errors": errors,
        "will_rescore_queries": sum(r.get("will_rescore_queries", 0) for r in runs),
        "runs": runs,
    }


ENQUEUED = {"run_id": "r1", "run_status": "done", "status": "enqueued",
            "ok_queries": 4, "queries_missing_scores": 1, "will_rescore_queries": 4}
SKIPPED = {"run_id": "r2", "run_status": "running", "status": "skipped",
           "reason": "run status is 'running', must be 'done'"}


class TestRediagnoseScript:

    def test_calls_experiment_endpoint_and_prints_summary(self, script, capsys, monkeypatch):
        monkeypatch.setenv("API_KEY", "secret")
        fake = _urlopen([_summary("e1", [ENQUEUED, SKIPPED]), _summary("e2", [])])
        with patch("urllib.request.urlopen", fake):
            code = script.main(["--experiment", "e1", "--experiment", "e2", "--api-url", "http://api.test/"])

        assert code == 0
        assert fake.calls == [
            ("POST", "http://api.test/api/v1/diagnostics/experiment/e1/rediagnose", "secret"),
            ("POST", "http://api.test/api/v1/diagnostics/experiment/e2/rediagnose", "secret"),
        ]
        out = capsys.readouterr().out
        assert "Experiment e1" in out
        assert "1 of 4 ok queries lack stored scores" in out
        assert "run status is 'running', must be 'done'" in out
        assert "enqueued 1, skipped 1, errors 0, queries to re-score 4" in out

    def test_no_api_key_sends_no_header(self, script, monkeypatch):
        monkeypatch.delenv("API_KEY", raising=False)
        fake = _urlopen([_summary("e1", [])])
        with patch("urllib.request.urlopen", fake):
            assert script.main(["--experiment", "e1", "--api-url", "http://api.test"]) == 0
        assert fake.calls[0][2] is None

    def test_http_error_exits_nonzero_but_continues(self, script, capsys):
        url = "http://api.test/api/v1/diagnostics/experiment/missing/rediagnose"
        err = urllib.error.HTTPError(url, 404, "Not Found", Message(),
                                     io.BytesIO(b'{"detail": "Experiment not found"}'))
        fake = _urlopen([err, _summary("e1", [ENQUEUED])])
        with patch("urllib.request.urlopen", fake):
            code = script.main(["--experiment", "missing", "--experiment", "e1", "--api-url", "http://api.test"])

        assert code == 1
        assert len(fake.calls) == 2
        captured = capsys.readouterr()
        assert "HTTP 404" in captured.err and "Experiment not found" in captured.err
        assert "Experiment e1" in captured.out

    def test_enqueue_error_in_summary_exits_nonzero(self, script):
        run = {"run_id": "r1", "run_status": "done", "status": "error", "reason": "could not enqueue diagnosis: x"}
        fake = _urlopen([_summary("e1", [run], errors=1)])
        with patch("urllib.request.urlopen", fake):
            assert script.main(["--experiment", "e1", "--api-url", "http://api.test"]) == 1

    def test_unreachable_api_exits_nonzero(self, script, capsys):
        fake = _urlopen([urllib.error.URLError("connection refused")])
        with patch("urllib.request.urlopen", fake):
            assert script.main(["--experiment", "e1", "--api-url", "http://api.test"]) == 1
        assert "cannot reach http://api.test" in capsys.readouterr().err

    def test_experiment_is_required(self, script):
        with pytest.raises(SystemExit) as exc:
            script.main([])
        assert exc.value.code == 2
