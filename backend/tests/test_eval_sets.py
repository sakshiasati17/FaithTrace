"""
Eval set tests (Known Issue #8).

Covers:
  - item validation (good set, duplicate id, bad date, bad modality, bad failure_type)
  - CSV parsing (header row, ';'-separated source_docs, booleans)
  - legacy path restriction to eval_sets/
  - POST/GET/DELETE /api/v1/eval-sets
  - experiment creation with eval_set_id / eval_set_path
  - re-evaluation and training use the experiment's eval set
  - the worker helper prefers DB items over paths
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.db.models import EvalSet, Experiment, Run
from app.db.session import get_db
from app.main import app
from app.services.evaluation import eval_sets as es

REPO_EVAL_SETS = Path(__file__).resolve().parents[2] / "eval_sets"


def _item(**overrides):
    base = {
        "id": "q1",
        "question": "What changed?",
        "ground_truth": "The threshold.",
        "source_docs": ["policy_v2.pdf"],
        "valid_from": "2024-07-01",
        "valid_to": None,
        "modality": "table",
        "difficulty": "medium",
        "answerable": True,
        "failure_type": None,
    }
    base.update(overrides)
    return base


# ─── Validation ───────────────────────────────────────────────────────────────

class TestValidation:

    def test_good_items_pass(self):
        res = es.validate_items([_item(), _item(id="q2", failure_type="UNANSWERABLE", answerable=False)])
        assert res.ok
        assert [i["id"] for i in res.items] == ["q1", "q2"]

    def test_repo_faithtrace_v1_is_valid(self):
        items = json.loads((REPO_EVAL_SETS / "faithtrace_v1.json").read_text())
        res = es.validate_items(items)
        assert res.ok, res.errors[:3]
        assert len(res.items) == len(items)

    def test_duplicate_id(self):
        res = es.validate_items([_item(), _item()])
        assert not res.ok
        assert res.errors == [{
            "row": 2, "id": "q1", "field": "id",
            "message": "duplicate id 'q1' (first seen in row 1)",
        }]

    def test_bad_date(self):
        res = es.validate_items([_item(valid_from="01/07/2024")])
        assert [(e["row"], e["field"]) for e in res.errors] == [(1, "valid_from")]

    def test_valid_to_before_valid_from(self):
        res = es.validate_items([_item(valid_from="2024-07-01", valid_to="2024-01-01")])
        assert [e["field"] for e in res.errors] == ["valid_to"]

    def test_bad_modality(self):
        res = es.validate_items([_item(), _item(id="q2", modality="video")])
        assert [(e["row"], e["id"], e["field"]) for e in res.errors] == [(2, "q2", "modality")]

    def test_bad_failure_type(self):
        res = es.validate_items([_item(failure_type="MADE_UP")])
        assert [e["field"] for e in res.errors] == ["failure_type"]

    def test_every_failure_category_is_accepted(self):
        from app.services.diagnostics.classifier import FailureCategory
        items = [_item(id=f"q{i}", failure_type=c.value) for i, c in enumerate(FailureCategory)]
        assert es.validate_items(items).ok

    def test_missing_required_fields_reported_per_row(self):
        res = es.validate_items([{"id": "q1"}, "not an object"])
        fields = {(e["row"], e["field"]) for e in res.errors}
        assert (1, "question") in fields and (1, "ground_truth") in fields
        assert (2, None) in fields

    def test_empty_and_non_list(self):
        assert not es.validate_items([]).ok
        assert not es.validate_items({"id": "q1"}).ok


class TestParsing:

    def test_csv_parsing(self):
        csv_text = (
            "id,question,ground_truth,source_docs,valid_from,valid_to,modality,answerable,failure_type\n"
            "q1,What?,This.,a.pdf; b.xlsx,2024-01-01,,spreadsheet,false,UNANSWERABLE\n"
            "q2,Why?,Because.,,,,text,TRUE,\n"
        )
        items = es.parse_upload("set.csv", csv_text.encode("utf-8"))
        assert items[0]["source_docs"] == ["a.pdf", "b.xlsx"]
        assert items[0]["answerable"] is False
        assert "valid_to" not in items[0]
        assert items[1]["answerable"] is True
        assert "source_docs" not in items[1] and "failure_type" not in items[1]
        assert es.validate_items(items).ok

    def test_csv_with_bom_and_bad_boolean(self):
        csv_text = "﻿id,question,ground_truth,answerable\nq1,Q?,A.,maybe\n"
        items = es.parse_upload("x.csv", csv_text.encode("utf-8"))
        res = es.validate_items(items)
        assert [e["field"] for e in res.errors] == ["answerable"]

    def test_json_must_be_list(self):
        with pytest.raises(es.EvalSetError):
            es.parse_upload("x.json", b'{"id": "q1"}')

    def test_rejects_other_extensions(self):
        with pytest.raises(es.EvalSetError):
            es.parse_upload("x.txt", b"[]")


# ─── Path restriction ─────────────────────────────────────────────────────────

class TestPathRestriction:

    @pytest.mark.parametrize("path", [
        "../eval_sets/faithtrace_v1.json",
        "eval_sets/../backend/requirements.txt",
        "eval_sets/../../etc/passwd",
        "/etc/passwd",
        str(REPO_EVAL_SETS / "faithtrace_v1.json"),  # absolute, even inside the folder
        "backend/requirements.txt",
        "corpus/README.md",
        "eval_sets/sub/x.json",
        "eval_sets\\faithtrace_v1.json",
        "eval_sets/missing.json",
        "eval_sets/README",
        "",
    ])
    def test_rejected(self, path):
        with pytest.raises(es.EvalSetError):
            es.resolve_eval_set_path(path)

    @pytest.mark.parametrize("path", ["eval_sets/faithtrace_v1.json", "faithtrace_v1.json"])
    def test_accepted(self, path):
        normalised, target = es.resolve_eval_set_path(path)
        assert normalised == "eval_sets/faithtrace_v1.json"
        assert target == (REPO_EVAL_SETS / "faithtrace_v1.json").resolve()

    def test_symlink_out_of_folder_rejected(self, tmp_path, monkeypatch):
        outside = tmp_path / "secret.json"
        outside.write_text("[]")
        folder = tmp_path / "eval_sets"
        folder.mkdir()
        (folder / "link.json").symlink_to(outside)
        monkeypatch.setenv("EVAL_SETS_DIR", str(folder))
        with pytest.raises(es.EvalSetError):
            es.resolve_eval_set_path("eval_sets/link.json")

    def test_default_constant_exists(self):
        assert es.DEFAULT_EVAL_SET_PATH == "eval_sets/faithtrace_v1.json"
        es.resolve_eval_set_path(es.DEFAULT_EVAL_SET_PATH)

    def test_worker_loader_refuses_paths_outside_folder(self, caplog):
        from app.workers import tasks
        assert tasks._load_eval_set("../backend/requirements.txt") == []
        assert tasks._load_eval_set("/etc/passwd") == []
        assert "Could not load eval set" in caplog.text


# ─── API ──────────────────────────────────────────────────────────────────────

def _result(scalars=None, scalar_one=None):
    res = MagicMock()
    res.scalars.return_value.all.return_value = scalars or []
    res.scalar_one.return_value = scalar_one
    return res


@pytest.fixture
def db():
    session = MagicMock()
    session.execute = AsyncMock()
    session.get = AsyncMock(return_value=None)
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.delete = AsyncMock()
    return session


@pytest.fixture
def client(db):
    async def _override():
        yield db
    app.dependency_overrides[get_db] = _override
    with patch("app.main.settings.API_KEY", ""):
        yield TestClient(app)
    app.dependency_overrides.pop(get_db, None)


class TestUploadEndpoint:

    def test_upload_201_with_warnings(self, client, db):
        db.execute.return_value = _result(scalars=["policy_v2.pdf"])
        items = [_item(), _item(id="q2", source_docs=["missing_doc.pdf", "policy_v2.pdf"])]

        resp = client.post(
            "/api/v1/eval-sets/",
            files={"file": ("mine.json", json.dumps(items), "application/json")},
            data={"name": "My set", "description": "d"},
        )

        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["item_count"] == 2
        assert body["source"] == "upload"
        assert body["name"] == "My set"
        assert body["id"]
        assert body["warnings"] == ["source_docs not found in uploaded documents: missing_doc.pdf"]
        stored = db.add.call_args.args[0]
        assert isinstance(stored, EvalSet)
        assert stored.items == items and stored.item_count == 2
        db.commit.assert_awaited()

    def test_upload_csv(self, client, db):
        db.execute.return_value = _result(scalars=[])
        csv_text = "id,question,ground_truth,source_docs\nq1,Q?,A.,a.pdf;b.pdf\n"

        resp = client.post("/api/v1/eval-sets/", files={"file": ("s.csv", csv_text, "text/csv")})

        assert resp.status_code == 201, resp.text
        assert resp.json()["name"] == "s.csv"
        assert len(resp.json()["warnings"]) == 2

    def test_upload_422_with_row_errors(self, client, db):
        items = [_item(), _item(), _item(id="q3", modality="video", valid_from="yesterday")]

        resp = client.post("/api/v1/eval-sets/",
                           files={"file": ("bad.json", json.dumps(items), "application/json")})

        assert resp.status_code == 422
        errors = resp.json()["detail"]["errors"]
        assert {(e["row"], e["field"]) for e in errors} == {(2, "id"), (3, "modality"), (3, "valid_from")}
        db.add.assert_not_called()

    def test_upload_422_on_unparseable_file(self, client, db):
        resp = client.post("/api/v1/eval-sets/",
                           files={"file": ("bad.json", "{not json", "application/json")})
        assert resp.status_code == 422
        assert "invalid JSON" in resp.json()["detail"]["message"]


class TestListGetDelete:

    def test_list_includes_builtin_files_and_uploads(self, client, db):
        uploaded = EvalSet(id="u1", name="Uploaded", description="", source="upload",
                           filename="u.json", items=[_item()], item_count=1)
        db.execute.return_value = _result(scalars=[uploaded])

        resp = client.get("/api/v1/eval-sets/")

        assert resp.status_code == 200
        body = resp.json()
        builtin = {e["filename"]: e for e in body if e["source"] == "builtin"}
        assert set(builtin) >= {"faithtrace_v1.json", "golden_regression_set.json"}
        assert builtin["faithtrace_v1.json"]["id"] == "builtin:faithtrace_v1.json"
        assert builtin["faithtrace_v1.json"]["path"] == "eval_sets/faithtrace_v1.json"
        assert builtin["faithtrace_v1.json"]["is_default"] is True
        assert builtin["golden_regression_set.json"]["item_count"] == 5
        assert [e["id"] for e in body if e["source"] == "upload"] == ["u1"]

    def test_get_uploaded_with_items(self, client, db):
        db.get.return_value = EvalSet(id="u1", name="U", source="upload", items=[_item()], item_count=1)
        resp = client.get("/api/v1/eval-sets/u1")
        assert resp.status_code == 200
        assert resp.json()["items"][0]["id"] == "q1"

    def test_get_builtin_with_items(self, client):
        resp = client.get("/api/v1/eval-sets/builtin:golden_regression_set.json")
        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 5

    def test_get_builtin_traversal_is_404(self, client):
        resp = client.get("/api/v1/eval-sets/builtin:..%2Fbackend%2Frequirements.txt")
        assert resp.status_code == 404

    def test_get_missing_404(self, client, db):
        assert client.get("/api/v1/eval-sets/nope").status_code == 404

    def test_delete_409_when_in_use(self, client, db):
        db.get.return_value = EvalSet(id="u1", name="U", source="upload", items=[], item_count=0)
        db.execute.return_value = _result(scalar_one=2)

        resp = client.delete("/api/v1/eval-sets/u1")

        assert resp.status_code == 409
        db.delete.assert_not_called()

    def test_delete_204_when_unused(self, client, db):
        row = EvalSet(id="u1", name="U", source="upload", items=[], item_count=0)
        db.get.return_value = row
        db.execute.return_value = _result(scalar_one=0)

        resp = client.delete("/api/v1/eval-sets/u1")

        assert resp.status_code == 204
        db.delete.assert_awaited_once_with(row)

    def test_delete_builtin_refused(self, client):
        assert client.delete("/api/v1/eval-sets/builtin:faithtrace_v1.json").status_code == 400


# ─── Experiments ──────────────────────────────────────────────────────────────

class TestExperimentCreate:

    def _post(self, client, db, payload):
        captured = {}

        def _add(obj):
            if isinstance(obj, Experiment):
                captured["experiment"] = obj

        db.add.side_effect = _add

        def _exp_result():
            exp = captured["experiment"]
            exp.runs = []
            from datetime import datetime
            exp.created_at = datetime(2026, 1, 1)
            res = MagicMock()
            res.scalar_one.return_value = exp
            return res

        db.execute.side_effect = lambda *_a, **_k: _exp_result()
        with patch("app.workers.tasks.run_experiment.delay") as delay:
            resp = client.post("/api/v1/experiments/", json={"name": "e", **payload})
        return resp, captured.get("experiment"), delay

    def test_eval_set_id_is_stored_and_preferred(self, client, db):
        db.get.return_value = EvalSet(id="u1", name="U", source="upload", items=[], item_count=0)

        resp, exp, delay = self._post(client, db, {"eval_set_id": "u1", "eval_set_path": "eval_sets/sample_eval_set.json"})

        assert resp.status_code == 201, resp.text
        assert exp.eval_set_id == "u1" and exp.eval_set_path is None
        assert resp.json()["eval_set_id"] == "u1"
        delay.assert_called_once_with(exp.id, None)

    def test_unknown_eval_set_id_422(self, client, db):
        resp, exp, delay = self._post(client, db, {"eval_set_id": "nope"})
        assert resp.status_code == 422
        delay.assert_not_called()

    def test_builtin_id_maps_to_path(self, client, db):
        resp, exp, delay = self._post(client, db, {"eval_set_id": "builtin:golden_regression_set.json"})
        assert resp.status_code == 201, resp.text
        assert exp.eval_set_id is None
        assert exp.eval_set_path == "eval_sets/golden_regression_set.json"
        delay.assert_called_once_with(exp.id, "eval_sets/golden_regression_set.json")

    def test_default_path(self, client, db):
        resp, exp, _ = self._post(client, db, {})
        assert resp.status_code == 201, resp.text
        assert exp.eval_set_path == "eval_sets/faithtrace_v1.json"

    def test_legacy_path_normalised(self, client, db):
        resp, exp, _ = self._post(client, db, {"eval_set_path": "sample_eval_set.json"})
        assert resp.status_code == 201
        assert exp.eval_set_path == "eval_sets/sample_eval_set.json"

    @pytest.mark.parametrize("path", ["../.env", "/etc/passwd", "backend/requirements.txt",
                                      "eval_sets/../corpus/x.json"])
    def test_bad_legacy_path_422(self, client, db, path):
        resp, exp, delay = self._post(client, db, {"eval_set_path": path})
        assert resp.status_code == 422
        delay.assert_not_called()


class TestReevaluateAndTrain:

    def test_reevaluate_uses_experiments_path(self, client, db):
        run = Run(id="r1", experiment_id="e1", config={}, status="done")
        exp = Experiment(id="e1", name="e", eval_set_path="eval_sets/golden_regression_set.json")
        db.execute.return_value = MagicMock(scalar_one_or_none=MagicMock(return_value=run))
        db.get.return_value = exp

        with patch("app.workers.tasks.evaluate_run.delay") as delay:
            resp = client.post("/api/v1/evaluation/run/r1")

        assert resp.status_code == 200, resp.text
        delay.assert_called_once_with("r1", "eval_sets/golden_regression_set.json")
        assert resp.json()["eval_set_path"] == "eval_sets/golden_regression_set.json"

    def test_reevaluate_uploaded_set_passes_no_path(self, client, db):
        run = Run(id="r1", experiment_id="e1", config={}, status="done")
        exp = Experiment(id="e1", name="e", eval_set_id="u1")
        db.execute.return_value = MagicMock(scalar_one_or_none=MagicMock(return_value=run))
        db.get.return_value = exp

        with patch("app.workers.tasks.evaluate_run.delay") as delay:
            resp = client.post("/api/v1/evaluation/run/r1")

        # No path: the worker loads the experiment's uploaded set (eval_set_id u1).
        delay.assert_called_once_with("r1", None)
        assert resp.json()["eval_set_id"] == "u1"

    def test_train_defaults_to_experiments_eval_set(self, client, db):
        db.execute.return_value = MagicMock(scalar_one_or_none=MagicMock(return_value=Experiment(id="e1", name="e")))
        with patch("app.workers.tasks.train_failure_classifier.delay",
                   return_value=SimpleNamespace(id="t1")) as delay:
            resp = client.post("/api/v1/diagnostics/classifier/train", params={"experiment_id": "e1"})
        assert resp.status_code == 200, resp.text
        delay.assert_called_once_with("e1", None)

    def test_train_rejects_path_outside_folder(self, client, db):
        db.execute.return_value = MagicMock(scalar_one_or_none=MagicMock(return_value=Experiment(id="e1", name="e")))
        with patch("app.workers.tasks.train_failure_classifier.delay") as delay:
            resp = client.post("/api/v1/diagnostics/classifier/train",
                               params={"experiment_id": "e1", "eval_set_path": "../.env"})
        assert resp.status_code == 422
        delay.assert_not_called()


# ─── Worker helper ────────────────────────────────────────────────────────────

class TestLoadEvalSetFor:

    DB_ITEMS = [{"id": "db_q1", "question": "Q?", "ground_truth": "A."}]

    def _db(self, row):
        db = MagicMock()
        db.get.return_value = row
        return db

    def test_db_items_win_over_paths(self):
        from app.workers import tasks
        exp = Experiment(id="e1", name="e", eval_set_id="u1",
                         eval_set_path="eval_sets/golden_regression_set.json")
        db = self._db(EvalSet(id="u1", name="U", items=self.DB_ITEMS, item_count=1))
        with patch.object(tasks, "_load_eval_set") as load:
            items = tasks.load_eval_set_for(db, exp, "eval_sets/sample_eval_set.json")
        assert items == self.DB_ITEMS
        db.get.assert_called_once_with(EvalSet, "u1")
        load.assert_not_called()

    def test_missing_db_set_returns_empty(self, caplog):
        from app.workers import tasks
        exp = Experiment(id="e1", name="e", eval_set_id="gone")
        assert tasks.load_eval_set_for(self._db(None), exp) == []
        assert "not found" in caplog.text

    def test_stored_path_then_task_arg_then_default(self):
        from app.workers import tasks
        golden = tasks.load_eval_set_for(
            MagicMock(), Experiment(id="e", name="e", eval_set_path="eval_sets/golden_regression_set.json"),
            "eval_sets/procurement_policy_eval.json")
        assert len(golden) == 5
        legacy = tasks.load_eval_set_for(MagicMock(), Experiment(id="e", name="e"),
                                         "eval_sets/procurement_policy_eval.json")
        assert len(legacy) == 10
        default = tasks.load_eval_set_for(MagicMock(), Experiment(id="e", name="e"))
        assert default and default[0]["id"] == json.loads(
            (REPO_EVAL_SETS / "faithtrace_v1.json").read_text())[0]["id"]

    def test_evaluate_run_uses_experiments_uploaded_set(self):
        from app.workers import tasks
        exp = Experiment(id="e1", name="e", eval_set_id="u1")
        run = MagicMock(experiment=exp)
        db = MagicMock()
        db.get.side_effect = lambda model, key: run if model is Run else EvalSet(
            id="u1", name="U", items=self.DB_ITEMS, item_count=1)
        db.execute.return_value = MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[]))))

        with patch("app.db.session.get_sync_db", return_value=db), \
             patch.object(tasks, "_load_eval_set") as load:
            out = tasks.evaluate_run.run("r1")

        # Eval set loaded from the DB (no file read); stops at "no query results".
        load.assert_not_called()
        assert out == {"error": "No query results found for this run"}

    def test_run_experiment_fails_cleanly_when_set_missing(self):
        from app.workers import tasks
        exp = Experiment(id="e1", name="e", eval_set_path="eval_sets/missing.json", status="pending")
        db = MagicMock()
        db.get.return_value = exp
        with patch("app.db.session.get_sync_db", return_value=db):
            out = tasks.run_experiment.run("e1")
        assert exp.status == "failed"
        assert "eval_sets/missing.json" in out["error"]


class TestUploadLimits:

    def test_oversized_upload_422(self, client, db):
        with patch.object(es, "MAX_EVAL_SET_BYTES", 10):
            resp = client.post("/api/v1/eval-sets/",
                               files={"file": ("big.json", json.dumps([_item()]), "application/json")})
        assert resp.status_code == 422
        assert "larger than" in resp.json()["detail"]["message"]
        db.add.assert_not_called()
