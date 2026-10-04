"""
API integrity tests (fix/api-integrity).

Covers:
  1. the baseline config is part of the default MVP preset (and the full matrix)
  2. deleting a document keeps the row (503) when its chunks can't be removed
  3. feedback validates correct_label and rejects errored query results
  4. the reasoning agent marks unparseable output, doesn't cache it, and never
     invents a confidence; the /reason endpoint rejects errored queries, passes
     UNDIAGNOSED for undiagnosed ones and tolerates duplicate query ids
  5. classifier training labels: feedback > eval set failure_type > classifier
  6. the leaderboard rejects an unknown sort_by

OpenAI, Qdrant and Celery are mocked; training runs against a real SQLite DB.
"""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import (
    Base, Document, Experiment, QueryFeedback, QueryResult as QueryResultModel, Run,
)
from app.db.session import get_db
from app.main import app
from app.services.diagnostics.classifier import FailureCategory


# ─── API plumbing ─────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    session = MagicMock()
    session.execute = AsyncMock()
    session.get = AsyncMock(return_value=None)
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
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


def _one(obj):
    res = MagicMock()
    res.scalar_one_or_none.return_value = obj
    return res


def _rows(objs):
    res = MagicMock()
    res.scalars.return_value.all.return_value = list(objs)
    res.scalars.return_value.first.return_value = objs[0] if objs else None
    return res


# ─── 1. Baseline is in the default preset ─────────────────────────────────────

class TestBaselineInPreset:

    def test_mvp_matrix_contains_baseline_exactly_once(self):
        from dataclasses import asdict
        from app.services.evaluation.baseline import is_baseline_config
        from app.services.experiment.config_matrix import build_mvp_matrix
        matrix = build_mvp_matrix()
        assert len(matrix) == 24
        assert sum(is_baseline_config(asdict(c)) for c in matrix) == 1

    def test_full_matrix_contains_baseline_exactly_once(self):
        from dataclasses import asdict
        from app.services.evaluation.baseline import is_baseline_config
        from app.services.experiment.config_matrix import build_matrix
        matrix = build_matrix()
        assert len(matrix) == 256
        assert sum(is_baseline_config(asdict(c)) for c in matrix) == 1

    def test_baseline_comparison_runs_on_mvp_experiment(self):
        from dataclasses import asdict
        from app.services.evaluation.baseline import compare_run_to_baseline
        from app.services.experiment.config_matrix import build_mvp_matrix
        runs = [
            {"run_id": f"r{i}", "config": asdict(c),
             "metrics": {"faithfulness": 0.5 + i / 100, "answer_correctness": 0.5,
                         "context_recall": 0.5}}
            for i, c in enumerate(build_mvp_matrix())
        ]
        result = compare_run_to_baseline(runs)
        assert "error" not in result
        assert result["baseline"]["config"]["chunking_strategy"] == "recursive"
        assert result["baseline"]["config"]["retrieval_strategy"] == "vector_only"


# ─── 2. Document deletion ────────────────────────────────────────────────────

class TestDeleteDocument:

    def _doc(self, tmp_path):
        stored = tmp_path / "doc1" / "policy.pdf"
        stored.parent.mkdir()
        stored.write_bytes(b"%PDF")
        return Document(id="doc1", filename="policy.pdf", file_type="pdf",
                        storage_path=str(stored)), stored

    def test_chunk_delete_failure_keeps_row_and_returns_503(self, client, db, tmp_path):
        doc, stored = self._doc(tmp_path)
        db.execute.return_value = _one(doc)
        with patch("app.services.ingestion.indexer.delete_doc_chunks",
                   side_effect=ConnectionError("qdrant down")):
            resp = client.delete("/api/v1/corpus/doc1")
        assert resp.status_code == 503
        assert "not deleted" in resp.json()["detail"]
        db.delete.assert_not_awaited()
        db.commit.assert_not_awaited()
        assert stored.exists(), "the stored file must be kept so a retry can succeed"

    def test_success_deletes_chunks_file_and_row(self, client, db, tmp_path):
        doc, stored = self._doc(tmp_path)
        db.execute.return_value = _one(doc)
        with patch("app.services.ingestion.indexer.delete_doc_chunks") as delete_chunks:
            resp = client.delete("/api/v1/corpus/doc1")
        assert resp.status_code == 204
        delete_chunks.assert_called_once_with("doc1")
        db.delete.assert_awaited_once_with(doc)
        db.commit.assert_awaited_once()
        assert not stored.parent.exists()

    def test_file_removal_failure_is_only_a_warning(self, client, db, tmp_path, caplog):
        doc, _ = self._doc(tmp_path)
        db.execute.return_value = _one(doc)
        with patch("app.services.ingestion.indexer.delete_doc_chunks"), \
                patch("shutil.rmtree", side_effect=PermissionError("ro fs")):
            resp = client.delete("/api/v1/corpus/doc1")
        assert resp.status_code == 204
        db.delete.assert_awaited_once_with(doc)
        assert "Could not remove stored file" in caplog.text

    def test_missing_document_404(self, client, db):
        db.execute.return_value = _one(None)
        resp = client.delete("/api/v1/corpus/nope")
        assert resp.status_code == 404


# ─── 3. Feedback validation ──────────────────────────────────────────────────

def _qr(status="ok", failure_category="LOW_RECALL_RETRIEVAL", **kw):
    base = dict(id="qr1", run_id="run1", query_id="q1", question="Q?",
                generated_answer="" if status == "error" else "A.",
                retrieved_chunks=[], status=status, failure_category=failure_category,
                diagnosis_evidence={"faithfulness": 0.2})
    base.update(kw)
    return QueryResultModel(**base)


class TestFeedback:

    def test_invalid_correct_label_rejected(self, client, db):
        db.get.return_value = _qr()
        resp = client.post("/api/v1/feedback/qr1",
                           json={"rating": "negative", "correct_label": "MADE_UP"})
        assert resp.status_code == 422
        assert "LOW_RECALL_RETRIEVAL" in resp.json()["detail"]
        db.add.assert_not_called()
        db.commit.assert_not_awaited()

    @pytest.mark.parametrize("label", [None, "TABLE_RETRIEVAL_MISS", "NO_FAILURE"])
    def test_valid_or_null_label_accepted(self, client, db, label):
        db.get.return_value = _qr()
        db.execute.return_value = _one(None)
        resp = client.post("/api/v1/feedback/qr1",
                           json={"rating": "negative", "correct_label": label})
        assert resp.status_code == 200, resp.text
        assert resp.json()["correct_label"] == label
        added = db.add.call_args.args[0]
        assert added.correct_label == label
        db.commit.assert_awaited_once()

    def test_unanswerable_is_not_a_label(self, client, db):
        # UNANSWERABLE is an eval-set failure_type, not a FailureCategory.
        db.get.return_value = _qr()
        resp = client.post("/api/v1/feedback/qr1",
                           json={"rating": "negative", "correct_label": "UNANSWERABLE"})
        assert resp.status_code == 422

    def test_errored_query_rejected(self, client, db):
        db.get.return_value = _qr(status="error", failure_category=None,
                                  error_message="ConnectionError")
        resp = client.post("/api/v1/feedback/qr1", json={"rating": "positive"})
        assert resp.status_code == 422
        assert "errored" in resp.json()["detail"]
        db.add.assert_not_called()

    def test_missing_query_result_404(self, client, db):
        resp = client.post("/api/v1/feedback/nope", json={"rating": "positive"})
        assert resp.status_code == 404


# ─── 4. Reasoning agent ──────────────────────────────────────────────────────

def _openai_returning(content):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return MagicMock(return_value=client), client


async def _call_reason(content, category="LOW_RECALL_RETRIEVAL"):
    from app.services.diagnostics import reasoning_agent
    factory, client = _openai_returning(content)
    with patch("openai.AsyncOpenAI", factory):
        out = await reasoning_agent.reason("Q?", "A.", [{"content": "ctx"}], {"faithfulness": 0.2}, category)
    return out, client


class TestReasoningAgent:

    @pytest.mark.asyncio
    async def test_invalid_json_is_marked_parse_error_without_confidence(self):
        out, _ = await _call_reason("not json at all")
        assert out["parse_error"] is True
        assert out["confidence"] is None
        assert out["root_cause"] == "not json at all"

    @pytest.mark.asyncio
    async def test_non_object_json_is_a_parse_error(self):
        out, _ = await _call_reason("[1, 2]")
        assert out["parse_error"] is True

    @pytest.mark.asyncio
    async def test_missing_confidence_stays_none(self):
        out, _ = await _call_reason(json.dumps({"root_cause": "retriever missed the table"}))
        assert out["parse_error"] is False
        assert out["confidence"] is None
        assert out["root_cause"] == "retriever missed the table"
        assert out["reasoning_steps"] == []

    @pytest.mark.asyncio
    async def test_reported_confidence_kept(self):
        out, _ = await _call_reason(json.dumps({"root_cause": "x", "confidence": 0.8}))
        assert out["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_undiagnosed_has_a_description(self):
        from app.services.diagnostics.reasoning_agent import _FAILURE_DESCRIPTIONS
        assert "UNDIAGNOSED" in _FAILURE_DESCRIPTIONS
        _, client = await _call_reason(json.dumps({"confidence": 0.5}), category="UNDIAGNOSED")
        prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "Category: UNDIAGNOSED" in prompt
        assert _FAILURE_DESCRIPTIONS["UNDIAGNOSED"] in prompt


REASON_URL = "/api/v1/diagnostics/run/run1/query/q1/reason"
GOOD_REASONING = {"reasoning_steps": ["s"], "root_cause": "r", "fix_suggestion": "f",
                  "stakeholder_summary": "s", "confidence": 0.7, "parse_error": False}
BAD_REASONING = {"reasoning_steps": [], "root_cause": "raw", "fix_suggestion": "",
                 "stakeholder_summary": "", "confidence": None, "parse_error": True}


class TestReasonEndpoint:

    def test_parse_error_not_cached(self, client, db):
        qr = _qr()
        db.execute.return_value = _rows([qr])
        with patch("app.services.diagnostics.reasoning_agent.reason",
                   AsyncMock(return_value=BAD_REASONING)):
            resp = client.post(REASON_URL)
        assert resp.status_code == 200
        assert resp.json()["reasoning"]["parse_error"] is True
        assert "reasoning" not in qr.diagnosis_evidence
        db.commit.assert_not_awaited()

    def test_good_result_cached_then_served(self, client, db):
        qr = _qr()
        db.execute.return_value = _rows([qr])
        llm = AsyncMock(return_value=GOOD_REASONING)
        with patch("app.services.diagnostics.reasoning_agent.reason", llm):
            first = client.post(REASON_URL)
            second = client.post(REASON_URL)
        assert first.json()["cached"] is False
        assert second.json()["cached"] is True
        assert qr.diagnosis_evidence["reasoning"] == GOOD_REASONING
        llm.assert_awaited_once()

    def test_legacy_cached_parse_error_is_retried(self, client, db):
        qr = _qr(diagnosis_evidence={"reasoning": BAD_REASONING})
        db.execute.return_value = _rows([qr])
        llm = AsyncMock(return_value=GOOD_REASONING)
        with patch("app.services.diagnostics.reasoning_agent.reason", llm):
            resp = client.post(REASON_URL)
        assert resp.json()["cached"] is False
        llm.assert_awaited_once()

    def test_errored_query_rejected(self, client, db):
        db.execute.return_value = _rows([_qr(status="error", failure_category=None)])
        llm = AsyncMock(return_value=GOOD_REASONING)
        with patch("app.services.diagnostics.reasoning_agent.reason", llm):
            resp = client.post(REASON_URL)
        assert resp.status_code == 422
        llm.assert_not_awaited()

    def test_null_category_passed_as_undiagnosed(self, client, db):
        db.execute.return_value = _rows([_qr(failure_category=None)])
        llm = AsyncMock(return_value=GOOD_REASONING)
        with patch("app.services.diagnostics.reasoning_agent.reason", llm):
            resp = client.post(REASON_URL)
        assert resp.status_code == 200
        assert llm.call_args.kwargs["failure_category"] == "UNDIAGNOSED"

    def test_duplicate_query_ids_do_not_500(self, client, db):
        errored = _qr(id="qr0", status="error", failure_category=None)
        ok = _qr(id="qr1")
        db.execute.return_value = _rows([errored, ok])
        llm = AsyncMock(return_value=GOOD_REASONING)
        with patch("app.services.diagnostics.reasoning_agent.reason", llm):
            resp = client.post(REASON_URL)
        assert resp.status_code == 200, resp.text
        assert ok.diagnosis_evidence["reasoning"] == GOOD_REASONING

    def test_missing_query_404(self, client, db):
        db.execute.return_value = _rows([])
        resp = client.post(REASON_URL)
        assert resp.status_code == 404


# ─── 5. Classifier training labels ───────────────────────────────────────────

def _fb(rating="negative", correct_label=None):
    return SimpleNamespace(id="fb", rating=rating, correct_label=correct_label)


class TestResolveTrainingLabel:

    def _resolve(self, *a):
        from app.workers.tasks import resolve_training_label
        return resolve_training_label(*a)

    def test_positive_feedback_wins(self):
        assert self._resolve(_fb("positive"), {"failure_type": "WRONG_VERSION"}, "STALE_ANSWER") \
            == ("NO_FAILURE", "feedback")

    def test_feedback_label_beats_eval_set(self):
        assert self._resolve(_fb(correct_label="STALE_ANSWER"), {"failure_type": "WRONG_VERSION"},
                             "LOW_RECALL_RETRIEVAL") == ("STALE_ANSWER", "feedback")

    def test_eval_set_beats_classifier(self):
        assert self._resolve(None, {"failure_type": "WRONG_VERSION"}, "LOW_RECALL_RETRIEVAL") \
            == ("WRONG_VERSION", "eval_set")

    def test_negative_feedback_without_label_falls_through(self):
        assert self._resolve(_fb(), {"failure_type": "WRONG_VERSION"}, "STALE_ANSWER") \
            == ("WRONG_VERSION", "eval_set")

    def test_invalid_feedback_label_ignored(self):
        assert self._resolve(_fb(correct_label="BOGUS"), {}, "STALE_ANSWER") \
            == ("STALE_ANSWER", "classifier")

    def test_eval_set_does_not_relabel_a_successful_run(self):
        # failure_type is what the question probes, not what happened: a run the
        # classifier judged NO_FAILURE stays NO_FAILURE.
        assert self._resolve(None, {"failure_type": "WRONG_VERSION"}, "NO_FAILURE") \
            == ("NO_FAILURE", "classifier")

    def test_eval_set_no_failure_does_not_hide_an_observed_failure(self):
        assert self._resolve(None, {"failure_type": "NO_FAILURE"}, "STALE_ANSWER") \
            == ("STALE_ANSWER", "classifier")

    def test_eval_set_ignored_when_run_undiagnosed(self):
        assert self._resolve(None, {"failure_type": "WRONG_VERSION"}, None) \
            == (None, "skipped_no_label")

    def test_unanswerable_skipped(self):
        assert self._resolve(None, {"failure_type": "UNANSWERABLE"}, "LOW_RECALL_RETRIEVAL") \
            == (None, "skipped_unanswerable")

    def test_classifier_fallback(self):
        assert self._resolve(None, {"failure_type": None}, "STALE_ANSWER") == ("STALE_ANSWER", "classifier")
        assert self._resolve(None, {}, None) == (None, "skipped_no_label")

    def test_describe_label_sources(self):
        from app.workers.tasks import describe_label_sources
        dominant, note = describe_label_sources({"feedback": 1, "eval_set": 6, "classifier": 3})
        assert dominant == "eval_set"
        assert "Dominant source: eval_set (60%)" in note
        dominant, note = describe_label_sources({"classifier": 4})
        assert dominant == "classifier" and "circular" in note
        assert describe_label_sources({}) == (None, "No training labels.")


TRAIN_EVAL_SET = (
    [{"id": f"wv{i}", "question": "?", "ground_truth": "a", "failure_type": "WRONG_VERSION"} for i in range(8)]
    + [{"id": "plain", "question": "?", "ground_truth": "a"}]
    + [{"id": "fbq", "question": "?", "ground_truth": "a", "failure_type": "WRONG_VERSION"}]
    + [{"id": "unans", "question": "?", "ground_truth": "", "failure_type": "UNANSWERABLE"}]
)


class TestTrainFailureClassifier:

    @pytest.fixture
    def Session(self, tmp_path):
        engine = create_engine(f"sqlite:///{tmp_path / 'train.db'}")
        Base.metadata.create_all(engine)
        yield sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
        engine.dispose()

    def _seed(self, Session):
        with Session() as db:
            exp = Experiment(id=str(uuid.uuid4()), name="exp", description="", status="done")
            run = Run(id=str(uuid.uuid4()), experiment_id=exp.id, config={}, status="done")
            db.add_all([exp, run])
            ids = [item["id"] for item in TRAIN_EVAL_SET]
            for qid in ids:
                db.add(QueryResultModel(id=f"qr_{qid}", run_id=run.id, query_id=qid, question="?",
                                        generated_answer="a", status="ok",
                                        failure_category="LOW_RECALL_RETRIEVAL",
                                        diagnosis_evidence={"faithfulness": 0.3}))
            # Errored rows never train (no answer); give it a stale label to prove it.
            db.add(QueryResultModel(id="qr_err", run_id=run.id, query_id="wv0", question="?",
                                    generated_answer="", status="error",
                                    failure_category="LOW_RECALL_RETRIEVAL"))
            db.add(QueryFeedback(query_result_id="qr_fbq", rating="negative",
                                 correct_label="STALE_ANSWER"))
            db.commit()
            return exp.id

    def test_labels_follow_priority_and_sources_are_reported(self, Session):
        from app.workers import tasks
        exp_id = self._seed(Session)
        ml_train = MagicMock(return_value={"accuracy": 0.9, "n_samples": 10})
        with patch("app.db.session.get_sync_db", side_effect=lambda: Session()), \
                patch.object(tasks, "load_eval_set_for", return_value=TRAIN_EVAL_SET), \
                patch("app.services.diagnostics.ml_classifier.train", ml_train), \
                patch("app.services.diagnostics.ml_classifier.reload_model"):
            out = tasks.train_failure_classifier.run(exp_id)

        labels_by_qid = {item.get("id"): label for item, label in
                         zip(ml_train.call_args.args[2], ml_train.call_args.args[3])}
        assert labels_by_qid["wv0"] == "WRONG_VERSION"           # eval set over classifier
        assert labels_by_qid["fbq"] == "STALE_ANSWER"            # feedback over eval set
        assert labels_by_qid["plain"] == "LOW_RECALL_RETRIEVAL"  # unlabelled item -> classifier
        assert "unans" not in labels_by_qid                      # unanswerable skipped
        assert len(ml_train.call_args.args[3]) == 10             # errored + unanswerable dropped

        assert out["label_sources"] == {
            "feedback": 1, "eval_set": 8, "classifier": 1,
            "skipped_unanswerable": 1, "skipped_no_label": 0, "skipped_errored": 1,
        }
        assert out["dominant_label_source"] == "eval_set"
        assert "Dominant source: eval_set" in out["label_source_summary"]
        assert out["samples_used"] == 10
        assert out["accuracy"] == 0.9


# ─── 6. Leaderboard sort ─────────────────────────────────────────────────────

class TestLeaderboardSort:

    def test_unknown_sort_by_rejected(self, client, db):
        resp = client.get("/api/v1/evaluation/leaderboard", params={"sort_by": "speed"})
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "speed" in detail
        for field in ("faithfulness", "latency_p50_ms", "multimodal_grounding_rate"):
            assert field in detail
        db.execute.assert_not_awaited()

    @pytest.mark.parametrize("sort_by", [None, "faithfulness", "avg_cost_usd"])
    def test_valid_sort_by_accepted(self, client, db, sort_by):
        res = MagicMock()
        res.all.return_value = []
        db.execute.return_value = res
        params = {"sort_by": sort_by} if sort_by else {}
        resp = client.get("/api/v1/evaluation/leaderboard", params=params)
        assert resp.status_code == 200
        assert resp.json() == []
