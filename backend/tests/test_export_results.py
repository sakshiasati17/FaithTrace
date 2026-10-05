"""
Tests for scripts/export_results.py.

The fixture JSON is built from the API's Pydantic response schemas so the
export is tested against the real field names, and urlopen is patched to
serve it.
"""

import csv
import importlib.util
import io
import json
import urllib.error
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from app.schemas.experiment_schemas import (
    ExperimentResponse,
    QueryResultResponse,
    RunMetricsResponse,
    RunResponse,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_results.py"
_spec = importlib.util.spec_from_file_location("export_results", SCRIPT)
export_results = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_results)

EXP_ID = "exp-1"
RUN_A = "run-a"
RUN_B = "run-b"

EVAL_ITEMS = [
    {"id": "q1", "question": "Q1", "ground_truth": "A1", "source_docs": ["a.html"],
     "valid_from": None, "valid_to": None, "modality": "text", "difficulty": "easy",
     "answerable": True, "failure_type": "WRONG_VERSION", "temporal_pair_id": "tp_01"},
    {"id": "q2", "question": "Q2", "ground_truth": "A2", "source_docs": ["b.html"],
     "valid_from": None, "valid_to": None, "modality": "text", "difficulty": "easy",
     "answerable": True, "failure_type": "STALE_ANSWER", "temporal_pair_id": "tp_01"},
    {"id": "q3", "question": "Q3", "ground_truth": "A3", "source_docs": ["c.csv"],
     "valid_from": None, "valid_to": None, "modality": "table", "difficulty": "medium",
     "answerable": True, "failure_type": None},
    {"id": "q4", "question": "Q4", "ground_truth": "A4", "source_docs": ["c.csv"],
     "valid_from": None, "valid_to": None, "modality": "table", "difficulty": "hard",
     "answerable": False, "failure_type": None},
]


def _scores(ac, faith=0.5, cp=0.5, cr=0.5, ar=0.5):
    return {"answer_correctness": ac, "faithfulness": faith, "context_precision": cp,
            "context_recall": cr, "answer_relevancy": ar}


def _qr(run_id, qid, category, scores, status="ok", error_message=None):
    evidence = {"scores": scores} if scores is not None else {}
    if status == "error":
        evidence = {"skipped": "query errored"}
    return QueryResultResponse(
        id=f"{run_id}-{qid}", run_id=run_id, query_id=qid, question=qid.upper(),
        generated_answer="" if status == "error" else f"answer {qid}",
        retrieved_chunks=[], latency_ms=100.0, input_tokens=10, output_tokens=5,
        cost_usd=0.001, failure_category=category, diagnosis_evidence=evidence,
        status=status, error_message=error_message,
    ).model_dump(mode="json")


def _run(run_id, retrieval, metrics, status="done"):
    return RunResponse(
        id=run_id, experiment_id=EXP_ID, status=status,
        config={"retrieval_strategy": retrieval, "chunking_strategy": "recursive",
                "parsing_strategy": "text_only", "freshness_policy": "none",
                "embedding_model": "text-embedding-3-small", "llm_model": "gpt-4o-mini",
                "top_k": 5, "reranker_enabled": False, "prompt_template": "default"},
        created_at=datetime(2026, 1, 1), completed_at=datetime(2026, 1, 1, 1),
        metrics=RunMetricsResponse(id=f"m-{run_id}", run_id=run_id, **metrics) if metrics else None,
    ).model_dump(mode="json")


def _complete_api():
    experiment = ExperimentResponse(
        id=EXP_ID, name="demo", description="", status="done",
        created_at=datetime(2026, 1, 1), eval_set_path="eval_sets/test_set.json",
        runs=[],
    ).model_dump(mode="json")
    experiment["runs"] = [
        _run(RUN_A, "vector_only", {"answer_correctness": 0.6, "faithfulness": 0.9,
                                    "latency_p50_ms": 100.0}),
        _run(RUN_B, "hybrid", {"answer_correctness": 0.4}),
    ]
    traces = {
        RUN_A: [
            _qr(RUN_A, "q1", "NO_FAILURE", _scores(1.0, faith=0.8)),
            _qr(RUN_A, "q2", "STALE_ANSWER", _scores(0.2, faith=None)),
            _qr(RUN_A, "q3", "TABLE_RETRIEVAL_MISS", _scores(0.6)),
            _qr(RUN_A, "q4", "NO_FAILURE", _scores(0.0)),
        ],
        RUN_B: [
            _qr(RUN_B, "q1", "WRONG_VERSION", _scores(0.1)),
            _qr(RUN_B, "q2", "WRONG_VERSION", _scores(0.3)),
            _qr(RUN_B, "q3", "NO_FAILURE", _scores(0.9)),
            _qr(RUN_B, "q4", "NO_FAILURE", _scores(0.5)),
        ],
    }
    return experiment, traces


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _serve(experiment, traces, seen_headers=None):
    def fake_urlopen(req, timeout=None):
        if seen_headers is not None:
            seen_headers.append(dict(req.header_items()))
        url = req.full_url
        prefix = f"http://api.test/api/v1/experiments/{EXP_ID}"
        if url == prefix:
            return _Resp(json.dumps(experiment).encode())
        for run_id, trace in traces.items():
            if url == f"{prefix}/runs/{run_id}/trace":
                return _Resp(json.dumps(trace).encode())
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b'{"detail":"nf"}'))
    return patch.object(export_results.urllib.request, "urlopen", side_effect=fake_urlopen)


@pytest.fixture
def eval_set(tmp_path):
    path = tmp_path / "test_set.json"
    path.write_text(json.dumps(EVAL_ITEMS))
    return path


def _run_export(eval_set, out, experiment, traces, *extra):
    with _serve(experiment, traces):
        return export_results.main([
            "--experiment", EXP_ID, "--eval-set", str(eval_set), "--out", str(out),
            "--api-url", "http://api.test", *extra,
        ])


def _read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_complete_export_writes_files_with_correct_numbers(eval_set, tmp_path):
    experiment, traces = _complete_api()
    out = tmp_path / "results" / "demo"
    assert _run_export(eval_set, out, experiment, traces) == 0

    assert json.loads((out / "raw" / "experiment.json").read_text())["id"] == EXP_ID
    assert len(json.loads((out / "raw" / "traces" / f"{RUN_A}.json").read_text())) == 4

    runs = {r["run_id"]: r for r in _read_csv(out / "runs.csv")}
    assert runs[RUN_A]["retrieval_strategy"] == "vector_only"
    assert runs[RUN_A]["llm_model"] == "gpt-4o-mini"
    assert runs[RUN_A]["answer_correctness"] == "0.6"
    assert runs[RUN_A]["latency_p50_ms"] == "100.0"
    assert runs[RUN_B]["faithfulness"] == ""  # not stored -> empty, not 0
    assert (runs[RUN_A]["n_ok"], runs[RUN_A]["n_error"], runs[RUN_A]["n_unscored"]) == ("4", "0", "0")

    mod = {(r["run_id"], r["group"], r["value"]): r for r in _read_csv(out / "by_modality.csv")}
    text_a = mod[(RUN_A, "modality", "text")]
    assert text_a["n"] == "2" and text_a["n_scored"] == "2"
    assert float(text_a["mean_answer_correctness"]) == pytest.approx(0.6)
    # q2's faithfulness is None: the mean is over q1 only, not (0.8 + 0) / 2.
    assert float(text_a["mean_faithfulness"]) == pytest.approx(0.8)
    assert text_a["n_faithfulness"] == "1"
    unans_b = mod[(RUN_B, "answerable", "false")]
    assert unans_b["n"] == "1" and float(unans_b["mean_answer_correctness"]) == pytest.approx(0.5)

    fails = {(r["run_id"], r["failure_category"]): r for r in _read_csv(out / "failures.csv")}
    assert fails[(RUN_B, "WRONG_VERSION")]["count"] == "2"
    assert fails[(RUN_A, "STALE_ANSWER")]["count"] == "1"
    assert fails[(RUN_A, "NO_FAILURE")]["n_diagnosed"] == "4"

    temporal = {r["run_id"]: r for r in _read_csv(out / "temporal.csv")}
    assert temporal[RUN_A]["n"] == "2"
    assert float(temporal[RUN_A]["mean_answer_correctness"]) == pytest.approx(0.6)
    assert temporal[RUN_A]["count_STALE_ANSWER"] == "1"
    assert temporal[RUN_B]["count_WRONG_VERSION"] == "2"

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["experiment_id"] == EXP_ID and manifest["experiment_name"] == "demo"
    assert manifest["eval_set_items"] == 4
    assert len(manifest["eval_set_sha256"]) == 64
    assert manifest["api_url"] == "http://api.test"
    assert manifest["generated_at"].endswith("Z")
    assert manifest["script_version"] == export_results.SCRIPT_VERSION

    summary = (out / "summary.md").read_text()
    assert "INCOMPLETE" not in summary
    assert "## Completeness" in summary and "| run_id |" in summary


def test_api_key_sent_as_header(eval_set, tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    experiment, traces = _complete_api()
    seen = []
    with _serve(experiment, traces, seen):
        assert export_results.main([
            "--experiment", EXP_ID, "--eval-set", str(eval_set), "--out", str(tmp_path / "o"),
            "--api-url", "http://api.test/",
        ]) == 0
    assert seen and all(h.get("X-api-key") == "secret" for h in seen)


def test_errored_query_exits_2_and_writes_nothing(eval_set, tmp_path, capsys):
    experiment, traces = _complete_api()
    traces[RUN_B][1] = _qr(RUN_B, "q2", None, None, status="error", error_message="timeout")
    out = tmp_path / "out"
    assert _run_export(eval_set, out, experiment, traces) == 2
    assert not out.exists()
    assert list(tmp_path.iterdir()) == [eval_set]  # no temp dir left behind
    printed = capsys.readouterr().out
    assert "Errored queries: 1" in printed and "timeout" in printed


@pytest.mark.parametrize("mutate, expected", [
    (lambda exp, tr: exp["runs"][1].update(status="failed"), "Runs not done: 1"),
    (lambda exp, tr: tr[RUN_A][0]["diagnosis_evidence"]["scores"].update(answer_correctness=None),
     "without an answer_correctness score: 1"),
    (lambda exp, tr: tr[RUN_A][0].update(failure_category=None), "(undiagnosed): 1"),
    (lambda exp, tr: tr[RUN_A].pop(), "have no result: 1"),
])
def test_each_incomplete_kind_blocks_export(eval_set, tmp_path, capsys, mutate, expected):
    experiment, traces = _complete_api()
    mutate(experiment, traces)
    out = tmp_path / "out"
    assert _run_export(eval_set, out, experiment, traces) == 2
    assert not out.exists()
    assert expected in capsys.readouterr().out


def test_allow_incomplete_writes_banner_and_flag(eval_set, tmp_path):
    experiment, traces = _complete_api()
    traces[RUN_B][1] = _qr(RUN_B, "q2", None, None, status="error", error_message="timeout")
    out = tmp_path / "out"
    assert _run_export(eval_set, out, experiment, traces, "--allow-incomplete") == 0

    summary = (out / "summary.md").read_text()
    assert summary.startswith("> **INCOMPLETE**")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["complete"] is False
    assert manifest["incomplete_counts"]["errored_queries"] == 1

    runs = {r["run_id"]: r for r in _read_csv(out / "runs.csv")}
    assert (runs[RUN_B]["n_ok"], runs[RUN_B]["n_error"]) == ("3", "1")
    temporal = {r["run_id"]: r for r in _read_csv(out / "temporal.csv")}
    # Errored q2 is excluded: mean over q1 only, with its n.
    assert temporal[RUN_B]["n"] == "2" and temporal[RUN_B]["n_error"] == "1"
    assert temporal[RUN_B]["n_scored"] == "1"
    assert float(temporal[RUN_B]["mean_answer_correctness"]) == pytest.approx(0.1)


def test_none_scores_never_become_zero(eval_set, tmp_path):
    experiment, traces = _complete_api()
    for qr in traces[RUN_B]:
        qr["diagnosis_evidence"]["scores"] = {k: None for k in export_results.RAGAS_KEYS}
    experiment["runs"][1]["metrics"] = None
    out = tmp_path / "out"
    assert _run_export(eval_set, out, experiment, traces, "--allow-incomplete") == 0

    runs = {r["run_id"]: r for r in _read_csv(out / "runs.csv")}
    assert all(runs[RUN_B][k] == "" for k in export_results.RUN_METRIC_KEYS)
    assert runs[RUN_B]["n_unscored"] == "4"
    for row in _read_csv(out / "by_modality.csv"):
        if row["run_id"] == RUN_B:
            assert row["n_scored"] == "0"
            for key in export_results.RAGAS_KEYS:
                assert row[f"mean_{key}"] == "" and row[f"n_{key}"] == "0"
    temporal = {r["run_id"]: r for r in _read_csv(out / "temporal.csv")}
    assert temporal[RUN_B]["mean_answer_correctness"] == ""

    summary = (out / "summary.md").read_text()
    run_b_line = next(l for l in summary.splitlines() if l.startswith(f"| {RUN_B} | hybrid"))
    assert "—" in run_b_line and "0.0000" not in run_b_line


def test_unmatched_query_id_is_an_error(eval_set, tmp_path, capsys):
    experiment, traces = _complete_api()
    traces[RUN_A].append(_qr(RUN_A, "q99", "NO_FAILURE", _scores(1.0)))
    out = tmp_path / "out"
    # Even --allow-incomplete does not let an unmatched id through.
    assert _run_export(eval_set, out, experiment, traces, "--allow-incomplete") == 1
    assert not out.exists()
    assert "'q99' matches no eval item" in capsys.readouterr().err


def test_refuses_non_empty_out_dir(eval_set, tmp_path):
    experiment, traces = _complete_api()
    out = tmp_path / "out"
    out.mkdir()
    (out / "old.csv").write_text("stale")
    assert _run_export(eval_set, out, experiment, traces) == 1
    assert [p.name for p in out.iterdir()] == ["old.csv"]


def test_http_error_is_reported(eval_set, tmp_path, capsys):
    experiment, traces = _complete_api()
    del traces[RUN_B]
    assert _run_export(eval_set, tmp_path / "out", experiment, traces) == 1
    assert "HTTP 404" in capsys.readouterr().err
