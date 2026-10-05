"""
scripts/seed.py pacing and 429 retries.

nginx limits /api/v1/corpus/upload to 5 requests/min (burst 3) and answers 429
beyond that. The seed script must space uploads and retry 429s. urlopen and
time.sleep are patched: nothing sleeps and nothing touches the network.
"""

import importlib.util
import io
import json
import urllib.error
from email.message import Message
from pathlib import Path
from unittest import mock

import pytest

SEED_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "seed.py"


@pytest.fixture
def seed():
    spec = importlib.util.spec_from_file_location("faithtrace_seed_script", SEED_PATH)
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


def _http_error(url, code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(url, code, "error", headers, io.BytesIO(b'{"detail": "error"}'))


def _urlopen(outcomes):
    """urlopen stand-in returning (or raising) each outcome in turn and recording request URLs."""
    calls = []
    queue = list(outcomes)

    def fake(req, timeout=None):
        calls.append((req.get_method(), req.full_url))
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)

    fake.calls = calls
    return fake


URL = "http://api.test/api/v1/corpus/upload"


class TestRequestRetry:
    def test_429_then_success_is_retried(self, seed, capsys):
        fake = _urlopen([_http_error(URL, 429), {"id": "doc-1"}])
        with mock.patch("urllib.request.urlopen", fake), mock.patch("time.sleep") as sleep:
            result = seed._request(URL, "", data=b"x", content_type="text/plain", max_retries=5)
        assert result == {"id": "doc-1"}
        assert len(fake.calls) == 2
        assert all(method == "POST" for method, _ in fake.calls)
        sleep.assert_called_once_with(15)
        assert f"429 from {URL}, retrying in 15s (attempt 1/5)" in capsys.readouterr().err

    def test_retry_after_header_is_honoured(self, seed, capsys):
        fake = _urlopen([_http_error(URL, 429, retry_after="7"), {"id": "doc-1"}])
        with mock.patch("urllib.request.urlopen", fake), mock.patch("time.sleep") as sleep:
            assert seed._request(URL, "", max_retries=3) == {"id": "doc-1"}
        sleep.assert_called_once_with(7)
        assert f"429 from {URL}, retrying in 7s (attempt 1/3)" in capsys.readouterr().err

    def test_non_integer_retry_after_falls_back_to_backoff(self, seed):
        fake = _urlopen(
            [
                _http_error(URL, 429, retry_after="Wed, 21 Oct 2026 07:28:00 GMT"),
                _http_error(URL, 429, retry_after="soon"),
                {"id": "doc-1"},
            ]
        )
        with mock.patch("urllib.request.urlopen", fake), mock.patch("time.sleep") as sleep:
            assert seed._request(URL, "", max_retries=5) == {"id": "doc-1"}
        assert [c.args[0] for c in sleep.call_args_list] == [15, 30]

    def test_retries_exhausted_raises_429(self, seed, capsys):
        fake = _urlopen([_http_error(URL, 429) for _ in range(3)])
        with mock.patch("urllib.request.urlopen", fake), mock.patch("time.sleep") as sleep:
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                seed._request(URL, "", max_retries=2)
        assert excinfo.value.code == 429
        assert len(fake.calls) == 3
        assert [c.args[0] for c in sleep.call_args_list] == [15, 30]
        err = capsys.readouterr().err
        assert "(attempt 1/2)" in err and "(attempt 2/2)" in err

    def test_non_429_error_is_not_retried(self, seed):
        fake = _urlopen([_http_error(URL, 500), {"id": "never"}])
        with mock.patch("urllib.request.urlopen", fake), mock.patch("time.sleep") as sleep:
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                seed._request(URL, "", max_retries=5)
        assert excinfo.value.code == 500
        assert len(fake.calls) == 1
        sleep.assert_not_called()


def _write_corpus(tmp_path, names):
    for name in names:
        (tmp_path / name).write_text("hello", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"documents": [{"filename": n, "version_label": "v1"} for n in names]}), encoding="utf-8"
    )
    return manifest


def _done(doc_id):
    return {"id": doc_id, "parse_status": "done", "index_status": "done"}


class TestMain:
    def test_delay_between_uploads_not_before_first_or_polls(self, seed, tmp_path):
        manifest = _write_corpus(tmp_path, ["a.txt", "b.txt", "c.txt"])
        uploads = [{"id": f"doc-{i}", "parse_status": "pending", "index_status": "pending"} for i in range(3)]
        fake = _urlopen(uploads + [_done(f"doc-{i}") for i in range(3)])
        events = []

        def recording_urlopen(req, timeout=None):
            events.append(("request", req.get_method()))
            return fake(req, timeout=timeout)

        def recording_sleep(seconds):
            events.append(("sleep", seconds))

        with mock.patch("urllib.request.urlopen", recording_urlopen), mock.patch("time.sleep", recording_sleep):
            code = seed.main(["--manifest", str(manifest), "--api-url", "http://api.test", "--delay", "13"])

        assert code == 0
        assert events == [
            ("request", "POST"),
            ("sleep", 13.0),
            ("request", "POST"),
            ("sleep", 13.0),
            ("request", "POST"),
            ("request", "GET"),
            ("request", "GET"),
            ("request", "GET"),
        ]

    def test_default_delay_is_13_seconds(self, seed, tmp_path):
        manifest = _write_corpus(tmp_path, ["a.txt", "b.txt"])
        fake = _urlopen([_done("doc-0"), _done("doc-1"), _done("doc-0"), _done("doc-1")])
        with mock.patch("urllib.request.urlopen", fake), mock.patch("time.sleep") as sleep:
            assert seed.main(["--manifest", str(manifest), "--api-url", "http://api.test"]) == 0
        assert [c.args[0] for c in sleep.call_args_list] == [13.0]

    def test_exhausted_429_is_reported_as_failure(self, seed, tmp_path, capsys):
        manifest = _write_corpus(tmp_path, ["a.txt", "b.txt"])
        upload_url = "http://api.test/api/v1/corpus/upload"
        fake = _urlopen(
            [
                _done("doc-0"),
                _http_error(upload_url, 429),
                _http_error(upload_url, 429),
                _done("doc-0"),
            ]
        )
        with mock.patch("urllib.request.urlopen", fake), mock.patch("time.sleep"):
            code = seed.main(
                ["--manifest", str(manifest), "--api-url", "http://api.test", "--delay", "0", "--max-retries", "1"]
            )
        assert code == 1
        out = capsys.readouterr().out
        assert "b.txt: HTTP 429" in out
        assert "1/2 documents ingested successfully." in out
