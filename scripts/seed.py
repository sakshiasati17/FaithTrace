#!/usr/bin/env python3
"""Upload the demo corpus to a running FaithTrace API and wait for ingestion.

Reads corpus/manifest.json, uploads each file with its version metadata to
POST {API_URL}/api/v1/corpus/upload, then polls GET /api/v1/corpus/{id} until
parse_status and index_status are both "done" or one of them is "failed".

Usage:
    python scripts/seed.py                       # API_URL defaults to http://localhost
    API_URL=http://localhost:8000 API_KEY=... python scripts/seed.py --timeout 900
    python scripts/seed.py --delay 0 --max-retries 0  # direct to the API, no nginx

Behind nginx (docker compose) uploads are rate limited to 5/min with a burst of 3,
so uploads are spaced by --delay seconds (default 13) and an HTTP 429 is retried
up to --max-retries times (default 5), waiting Retry-After seconds when the header
is an integer, otherwise 15 s * attempt.

Standard library only. Exits non-zero if any upload or ingestion fails or times out.
"""

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "corpus" / "manifest.json"
TERMINAL = {"done", "failed"}
DEFAULT_DELAY = 13.0
DEFAULT_MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 15


def _headers(api_key):
    return {"X-API-Key": api_key} if api_key else {}


def _multipart(fields, file_field, file_path):
    boundary = f"----faithtrace{uuid.uuid4().hex}"
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    parts.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
            f'filename="{file_path.name}"\r\nContent-Type: {content_type}\r\n\r\n'
        ).encode()
        + file_path.read_bytes()
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _retry_after(exc, attempt):
    """Seconds to wait before retrying a 429: Retry-After if it is an integer, else 15 s * attempt."""
    value = (exc.headers.get("Retry-After") if exc.headers is not None else None) or ""
    try:
        seconds = int(value.strip())
    except ValueError:
        seconds = -1
    return seconds if seconds >= 0 else RETRY_BACKOFF_SECONDS * attempt


def _request(url, api_key, data=None, content_type=None, timeout=120, max_retries=DEFAULT_MAX_RETRIES):
    headers = _headers(api_key)
    if content_type:
        headers["Content-Type"] = content_type
    attempt = 0
    while True:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt >= max_retries:
                raise
            attempt += 1
            wait = _retry_after(exc, attempt)
            exc.close()
            print(f"429 from {url}, retrying in {wait}s (attempt {attempt}/{max_retries})", file=sys.stderr)
            time.sleep(wait)


def upload(api_url, api_key, corpus_dir, entry, max_retries=DEFAULT_MAX_RETRIES):
    path = corpus_dir / entry["filename"]
    fields = {
        "version_label": entry.get("version_label") or "v1",
        "effective_from": entry.get("effective_from") or "",
        "effective_to": entry.get("effective_to") or "",
    }
    body, content_type = _multipart(fields, "file", path)
    return _request(
        f"{api_url}/api/v1/corpus/upload", api_key, data=body, content_type=content_type, max_retries=max_retries
    )


def _non_negative(cast):
    def parse(value):
        number = cast(value)
        if number < 0:
            raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
        return number

    return parse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost"))
    parser.add_argument("--timeout", type=int, default=600, help="seconds to wait for ingestion (default 600)")
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument(
        "--delay",
        type=_non_negative(float),
        default=DEFAULT_DELAY,
        help=f"seconds between uploads, keeps them under nginx's 5/min limit (default {DEFAULT_DELAY:g})",
    )
    parser.add_argument(
        "--max-retries",
        type=_non_negative(int),
        default=DEFAULT_MAX_RETRIES,
        help=f"retries per request after an HTTP 429 (default {DEFAULT_MAX_RETRIES})",
    )
    args = parser.parse_args(argv)

    api_url = args.api_url.rstrip("/")
    api_key = os.environ.get("API_KEY", "")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    corpus_dir = args.manifest.parent
    entries = manifest.get("documents", [])

    results = {}  # filename -> dict(id, parse, index, error)
    uploads_attempted = 0
    for entry in entries:
        name = entry["filename"]
        if not (corpus_dir / name).exists():
            results[name] = {"id": None, "parse": "-", "index": "-", "error": "file missing"}
            print(f"[skip]   {name}: file missing")
            continue
        if uploads_attempted and args.delay:
            time.sleep(args.delay)
        uploads_attempted += 1
        try:
            doc = upload(api_url, api_key, corpus_dir, entry, max_retries=args.max_retries)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            results[name] = {"id": None, "parse": "-", "index": "-", "error": f"HTTP {exc.code}: {detail}"}
            print(f"[error]  {name}: HTTP {exc.code}: {detail}")
            continue
        except (urllib.error.URLError, OSError) as exc:
            print(f"[error]  {name}: cannot reach {api_url}: {exc}")
            print("Is the stack running? Try: docker compose up -d --build")
            return 2
        results[name] = {"id": doc["id"], "parse": doc.get("parse_status"), "index": doc.get("index_status"), "error": None}
        print(f"[upload] {name} -> {doc['id']} (version {entry.get('version_label')})")

    deadline = time.time() + args.timeout
    pending = {n for n, r in results.items() if r["id"]}
    while pending and time.time() < deadline:
        for name in sorted(pending):
            res = results[name]
            try:
                doc = _request(f"{api_url}/api/v1/corpus/{res['id']}", api_key, max_retries=args.max_retries)
            except (urllib.error.URLError, OSError) as exc:
                res["error"] = f"poll failed: {exc}"
                print(f"[warn]   {name}: poll failed: {exc}")
                continue
            res["parse"], res["index"], res["error"] = doc.get("parse_status"), doc.get("index_status"), None
            if "failed" in (res["parse"], res["index"]) or (res["parse"] == "done" and res["index"] == "done"):
                pending.discard(name)
                print(f"[{res['parse']:>6}] {name} (parse={res['parse']}, index={res['index']})")
        if pending:
            time.sleep(args.poll_interval)

    for name in pending:
        results[name]["error"] = f"timed out after {args.timeout}s"

    print("\nSummary")
    print(f"{'file':48} {'parse':8} {'index':8} note")
    failures = 0
    for name, res in results.items():
        ok = res["parse"] == "done" and res["index"] == "done" and not res["error"]
        failures += 0 if ok else 1
        print(f"{name:48} {str(res['parse']):8} {str(res['index']):8} {res['error'] or ''}")
    print(f"\n{len(results) - failures}/{len(results)} documents ingested successfully.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
