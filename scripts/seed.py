#!/usr/bin/env python3
"""Upload the demo corpus to a running FaithTrace API and wait for ingestion.

Reads corpus/manifest.json, uploads each file with its version metadata to
POST {API_URL}/api/v1/corpus/upload, then polls GET /api/v1/corpus/{id} until
parse_status and index_status are both "done" or one of them is "failed".

Usage:
    python scripts/seed.py                       # API_URL defaults to http://localhost
    API_URL=http://localhost:8000 API_KEY=... python scripts/seed.py --timeout 900

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


def _request(url, api_key, data=None, content_type=None, timeout=120):
    headers = _headers(api_key)
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def upload(api_url, api_key, corpus_dir, entry):
    path = corpus_dir / entry["filename"]
    fields = {
        "version_label": entry.get("version_label") or "v1",
        "effective_from": entry.get("effective_from") or "",
        "effective_to": entry.get("effective_to") or "",
    }
    body, content_type = _multipart(fields, "file", path)
    return _request(f"{api_url}/api/v1/corpus/upload", api_key, data=body, content_type=content_type)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost"))
    parser.add_argument("--timeout", type=int, default=600, help="seconds to wait for ingestion (default 600)")
    parser.add_argument("--poll-interval", type=float, default=3.0)
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    api_key = os.environ.get("API_KEY", "")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    corpus_dir = args.manifest.parent
    entries = manifest.get("documents", [])

    results = {}  # filename -> dict(id, parse, index, error)
    for entry in entries:
        name = entry["filename"]
        if not (corpus_dir / name).exists():
            results[name] = {"id": None, "parse": "-", "index": "-", "error": "file missing"}
            print(f"[skip]   {name}: file missing")
            continue
        try:
            doc = upload(api_url, api_key, corpus_dir, entry)
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
                doc = _request(f"{api_url}/api/v1/corpus/{res['id']}", api_key)
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
