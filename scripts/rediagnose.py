#!/usr/bin/env python3
"""Re-run only diagnosis for the runs of one or more experiments.

Calls POST {API_URL}/api/v1/diagnostics/experiment/{id}/rediagnose for each
--experiment and prints the per-run result. Diagnosis reuses the per-query
Ragas scores stored at evaluation; runs whose ok queries lack stored scores
are re-scored by the worker, and the summary shows how many queries that is
(will_rescore_queries) so the OpenAI cost is visible.

Only runs with status "done" that were evaluated are enqueued; the others are
listed as skipped with the reason (not an error).

Usage:
    python scripts/rediagnose.py --experiment <id> [--experiment <id> ...]
    API_URL=http://localhost:8000 API_KEY=... python scripts/rediagnose.py --experiment <id>

Standard library only. Exits 1 on any error (API unreachable, HTTP error, or a
run whose diagnosis could not be enqueued), 0 otherwise.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def _headers(api_key):
    return {"X-API-Key": api_key} if api_key else {}


def rediagnose_experiment(api_url, api_key, experiment_id, timeout=120):
    url = f"{api_url}/api/v1/diagnostics/experiment/{urllib.parse.quote(experiment_id, safe='')}/rediagnose"
    req = urllib.request.Request(url, data=b"", headers=_headers(api_key), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def print_summary(result):
    print(f"\nExperiment {result.get('experiment_id')}")
    eval_set = result.get("eval_set_id") or result.get("eval_set_path") or "default"
    print(f"eval set: {eval_set}")
    print(f"{'run':38} {'result':9} {'rescore':>7} note")
    for run in result.get("runs", []):
        rescore = run.get("will_rescore_queries")
        note = run.get("reason") or ""
        if rescore:
            note = f"{run.get('queries_missing_scores')} of {run.get('ok_queries')} ok queries lack stored scores"
        print(f"{run.get('run_id', ''):38} {run.get('status', ''):9} {'' if rescore is None else rescore:>7} {note}")
    print(
        f"enqueued {result.get('enqueued', 0)}, skipped {result.get('skipped', 0)}, "
        f"errors {result.get('errors', 0)}, queries to re-score {result.get('will_rescore_queries', 0)}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", action="append", required=True, metavar="ID",
                        help="experiment id (repeat for several)")
    parser.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost"))
    args = parser.parse_args(argv)

    api_url = args.api_url.rstrip("/")
    api_key = os.environ.get("API_KEY", "")

    failures = 0
    for experiment_id in args.experiment:
        try:
            result = rediagnose_experiment(api_url, api_key, experiment_id)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            print(f"[error]  experiment {experiment_id}: HTTP {exc.code}: {detail}", file=sys.stderr)
            failures += 1
            continue
        except (urllib.error.URLError, OSError) as exc:
            print(f"[error]  cannot reach {api_url}: {exc}", file=sys.stderr)
            print("Is the stack running? Try: docker compose up -d --build", file=sys.stderr)
            return 1
        print_summary(result)
        if result.get("errors"):
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
