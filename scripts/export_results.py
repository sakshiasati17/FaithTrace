#!/usr/bin/env python3
"""Export one experiment's results from a running FaithTrace API into reproducible files.

Fetches GET /api/v1/experiments/{id} and, for every run,
GET /api/v1/experiments/{id}/runs/{run_id}/trace (the trace endpoint returns
every query result of the run in one response; it does not paginate). Each
query result is joined to its eval item by query_id == id.

Usage:
    python scripts/export_results.py --experiment ID --eval-set eval_sets/faithtrace_v1.json \\
        --out results/<name> [--allow-incomplete]
    API_URL=http://localhost:8000 API_KEY=... python scripts/export_results.py ...

Before writing, the export checks completeness: runs not "done", errored
queries, ok queries without an answer_correctness score, ok queries without a
failure_category, and eval items with no result in a run. If anything is
incomplete it prints the report, writes nothing and exits 2, unless
--allow-incomplete is given; then everything is written, summary.md starts
with an INCOMPLETE banner and manifest.json has "complete": false.

A query_id that matches no eval item, a duplicate query_id within a run, or
an API error is a hard error (exit 1, nothing written).

Missing values are empty CSV cells and "—" in Markdown, never 0. Means use
only non-null values and are reported with their n.

Standard library only.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_VERSION = "1.0.0"

CONFIG_AXES = ["retrieval_strategy", "chunking_strategy", "parsing_strategy", "freshness_policy"]
# RunMetricsResponse fields, in schema order; any extra keys are appended sorted.
RUN_METRIC_KEYS = [
    "answer_correctness", "faithfulness", "context_precision", "context_recall",
    "answer_relevance", "latency_p50_ms", "latency_p95_ms", "avg_token_usage",
    "avg_cost_usd", "freshness_validity", "temporal_citation_accuracy",
    "multimodal_grounding_rate", "root_cause_diagnostic_accuracy",
]
# Per-query Ragas keys stored by evaluate_run in diagnosis_evidence["scores"].
RAGAS_KEYS = [
    "answer_correctness", "faithfulness", "context_precision", "context_recall",
    "answer_relevancy",
]
FAILURE_CATEGORIES = [
    "STALE_ANSWER", "WRONG_VERSION", "TABLE_RETRIEVAL_MISS", "CHART_LAYOUT_BLINDNESS",
    "CHUNKING_BOUNDARY_ERROR", "LOW_RECALL_RETRIEVAL", "IRRELEVANT_CONTEXT_POLLUTION",
    "UNSUPPORTED_SYNTHESIS", "NO_FAILURE",
]
MISSING_MD = "—"


class ExportError(Exception):
    """A hard error: nothing is written, exit 1."""


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

def _get(url, api_key, timeout=120):
    """GET a URL; return (raw bytes, parsed JSON)."""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise ExportError(f"GET {url} failed: HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ExportError(f"GET {url} failed: {exc}. Is the stack running?") from exc
    try:
        return raw, json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportError(f"GET {url} returned invalid JSON: {exc}") from exc


def fetch(api_url, api_key, experiment_id):
    """Fetch the experiment and every run's trace. Returns (exp_raw, exp, {run_id: (raw, trace)})."""
    exp_raw, exp = _get(f"{api_url}/api/v1/experiments/{experiment_id}", api_key)
    if not isinstance(exp, dict) or not isinstance(exp.get("runs"), list):
        raise ExportError("experiment response has no 'runs' list")
    traces = {}
    for run in exp["runs"]:
        run_id = run["id"]
        raw, trace = _get(
            f"{api_url}/api/v1/experiments/{experiment_id}/runs/{run_id}/trace", api_key
        )
        if not isinstance(trace, list):
            raise ExportError(f"trace for run {run_id} is not a list")
        traces[run_id] = (raw, trace)
    return exp_raw, exp, traces


# --------------------------------------------------------------------------- #
# Eval set
# --------------------------------------------------------------------------- #

def load_eval_set(path):
    """Return (items, sha256 hex). Accepts a JSON list or an object with "items"."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExportError(f"cannot read eval set {path}: {exc}") from exc
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportError(f"eval set {path} is not valid JSON: {exc}") from exc
    items = parsed.get("items") if isinstance(parsed, dict) else parsed
    if not isinstance(items, list) or not items:
        raise ExportError(f"eval set {path} has no items")
    seen = set()
    for item in items:
        if not isinstance(item, dict) or not item.get("id"):
            raise ExportError(f"eval set {path}: every item needs an 'id'")
        if item["id"] in seen:
            raise ExportError(f"eval set {path}: duplicate id {item['id']!r}")
        seen.add(item["id"])
    return items, hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# Join + completeness
# --------------------------------------------------------------------------- #

def _scores(qr):
    ev = qr.get("diagnosis_evidence") or {}
    sc = ev.get("scores") if isinstance(ev, dict) else None
    return sc if isinstance(sc, dict) else {}


def _is_error(qr):
    return qr.get("status") == "error"


def join(exp, traces, items):
    """Return {run_id: [(query_result, eval_item), ...]}. Raises on unmatched/duplicate query_id."""
    by_id = {item["id"]: item for item in items}
    joined = {}
    problems = []
    for run in exp["runs"]:
        rows, seen = [], set()
        for qr in traces[run["id"]][1]:
            qid = qr.get("query_id")
            if qid not in by_id:
                problems.append(f"run {run['id']}: query_id {qid!r} matches no eval item")
                continue
            if qid in seen:
                problems.append(f"run {run['id']}: duplicate query_id {qid!r}")
                continue
            seen.add(qid)
            rows.append((qr, by_id[qid]))
        joined[run["id"]] = rows
    if problems:
        raise ExportError("cannot join query results to the eval set:\n  " + "\n  ".join(problems))
    return joined


def check_completeness(exp, joined, items):
    """Return a dict of issue lists; all empty means complete."""
    issues = {
        "runs_not_done": [],
        "errored_queries": [],
        "unscored_queries": [],
        "undiagnosed_queries": [],
        "missing_results": [],
    }
    item_ids = [item["id"] for item in items]
    for run in exp["runs"]:
        rid = run["id"]
        if run.get("status") != "done":
            issues["runs_not_done"].append({"run_id": rid, "status": run.get("status")})
        rows = joined[rid]
        for qr, _ in rows:
            qid = qr["query_id"]
            if _is_error(qr):
                issues["errored_queries"].append(
                    {"run_id": rid, "query_id": qid, "error_message": qr.get("error_message")}
                )
                continue
            if _scores(qr).get("answer_correctness") is None:
                issues["unscored_queries"].append({"run_id": rid, "query_id": qid})
            if qr.get("failure_category") is None:
                issues["undiagnosed_queries"].append({"run_id": rid, "query_id": qid})
        present = {qr["query_id"] for qr, _ in rows}
        missing = [i for i in item_ids if i not in present]
        if missing:
            issues["missing_results"].append({"run_id": rid, "query_ids": missing})
    return issues


def is_complete(issues):
    return not any(issues.values())


def completeness_lines(issues, n_runs):
    """Human-readable completeness report (plain text, also used in summary.md)."""
    lines = []
    nd = issues["runs_not_done"]
    lines.append(f"Runs not done: {len(nd)} of {n_runs}")
    lines += [f"  - {r['run_id']}: status {r['status']}" for r in nd]
    err = issues["errored_queries"]
    lines.append(f"Errored queries: {len(err)}")
    lines += [f"  - {e['run_id']} / {e['query_id']}: {e['error_message'] or '(no message)'}" for e in err]
    uns = issues["unscored_queries"]
    lines.append(f"OK queries without an answer_correctness score: {len(uns)}")
    lines += [f"  - {u['run_id']} / {u['query_id']}" for u in uns]
    und = issues["undiagnosed_queries"]
    lines.append(f"OK queries without a failure_category (undiagnosed): {len(und)}")
    lines += [f"  - {u['run_id']} / {u['query_id']}" for u in und]
    mis = issues["missing_results"]
    lines.append(f"Runs with eval items that have no result: {len(mis)}")
    lines += [f"  - {m['run_id']}: {len(m['query_ids'])} missing ({', '.join(m['query_ids'])})" for m in mis]
    return lines


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _mean(values):
    """(mean, n) over non-null values; mean is None when n == 0."""
    vals = [float(v) for v in values if v is not None]
    return (sum(vals) / len(vals) if vals else None), len(vals)


def build_runs_rows(exp, joined):
    extra = set()
    for run in exp["runs"]:
        extra |= set((run.get("metrics") or {}).keys())
    extra -= set(RUN_METRIC_KEYS) | {"id", "run_id"}
    metric_keys = RUN_METRIC_KEYS + sorted(extra)
    header = ["run_id"] + CONFIG_AXES + ["llm_model", "status"] + metric_keys + [
        "n_ok", "n_error", "n_unscored", "n_undiagnosed",
    ]
    rows = []
    for run in exp["runs"]:
        cfg = run.get("config") or {}
        metrics = run.get("metrics") or {}
        qrs = [qr for qr, _ in joined[run["id"]]]
        ok = [qr for qr in qrs if not _is_error(qr)]
        rows.append(
            [run["id"]] + [cfg.get(a) for a in CONFIG_AXES] + [cfg.get("llm_model"), run.get("status")]
            + [metrics.get(k) for k in metric_keys]
            + [
                len(ok),
                len(qrs) - len(ok),
                sum(1 for qr in ok if _scores(qr).get("answer_correctness") is None),
                sum(1 for qr in ok if qr.get("failure_category") is None),
            ]
        )
    return header, rows


def _group_stats(rows):
    """n, n_error, n_scored (answer_correctness) and mean/n per Ragas key over ok queries."""
    ok = [qr for qr, _ in rows if not _is_error(qr)]
    out = {
        "n": len(rows),
        "n_error": len(rows) - len(ok),
        "n_scored": sum(1 for qr in ok if _scores(qr).get("answer_correctness") is not None),
    }
    for key in RAGAS_KEYS:
        out[f"mean_{key}"], out[f"n_{key}"] = _mean(_scores(qr).get(key) for qr in ok)
    return out


def build_modality_rows(exp, joined):
    stat_cols = ["n", "n_error", "n_scored"] + [c for k in RAGAS_KEYS for c in (f"mean_{k}", f"n_{k}")]
    header = ["run_id", "group", "value"] + stat_cols
    rows = []
    for run in exp["runs"]:
        jr = joined[run["id"]]
        for group, keyfn in (
            ("modality", lambda item: str(item.get("modality"))),
            ("answerable", lambda item: str(bool(item.get("answerable"))).lower()),
        ):
            values = sorted({keyfn(item) for _, item in jr})
            for value in values:
                stats = _group_stats([(qr, item) for qr, item in jr if keyfn(item) == value])
                rows.append([run["id"], group, value] + [stats[c] for c in stat_cols])
    return header, rows


def build_failure_rows(exp, joined):
    header = ["run_id", "failure_category", "count", "n_diagnosed"]
    rows = []
    for run in exp["runs"]:
        cats = [qr.get("failure_category") for qr, _ in joined[run["id"]] if not _is_error(qr)]
        diagnosed = [c for c in cats if c is not None]
        known = FAILURE_CATEGORIES + sorted(set(diagnosed) - set(FAILURE_CATEGORIES))
        for cat in known:
            rows.append([run["id"], cat, diagnosed.count(cat), len(diagnosed)])
    return header, rows


def build_temporal_rows(exp, joined):
    header = [
        "run_id", "n", "n_error", "n_scored", "mean_answer_correctness",
        "n_diagnosed", "count_STALE_ANSWER", "count_WRONG_VERSION",
    ]
    rows = []
    for run in exp["runs"]:
        jr = [(qr, item) for qr, item in joined[run["id"]] if item.get("temporal_pair_id")]
        ok = [qr for qr, _ in jr if not _is_error(qr)]
        mean, n_scored = _mean(_scores(qr).get("answer_correctness") for qr in ok)
        cats = [qr.get("failure_category") for qr in ok if qr.get("failure_category") is not None]
        rows.append([
            run["id"], len(jr), len(jr) - len(ok), n_scored, mean,
            len(cats), cats.count("STALE_ANSWER"), cats.count("WRONG_VERSION"),
        ])
    return header, rows


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

def _csv_cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return repr(round(v, 6))
    return str(v)


def _md_cell(v):
    if v is None:
        return MISSING_MD
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v).replace("|", "\\|")


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            w.writerow([_csv_cell(v) for v in row])


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_md_cell(v) for v in row) + " |" for row in rows]
    return "\n".join(out)


def render_summary(exp, manifest, tables, issues, complete):
    parts = []
    if not complete:
        parts.append(
            "> **INCOMPLETE** — this export was written with `--allow-incomplete`. "
            "Some runs, queries, scores or diagnoses are missing; see Completeness below. "
            "Do not report these numbers as final.\n"
        )
    parts.append(f"# Results: {exp.get('name')}\n")
    parts.append(
        f"- Experiment: `{exp.get('id')}`\n"
        f"- Eval set: `{manifest['eval_set_path']}` (sha256 `{manifest['eval_set_sha256']}`, "
        f"{manifest['eval_set_items']} items)\n"
        f"- Generated: {manifest['generated_at']} from {manifest['api_url']}\n"
        f"- Runs: {len(exp['runs'])}\n"
        f"\nMissing values are shown as {MISSING_MD}. Means are over non-null values only; "
        "each mean is shown with its n.\n"
    )
    titles = {
        "runs": "Runs",
        "by_modality": "By modality and answerability",
        "failures": "Failure categories (ok queries with a diagnosis)",
        "temporal": "Temporal items (eval items with a temporal_pair_id)",
    }
    for key, title in titles.items():
        header, rows = tables[key]
        parts.append(f"## {title}\n\n{md_table(header, rows)}\n")
    parts.append("## Completeness\n")
    parts.append("Complete: " + ("yes" if complete else "**no**") + "\n")
    parts.append("```\n" + "\n".join(completeness_lines(issues, len(exp["runs"]))) + "\n```\n")
    return "\n".join(parts)


def write_outputs(out_dir, exp_raw, exp, traces, tables, manifest, issues, complete):
    """Write everything into a temp dir next to out_dir, then move it into place."""
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        (tmp / "raw" / "traces").mkdir(parents=True)
        (tmp / "raw" / "experiment.json").write_bytes(exp_raw)
        for run_id, (raw, _) in traces.items():
            (tmp / "raw" / "traces" / f"{run_id}.json").write_bytes(raw)
        for key in ("runs", "by_modality", "failures", "temporal"):
            write_csv(tmp / f"{key}.csv", *tables[key])
        (tmp / "summary.md").write_text(render_summary(exp, manifest, tables, issues, complete), encoding="utf-8")
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if out_dir.exists():
            out_dir.rmdir()  # empty (checked before fetching)
        os.replace(tmp, out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _display_path(path):
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="experiment id")
    parser.add_argument("--eval-set", required=True, type=Path, help="eval set JSON the experiment ran on")
    parser.add_argument("--out", required=True, type=Path, help="output directory (must not exist or be empty)")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="write incomplete results, marked INCOMPLETE")
    parser.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost"))
    args = parser.parse_args(argv)

    api_url = args.api_url.rstrip("/")
    api_key = os.environ.get("API_KEY", "")

    try:
        if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
            raise ExportError(f"--out {args.out} exists and is not an empty directory")
        items, sha = load_eval_set(args.eval_set)
        exp_raw, exp, traces = fetch(api_url, api_key, args.experiment)
        joined = join(exp, traces, items)
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    exp_eval_path = exp.get("eval_set_path")
    if exp_eval_path and Path(exp_eval_path).name != args.eval_set.name:
        print(f"warning: experiment ran on eval_set_path {exp_eval_path!r}, "
              f"exporting against {args.eval_set}", file=sys.stderr)

    issues = check_completeness(exp, joined, items)
    complete = is_complete(issues)
    print(f"Experiment {exp.get('id')} ({exp.get('name')}): {len(exp['runs'])} runs, "
          f"{len(items)} eval items\n")
    print("Completeness")
    print("\n".join(completeness_lines(issues, len(exp["runs"]))))
    print()

    if not complete and not args.allow_incomplete:
        print("INCOMPLETE: nothing written. Re-run when the experiment is done, "
              "or pass --allow-incomplete to write results marked INCOMPLETE.")
        return 2

    tables = {
        "runs": build_runs_rows(exp, joined),
        "by_modality": build_modality_rows(exp, joined),
        "failures": build_failure_rows(exp, joined),
        "temporal": build_temporal_rows(exp, joined),
    }
    manifest = {
        "experiment_id": exp.get("id"),
        "experiment_name": exp.get("name"),
        "eval_set_path": _display_path(args.eval_set),
        "eval_set_sha256": sha,
        "eval_set_items": len(items),
        "n_runs": len(exp["runs"]),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "api_url": api_url,
        "complete": complete,
        "incomplete_counts": {k: len(v) for k, v in issues.items()},
        "script": "scripts/export_results.py",
        "script_version": SCRIPT_VERSION,
    }
    try:
        write_outputs(args.out, exp_raw, exp, traces, tables, manifest, issues, complete)
    except OSError as exc:
        print(f"error: writing {args.out} failed: {exc}", file=sys.stderr)
        return 1
    print(("Wrote" if complete else "Wrote INCOMPLETE results to") + f" {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
