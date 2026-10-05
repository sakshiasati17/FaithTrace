# Scripts

Run these from the repo root.

| Script | Purpose | Dependencies |
|---|---|---|
| `validate_eval_set.py` | Checks an eval set against `docs/dataset/eval_set_schema.md`: required fields, unique ids, enum values, ISO dates, and that every `source_docs` entry is listed in `corpus/manifest.json`. With `--check-evidence` it also reopens each source file and confirms the item's `evidence` quote or cells are there. Exits non-zero on any error. | stdlib; `--check-evidence` also needs `pdfplumber` and `openpyxl` from `backend/requirements.txt` |
| `seed.py` | Uploads every file in `corpus/manifest.json` to a running API (`POST /api/v1/corpus/upload` with `version_label`, `effective_from`, `effective_to`). Then it polls each document until parsing and indexing finish or fail, and prints a summary. Exits non-zero if any document fails or times out. | stdlib |
| `export_results.py` | Exports one experiment from a running API into reproducible files: `raw/experiment.json` and `raw/traces/<run_id>.json` (exactly as fetched), `runs.csv`, `by_modality.csv`, `failures.csv`, `temporal.csv`, `summary.md` and `manifest.json` (eval set path + sha256, item count, UTC time, API URL, `complete` flag, script version). Query results are joined to eval items by `query_id` == `id`; an unmatched or duplicate id is an error. It refuses incomplete data (runs not `done`, errored queries, ok queries without an `answer_correctness` score or a `failure_category`, eval items with no result): it prints the report, writes nothing and exits 2, unless `--allow-incomplete`, which writes everything with an INCOMPLETE banner and `"complete": false`. Missing values are empty CSV cells / `—`, never 0; means use non-null values only and come with their n. | stdlib |

```bash
python scripts/validate_eval_set.py eval_sets/faithtrace_v1.json --check-evidence

docker compose up -d --build
python scripts/seed.py                                   # API_URL defaults to http://localhost
API_URL=http://localhost:8000 API_KEY=... python scripts/seed.py --timeout 900

# after an experiment finishes (status done):
python scripts/export_results.py --experiment <experiment-id> --eval-set eval_sets/faithtrace_v1.json --out results/<name>
```

`export_results.py` exit codes: 0 written, 1 error (API unreachable or HTTP error, bad eval set, unmatched query id, `--out` not empty), 2 incomplete and nothing written. `--out` must not exist or be empty; files are written to a temporary directory first and moved into place, so a failed export leaves no partial output. The trace endpoint returns all of a run's query results in one response (no pagination). `API_URL` and `API_KEY` work as for `seed.py`.

Note: `.gitignore` ignores `*.csv`, so committing an export needs `git add -f results/<name>`.
