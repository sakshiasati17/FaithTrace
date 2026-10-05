# Scripts

Run these from the repo root.

| Script | Purpose | Dependencies |
|---|---|---|
| `validate_eval_set.py` | Checks an eval set against `docs/dataset/eval_set_schema.md`: required fields, unique ids, enum values, ISO dates, and that every `source_docs` entry is listed in `corpus/manifest.json`. With `--check-evidence` it also reopens each source file and confirms the item's `evidence` quote or cells are there. Exits non-zero on any error. | stdlib; `--check-evidence` also needs `pdfplumber` and `openpyxl` from `backend/requirements.txt` |
| `make_eval_subset.py` | Builds the focused subsets below from `eval_sets/faithtrace_v1.json`. Items are copied unchanged and kept in v1 order, and temporal pairs are never split. With `--check` it regenerates them in memory and exits 1 if a committed file differs (the backend tests run this). | stdlib |
| `seed.py` | Uploads every file in `corpus/manifest.json` to a running API (`POST /api/v1/corpus/upload` with `version_label`, `effective_from`, `effective_to`). Then it polls each document until parsing and indexing finish or fail, and prints a summary. Exits non-zero if any document fails or times out. Behind nginx (docker compose) uploads are limited to 5/min with a burst of 3, so uploads are spaced by `--delay` seconds (default 13; not before the first upload or between status polls) and an HTTP 429 is retried up to `--max-retries` times (default 5), waiting `Retry-After` seconds when it is an integer, otherwise 15 s × attempt. A 429 left after the last retry counts as a failure. | stdlib |
| `export_results.py` | Exports one experiment from a running API into reproducible files: `raw/experiment.json` and `raw/traces/<run_id>.json` (exactly as fetched), `runs.csv`, `by_modality.csv`, `failures.csv`, `temporal.csv`, `summary.md` and `manifest.json` (eval set path + sha256, item count, UTC time, API URL, `complete` flag, script version). Query results are joined to eval items by `query_id` == `id`; an unmatched or duplicate id is an error. It refuses incomplete data (runs not `done`, errored queries, ok queries without an `answer_correctness` score or a `failure_category`, eval items with no result): it prints the report, writes nothing and exits 2, unless `--allow-incomplete`, which writes everything with an INCOMPLETE banner and `"complete": false`. Missing values are empty CSV cells / `—`, never 0; means use non-null values only and come with their n. | stdlib |

```bash
python scripts/validate_eval_set.py eval_sets/faithtrace_v1.json --check-evidence
python scripts/make_eval_subset.py            # rewrite the subset files
python scripts/make_eval_subset.py --check    # fail if they are out of date

docker compose up -d --build
python scripts/seed.py                                   # API_URL defaults to http://localhost
API_URL=http://localhost:8000 API_KEY=... python scripts/seed.py --timeout 900
API_URL=http://localhost:8000 python scripts/seed.py --delay 0   # straight to the API, no nginx rate limit

# after an experiment finishes (status done):
python scripts/export_results.py --experiment <experiment-id> --eval-set eval_sets/faithtrace_v1.json --out results/<name>
```

`export_results.py` exit codes: 0 written, 1 error (API unreachable or HTTP error, bad eval set, unmatched query id, `--out` not empty), 2 incomplete and nothing written. `--out` must not exist or be empty; files are written to a temporary directory first and moved into place, so a failed export leaves no partial output. The trace endpoint returns all of a run's query results in one response (no pagination). `API_URL` and `API_KEY` work as for `seed.py`.

Note: `.gitignore` ignores `*.csv`, so committing an export needs `git add -f results/<name>`.

## Eval subsets

All four are built-in eval sets (`builtin:<file>` in the API). The default eval set is still `faithtrace_v1.json`.

| File | Items | Selection |
|---|---|---|
| `faithtrace_smoke.json` | 5 | The first temporal pair (by first appearance of `temporal_pair_id`), the first answerable table item, the first answerable chart item, the first unanswerable item. |
| `faithtrace_temporal.json` | 30 | Every item with a `temporal_pair_id` (15 pairs, all text). |
| `faithtrace_multimodal.json` | 38 | Answerable items without a `temporal_pair_id` whose modality is table (16), spreadsheet (13) or chart (9). |
| `faithtrace_core40.json` | 40 | The first 8 temporal pairs. Then, from the remaining items in v1 order: the first 6 unanswerable items, then the first 8 table, 4 spreadsheet and 6 chart items that are not unanswerable. Result: 16 temporal and 6 unanswerable items; modality text 18, table 9, chart 7, spreadsheet 6. |
