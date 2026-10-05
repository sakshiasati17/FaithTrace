# Scripts

Run these from the repo root.

| Script | Purpose | Dependencies |
|---|---|---|
| `validate_eval_set.py` | Checks an eval set against `docs/dataset/eval_set_schema.md`: required fields, unique ids, enum values, ISO dates, and that every `source_docs` entry is listed in `corpus/manifest.json`. With `--check-evidence` it also reopens each source file and confirms the item's `evidence` quote or cells are there. Exits non-zero on any error. | stdlib; `--check-evidence` also needs `pdfplumber` and `openpyxl` from `backend/requirements.txt` |
| `make_eval_subset.py` | Builds the focused subsets below from `eval_sets/faithtrace_v1.json`. Items are copied unchanged and kept in v1 order, and temporal pairs are never split. With `--check` it regenerates them in memory and exits 1 if a committed file differs (the backend tests run this). | stdlib |
| `seed.py` | Uploads every file in `corpus/manifest.json` to a running API (`POST /api/v1/corpus/upload` with `version_label`, `effective_from`, `effective_to`). Then it polls each document until parsing and indexing finish or fail, and prints a summary. Exits non-zero if any document fails or times out. | stdlib |

```bash
python scripts/validate_eval_set.py eval_sets/faithtrace_v1.json --check-evidence
python scripts/make_eval_subset.py            # rewrite the subset files
python scripts/make_eval_subset.py --check    # fail if they are out of date

docker compose up -d --build
python scripts/seed.py                                   # API_URL defaults to http://localhost
API_URL=http://localhost:8000 API_KEY=... python scripts/seed.py --timeout 900
```

## Eval subsets

All four are built-in eval sets (`builtin:<file>` in the API). The default eval set is still `faithtrace_v1.json`.

| File | Items | Selection |
|---|---|---|
| `faithtrace_smoke.json` | 5 | The first temporal pair (by first appearance of `temporal_pair_id`), the first answerable table item, the first answerable chart item, the first unanswerable item. |
| `faithtrace_temporal.json` | 30 | Every item with a `temporal_pair_id` (15 pairs, all text). |
| `faithtrace_multimodal.json` | 38 | Answerable items without a `temporal_pair_id` whose modality is table (16), spreadsheet (13) or chart (9). |
| `faithtrace_core40.json` | 40 | The first 8 temporal pairs. Then, from the remaining items in v1 order: the first 6 unanswerable items, then the first 8 table, 4 spreadsheet and 6 chart items that are not unanswerable. Result: 16 temporal and 6 unanswerable items; modality text 18, table 9, chart 7, spreadsheet 6. |
