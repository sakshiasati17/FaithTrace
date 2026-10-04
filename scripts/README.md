# Scripts

Run these from the repo root.

| Script | Purpose | Dependencies |
|---|---|---|
| `validate_eval_set.py` | Checks an eval set against `docs/dataset/eval_set_schema.md`: required fields, unique ids, enum values, ISO dates, and that every `source_docs` entry is listed in `corpus/manifest.json`. With `--check-evidence` it also reopens each source file and confirms the item's `evidence` quote or cells are there. Exits non-zero on any error. | stdlib; `--check-evidence` also needs `pdfplumber` and `openpyxl` from `backend/requirements.txt` |
| `seed.py` | Uploads every file in `corpus/manifest.json` to a running API (`POST /api/v1/corpus/upload` with `version_label`, `effective_from`, `effective_to`). Then it polls each document until parsing and indexing finish or fail, and prints a summary. Exits non-zero if any document fails or times out. | stdlib |

```bash
python scripts/validate_eval_set.py eval_sets/faithtrace_v1.json --check-evidence

docker compose up -d --build
python scripts/seed.py                                   # API_URL defaults to http://localhost
API_URL=http://localhost:8000 API_KEY=... python scripts/seed.py --timeout 900
```
