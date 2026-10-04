# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Git rules (mandatory)

- Commit only as `sakshiasati17 <sakshiasati51@gmail.com>`. Check `git config user.name` / `user.email` before committing.
- Branch names: `fix/<topic>`, `feat/<topic>`, `chore/<topic>`, `docs/<topic>`.
- Commit messages: conventional style, e.g. `fix(runner): record query errors instead of scoring them`.
- Never push to `sakshi/main` directly. Never force-push or rewrite shared history.
- Do not commit or push unless the user asks.

## Project

FaithTrace benchmarks and diagnoses RAG pipelines, focusing on temporal drift (stale or wrong-version answers) and multimodal failures (tables, charts, spreadsheets). It runs pipeline configurations over an eval set, scores them with Ragas plus custom metrics, classifies failures into 9 categories, and recommends a config per objective.

## Layout

```
backend/app/
  api/v1/endpoints/   corpus, experiments, evaluation, eval_sets, diagnostics, recommendations, feedback
  db/                 SQLAlchemy models (7 tables); migrations in backend/alembic/versions
  services/
    ingestion/        parser.py, chunker.py, indexer.py (Qdrant)
    experiment/       runner.py (retrieval + generation), config_matrix.py (256 / 24 configs)
    evaluation/       ragas_runner.py, metrics.py, baseline.py
    diagnostics/      classifier.py (rules), ml_classifier.py (XGBoost), reasoning_agent.py (LLM)
    recommendation/   engine.py (7 objectives)
  workers/tasks.py    Celery: ingest_document → run_experiment → evaluate_run → diagnose_run; train_failure_classifier
backend/tests/        test_regression.py, test_failure_injection.py
frontend/src/app/     Next.js 14 pages: /, corpus, experiments, experiments/[id], runs/[runId], leaderboard, diagnostics, recommendations
eval_sets/            JSON question sets (golden_regression_set.json is frozen for tests)
```

## Commands

```bash
cp .env.example .env            # set OPENAI_API_KEY
docker compose up -d --build    # UI http://localhost, API docs http://localhost/docs

cd backend && pip install -r requirements.txt && pytest -q
# Concurrency test for the experiment status needs PostgreSQL (skipped otherwise):
#   FAITHTRACE_TEST_PG_URL=postgresql+psycopg2://user:pw@localhost:5432/faithtrace_test pytest -q
# If system pip fails building langdetect/antlr4/iopath ("install_layout"), use a venv:
#   python -m venv .venv && .venv/bin/pip install -U pip setuptools wheel && .venv/bin/pip install -r backend/requirements.txt
cd frontend && npm ci && npm test -- --run && npm run build
```

New DB columns/tables need an Alembic migration in `backend/alembic/versions` (next is `007_...`).

## Working rules (do not disrupt the current project)

- Each fix is its own small branch + PR; tests must stay green after each.
- Changes to schema are additive only: new nullable columns or new tables, with defaults. No renames or drops.
- Keep API responses backward compatible: add fields, don't remove or rename.
- Keep the 24-config MVP preset and the full 256-config matrix working; `test_regression.py` asserts both.
- Do not edit `eval_sets/golden_regression_set.json` (tests assert 5 items). Add new eval sets as new files.
- New behaviour that costs money (vision parsing, multi-strategy indexing, reranker) must be opt-in or bounded.
- Never swallow exceptions silently (`except: pass`). Log and record them.
- Don't claim results that aren't reproducible from data in the repo.

## Known issues (fix in this order)

1. **No real data.** No documents in repo; `sample_eval_set.json` and the golden set are the same 5 invented questions; `procurement_policy_eval.json` is 10 text-only questions. `scripts/` scripts listed in its README don't exist. Add a public `corpus/`, a verified eval set (60–100 questions incl. table, chart, spreadsheet, temporal pairs, unanswerable) and `scripts/seed.py`.
2. ~~**Diagnosis correctness.**~~ Fixed in PR #19 (`fix/diagnosis-correctness`). Eval items are matched by `id` (`classifier.index_eval_set`); `root_cause_diagnostic_accuracy` is computed in `diagnose_run` from stored diagnoses vs `failure_type` labels and is `None` when nothing is labelled. Retrain old classifier models.
3. ~~**Errors scored as answers.**~~ Fixed in PR #23 (`fix/query-error-handling`): queries carry `status`/`error_message`; errored rows are excluded from metrics and diagnosis; a run with >50% errored queries is `failed`.
4. ~~**BM25 ignores filters.**~~ Fixed in `fix/hybrid-filters`. BM25 candidates pass `runner.chunk_passes_filters` (same rules as the Qdrant filter, which now also checks `effective_to`); chunks are fetched once per `run_pipeline`.
5. ~~**Reranker never runs.**~~ Fixed in `fix/reranker`. `sentence-transformers` is pinned; the cross-encoder (`RERANKER_MODEL`) loads once per process; each chunk carries `reranked` (and `rerank_error` on failure) in `retrieved_chunks`, and failures are logged.
6. **Parsing fixed at upload.** `corpus._default_strategy` never selects `text_table_vision`, so no image chunks exist. ~~`spreadsheet_aware` filter used `"spreadsheet"`~~ (fixed in `fix/hybrid-filters`: now `"spreadsheet_cell"`).
7. **Chunking axis is a no-op.** Ingestion always chunks `recursive`; runner never reads `chunking_strategy`.
8. ~~**Eval set is a file path.**~~ Fixed in `feat/eval-set-upload`: `eval_sets` table + `/api/v1/eval-sets` upload (JSON/CSV, validated); experiments store `eval_set_id`/`eval_set_path` (restricted to `eval_sets/`); workers use `tasks.load_eval_set_for`; one default `eval_sets/faithtrace_v1.json`.
9. ~~**Lifecycle/cost.**~~ Fixed in `fix/experiment-lifecycle`: experiment status `pending → running → evaluating → diagnosing → done | failed`, recomputed from runs' `evaluated_at`/`diagnosed_at` under a row lock (`tasks.refresh_experiment_status`); `MAX_COST_PER_RUN_USD` stops a run's queries (`"budget exceeded"`, run `failed`); optional `experiments.document_ids` scopes retrieval via `runner.chunk_passes_filters`.
10. **Honesty/CI.** ~~Hard-coded status indicators, no CI~~ fixed in PR #20 (`/api/v1/system/status`, `.github/workflows/ci.yml`). Still open: no results committed.
