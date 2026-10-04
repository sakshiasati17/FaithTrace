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
  api/v1/endpoints/   corpus, experiments, evaluation, diagnostics, recommendations, feedback
  db/                 SQLAlchemy models (6 tables); migrations in backend/alembic/versions
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
cd frontend && npm ci && npm test -- --run && npm run build
```

New DB columns/tables need an Alembic migration in `backend/alembic/versions` (next is `004_...`).

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
2. **Diagnosis correctness.** `tasks.py` `train_failure_classifier` keys eval items by `query_id` but sets use `id`. `diagnose_run` pairs rows with questions by position, not `query_id`. `metrics._compute_diagnostic_accuracy` uses empty metrics and returns 1.0 when unlabelled.
3. **Errors scored as answers.** `runner.py` turns exceptions into `"Error: ..."` answers that get scored and diagnosed. Record per-query error status; exclude from metrics.
4. **BM25 ignores filters.** `_build_bm25_retriever` skips freshness and chunk-type filters, so hybrid configs leak stale chunks. Also rebuilt per question.
5. **Reranker never runs.** `sentence-transformers` is missing from requirements; failure is hidden by `except: pass`.
6. **Parsing fixed at upload.** `corpus._default_strategy` never selects `text_table_vision`, so no image chunks exist. `spreadsheet_aware` filter uses `"spreadsheet"` but parser emits `"spreadsheet_cell"`.
7. **Chunking axis is a no-op.** Ingestion always chunks `recursive`; runner never reads `chunking_strategy`.
8. **Eval set is a file path.** Defaults differ (frontend vs API vs training); `POST /evaluation/run/{id}` re-scores with the default set; path is unrestricted.
9. **Lifecycle/cost.** Experiment marked `done` before evaluation/diagnosis finish; `MAX_COST_PER_RUN_USD` unused; all experiments search all documents.
10. **Honesty/CI.** Frontend status indicators are hard-coded; no CI workflow; no results committed.
