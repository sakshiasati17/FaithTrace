# FaithTrace

[![CI](https://github.com/sakshiasati17/FaithTrace/actions/workflows/ci.yml/badge.svg?branch=sakshi%2Fmain)](https://github.com/sakshiasati17/FaithTrace/actions/workflows/ci.yml)

**Know why your RAG pipeline failed.** FaithTrace runs retrieval pipelines over a question set, scores every answer, classifies the root cause of each failure, and recommends a configuration per objective (quality, cost, latency, tables, drift, long PDFs).

It targets two gaps most RAG evaluation tools miss:

1. **Temporal drift**: answers that come from an outdated or not-yet-effective document version.
2. **Multimodal failures**: evidence that lives in tables, charts or spreadsheets instead of plain text.

---

## How it works

```
Upload documents ─► parse (text / tables / opt-in vision / spreadsheets)
                 ─► chunk (every configured chunking strategy) ─► embed + index in Qdrant
Create experiment ─► for each pipeline config, for each question:
                      retrieve (vector / BM25 / hybrid / hybrid + reranker, with freshness,
                      parsing, chunking and document-scope filters) ─► generate
                 ─► evaluate (Ragas + operational + custom metrics)
                 ─► diagnose each query (9 failure categories)
                 ─► leaderboard, baseline comparison, recommendations
```

Experiments move through `pending → running → evaluating → diagnosing → done | failed`; the UI polls until they finish.

## Pipeline configuration matrix

| Axis | Options |
|---|---|
| Retrieval | `vector_only`, `bm25`, `hybrid`, `hybrid_reranker` |
| Chunking | `fixed_size`, `recursive`, `semantic`, `structure_aware` |
| Parsing | `text_only`, `text_table`, `text_table_vision`, `spreadsheet_aware` |
| Freshness | `none`, `recency_biased`, `effective_date_filter`, `version_aware` |

Full matrix: 4 × 4 × 4 × 4 = 256 configurations. MVP preset: 24 (3 retrieval × 2 chunking × 2 parsing × 2 freshness).

- **Chunking.** Each document's text is indexed once per strategy in `INGEST_CHUNKING_STRATEGIES` (default `fixed_size,recursive,structure_aware`; `semantic` calls the embeddings API while chunking, so it is opt-in). Each listed strategy is embedded separately, so embedding cost and index size grow with it. A run whose strategy was not indexed fails with a clear reason instead of silently answering from other chunks; documents indexed before this change count as `recursive` until reindexed.
- **Vision.** Image chunks exist only for PDFs uploaded with **Enable vision parsing** (`enable_vision=true`): up to `VISION_MAX_PAGES` pages go to `VISION_MODEL`. It costs money, so it is off by default and needs `OPENAI_API_KEY`. Only `text_table_vision` runs retrieve image chunks; `text_table` retrieves text and tables.
- **Freshness.** Vector and BM25 retrieval apply the same date, chunk-type, chunking and document-scope rules (`runner.chunk_passes_filters`).
- **Cost.** `MAX_COST_PER_RUN_USD` stops a run's remaining queries once exceeded; the run is marked `failed` with the reason.

## Diagnostics

- **9 failure categories:** `STALE_ANSWER`, `WRONG_VERSION`, `TABLE_RETRIEVAL_MISS`, `CHART_LAYOUT_BLINDNESS`, `CHUNKING_BOUNDARY_ERROR`, `LOW_RECALL_RETRIEVAL`, `IRRELEVANT_CONTEXT_POLLUTION`, `UNSUPPORTED_SYNTHESIS`, `NO_FAILURE`.
- **Classifier:** an XGBoost model on 14 features (5 Ragas scores, latency, cost, chunk count, table/image/vision flags, question modality) once trained; otherwise priority-ordered rules (temporal → version → modality → recall → synthesis).
- **Human feedback** (thumbs up/down or a corrected label) overrides classifier labels when retraining.
- **Reasoning agent:** on demand, an LLM explains a failed query step by step.
- Crashed queries are recorded as `status="error"` with the error message and are excluded from metrics and diagnosis; they are never scored as answers.

## Metrics

- **Ragas:** faithfulness, answer correctness, context recall, context precision, answer relevance.
- **Operational:** latency p50/p95, tokens, cost per query.
- **Custom:** freshness validity, temporal citation accuracy, multimodal grounding rate, root-cause diagnostic accuracy (stored diagnoses vs labelled `failure_type`).

A metric that could not be computed is stored as empty (shown as "—"), never as a fake 0 or 1.

## Recommendations

7 objectives: `best_overall`, `lowest_cost`, `best_latency`, `best_faithfulness`, `best_for_tables`, `best_for_drift`, `best_for_long_pdfs`. Each uses a fixed rule (for example `best_overall` = 0.4 × faithfulness + 0.3 × answer correctness + 0.2 × context recall + 0.1 × cost score) and returns the winning config with its rationale. A naive baseline comparison is available at `GET /api/v1/evaluation/baseline-comparison`.

## Demo data

- `corpus/`: 10 openly licensed documents (Creative Commons BY and BY-SA legal code in versions 3.0 and 4.0 as dated version pairs; CNCF survey PDFs with tables and image-only chart labels; an XLSX and a CSV). Sources, licenses and effective dates are in `corpus/manifest.json` and `corpus/README.md`.
- `eval_sets/faithtrace_v1.json`: 86 questions (text, table, chart, spreadsheet; 15 temporal pairs; 10 unanswerable), each answerable one with a verbatim evidence quote. Check a set with `python scripts/validate_eval_set.py <file> --check-evidence`.
- Eval sets can also be uploaded (JSON or CSV) from the New Experiment form or `POST /api/v1/eval-sets/`.

## Architecture

8 Docker Compose services: `postgres` (PostgreSQL 16, 7 tables, Alembic migrations), `redis` (Celery broker), `qdrant` (vectors), `api` (FastAPI, 8 route groups: corpus, experiments, evaluation, eval-sets, diagnostics, recommendations, feedback, system), `worker-ingestion` and `worker-experiments` (Celery: ingest, run experiment, evaluate, diagnose, train classifier), `frontend` (Next.js 14), `nginx` (reverse proxy, rate limits).

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, React Query, Recharts |
| Backend | Python 3.11, FastAPI, Pydantic v2, Celery, Redis |
| LLM / retrieval | LangChain, OpenAI (gpt-4o-mini by default, text-embedding-3-small), rank-bm25, sentence-transformers cross-encoder |
| Evaluation | Ragas + custom metrics |
| Diagnostics | XGBoost, scikit-learn |
| Storage | Qdrant, PostgreSQL 16, SQLAlchemy 2.0, Alembic |
| Parsing | PyMuPDF, pdfplumber, unstructured, openpyxl, python-docx |

## Quick start

```bash
git clone https://github.com/sakshiasati17/FaithTrace.git
cd FaithTrace
cp .env.example .env            # set OPENAI_API_KEY
docker compose up -d --build
python scripts/seed.py          # upload the demo corpus (API_URL defaults to http://localhost)
```

Then open http://localhost, create an experiment with the `faithtrace_v1` eval set, and watch it move to `done`. API docs: http://localhost/docs. Live component status: `GET /api/v1/system/status`.

Documents indexed before the per-strategy chunking and nested-metadata changes must be reindexed (`POST /api/v1/corpus/{id}/reindex`).

## Testing

CI (`.github/workflows/ci.yml`) runs on every push and pull request: backend pytest against a PostgreSQL 16 service, frontend lint, Vitest and a production build.

```bash
cd backend && pytest -q        # FAITHTRACE_TEST_PG_URL=... also runs the PostgreSQL concurrency test
cd frontend && npm ci && npm run lint && npm test -- --run && npm run build
```

TEST_COUNTS_PLACEHOLDER

## Bugs found and fixed

Each of these produced plausible-looking but wrong results rather than an error:

- **Reranker never ran.** `sentence-transformers` was not installed and an `except: pass` hid the import error, so every "hybrid + reranker" run was plain hybrid.
- **Chunking axis did nothing.** Ingestion always chunked `recursive` and retrieval ignored the setting, so all chunking strategies returned identical chunks.
- **Hybrid search leaked stale chunks.** BM25 ignored the freshness and chunk-type filters that vector search applied.
- **Vector-retrieved chunks lost their metadata.** LangChain reads only the payload's `metadata` key, so chunk type, version and dates were missing and freshness checks and diagnosis were blind on vector configs.
- **Crashed queries were scored as answers.** Exceptions became `"Error: …"` answers that Ragas scored and the classifier diagnosed.
- **Silent zeros and fake perfect scores.** Failed Ragas metrics became 0.0, "not applicable" metrics became 1.0, and unscored queries were labelled `NO_FAILURE`.
- **Diagnosis used the wrong question.** Results were paired with questions by list position, and classifier training looked up questions by a key the eval sets don't use.
- **Experiments were `done` before they were scored.** Status flipped when generation finished, while evaluation and diagnosis were still queued.
- **Zero-value bug.** `float(score or 1.0)` turned real 0.0 scores into 1.0.
- **Ingestion races.** Concurrent workers hit a 409 creating the Qdrant collection, retries duplicated chunks, and documents showed `failed` while a retry was still pending.

## Not yet done

- No benchmark results are committed yet: run the 24-config preset on the demo corpus with a real OpenAI key to produce them.

## License

MIT
