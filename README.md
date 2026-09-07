# FaithTrace

**RAG Diagnostics Platform — Temporal Drift + Multimodal Failure Detection**

---

## What It Does

FaithTrace benchmarks and diagnoses enterprise RAG pipelines. It targets two gaps most RAG evaluation tools miss:

1. **Temporal drift** — answers that go stale when underlying documents change
2. **Multimodal failures** — breakdowns when evidence lives in tables, charts, or spreadsheets instead of plain text

It runs experiments across 256 pipeline configurations (4 retrieval × 4 chunking × 4 parsing × 4 freshness), evaluates each with Ragas metrics, classifies root-cause failures, and recommends the best config per objective.

---

## Architecture

8-container Docker Compose deployment:

| Container | Role |
|---|---|
| **FastAPI** | REST API (6 route groups) |
| **Next.js** | Frontend dashboard |
| **PostgreSQL** | State + metrics (6 models) |
| **Qdrant** | Vector search |
| **Redis** | Celery broker |
| **Celery Workers** | 5 async tasks (ingest, experiment, evaluate, diagnose, train) |
| **Nginx** | Reverse proxy, rate limiting |
| **Storage** | Local Docker volume |

---

## Pipeline Configuration Matrix

| Axis | Options |
|---|---|
| **Retrieval** | vector_only, bm25, hybrid, hybrid_reranker |
| **Chunking** | fixed_size, recursive, semantic, structure_aware |
| **Parsing** | text_only, text_table, text_table_vision, spreadsheet_aware |
| **Freshness** | none, recency_biased, effective_date_filter, version_aware |

Full matrix: `itertools.product` → 256 configs. MVP subset: 24 configs.

---

## Diagnostics Engine

**Two-tier classifier:**
1. **XGBoost ML** — 14 features (5 Ragas scores + operational metrics + chunk metadata), trained on labeled query results with human feedback
2. **Heuristic fallback** — priority-ordered if/elif chain with metric thresholds

**9 failure categories:** STALE_ANSWER, WRONG_VERSION, TABLE_RETRIEVAL_MISS, CHART_LAYOUT_BLINDNESS, CHUNKING_BOUNDARY_ERROR, LOW_RECALL_RETRIEVAL, IRRELEVANT_CONTEXT_POLLUTION, UNSUPPORTED_SYNTHESIS, NO_FAILURE

**Diagnostic Reasoning Agent:** GPT-4o step-by-step root-cause analysis on failed queries, invoked on-demand via API.

---

## Recommendation Engine

7 objectives: best_overall, lowest_cost, best_latency, best_faithfulness, best_for_tables, best_for_drift, best_for_long_pdfs.

Weighted scoring with configurable weights per objective. Returns ranked configs with metric breakdowns.

---

## Metrics

**Standard (via Ragas):** faithfulness, answer_correctness, context_recall, context_precision, answer_relevance

**Operational:** latency_p50/p95, token usage, cost_per_query

**Custom:** freshness_validity, temporal_citation_accuracy, multimodal_grounding_rate, root_cause_diagnostic_accuracy

---

## Testing

**42 tests, 0 skipped, 0 failures** (all dependencies installed)

| Suite | Tests | Validates |
|---|---|---|
| **Regression** (5 classes) | 21 | Golden eval set integrity, 256-config matrix, baseline comparison deltas, diagnostics stability, recommendation engine |
| **Failure injection** (6 classes) | 21 | Qdrant failures, malformed docs, Celery retry behavior, graceful degradation, ML feature extraction, zero-value preservation |

### Baseline Comparison

Compares any config against a naive baseline (vector_only, fixed_size, text_only, no freshness). Reports 9-metric deltas with absolute/relative improvement.

API: `GET /api/v1/evaluation/baseline-comparison?experiment_id=<id>`

### Bugs Found & Fixed

- **None metrics crash:** `_heuristic_diagnose` threw `TypeError` on `None` metric values. Fixed with null-safe extraction.
- **Zero-value bug:** `float(0.0 or 1.0)` silently converts legitimate `0.0` scores to `1.0` (Python falsy). Fixed with explicit `None` check: `1.0 if _f is None else float(_f)`.

### Known Gap

`ingest_document` Celery retry does NOT call `delete_doc_chunks` before re-upserting — partial failures can create duplicate chunks. `reindex_document` IS idempotent (deletes first). Documented, not yet fixed.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, shadcn/ui, Recharts |
| Backend | Python 3.11, FastAPI, Pydantic v2, Celery, Redis |
| LLM | LangChain, OpenAI (GPT-4o-mini), text-embedding-3-small |
| Evaluation | Ragas (5 metrics), custom evaluators |
| Diagnostics | XGBoost, scikit-learn |
| Vector DB | Qdrant |
| Relational DB | PostgreSQL 16, SQLAlchemy 2.0, Alembic |
| Parsing | PyMuPDF, pdfplumber, unstructured, openpyxl, python-docx |
| Deployment | Docker Compose (8 containers), Nginx reverse proxy |

---

## Quick Start

```bash
git clone https://github.com/sakshiasati17/FaithTrace.git
cd FaithTrace
cp .env.example .env       # add OPENAI_API_KEY
docker compose up -d --build
```

- **UI:** http://localhost
- **API docs:** http://localhost/docs
- **Health:** http://localhost/health

---

## Repository Structure

```
FaithTrace/
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints/       # 6 route groups
│   │   ├── db/                     # 6 SQLAlchemy models, Alembic migrations
│   │   ├── services/
│   │   │   ├── ingestion/          # parser, chunker, indexer
│   │   │   ├── experiment/         # runner, config_matrix
│   │   │   ├── evaluation/         # ragas_runner, metrics, baseline
│   │   │   ├── diagnostics/        # classifier, ml_classifier, reasoning_agent
│   │   │   └── recommendation/     # engine
│   │   └── workers/                # 5 Celery tasks
│   ├── tests/                      # 42 tests (regression + failure injection)
│   └── requirements.txt
├── frontend/                       # Next.js dashboard
├── eval_sets/                      # Benchmark + frozen golden regression set
├── docker-compose.yml              # 8-container orchestration
└── nginx.conf
```

---

## License

MIT
