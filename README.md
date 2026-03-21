# FaithTrace

**Temporal + Multimodal RAG Diagnostics Platform for Enterprise Knowledge Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-purple.svg)](https://langchain.com)

---

## Overview

FaithTrace is a diagnostics and benchmarking platform for enterprise RAG (Retrieval-Augmented Generation) systems. It goes beyond standard quality metrics by targeting two critical gaps that existing tools largely ignore:

1. **Temporal drift** — answers that become wrong because the underlying knowledge base changed over time
2. **Multimodal retrieval failures** — breakdowns caused by evidence that lives in tables, charts, spreadsheets, or visually structured PDFs rather than plain text

Enterprise corpora are not static and not purely textual. Policies get revised, SOPs are updated, manuals contain diagrams, reports embed tables, and spreadsheets encode operational logic. Standard RAG evaluation pipelines measure answer quality in a snapshot, but they rarely tell you *why* a pipeline failed or whether the failure was caused by stale knowledge, a document-version mismatch, or a retriever that cannot read a table.

FaithTrace fills this gap with a structured experiment engine, a root-cause diagnostics module, and a configuration recommendation system tuned for time-sensitive, multimodal enterprise knowledge.

---

## Key Features

| Feature | Description |
|---|---|
| **Multi-pipeline benchmarking** | Compare vector-only, BM25, hybrid, and hybrid+reranker retrievers across identical question sets |
| **Temporal freshness evaluation** | Score answers against version-correct knowledge using effective-date filtering and recency-biased ranking |
| **Multimodal document parsing** | Extract and index text, tables, page layouts, and spreadsheet cells as first-class retrieval units |
| **Root-cause diagnostics** | Classify each failure as stale content, version mismatch, table miss, chart blindness, chunking error, or ranking failure |
| **Configuration recommendation** | Recommend the best pipeline per objective: lowest cost, highest faithfulness, best latency, best for tables, best for drift-heavy corpora |
| **Leaderboard + trace viewer** | Side-by-side metric comparison with cited chunk/page/table highlights per answer |

---

## Problem Statement

Organizations build RAG systems over internal knowledge bases, but enterprise corpora present challenges that generic RAG evaluation does not address:

- **Knowledge changes**: Policy versions, SOP revisions, pricing rule updates, and amended contracts make retrieving the *correct version* as important as retrieving relevant content
- **Non-textual evidence**: Critical answers often live in a table row, a chart value, or a spreadsheet cell — content that text-only chunking strategies routinely lose or misrepresent
- **Opaque failures**: Current tooling surfaces that a pipeline scored 0.62 on faithfulness, but not *why* — was it a ranking problem, a chunking boundary, a stale document, or a missing table?

FaithTrace treats each of these as a measurable, diagnosable, and improvable system property.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FaithTrace Platform                         │
│                                                                     │
│  ┌──────────────┐      ┌──────────────────────────────────────────┐ │
│  │   Frontend   │      │               Backend (FastAPI)          │ │
│  │  (Next.js)   │◄────►│                                          │ │
│  │              │      │  ┌────────────┐   ┌────────────────────┐ │ │
│  │ - Experiment │      │  │ Ingestion  │   │ Experiment Engine  │ │ │
│  │   Setup      │      │  │ API        │   │                    │ │ │
│  │ - Corpus     │      │  │            │   │ - Pipeline runner  │ │ │
│  │   Upload     │      │  │ - Parse    │   │ - Config matrix    │ │ │
│  │ - Dashboard  │      │  │ - Chunk    │   │ - Parallel jobs    │ │ │
│  │ - Trace      │      │  │ - Embed    │   └────────────────────┘ │ │
│  │   Viewer     │      │  │ - Version  │                          │ │
│  │ - Leaderboard│      │  │   tracking │   ┌────────────────────┐ │ │
│  └──────────────┘      │  └────────────┘   │ Evaluation Engine  │ │ │
│                        │                   │                    │ │ │
│                        │  ┌────────────┐   │ - Ragas metrics    │ │ │
│                        │  │Diagnostics │   │ - Freshness score  │ │ │
│                        │  │Engine      │   │ - Multimodal       │ │ │
│                        │  │            │   │   grounding rate   │ │ │
│                        │  │ - Failure  │   │ - Latency/cost     │ │ │
│                        │  │   classifier│  └────────────────────┘ │ │
│                        │  │ - Root     │                          │ │ │
│                        │  │   cause    │   ┌────────────────────┐ │ │
│                        │  │   labeling │   │ Recommendation     │ │ │
│                        │  └────────────┘   │ Engine             │ │ │
│                        │                   │                    │ │ │
│                        │                   │ - Best overall     │ │ │
│                        │                   │ - Best for tables  │ │ │
│                        │                   │ - Lowest cost      │ │ │
│                        │                   │ - Best for drift   │ │ │
│                        │                   └────────────────────┘ │ │
│                        └──────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                        Storage Layer                          │  │
│  │                                                               │  │
│  │  PostgreSQL              Vector DB           Object Storage   │  │
│  │  (runs, metrics,         (embeddings,        (raw docs,       │  │
│  │   configs, versions)      chunks, index)      parsed artifacts)│  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Metrics

### Standard RAG Metrics
| Metric | Description |
|---|---|
| Answer Correctness | Semantic similarity between generated and ground-truth answer |
| Faithfulness | Fraction of answer claims grounded in retrieved context |
| Context Precision | Fraction of retrieved chunks that were actually relevant |
| Context Recall | Fraction of relevant information that was retrieved |
| Answer Relevance | How directly the answer addresses the question |

### Operational Metrics
| Metric | Description |
|---|---|
| Latency (p50 / p95) | End-to-end query response time |
| Token Usage | Input + output tokens per query |
| Cost per Query | Estimated cost based on model pricing |
| Reranker Overhead | Latency delta introduced by reranking step |

### FaithTrace Custom Metrics
| Metric | Description |
|---|---|
| **Freshness Validity** | Did the answer draw from the document version valid at query time? |
| **Temporal Citation Accuracy** | Were cited documents not only relevant but also time-correct? |
| **Multimodal Grounding Rate** | For table/chart/spreadsheet questions, did the answer use the correct non-text evidence? |
| **Root-Cause Diagnostic Accuracy** | How accurately does the diagnostics module classify failure type? |

---

## Failure Categories (Diagnostics Engine)

The diagnostics engine classifies each retrieval or generation failure into one of the following root-cause categories:

- `STALE_ANSWER` — answer drew from an outdated document version
- `WRONG_VERSION` — answer used the wrong policy/SOP version relative to the query date
- `TABLE_RETRIEVAL_MISS` — answer required table data that was not retrieved or not indexed
- `CHART_LAYOUT_BLINDNESS` — answer required chart or page-layout evidence that was missed
- `CHUNKING_BOUNDARY_ERROR` — relevant content was split across chunk boundaries
- `LOW_RECALL_RETRIEVAL` — top-k retrieval did not surface the relevant passage
- `IRRELEVANT_CONTEXT_POLLUTION` — retrieved chunks introduced off-topic content that misled generation
- `UNSUPPORTED_SYNTHESIS` — answer made claims that no retrieved chunk supported

---

## Pipeline Variants

FaithTrace benchmarks combinations of the following configuration axes:

### Retrieval Strategy
- `vector_only` — dense embedding search
- `bm25` — sparse keyword retrieval
- `hybrid` — combined dense + sparse
- `hybrid_reranker` — hybrid with cross-encoder reranking

### Chunking Strategy
- `fixed_size` — fixed token/character window
- `recursive` — recursive character text splitter
- `semantic` — embedding-similarity-based boundary detection
- `structure_aware` — layout- and heading-aware splitting for tables and sections

### Multimodal Parsing
- `text_only` — plain text extraction
- `text_table` — text + structured table extraction
- `text_table_vision` — text + tables + vision-assisted page understanding
- `spreadsheet_aware` — workbook-level cell and sheet relationship extraction

### Freshness Policy
- `none` — no temporal filtering
- `recency_biased` — recency-weighted ranking boost
- `effective_date_filter` — strict cutoff based on document effective date
- `version_aware` — version-tagged retrieval with date-range matching

---

## Dataset Design

### Corpus Structure
```
corpus/
├── policies/
│   ├── procurement_policy_v1.pdf       # version 1 — effective Jan 2024
│   ├── procurement_policy_v2.pdf       # version 2 — effective Jul 2024
│   └── procurement_policy_v3.pdf       # version 3 — effective Jan 2025
├── manuals/
│   ├── vendor_manual_A.pdf             # text + diagrams
│   └── equipment_ops_manual.pdf        # text + tables + images
├── sops/
│   ├── exception_handling_v1.docx
│   └── exception_handling_v2.docx
├── reports/
│   ├── monthly_report_jan2025.pdf      # tables + charts
│   └── monthly_report_mar2025.pdf
└── spreadsheets/
    ├── inventory_thresholds.xlsx
    └── pricing_rules_q1_2025.xlsx
```

### Evaluation Set Fields
Each test item includes:
| Field | Description |
|---|---|
| `question` | Natural language query |
| `ground_truth` | Correct answer |
| `source_docs` | List of supporting document IDs |
| `valid_from` / `valid_to` | Effective date range for the correct answer |
| `modality` | `text`, `table`, `chart`, `spreadsheet`, or `mixed` |
| `difficulty` | `easy`, `medium`, `hard` |
| `answerable` | Whether the corpus contains a correct answer |
| `failure_type` | Expected failure category if known |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, shadcn/ui, Recharts |
| Backend API | Python 3.11, FastAPI, Pydantic v2, Celery, Redis |
| RAG / LLM | LangChain, LangGraph, OpenAI API, HuggingFace models |
| Evaluation | Ragas, custom evaluators, LangSmith (optional) |
| Vector DB | Qdrant (default), Weaviate or Chroma (configurable) |
| Relational DB | PostgreSQL 16, SQLAlchemy, Alembic |
| Object Storage | Local filesystem (dev), S3-compatible (prod) |
| Parsing | PyMuPDF, pdfplumber, unstructured, openpyxl, Camelot |
| Containerization | Docker, Docker Compose |
| Testing | Pytest, Vitest, Playwright |

---

## Research Questions

1. How much does temporal drift degrade RAG performance on operational knowledge bases over time?
2. Do multimodal parsing and retrieval strategies improve answer faithfulness for table- and chart-heavy enterprise documents?
3. Can root-cause diagnostics distinguish retrieval failures caused by stale content, ranking errors, and multimodal blind spots?
4. Which retrieval configuration offers the best quality–cost–latency tradeoff under operational constraints?
5. Can automated recommendation reliably select better RAG configurations than manual tuning?

---

## Project Milestones

### MVP (Phase 1)
- [ ] Corpus ingestion pipeline for PDF, DOCX, and XLSX
- [ ] 4–6 RAG pipeline configurations (vector, hybrid, hybrid+rerank)
- [ ] Faithfulness, context precision, latency, and cost metrics via Ragas
- [ ] Basic freshness validity metric
- [ ] Leaderboard UI and config comparison view

### Phase 2
- [ ] Table-aware and structure-aware chunking
- [ ] Version-aware retrieval with effective-date metadata
- [ ] Root-cause failure classifier
- [ ] Recommendation engine (best overall, lowest cost, best for tables, best for drift)
- [ ] Report export (PDF / JSON)

### Phase 3
- [ ] Vision-assisted PDF parsing (chart and image understanding)
- [ ] Temporal drift trend dashboard (corpus snapshot comparison over time)
- [ ] Automated regression gates for pipeline re-evaluation
- [ ] Explainable recommendation rationale

---

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 20+
- Docker and Docker Compose
- PostgreSQL 16
- An OpenAI API key (or compatible LLM endpoint)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/sakshiasati17/FaithTrace.git
cd FaithTrace

# Backend
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # fill in your API keys and DB URL
alembic upgrade head
uvicorn app.main:app --reload

# Frontend
cd ../frontend
npm install
cp .env.local.example .env.local
npm run dev
```

### Docker (recommended for full stack)

```bash
cp .env.example .env
docker compose up --build
```

The app will be available at `http://localhost:3000`. The API docs are at `http://localhost:8000/docs`.

---

## Repository Structure

```
FaithTrace/
├── backend/                    # FastAPI application
│   ├── app/
│   │   ├── api/                # Route handlers
│   │   ├── core/               # Config, security, dependencies
│   │   ├── db/                 # SQLAlchemy models and Alembic migrations
│   │   ├── services/           # Business logic modules
│   │   │   ├── ingestion/      # Document parsing and chunking
│   │   │   ├── experiment/     # Pipeline runner and config matrix
│   │   │   ├── evaluation/     # Metrics computation
│   │   │   ├── diagnostics/    # Failure classification
│   │   │   └── recommendation/ # Config recommendation logic
│   │   ├── workers/            # Celery task definitions
│   │   └── main.py
│   ├── tests/
│   ├── alembic/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                   # Next.js application
│   ├── src/
│   │   ├── app/                # App router pages
│   │   ├── components/         # Reusable UI components
│   │   ├── lib/                # API client and utilities
│   │   └── types/              # TypeScript type definitions
│   ├── tests/
│   ├── package.json
│   └── Dockerfile
├── docs/                       # Architecture diagrams and research notes
│   ├── architecture/
│   ├── metrics/
│   └── dataset/
├── corpus/                     # Sample enterprise document corpus
│   ├── policies/
│   ├── manuals/
│   ├── sops/
│   ├── reports/
│   └── spreadsheets/
├── eval_sets/                  # Benchmark question sets and ground truth
├── scripts/                    # Data generation and migration utilities
├── docker-compose.yml
├── docker-compose.prod.yml
├── .env.example
└── README.md
```

---

## Contributing

This is a master's research project. Contributions, feedback, and issue reports are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit with clear messages: `git commit -m "feat: add table-aware chunking strategy"`
4. Push and open a pull request

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Ragas](https://ragas.io) — RAG evaluation metrics
- [LangChain](https://langchain.com) — RAG pipeline framework
- [LangSmith](https://smith.langchain.com) — experiment tracing
- [Unstructured](https://unstructured.io) — document parsing
- Research on temporal drift in retrieval benchmarks and multimodal enterprise document understanding
