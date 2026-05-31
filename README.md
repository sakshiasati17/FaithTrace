# FaithTrace

**Temporal + Multimodal RAG Diagnostics Platform for Enterprise Knowledge Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c.svg)](https://pytorch.org)
[![TensorRT](https://img.shields.io/badge/TensorRT-optimized-76b900.svg)](https://developer.nvidia.com/tensorrt)

---

## Overview

FaithTrace is a diagnostics and benchmarking platform for enterprise RAG (Retrieval-Augmented Generation) systems. It goes beyond standard quality metrics by targeting two critical gaps that existing tools largely ignore:

1. **Temporal drift** — answers that become wrong because the underlying knowledge base changed over time
2. **Multimodal retrieval failures** — breakdowns caused by evidence that lives in tables, charts, spreadsheets, or visually structured PDFs rather than plain text

Enterprise corpora are not static and not purely textual. Policies get revised, SOPs are updated, manuals contain diagrams, reports embed tables, and spreadsheets encode operational logic. Standard RAG evaluation pipelines measure answer quality in a snapshot, but they rarely tell you *why* a pipeline failed or whether the failure was caused by stale knowledge, a document-version mismatch, or a retriever that cannot read a table.

FaithTrace fills this gap with a structured experiment engine, a root-cause diagnostics module, a configuration recommendation system, GPU-accelerated inference optimization, and a voice-driven diagnostic interface — all tuned for time-sensitive, multimodal enterprise knowledge.

---

## Key Features

| Feature | Description |
|---|---|
| **Multi-pipeline benchmarking** | Compare vector-only, BM25, hybrid, and hybrid+reranker retrievers across identical question sets |
| **Temporal freshness evaluation** | Score answers against version-correct knowledge using effective-date filtering and recency-biased ranking |
| **Multimodal document parsing** | Extract and index text, tables, page layouts, and spreadsheet cells as first-class retrieval units |
| **Root-cause diagnostics** | Classify each failure as stale content, version mismatch, table miss, chart blindness, chunking error, or ranking failure |
| **Configuration recommendation** | Recommend the best pipeline per objective: lowest cost, highest faithfulness, best latency, best for tables, best for drift-heavy corpora |
| **Diagnostic Reasoning Agent** | LLM-powered agent (GPT-4o) that performs step-by-step root-cause analysis on failed queries with actionable fix suggestions |
| **Autonomous Optimizer Agent** | Iterative search agent that automatically benchmarks pipeline configurations using neighbor-search to converge on the best config |
| **PyTorch Failure Classifier** | Custom DistilBERT fine-tuned classifier with RAGAS score concatenation, focal loss, and label smoothing for 6-class failure categorization |
| **TensorRT Inference Optimization** | GPU auto-profiling, ONNX export, TensorRT FP32/FP16/INT8 engine conversion, and NVIDIA Triton serving with dynamic batching |
| **Voice Interface** | Whisper-powered speech-to-text with intent parsing, API execution, and text-to-speech response — accessible via browser microphone |
| **Leaderboard + trace viewer** | Side-by-side metric comparison with cited chunk/page/table highlights per answer |

---

## Problem Statement

Organizations build RAG systems over internal knowledge bases, but enterprise corpora present challenges that generic RAG evaluation does not address:

- **Knowledge changes**: Policy versions, SOP revisions, pricing rule updates, and amended contracts make retrieving the *correct version* as important as retrieving relevant content
- **Non-textual evidence**: Critical answers often live in a table row, a chart value, or a spreadsheet cell — content that text-only chunking strategies routinely lose or misrepresent
- **Opaque failures**: Current tooling surfaces that a pipeline scored 0.62 on faithfulness, but not *why* — was it a ranking problem, a chunking boundary, a stale document, or a missing table?
- **Inference latency**: Production RAG systems need sub-100ms classification to route failures in real time — standard PyTorch inference is too slow at scale

FaithTrace treats each of these as a measurable, diagnosable, and improvable system property.

---

## Architecture

```
                              ┌───────────┐
                              │   Nginx   │ :80
                              │  (Reverse │
                              │   Proxy)  │
                              └─────┬─────┘
                          ┌─────────┼─────────┐
                          ▼                   ▼
                    ┌──────────┐        ┌──────────┐
                    │ Frontend │ :3000  │ FastAPI  │ :8000
                    │ Next.js  │        │  API     │
                    │ (React)  │        │(16 routes│
                    └──────────┘        └────┬─────┘
                                             │
              ┌──────────────┬───────────────┼───────────────┐
              ▼              ▼               ▼               ▼
       ┌────────────┐ ┌────────────┐  ┌────────────┐  ┌──────────┐
       │ PostgreSQL │ │   Redis    │  │   Qdrant   │  │  Celery  │
       │   (state)  │ │  (broker)  │  │ (vectors)  │  │ Workers  │
       └────────────┘ └────────────┘  └────────────┘  └────┬─────┘
                                                           │
                                            ┌──────────────┼──────────────┐
                                            ▼              ▼              ▼
                                      ┌──────────┐  ┌──────────┐  ┌──────────┐
                                      │Ingestion │  │Experiment│  │Evaluation│
                                      │ Worker   │  │ Worker   │  │& Diag    │
                                      └──────────┘  └──────────┘  └──────────┘

   ┌─────────────────────────────────────────────────────────────────────────┐
   │                         AI & Inference Layer                           │
   │                                                                       │
   │  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────────────┐  │
   │  │   Diagnostic    │  │   Autonomous    │  │   PyTorch DistilBERT  │  │
   │  │   Reasoning     │  │   Optimizer     │  │   Failure Classifier  │  │
   │  │   Agent (GPT-4o)│  │   Agent         │  │   (6-class, focal loss│  │
   │  └─────────────────┘  └─────────────────┘  └───────────┬───────────┘  │
   │                                                        │              │
   │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────▼──────────┐   │
   │  │  Voice Interface│  │  GPU Auto-      │  │  ONNX → TensorRT   │   │
   │  │  Whisper STT    │  │  Profiler       │  │  FP32 / FP16 / INT8│   │
   │  │  + TTS Engine   │  │  (pynvml)       │  │  Engine Conversion │   │
   │  └─────────────────┘  └─────────────────┘  └──────────┬──────────┘   │
   │                                                        │              │
   │                                            ┌───────────▼───────────┐  │
   │                                            │  NVIDIA Triton       │  │
   │                                            │  Inference Server    │  │
   │                                            │  (dynamic batching)  │  │
   │                                            └──────────────────────┘  │
   └─────────────────────────────────────────────────────────────────────────┘
```

---

## NVIDIA Inference Extensions

### 1. Custom PyTorch DistilBERT Failure Classifier

Replaces the XGBoost heuristic classifier with a fine-tuned deep learning model for 6-class failure categorization.

**Architecture:**
- **Encoder:** DistilBERT (`distilbert-base-uncased`) with first 4/6 transformer layers frozen
- **Feature fusion:** 768-dim CLS token concatenated with 5 RAGAS scores → 773-dim input
- **Head:** Linear(773→256) → LayerNorm → GELU → Dropout(0.3) → Linear(256→6)
- **Loss:** Focal Loss (γ=2.0) with label smoothing (ε=0.1) and per-class alpha weights for imbalanced data
- **Optimizer:** AdamW (lr=3e-5) with CosineAnnealingLR and gradient clipping (max_norm=1.0)
- **Early stopping:** patience=7 on macro F1 score

**6 failure classes:** `no_failure`, `retrieval_miss`, `context_insufficient`, `hallucination`, `prompt_weakness`, `ranking_failure`

**Integration:** The diagnostics service tries PyTorch first, falls back to XGBoost, then to rule-based heuristics.

### 2. TensorRT + Triton + GPU Auto-Profiling

Full inference optimization pipeline from PyTorch checkpoint to production serving.

**GPU Auto-Profiler:**
- Detects GPU hardware via `pynvml` (VRAM, compute capability, SM count)
- Recommends optimal precision (FP32/FP16/INT8) and max batch size per device
- Returns human-readable reasoning for precision selection

**ONNX Export:**
- Exports PyTorch classifier with dynamic batch axes for variable runtime batch sizes
- Validates ONNX output against PyTorch reference with configurable tolerance

**TensorRT Conversion:**
- Builds optimized engines at FP32, FP16, and INT8 precision levels
- Skips unsupported precisions based on GPU compute capability

**Inference Benchmarking:**
- Measures p50, p95, p99 latency and throughput (queries/sec)
- Compares raw PyTorch vs TensorRT at all precision levels
- Generates JSON reports with speedup factors

**Triton Inference Server:**
- Model config with dynamic batching (preferred batch sizes: 4, 8, 16)
- Max queue delay: 10ms for latency-sensitive serving
- gRPC client for health checks and inference requests

### 3. Voice Interface (Whisper + TTS)

Natural language voice interface for hands-free diagnostics interaction.

**Speech-to-Text:**
- OpenAI Whisper (`base` model) with GPU acceleration
- Energy-based Voice Activity Detection for noise filtering
- Multi-language support with automatic language detection

**Intent Parser:**
- 12 deterministic regex patterns mapping voice commands to API endpoints
- Extracts parameters (experiment IDs, metric names) from natural language
- Example: *"what went wrong in my last experiment"* → `GET /diagnostics/summary`

**API Execution:**
- Voice commands trigger actual FaithTrace API calls via `httpx`
- Results formatted into natural language spoken responses

**Text-to-Speech:**
- Coqui TTS with `pyttsx3` fallback for environments without GPU
- Diagnostic-specific response formatting

**Browser Integration:**
- MediaRecorder API for microphone capture in the frontend
- Real-time transcription display with confidence scores and detected intent

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

### Inference Optimization Metrics
| Metric | Description |
|---|---|
| **TensorRT Speedup** | Latency reduction vs raw PyTorch (measured at p50/p95/p99) |
| **Throughput (qps)** | Queries per second at each precision level |
| **VRAM Utilization** | GPU memory usage during batch inference |

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

## AI Agents

### Diagnostic Reasoning Agent
An LLM-powered agent that uses GPT-4o to perform step-by-step root-cause analysis on failed RAG queries. Given a question, the generated answer, retrieved chunks, evaluation metrics, and the pre-classified failure category, the agent:
1. Reasons through what went wrong in a structured chain-of-thought
2. Identifies the true root cause
3. Suggests one concrete, actionable fix for the engineering team
4. Produces a plain-English summary for non-technical stakeholders

Invoked on-demand via `POST /api/v1/diagnostics/run/{run_id}/query/{query_id}/reason`. Results are cached to avoid repeated API calls.

### Autonomous Optimizer Agent
An iterative search agent that automatically discovers the best RAG pipeline configuration for a given optimization goal:
1. **Iteration 1:** Runs a broad sweep across the MVP config matrix (24 configurations)
2. **Iteration 2+:** Analyzes the top 3 performers, generates neighbor configs (varying one axis at a time), and benchmarks only those variants
3. **Convergence:** Stops when the target metric threshold is met, the budget is exhausted, or max iterations are reached

Goal definition includes target metric (e.g., faithfulness ≥ 0.85), iteration budget, and cost cap. The agent updates its state in the database after each iteration, enabling real-time progress tracking from the frontend.

Invoked via `POST /api/v1/optimizer/`. Runs as a Celery background task.

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
| **Frontend** | Next.js 14, TypeScript, Tailwind CSS, Recharts, JetBrains Mono |
| **Backend API** | Python 3.11, FastAPI, Pydantic v2, Celery, Redis |
| **RAG / LLM** | LangChain, OpenAI API (GPT-4o, GPT-4o-mini), text-embedding-3-small |
| **Evaluation** | Ragas, custom metric evaluators, LangSmith (optional tracing) |
| **AI Agents** | GPT-4o diagnostic reasoning, autonomous optimizer agent |
| **ML / Classification** | PyTorch, DistilBERT, Focal Loss, XGBoost fallback, scikit-learn |
| **Inference Optimization** | ONNX, TensorRT (FP32/FP16/INT8), NVIDIA Triton Inference Server |
| **GPU Profiling** | pynvml, CUDA compute capability detection |
| **Voice Interface** | OpenAI Whisper (STT), Coqui TTS, energy-based VAD, MediaRecorder API |
| **Vector DB** | Qdrant |
| **Relational DB** | PostgreSQL 16, SQLAlchemy 2.0, Alembic |
| **Parsing** | PyMuPDF, pdfplumber, unstructured, openpyxl, python-docx |
| **Reranking** | Cross-encoder (ms-marco-MiniLM-L-6-v2) |
| **Containerization** | Docker, Docker Compose |
| **Reverse Proxy** | Nginx (rate limiting, SSL termination) |

---

## API Endpoints

### Core Platform
| Method | Endpoint | Description |
|---|---|---|
| `GET/POST` | `/api/v1/corpus/` | Document CRUD and upload |
| `GET/POST` | `/api/v1/experiments/` | Experiment management and execution |
| `GET` | `/api/v1/evaluation/leaderboard` | Ranked pipeline comparison |
| `GET` | `/api/v1/diagnostics/summary` | Failure category breakdown |
| `POST` | `/api/v1/diagnostics/.../reason` | LLM-powered root-cause reasoning |
| `GET` | `/api/v1/recommendations/` | Best config per objective |
| `POST` | `/api/v1/optimizer/` | Launch autonomous optimizer agent |
| `POST` | `/api/v1/feedback/{id}` | Human feedback submission |

### Inference Optimization
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/optimization/gpu-profile` | GPU detection and precision recommendation |
| `POST` | `/api/v1/optimization/export/classifier` | Export PyTorch model to ONNX |
| `POST` | `/api/v1/optimization/export/optimize` | Full PyTorch → ONNX → TensorRT pipeline |
| `POST` | `/api/v1/optimization/benchmark` | Run latency/throughput benchmark |
| `GET` | `/api/v1/optimization/benchmark/latest` | Retrieve benchmark results |
| `POST` | `/api/v1/optimization/train/pytorch` | Train DistilBERT failure classifier |
| `GET` | `/api/v1/optimization/triton/health` | Triton server health check |

### Voice Interface
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/voice/command` | Audio → transcription → intent → API execution |
| `POST` | `/api/v1/voice/speak` | Text → synthesized speech (WAV) |
| `GET` | `/api/v1/voice/health` | Voice model status (Whisper + TTS) |

---

## Research Questions

1. How much does temporal drift degrade RAG performance on operational knowledge bases over time?
2. Do multimodal parsing and retrieval strategies improve answer faithfulness for table- and chart-heavy enterprise documents?
3. Can root-cause diagnostics distinguish retrieval failures caused by stale content, ranking errors, and multimodal blind spots?
4. Which retrieval configuration offers the best quality–cost–latency tradeoff under operational constraints?
5. Can automated recommendation reliably select better RAG configurations than manual tuning?
6. How much inference speedup does TensorRT provide over raw PyTorch for failure classification at scale?

---

## Project Milestones

### MVP (Phase 1) ✅
- [x] Corpus ingestion pipeline for PDF, DOCX, XLSX, CSV, and HTML
- [x] 4 retrieval strategies × 4 chunking × 4 parsing × 4 freshness = 256 pipeline configurations
- [x] Faithfulness, context precision/recall, answer correctness/relevance via Ragas
- [x] Freshness validity and temporal citation accuracy metrics
- [x] Leaderboard UI and config comparison view
- [x] Docker Compose deployment

### Phase 2 ✅
- [x] Table-aware and structure-aware chunking
- [x] Version-aware retrieval with effective-date metadata
- [x] Root-cause failure classifier (XGBoost ML + heuristic fallback)
- [x] Recommendation engine (7 objectives: best overall, lowest cost, best latency, best faithfulness, best for tables, best for drift, best for long PDFs)
- [x] Human feedback loop with classifier retraining

### Phase 3 ✅
- [x] Vision-assisted PDF parsing (GPT-4o Vision for charts and diagrams)
- [x] LLM-powered Diagnostic Reasoning Agent (GPT-4o step-by-step root-cause analysis)
- [x] Autonomous Optimizer Agent (iterative config search with neighbor-expansion and convergence detection)
- [x] Production hardening: Nginx reverse proxy, rate limiting, API key auth

### Phase 4 — NVIDIA Inference Extensions ✅
- [x] Custom PyTorch DistilBERT failure classifier (frozen encoder + RAGAS score fusion + focal loss)
- [x] ONNX export with dynamic batch axes and output validation
- [x] TensorRT engine conversion at FP32, FP16, INT8 precision levels
- [x] GPU auto-profiler with hardware-aware precision selection (pynvml)
- [x] Inference benchmarking: p50/p95/p99 latency + throughput comparison
- [x] NVIDIA Triton Inference Server config with dynamic batching
- [x] Voice interface: Whisper STT + energy-based VAD + regex intent parser
- [x] Voice → API execution: voice commands trigger actual diagnostic API calls
- [x] Browser microphone integration via MediaRecorder API
- [x] Text-to-speech response with Coqui TTS (pyttsx3 fallback)
- [x] Frontend UI polish: stagger animations, card interactions, glass morphism, gradient accents

---

## Getting Started

### Prerequisites
- Docker and Docker Compose
- An OpenAI API key
- NVIDIA GPU + CUDA (optional, for TensorRT/Triton/Whisper GPU acceleration)

### Quick Start (Docker — recommended)

```bash
git clone https://github.com/sakshiasati17/FaithTrace.git
cd FaithTrace
cp .env.example .env       # add your OPENAI_API_KEY
docker compose up -d --build
```

The full platform will be available at:
- **Frontend:** http://localhost (proxied via Nginx)
- **API Docs:** http://localhost/docs (development mode)
- **Health Check:** http://localhost/health

### Manual Setup (development)

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
alembic upgrade head
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

### GPU Acceleration (optional)

```bash
# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install TensorRT (requires NVIDIA GPU)
pip install tensorrt pycuda pynvml

# Enable Triton container in docker-compose.yml (uncomment triton service)
docker compose --profile gpu up -d
```

---

## Repository Structure

```
FaithTrace/
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints/      # REST endpoints
│   │   │   ├── corpus.py          # Document CRUD
│   │   │   ├── experiments.py     # Experiment management
│   │   │   ├── evaluation.py      # Metrics and leaderboard
│   │   │   ├── diagnostics.py     # Failure analysis
│   │   │   ├── recommendations.py # Config recommendations
│   │   │   ├── feedback.py        # Human feedback
│   │   │   ├── optimizer.py       # Autonomous optimizer agent
│   │   │   ├── optimization.py    # GPU profiling, ONNX, TensorRT, benchmarks
│   │   │   └── voice.py           # Whisper STT, TTS, intent execution
│   │   ├── core/                  # Config, security
│   │   ├── db/                    # SQLAlchemy models, Alembic migrations
│   │   ├── schemas/               # Pydantic request/response schemas
│   │   ├── models/
│   │   │   └── failure_classifier/  # PyTorch DistilBERT classifier
│   │   │       ├── model.py       # Architecture (encoder + fusion + head)
│   │   │       ├── dataset.py     # Training dataset with RAGAS score features
│   │   │       ├── losses.py      # Focal loss with label smoothing
│   │   │       └── train.py       # Training loop (AdamW, CosineAnnealingLR)
│   │   ├── optimization/
│   │   │   ├── gpu_profiler.py    # pynvml GPU detection + precision selection
│   │   │   ├── onnx_export.py     # PyTorch → ONNX with dynamic axes
│   │   │   ├── tensorrt_convert.py # ONNX → TensorRT FP32/FP16/INT8
│   │   │   ├── benchmark.py       # p50/p95/p99 latency + throughput
│   │   │   ├── triton_client.py   # Triton gRPC inference client
│   │   │   └── triton_config/     # Triton model repository configs
│   │   ├── voice/
│   │   │   ├── stt.py             # Whisper speech-to-text + VAD
│   │   │   ├── tts.py             # Coqui TTS + pyttsx3 fallback
│   │   │   ├── intent_parser.py   # Regex intent → API endpoint mapping
│   │   │   └── controller.py      # Voice workflow orchestration
│   │   ├── services/
│   │   │   ├── ingestion/         # parser.py, chunker.py, indexer.py
│   │   │   ├── experiment/        # runner.py, config_matrix.py, optimizer.py
│   │   │   ├── evaluation/        # ragas_runner.py, metrics.py
│   │   │   ├── diagnostics/       # classifier.py, ml_classifier.py, reasoning_agent.py
│   │   │   └── recommendation/    # engine.py
│   │   ├── workers/               # Celery tasks (ingestion, experiments, evaluation,
│   │   │                          #   diagnostics, optimizer, training, benchmarks)
│   │   └── main.py
│   ├── alembic/versions/          # Database migrations
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── app/                   # Pages
│   │   │   ├── page.tsx           # Dashboard with KPI strip, pipeline visualization
│   │   │   ├── corpus/            # Document upload and management
│   │   │   ├── experiments/       # Experiment list + detail + run trace
│   │   │   ├── leaderboard/       # Ranked configuration comparison
│   │   │   ├── diagnostics/       # Failure analysis with pie chart + breakdown
│   │   │   ├── recommendations/   # Best config per objective
│   │   │   ├── optimizer/         # Autonomous optimizer job management
│   │   │   ├── inference/         # GPU profile, benchmarks, classifier training, voice
│   │   │   ├── ClientLayout.tsx   # Glass morphism sidebar with grouped navigation
│   │   │   ├── globals.css        # Custom CSS: animations, interactions, effects
│   │   │   ├── error.tsx          # Global error boundary
│   │   │   └── not-found.tsx      # Custom 404 page
│   │   ├── components/ui/
│   │   │   ├── VoicePanel.tsx     # Browser microphone capture + command display
│   │   │   ├── StatusBadge.tsx    # Animated status indicators
│   │   │   └── MetricCard.tsx     # Score cards with conditional coloring
│   │   ├── lib/api.ts             # Typed API client (11 endpoint groups)
│   │   └── types/index.ts         # TypeScript interfaces (20+ types)
│   ├── tailwind.config.js
│   ├── package.json
│   └── Dockerfile
├── eval_sets/                     # Benchmark question sets with ground truth
├── nginx.conf                     # Reverse proxy configuration
├── docker-compose.yml             # Container orchestration (+ optional Triton)
├── .env.example
└── README.md
```

---

## Contributing

Contributions, feedback, and issue reports are welcome.

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
- [OpenAI](https://openai.com) — GPT-4o, Whisper, text-embedding-3-small
- [PyTorch](https://pytorch.org) — Deep learning framework
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) — Inference optimization
- [NVIDIA Triton](https://developer.nvidia.com/triton-inference-server) — Model serving
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — DistilBERT encoder
- [Qdrant](https://qdrant.tech) — Vector database
- [XGBoost](https://xgboost.readthedocs.io) — Gradient-boosted failure classification
- [Coqui TTS](https://github.com/coqui-ai/TTS) — Text-to-speech synthesis
- [Unstructured](https://unstructured.io) — Document parsing
