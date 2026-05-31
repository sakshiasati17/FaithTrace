# FaithTrace

**Temporal + Multimodal RAG Diagnostics Platform for Enterprise Knowledge Systems**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c.svg)](https://pytorch.org)
[![TensorRT](https://img.shields.io/badge/TensorRT-optimized-76b900.svg)](https://developer.nvidia.com/tensorrt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

FaithTrace benchmarks and diagnoses enterprise RAG pipelines with focus on two gaps existing tools ignore:

1. **Temporal drift** — answers that go stale when the knowledge base changes over time
2. **Multimodal retrieval failures** — breakdowns from tables, charts, and spreadsheets that text-only retrievers miss

The platform runs structured experiments across 256 pipeline configurations, classifies root-cause failures, recommends optimal configs per objective, and accelerates inference with GPU-optimized models — all accessible through a polished dashboard or voice commands.

---

## Key Features

| Feature | Description |
|---|---|
| **Multi-pipeline benchmarking** | Compare vector, BM25, hybrid, and hybrid+reranker retrievers (256 configs) |
| **Temporal freshness evaluation** | Version-aware retrieval with effective-date filtering |
| **Multimodal document parsing** | Text + tables + charts + spreadsheet cells as first-class retrieval units |
| **Root-cause diagnostics** | 8-category failure classification (stale content, table miss, chunking error, etc.) |
| **Configuration recommendation** | Best pipeline per objective across 7 goals |
| **Diagnostic Reasoning Agent** | GPT-4o step-by-step root-cause analysis with actionable fixes |
| **Autonomous Optimizer** | Iterative search agent that converges on the best config automatically |
| **PyTorch Failure Classifier** | DistilBERT fine-tuned with RAGAS score fusion + focal loss (6-class) |
| **TensorRT Optimization** | GPU auto-profiling → ONNX → TensorRT FP32/FP16/INT8 with Triton serving |
| **Voice Interface** | Whisper STT → intent parsing → API execution → TTS response via browser mic |

---

## Architecture

```
                         ┌───────────┐
                         │   Nginx   │ :80
                         └─────┬─────┘
                     ┌─────────┼─────────┐
                     ▼                   ▼
               ┌──────────┐        ┌──────────┐
               │ Next.js  │ :3000  │ FastAPI  │ :8000
               │ Frontend │        │  (16 API │
               └──────────┘        │  routes) │
                                   └────┬─────┘
            ┌──────────┬────────────────┼────────────────┐
            ▼          ▼                ▼                ▼
      ┌──────────┐ ┌────────┐  ┌────────────┐  ┌─────────────┐
      │PostgreSQL│ │ Redis  │  │   Qdrant   │  │Celery Workers│
      └──────────┘ └────────┘  └────────────┘  └──────┬──────┘
                                                       │
                  ┌────────────────────────────────────┘
                  ▼
   ┌──────────────────────────────────────────────────────────┐
   │                   AI & Inference Layer                   │
   │                                                         │
   │  Diagnostic Reasoning Agent  │  Autonomous Optimizer     │
   │  PyTorch DistilBERT Classifier → ONNX → TensorRT        │
   │  GPU Auto-Profiler (pynvml)  │  Triton Inference Server  │
   │  Whisper STT + Coqui TTS    │  Voice Intent Parser       │
   └──────────────────────────────────────────────────────────┘
```

---

## NVIDIA Inference Extensions

### PyTorch DistilBERT Failure Classifier
- Frozen encoder (4/6 layers) + 5 RAGAS scores concatenated → 773-dim → 6-class head
- Focal Loss (γ=2.0) with label smoothing (ε=0.1) for class imbalance
- AdamW + CosineAnnealingLR + early stopping on macro F1
- Falls back to XGBoost → heuristic rules when no GPU available

### TensorRT + Triton + GPU Profiling
- **GPU Auto-Profiler:** Detects hardware via pynvml, recommends precision and batch size
- **ONNX Export:** Dynamic batch axes with output validation against PyTorch reference
- **TensorRT:** FP32/FP16/INT8 engine conversion, skips unsupported precisions per GPU
- **Benchmarking:** p50/p95/p99 latency + throughput comparison across all backends
- **Triton Server:** Dynamic batching (preferred: 4, 8, 16), 10ms max queue delay

### Voice Interface
- **STT:** OpenAI Whisper with energy-based VAD and language detection
- **Intent Parser:** 12 regex patterns mapping commands to API endpoints with parameter extraction
- **Execution:** Voice commands trigger actual API calls and return natural language responses
- **TTS:** Coqui TTS with pyttsx3 fallback
- **Browser:** MediaRecorder API for microphone capture in the frontend

---

## Technology Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 14, TypeScript, Tailwind CSS, Recharts |
| **Backend** | Python 3.11, FastAPI, Pydantic v2, Celery, Redis |
| **RAG / LLM** | LangChain, OpenAI (GPT-4o, text-embedding-3-small) |
| **Evaluation** | Ragas, custom metrics |
| **ML** | PyTorch, DistilBERT, XGBoost, scikit-learn |
| **Inference** | ONNX, TensorRT, NVIDIA Triton, pynvml |
| **Voice** | OpenAI Whisper, Coqui TTS, pyttsx3 |
| **Storage** | PostgreSQL 16, Qdrant, SQLAlchemy 2.0, Alembic |
| **Parsing** | PyMuPDF, pdfplumber, unstructured, openpyxl |
| **Infrastructure** | Docker Compose, Nginx |

---

## API Endpoints

| Group | Routes | Description |
|---|---|---|
| Corpus | `GET/POST /corpus/` | Document upload, parsing, indexing |
| Experiments | `GET/POST /experiments/` | Pipeline benchmarking (256 configs) |
| Evaluation | `GET /evaluation/leaderboard` | Ranked metric comparison |
| Diagnostics | `GET/POST /diagnostics/` | Failure analysis + GPT-4o reasoning |
| Recommendations | `GET /recommendations/` | Best config per objective (7 goals) |
| Optimizer | `POST /optimizer/` | Autonomous config search agent |
| Optimization | `GET/POST /optimization/` | GPU profiling, ONNX, TensorRT, benchmarks |
| Voice | `POST /voice/command` | Audio → intent → API execution → speech |
| Feedback | `POST /feedback/{id}` | Human-in-the-loop correction |

---

## Failure Categories

| Category | Root Cause |
|---|---|
| `STALE_ANSWER` | Outdated document version used |
| `WRONG_VERSION` | Incorrect policy/SOP version for query date |
| `TABLE_RETRIEVAL_MISS` | Table data not retrieved or indexed |
| `CHART_LAYOUT_BLINDNESS` | Chart/layout evidence missed |
| `CHUNKING_BOUNDARY_ERROR` | Content split across chunk boundaries |
| `LOW_RECALL_RETRIEVAL` | Relevant passage not in top-k |
| `IRRELEVANT_CONTEXT_POLLUTION` | Off-topic chunks contaminating context |
| `UNSUPPORTED_SYNTHESIS` | Claims not supported by any chunk |

---

## Getting Started

### Prerequisites
- Docker and Docker Compose
- OpenAI API key
- NVIDIA GPU + CUDA (optional — for TensorRT, Triton, Whisper GPU)

### Quick Start

```bash
git clone https://github.com/sakshiasati17/FaithTrace.git
cd FaithTrace
cp .env.example .env       # add OPENAI_API_KEY
docker compose up -d --build
```

- **Frontend:** http://localhost
- **API Docs:** http://localhost/docs

### Development Setup

```bash
# Backend
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && alembic upgrade head
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

---

## Repository Structure

```
FaithTrace/
├── backend/app/
│   ├── api/v1/endpoints/          # 9 endpoint modules
│   ├── models/failure_classifier/ # PyTorch DistilBERT (model, dataset, loss, train)
│   ├── optimization/              # GPU profiler, ONNX, TensorRT, benchmark, Triton
│   ├── voice/                     # Whisper STT, TTS, intent parser, controller
│   ├── services/                  # Ingestion, experiment, evaluation, diagnostics,
│   │                              #   recommendation, optimizer
│   ├── workers/                   # Celery tasks (7 task types)
│   └── db/                        # SQLAlchemy models, Alembic migrations
├── frontend/src/
│   ├── app/                       # 11 pages (dashboard, corpus, experiments,
│   │                              #   leaderboard, diagnostics, recommendations,
│   │                              #   optimizer, inference, error, 404)
│   ├── components/ui/             # VoicePanel, StatusBadge, MetricCard
│   ├── lib/api.ts                 # Typed API client (11 endpoint groups)
│   └── types/index.ts             # 20+ TypeScript interfaces
├── docker-compose.yml
├── nginx.conf
└── eval_sets/
```

---

## Project Milestones

- **Phase 1** ✅ — Corpus ingestion, 256 pipeline configs, Ragas evaluation, leaderboard, Docker
- **Phase 2** ✅ — Table-aware chunking, version-aware retrieval, XGBoost classifier, recommendations, feedback loop
- **Phase 3** ✅ — GPT-4o diagnostic reasoning agent, autonomous optimizer agent, Nginx, rate limiting
- **Phase 4** ✅ — PyTorch DistilBERT classifier, TensorRT optimization, Triton serving, voice interface, UI polish

---

## License

MIT License. See [LICENSE](LICENSE) for details.
