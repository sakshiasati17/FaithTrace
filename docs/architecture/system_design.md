# DriftLens — System Design

## Component Overview

### Frontend (Next.js 14)

| Page | Purpose |
|---|---|
| `/` | Landing / dashboard summary |
| `/corpus` | Corpus management: upload, version, and track documents |
| `/experiments` | Create, configure, and launch experiments |
| `/experiments/[id]` | Experiment detail: run list, leaderboard |
| `/experiments/[id]/runs/[runId]` | Run detail: per-query trace with chunk highlights |
| `/diagnostics` | Failure category breakdown across experiments |
| `/leaderboard` | Cross-experiment pipeline rankings |

### Backend (FastAPI + Celery)

#### Ingestion flow
```
POST /corpus/upload
  → save file to object storage
  → enqueue ingest_document task
    → parser.parse_document(strategy)
    → chunker.chunk(strategy)
    → embed chunks (OpenAI embeddings)
    → upsert to Qdrant
    → update document.index_status = done
```

#### Experiment flow
```
POST /experiments/
  → create Experiment + Run rows per config
  → enqueue run_experiment task
    → for each Run:
        runner.run_pipeline(config, eval_set)
        → retrieve context per query (LangChain retriever)
        → generate answer (LLM)
        → record QueryResult
    → enqueue evaluate_run
        → metrics.compute_metrics(results, eval_set)
        → persist RunMetrics
    → enqueue diagnose_run
        → classifier.diagnose_run(results, eval_set, query_metrics)
        → persist failure categories
    → generate Recommendations
```

### Data Flow Diagram

```
[User Upload]
    │
    ▼
[Object Storage]──────►[Parser]──────►[Chunker]──────►[Embedder]──────►[Qdrant]
                                                                           │
[Eval Set Upload]                                                          │
    │                                                                      │
    ▼                                                                      ▼
[Experiment Config]────►[Pipeline Runner]◄─────────────────────────[Retriever]
                              │
                              ▼
                        [LLM Generator]
                              │
                              ▼
                        [QueryResults]────►[Evaluation Engine]────►[RunMetrics]
                              │                                        │
                              └────────────►[Diagnostics Engine]───►[FailureLabels]
                                                                        │
                                                                        ▼
                                                               [Recommendation Engine]
                                                                        │
                                                                        ▼
                                                               [PostgreSQL: Recommendations]
```

## Database Schema (Simplified)

```
documents
  id, filename, file_type, version_label, effective_from, effective_to,
  storage_path, parse_status, index_status, created_at, doc_metadata

experiments
  id, name, description, status, created_at, completed_at

runs
  id, experiment_id (FK), config (JSON), status, created_at, completed_at

run_metrics
  id, run_id (FK),
  answer_correctness, faithfulness, context_precision, context_recall, answer_relevance,
  latency_p50_ms, latency_p95_ms, avg_token_usage, avg_cost_usd,
  freshness_validity, temporal_citation_accuracy, multimodal_grounding_rate

query_results
  id, run_id (FK), query_id, question, generated_answer, retrieved_chunks (JSON),
  latency_ms, input_tokens, output_tokens, cost_usd,
  failure_category, diagnosis_evidence (JSON)
```

## Key Design Decisions

### Why Celery over FastAPI background tasks?
Long-running ingestion and experiment jobs need persistent queues, retry logic, and independent worker scaling. Celery with Redis provides this cleanly. Short health checks and metadata reads stay synchronous in FastAPI.

### Why Qdrant?
Qdrant supports payload filtering, which is required for version-aware and effective-date-filtered retrieval. It also supports hybrid (dense + sparse) search natively.

### Why LangChain + LangGraph?
LangChain provides ready-made loaders, retrievers, and eval chains. LangGraph is used for multi-step diagnostic workflows that need branching (e.g., route to multimodal path if modality=table).

### Multimodal chunk types
Each chunk in Qdrant carries a `chunk_type` payload: `text`, `table`, `image`, or `spreadsheet_cell`. Retrievers can filter or weight by chunk type based on the pipeline config, enabling modality-specific ablations.

### Freshness metadata
Each chunk is tagged with `doc_version`, `effective_from`, and `effective_to` payload fields. The `version_aware` freshness policy uses these to filter the Qdrant search to only chunks valid at the query's `query_date` parameter.
