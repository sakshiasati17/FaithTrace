// ─── Document / Corpus ────────────────────────────────────────────────────────

export type ParseStatus = "pending" | "running" | "done" | "failed";

export interface Document {
  id: string;
  filename: string;
  file_type: string;
  version_label: string;
  effective_from: string | null;
  effective_to: string | null;
  parse_status: ParseStatus;
  index_status: ParseStatus;
  created_at: string;
}

// ─── Experiment / Run ─────────────────────────────────────────────────────────

export type ExperimentStatus = "pending" | "running" | "done" | "failed";

export interface PipelineConfig {
  retrieval_strategy: "vector_only" | "bm25" | "hybrid" | "hybrid_reranker";
  chunking_strategy: "fixed_size" | "recursive" | "semantic" | "structure_aware";
  parsing_strategy: "text_only" | "text_table" | "text_table_vision" | "spreadsheet_aware";
  freshness_policy: "none" | "recency_biased" | "effective_date_filter" | "version_aware";
  embedding_model: string;
  llm_model: string;
  top_k: number;
  reranker_enabled: boolean;
}

export interface Run {
  id: string;
  experiment_id: string;
  config: PipelineConfig;
  status: ExperimentStatus;
  created_at: string;
  completed_at: string | null;
  metrics?: RunMetrics;
}

export interface Experiment {
  id: string;
  name: string;
  description: string;
  status: ExperimentStatus;
  created_at: string;
  completed_at: string | null;
  runs: Run[];
}

// ─── Metrics ─────────────────────────────────────────────────────────────────

export interface RunMetrics {
  // Standard RAG
  answer_correctness: number | null;
  faithfulness: number | null;
  context_precision: number | null;
  context_recall: number | null;
  answer_relevance: number | null;

  // Operational
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
  avg_token_usage: number | null;
  avg_cost_usd: number | null;

  // DriftLens custom
  freshness_validity: number | null;
  temporal_citation_accuracy: number | null;
  multimodal_grounding_rate: number | null;
}

// ─── Diagnostics ──────────────────────────────────────────────────────────────

export type FailureCategory =
  | "STALE_ANSWER"
  | "WRONG_VERSION"
  | "TABLE_RETRIEVAL_MISS"
  | "CHART_LAYOUT_BLINDNESS"
  | "CHUNKING_BOUNDARY_ERROR"
  | "LOW_RECALL_RETRIEVAL"
  | "IRRELEVANT_CONTEXT_POLLUTION"
  | "UNSUPPORTED_SYNTHESIS"
  | "NO_FAILURE";

export interface QueryDiagnosis {
  query_id: string;
  primary_failure: FailureCategory;
  secondary_failures: FailureCategory[];
  confidence: number;
  evidence: Record<string, unknown>;
}

// ─── Recommendations ─────────────────────────────────────────────────────────

export type Objective =
  | "best_overall"
  | "lowest_cost"
  | "best_latency"
  | "best_faithfulness"
  | "best_for_tables"
  | "best_for_drift"
  | "best_for_long_pdfs";

export interface Recommendation {
  objective: Objective;
  best_config: PipelineConfig;
  run_id: string;
  score: number;
  rationale: string;
}
