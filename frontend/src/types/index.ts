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

// pending → running (generating) → evaluating → diagnosing → done, or failed
export type ExperimentStatus =
  | "pending"
  | "running"
  | "evaluating"
  | "diagnosing"
  | "done"
  | "failed";

// Runs stay pending/running/done/failed; evaluated_at/diagnosed_at mark progress.
export type RunStatus = "pending" | "running" | "done" | "failed";

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
  run_id?: string;
  experiment_id: string;
  config: PipelineConfig;
  status: RunStatus;
  created_at: string;
  completed_at: string | null;
  // Absent on older API responses.
  evaluated_at?: string | null;
  diagnosed_at?: string | null;
  metrics?: RunMetrics;
}

export interface Experiment {
  id: string;
  name: string;
  description: string;
  status: ExperimentStatus;
  created_at: string;
  completed_at: string | null;
  // Null on experiments created before eval sets were stored.
  eval_set_id?: string | null;
  eval_set_path?: string | null;
  // Documents retrieval is limited to; null/absent = every document.
  document_ids?: string[] | null;
  runs: Run[];
}

export interface ExperimentCreatePayload {
  name: string;
  description?: string;
  eval_set_id?: string;
  config_preset?: string;
  document_ids?: string[] | null;
}

// ─── Eval sets ────────────────────────────────────────────────────────────────

export type EvalSetSource = "upload" | "builtin";

export interface EvalSetSummary {
  id: string; // uploaded: uuid; built-in: "builtin:<file>.json"
  name: string;
  description: string | null;
  source: EvalSetSource;
  filename: string | null;
  path: string | null; // built-in only: "eval_sets/<file>.json"
  item_count: number;
  created_at: string | null;
  is_default: boolean;
}

export interface EvalSetDetail extends EvalSetSummary {
  items: Record<string, unknown>[];
}

export interface EvalSetUploadResult extends EvalSetSummary {
  warnings: string[];
}

export interface EvalSetRowError {
  row: number | null;
  id: string | null;
  field: string | null;
  message: string;
}

// ─── Query trace ──────────────────────────────────────────────────────────────

export type QueryResultStatus = "ok" | "error";

export interface QueryResult {
  id: string;
  run_id: string;
  query_id: string;
  question: string;
  generated_answer: string;
  retrieved_chunks: Record<string, unknown>[];
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  failure_category: FailureCategory | null;
  diagnosis_evidence: Record<string, unknown>;
  // Absent on older API responses; "error" means the pipeline raised and
  // the query has no answer, metrics, or diagnosis.
  status?: QueryResultStatus | null;
  error_message?: string | null;
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

  // FaithTrace custom
  freshness_validity: number | null;
  temporal_citation_accuracy: number | null;
  multimodal_grounding_rate: number | null;
  root_cause_diagnostic_accuracy: number | null;
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

export interface DiagnosticReasoning {
  reasoning_steps: string[];
  root_cause: string;
  fix_suggestion: string;
  stakeholder_summary: string;
  confidence: number;
}

export interface QueryDiagnosis {
  query_id: string;
  failure_category: FailureCategory;
  secondary_failures: FailureCategory[];
  confidence: number;
  evidence: Record<string, unknown>;
  reasoning?: DiagnosticReasoning;
}

// ─── Feedback ────────────────────────────────────────────────────────────────

export type FeedbackRating = "positive" | "negative";

export interface QueryFeedback {
  id: string;
  rating: FeedbackRating;
  correct_label: FailureCategory | null;
  comment: string | null;
}

export interface FeedbackSummary {
  positive: number;
  negative: number;
  total: number;
  coverage: number;
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



// ─── System status ────────────────────────────────────────────────────────────

export interface ComponentStatus {
  ok: boolean;
  detail?: string;
  latency_ms?: number;
  trained?: boolean;
  classifier_type?: string;
  [extra: string]: unknown;
}

export type SystemComponent = "database" | "qdrant" | "redis" | "workers" | "ml_classifier";

export interface SystemStatus {
  ok: boolean;
  version: string;
  env: string;
  checked_at: string;
  components: Partial<Record<SystemComponent, ComponentStatus>>;
}
