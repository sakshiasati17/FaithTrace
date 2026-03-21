"""
Experiment runner.

Executes a RAG pipeline configuration against a benchmark question set
and records per-query results for downstream evaluation and diagnostics.
"""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Full specification of a single RAG pipeline variant."""
    retrieval_strategy: str          # vector_only | bm25 | hybrid | hybrid_reranker
    chunking_strategy: str           # fixed_size | recursive | semantic | structure_aware
    parsing_strategy: str            # text_only | text_table | text_table_vision | spreadsheet_aware
    freshness_policy: str            # none | recency_biased | effective_date_filter | version_aware
    embedding_model: str             # e.g. text-embedding-3-small
    llm_model: str                   # e.g. gpt-4o
    top_k: int = 5
    reranker_enabled: bool = False
    prompt_template: str = "default"


@dataclass
class QueryResult:
    """Per-query output from a single pipeline run."""
    query_id: str
    question: str
    generated_answer: str
    retrieved_chunks: list[dict]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


def run_pipeline(config: PipelineConfig, eval_set: list[dict]) -> list[QueryResult]:
    """
    Execute a RAG pipeline over an evaluation question set.

    Args:
        config: Full pipeline configuration
        eval_set: List of question dicts with keys: question, ground_truth, valid_from, valid_to, modality

    Returns:
        List of QueryResult objects, one per question
    """
    raise NotImplementedError
