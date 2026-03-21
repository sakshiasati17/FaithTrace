"""
Configuration matrix generator.

Produces the set of PipelineConfig combinations to benchmark in an experiment.
Supports full grid search or a curated subset of high-signal variants.
"""

from itertools import product
from app.services.experiment.runner import PipelineConfig


RETRIEVAL_STRATEGIES = ["vector_only", "bm25", "hybrid", "hybrid_reranker"]
CHUNKING_STRATEGIES = ["fixed_size", "recursive", "semantic", "structure_aware"]
PARSING_STRATEGIES = ["text_only", "text_table", "text_table_vision", "spreadsheet_aware"]
FRESHNESS_POLICIES = ["none", "recency_biased", "effective_date_filter", "version_aware"]


def build_matrix(
    retrieval: list[str] | None = None,
    chunking: list[str] | None = None,
    parsing: list[str] | None = None,
    freshness: list[str] | None = None,
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-4o",
) -> list[PipelineConfig]:
    """
    Generate all combinations of the specified config axes.

    Omit any axis to use all available values for that axis.
    Returns a list of PipelineConfig objects ready for the experiment runner.
    """
    r = retrieval or RETRIEVAL_STRATEGIES
    c = chunking or CHUNKING_STRATEGIES
    p = parsing or PARSING_STRATEGIES
    f = freshness or FRESHNESS_POLICIES

    return [
        PipelineConfig(
            retrieval_strategy=ret,
            chunking_strategy=chk,
            parsing_strategy=prs,
            freshness_policy=frsh,
            embedding_model=embedding_model,
            llm_model=llm_model,
            reranker_enabled=(ret == "hybrid_reranker"),
        )
        for ret, chk, prs, frsh in product(r, c, p, f)
    ]


def build_mvp_matrix(embedding_model: str = "text-embedding-3-small", llm_model: str = "gpt-4o") -> list[PipelineConfig]:
    """
    Curated 6-config MVP matrix covering the most informative combinations.
    """
    return build_matrix(
        retrieval=["vector_only", "hybrid", "hybrid_reranker"],
        chunking=["recursive", "structure_aware"],
        parsing=["text_only", "text_table"],
        freshness=["none", "effective_date_filter"],
        embedding_model=embedding_model,
        llm_model=llm_model,
    )
