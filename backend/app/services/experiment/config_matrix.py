"""
Configuration matrix generator.

Produces the set of PipelineConfig combinations to benchmark in an experiment.
Supports full grid search or a curated subset of high-signal variants.
"""

from dataclasses import fields
from itertools import product
from typing import Iterable, Mapping

from app.services.experiment.runner import PipelineConfig


RETRIEVAL_STRATEGIES = ["vector_only", "bm25", "hybrid", "hybrid_reranker"]
CHUNKING_STRATEGIES = ["fixed_size", "recursive", "semantic", "structure_aware"]
PARSING_STRATEGIES = ["text_only", "text_table", "text_table_vision", "spreadsheet_aware"]
FRESHNESS_POLICIES = ["none", "recency_biased", "effective_date_filter", "version_aware"]

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
MVP_LLM_MODEL = "gpt-4o-mini"
DEFAULT_TOP_K: int = next(f.default for f in fields(PipelineConfig) if f.name == "top_k")


def build_configs(
    specs: Iterable[Mapping],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[PipelineConfig]:
    """
    Turn config specs into PipelineConfig objects.

    Each spec has retrieval_strategy, chunking_strategy, parsing_strategy,
    freshness_policy and llm_model; top_k is optional (PipelineConfig default).
    The reranker runs exactly when retrieval_strategy is "hybrid_reranker".
    Values are not validated here; the API schema checks them.
    """
    configs = []
    for spec in specs:
        extra = {"top_k": spec["top_k"]} if spec.get("top_k") is not None else {}
        configs.append(PipelineConfig(
            retrieval_strategy=spec["retrieval_strategy"],
            chunking_strategy=spec["chunking_strategy"],
            parsing_strategy=spec["parsing_strategy"],
            freshness_policy=spec["freshness_policy"],
            embedding_model=embedding_model,
            llm_model=spec["llm_model"],
            reranker_enabled=(spec["retrieval_strategy"] == "hybrid_reranker"),
            **extra,
        ))
    return configs


def build_matrix(
    retrieval: list[str] | None = None,
    chunking: list[str] | None = None,
    parsing: list[str] | None = None,
    freshness: list[str] | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
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

    return build_configs(
        (
            {
                "retrieval_strategy": ret,
                "chunking_strategy": chk,
                "parsing_strategy": prs,
                "freshness_policy": frsh,
                "llm_model": llm_model,
            }
            for ret, chk, prs, frsh in product(r, c, p, f)
        ),
        embedding_model=embedding_model,
    )


def build_mvp_matrix(embedding_model: str = DEFAULT_EMBEDDING_MODEL, llm_model: str = MVP_LLM_MODEL) -> list[PipelineConfig]:
    """
    Curated MVP matrix: 3 retrieval × 2 chunking × 2 parsing × 2 freshness = 24 configs.
    Covers the most informative combinations for Phase 1 benchmarking.
    Uses gpt-4o-mini by default — 3x faster and 10x cheaper than gpt-4o for
    optimizer sweeps. Switch to gpt-4o for the final best-config validation run.
    """
    return build_matrix(
        retrieval=["vector_only", "hybrid", "hybrid_reranker"],
        chunking=["recursive", "structure_aware"],
        parsing=["text_only", "text_table"],
        freshness=["none", "effective_date_filter"],
        embedding_model=embedding_model,
        llm_model=llm_model,
    )
