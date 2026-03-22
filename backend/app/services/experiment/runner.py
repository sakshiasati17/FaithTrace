"""
Experiment runner.

Executes a RAG pipeline configuration against a benchmark question set
and records per-query results for downstream evaluation and diagnostics.
Uses LangChain for retrieval and LLM orchestration.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.core.config import settings


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


# ─── LLM cost table (per 1M tokens) ─────────────────────────────────────────

_COST_TABLE = {
    "gpt-4o":            {"input": 5.0,   "output": 15.0},
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":       {"input": 10.0,  "output": 30.0},
    "gpt-3.5-turbo":     {"input": 0.50,  "output": 1.50},
}


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = _COST_TABLE.get(model, {"input": 5.0, "output": 15.0})
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


# ─── RAG Prompt ───────────────────────────────────────────────────────────────

_DEFAULT_PROMPT = """You are an expert assistant for enterprise document analysis.
Use ONLY the context below to answer the question. If the context does not contain
the answer, say "I cannot find this information in the provided documents."

Context:
{context}

Question: {question}

Answer:"""


# ─── Retriever builders ───────────────────────────────────────────────────────

def _build_vector_retriever(embedding_model: str, top_k: int, qdrant_filter=None):
    from langchain_community.vectorstores import Qdrant as LCQdrant
    from langchain_openai import OpenAIEmbeddings
    from qdrant_client import QdrantClient

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    client = QdrantClient(url=settings.QDRANT_URL)
    vectorstore = LCQdrant(
        client=client,
        collection_name=settings.QDRANT_COLLECTION,
        embeddings=embeddings,
    )
    search_kwargs = {"k": top_k}
    if qdrant_filter:
        search_kwargs["filter"] = qdrant_filter
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def _build_bm25_retriever(top_k: int):
    from langchain_community.retrievers import BM25Retriever
    from langchain.schema import Document as LCDoc
    from app.services.ingestion.indexer import fetch_all_chunks

    all_chunks = fetch_all_chunks()
    if not all_chunks:
        return None

    docs = [
        LCDoc(
            page_content=c.get("content", ""),
            metadata={k: v for k, v in c.items() if k != "content"},
        )
        for c in all_chunks
        if c.get("content")
    ]
    if not docs:
        return None

    retriever = BM25Retriever.from_documents(docs, k=top_k)
    return retriever


def _build_freshness_filter(freshness_policy: str, eval_item: dict):
    """Build a Qdrant payload filter for freshness policies."""
    from qdrant_client.models import Filter, FieldCondition, Range

    if freshness_policy == "none" or freshness_policy == "recency_biased":
        return None

    valid_from_str = eval_item.get("valid_from")
    if not valid_from_str:
        return None

    try:
        query_date = datetime.fromisoformat(valid_from_str)
        query_epoch = int(query_date.timestamp())
    except (ValueError, TypeError):
        return None

    if freshness_policy in ("effective_date_filter", "version_aware"):
        # effective_from <= query_date AND (effective_to IS NULL OR effective_to >= query_date)
        # Qdrant: only filter on effective_from <= query_date; we post-filter effective_to
        return Filter(
            must=[
                FieldCondition(
                    key="effective_from",
                    range=Range(lte=query_epoch),
                )
            ]
        )

    return None


def _post_filter_by_effective_to(chunks: list[dict], eval_item: dict, freshness_policy: str) -> list[dict]:
    """Post-filter chunks that have expired before the query date."""
    if freshness_policy not in ("effective_date_filter", "version_aware"):
        return chunks

    valid_from_str = eval_item.get("valid_from")
    if not valid_from_str:
        return chunks

    try:
        query_date = datetime.fromisoformat(valid_from_str)
        query_epoch = int(query_date.timestamp())
    except (ValueError, TypeError):
        return chunks

    filtered = []
    for chunk in chunks:
        eff_to = chunk.get("effective_to")
        if eff_to is None or eff_to >= query_epoch:
            filtered.append(chunk)
    return filtered or chunks  # fallback to all if nothing passes


def _sort_by_recency(chunks: list[dict]) -> list[dict]:
    """Sort chunks by effective_from descending (most recent first)."""
    return sorted(chunks, key=lambda c: c.get("effective_from") or 0, reverse=True)


# ─── Main runner ─────────────────────────────────────────────────────────────

def run_pipeline(config: PipelineConfig, eval_set: list[dict]) -> list[QueryResult]:
    """
    Execute a RAG pipeline over an evaluation question set.

    Args:
        config: Full pipeline configuration
        eval_set: List of question dicts with keys: question, ground_truth, valid_from, valid_to, modality

    Returns:
        List of QueryResult objects, one per question
    """
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage

    llm = ChatOpenAI(
        model=config.llm_model,
        openai_api_key=settings.OPENAI_API_KEY,
        temperature=0,
    )

    results = []

    for eval_item in eval_set:
        question = eval_item.get("question", "")
        query_id = eval_item.get("id", f"q_{len(results)}")

        start_time = time.monotonic()

        try:
            # Build freshness filter for this eval item
            qdrant_filter = _build_freshness_filter(config.freshness_policy, eval_item)

            # Build retriever
            if config.retrieval_strategy == "vector_only":
                retriever = _build_vector_retriever(
                    config.embedding_model, config.top_k, qdrant_filter
                )
                lc_docs = retriever.invoke(question)

            elif config.retrieval_strategy == "bm25":
                bm25 = _build_bm25_retriever(config.top_k)
                if bm25:
                    lc_docs = bm25.invoke(question)
                else:
                    lc_docs = []

            elif config.retrieval_strategy in ("hybrid", "hybrid_reranker"):
                from langchain.retrievers import EnsembleRetriever

                vector_ret = _build_vector_retriever(
                    config.embedding_model, config.top_k, qdrant_filter
                )
                bm25_ret = _build_bm25_retriever(config.top_k)

                if bm25_ret:
                    ensemble = EnsembleRetriever(
                        retrievers=[vector_ret, bm25_ret],
                        weights=[0.6, 0.4],
                    )
                    lc_docs = ensemble.invoke(question)
                else:
                    lc_docs = vector_ret.invoke(question)

                if config.retrieval_strategy == "hybrid_reranker" and lc_docs:
                    try:
                        from langchain.retrievers.document_compressors import CrossEncoderReranker
                        from langchain.retrievers import ContextualCompressionRetriever
                        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

                        model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
                        compressor = CrossEncoderReranker(model=model, top_n=config.top_k)
                        # Apply reranker directly to docs
                        reranked = compressor.compress_documents(lc_docs, question)
                        lc_docs = reranked if reranked else lc_docs
                    except Exception:
                        pass  # reranker is optional enhancement

            else:
                retriever = _build_vector_retriever(
                    config.embedding_model, config.top_k, qdrant_filter
                )
                lc_docs = retriever.invoke(question)

            # Convert LangChain docs to chunk dicts
            retrieved_chunks = []
            for doc in lc_docs:
                chunk = {"content": doc.page_content}
                chunk.update(doc.metadata)
                retrieved_chunks.append(chunk)

            # Apply post-filtering for date range
            retrieved_chunks = _post_filter_by_effective_to(
                retrieved_chunks, eval_item, config.freshness_policy
            )

            # Recency bias: re-sort by effective_from
            if config.freshness_policy == "recency_biased":
                retrieved_chunks = _sort_by_recency(retrieved_chunks)

            # Build context and call LLM
            context_text = "\n\n---\n\n".join(
                c.get("content", "") for c in retrieved_chunks[:config.top_k]
            )

            prompt = _DEFAULT_PROMPT.format(context=context_text, question=question)
            response = llm.invoke([HumanMessage(content=prompt)])

            generated_answer = response.content or ""
            input_tokens = getattr(response, "usage_metadata", {}).get("input_tokens", 0) if hasattr(response, "usage_metadata") else 0
            output_tokens = getattr(response, "usage_metadata", {}).get("output_tokens", 0) if hasattr(response, "usage_metadata") else 0

            # Fallback token count from response_metadata
            if input_tokens == 0 and hasattr(response, "response_metadata"):
                rm = response.response_metadata or {}
                token_usage = rm.get("token_usage", rm.get("usage", {}))
                if isinstance(token_usage, dict):
                    input_tokens = token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0))
                    output_tokens = token_usage.get("completion_tokens", token_usage.get("output_tokens", 0))

        except Exception as e:
            generated_answer = f"Error: {str(e)}"
            retrieved_chunks = []
            input_tokens = 0
            output_tokens = 0

        latency_ms = (time.monotonic() - start_time) * 1000
        cost_usd = _compute_cost(config.llm_model, input_tokens, output_tokens)

        results.append(QueryResult(
            query_id=query_id,
            question=question,
            generated_answer=generated_answer,
            retrieved_chunks=retrieved_chunks,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        ))

    return results
