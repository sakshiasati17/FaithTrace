"""
Experiment runner.

Executes a RAG pipeline configuration against a benchmark question set
and records per-query results for downstream evaluation and diagnostics.
Uses LangChain for retrieval and LLM orchestration.
"""

import functools
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings

logger = logging.getLogger(__name__)

# Build the rate-limit retry predicate once at module load time.
try:
    from openai import RateLimitError as _OAIRateLimitError
    _retry_on_rate_limit = retry_if_exception_type(_OAIRateLimitError)
except ImportError:
    # openai not installed — define a predicate that never fires so the
    # decorator becomes a no-op instead of incorrectly retrying everything.
    _retry_on_rate_limit = retry_if_exception_type(type(None))


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
    status: str = "ok"                   # "ok" | "error"
    error_message: str | None = None     # set when status == "error"


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
Answer the question using the context below. Extract specific values, thresholds,
names, and dates directly from the text — including inferences like deriving a
maximum from a "less than X" statement. Be concise and precise.
Only say "I cannot find this information in the provided documents." if the context
genuinely contains no relevant information at all.

Context:
{context}

Question: {question}

Answer:"""


# ─── Retrieval filter rules ──────────────────────────────────────────────────
#
# One set of rules drives both retrieval paths: the Qdrant payload filter used
# by vector search, and the in-memory predicate applied to BM25 candidates.

# Chunk types each parsing strategy may retrieve; strategies not listed
# (text_table, text_table_vision) retrieve every chunk type. Values match the
# chunk_type labels the parser emits: text, table, image, spreadsheet_cell.
_ALLOWED_CHUNK_TYPES: dict[str, tuple[str, ...]] = {
    "text_only": ("text",),
    "spreadsheet_aware": ("spreadsheet_cell", "text"),
}

# Freshness policies that restrict chunks to those effective on the query date.
_DATE_FILTERED_POLICIES = ("effective_date_filter", "version_aware")


def _allowed_chunk_types(parsing_strategy: str) -> Optional[tuple[str, ...]]:
    """Chunk types allowed for a parsing strategy, or None for no restriction."""
    return _ALLOWED_CHUNK_TYPES.get(parsing_strategy)


def _query_epoch(freshness_policy: str, eval_item: dict) -> Optional[int]:
    """
    Query date (epoch seconds, as stored by indexer.py) to filter on, or None
    when the policy does not filter by date or the item has no usable date.
    """
    if freshness_policy not in _DATE_FILTERED_POLICIES:
        return None
    valid_from_str = eval_item.get("valid_from")
    if not valid_from_str:
        return None
    try:
        return int(datetime.fromisoformat(valid_from_str).timestamp())
    except (ValueError, TypeError):
        return None


def chunk_passes_filters(
    payload: dict, parsing_strategy: str, freshness_policy: str, eval_item: dict
) -> bool:
    """
    Whether a chunk payload passes the retrieval filters for this config and
    question. Mirrors the Qdrant filter built by _build_freshness_filter +
    _build_chunk_type_filter:

    - chunk_type must be in _allowed_chunk_types(parsing_strategy), if restricted
    - under effective_date_filter / version_aware with a valid query date:
      effective_from <= query date (a missing effective_from does not match,
      as in a Qdrant range condition) and effective_to is null or >= query date
    """
    allowed = _allowed_chunk_types(parsing_strategy)
    if allowed is not None and payload.get("chunk_type") not in allowed:
        return False

    query_epoch = _query_epoch(freshness_policy, eval_item)
    if query_epoch is None:
        return True
    eff_from = payload.get("effective_from")
    if eff_from is None or eff_from > query_epoch:
        return False
    eff_to = payload.get("effective_to")
    return eff_to is None or eff_to >= query_epoch


# ─── Retriever builders ───────────────────────────────────────────────────────

def _build_chunk_type_filter(parsing_strategy: str, existing_filter=None):
    """
    Build a Qdrant chunk_type filter based on parsing_strategy.

    - text_only          → only "text" chunks (no tables)
    - text_table         → all chunk types (no filter)
    - text_table_vision  → all chunk types (no filter)
    - spreadsheet_aware  → "spreadsheet_cell" + "text" chunks
    """
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    allowed = _allowed_chunk_types(parsing_strategy)
    if allowed is None:
        return existing_filter

    chunk_filter = Filter(
        must=[FieldCondition(key="chunk_type", match=MatchAny(any=list(allowed)))]
    )
    if existing_filter is None:
        return chunk_filter
    # Merge: both filters must hold
    return Filter(must=list(existing_filter.must or []) + list(chunk_filter.must or []))


def _build_vector_retriever(embedding_model: str, top_k: int, qdrant_filter=None, parsing_strategy: str = "text_table"):
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
        content_payload_key="content",
    )
    combined_filter = _build_chunk_type_filter(parsing_strategy, qdrant_filter)
    search_kwargs = {"k": top_k}
    if combined_filter:
        search_kwargs["filter"] = combined_filter
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def _load_bm25_corpus() -> Optional[list]:
    """
    Fetch every chunk from Qdrant once and wrap it as a LangChain Document.

    Returns None when BM25 is unavailable (rank-bm25 not installed), so hybrid
    falls back to vector-only. Filtering happens later, per question.
    """
    try:
        from langchain_community.retrievers import BM25Retriever  # noqa: F401
    except ImportError:
        logger.warning("BM25Retriever unavailable; bm25/hybrid configs use no BM25 results")
        return None
    from langchain.schema import Document as LCDoc
    from app.services.ingestion.indexer import fetch_all_chunks

    return [
        LCDoc(
            page_content=str(c.get("content", "") or ""),
            metadata={k: v for k, v in c.items() if k != "content"},
        )
        for c in fetch_all_chunks()
        if c.get("content")
    ]


def _build_bm25_retriever(
    top_k: int,
    parsing_strategy: str,
    freshness_policy: str,
    eval_item: dict,
    cache: dict,
):
    """
    Build a BM25 retriever over the chunks that pass the same chunk-type and
    freshness filters as the vector path (chunk_passes_filters).

    `cache` lives for one run_pipeline call: the corpus is fetched from Qdrant
    once, and a retriever is built once per distinct query date (the only
    filter input that varies between questions of a run).

    Returns None when BM25 is unavailable or no chunk passes the filters.
    """
    if "corpus" not in cache:
        cache["corpus"] = _load_bm25_corpus()
    corpus = cache["corpus"]
    if not corpus:
        return None

    retrievers = cache.setdefault("retrievers", {})
    key = _query_epoch(freshness_policy, eval_item)
    if key not in retrievers:
        from langchain_community.retrievers import BM25Retriever

        docs = [
            d for d in corpus
            if chunk_passes_filters(d.metadata, parsing_strategy, freshness_policy, eval_item)
        ]
        retrievers[key] = BM25Retriever.from_documents(docs, k=top_k) if docs else None
    return retrievers[key]


def _copy_docs(docs: list) -> list:
    """Shallow-copy LangChain docs (new metadata dicts) so cached docs stay unmodified."""
    return [d.copy(update={"metadata": dict(d.metadata or {})}) for d in docs]


def _build_freshness_filter(freshness_policy: str, eval_item: dict):
    """
    Build a Qdrant payload filter for freshness policies:
    effective_from <= query_date AND (effective_to IS NULL OR effective_to >= query_date).
    """
    from qdrant_client.models import (
        Filter, FieldCondition, Range, IsEmptyCondition, IsNullCondition, PayloadField,
    )

    query_epoch = _query_epoch(freshness_policy, eval_item)
    if query_epoch is None:
        return None

    return Filter(
        must=[
            FieldCondition(key="effective_from", range=Range(lte=query_epoch)),
            Filter(
                should=[
                    IsNullCondition(is_null=PayloadField(key="effective_to")),
                    IsEmptyCondition(is_empty=PayloadField(key="effective_to")),
                    FieldCondition(key="effective_to", range=Range(gte=query_epoch)),
                ]
            ),
        ]
    )


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


# ─── Reranker ────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def _get_cross_encoder(model_name: str):
    """Load a cross-encoder once per process (lazily, on first use)."""
    from sentence_transformers import CrossEncoder

    logger.info("Loading reranker model %s", model_name)
    return CrossEncoder(model_name)


def _rerank_docs(docs: list, question: str, top_k: int) -> list:
    """
    Rerank retrieved docs with a cross-encoder and keep the top_k by score.

    Every returned doc gets metadata["reranked"] (True/False). On failure the
    original docs are returned unchanged, marked reranked=False with
    metadata["rerank_error"], and a warning is logged.
    """
    model_name = settings.RERANKER_MODEL
    try:
        model = _get_cross_encoder(model_name)
        scores = model.predict(
            [(question, d.page_content or "") for d in docs], show_progress_bar=False
        )
        ranked = sorted(zip(docs, scores), key=lambda pair: float(pair[1]), reverse=True)
        out = []
        for doc, score in ranked[:top_k]:
            doc.metadata = {**(doc.metadata or {}), "reranked": True, "rerank_score": float(score)}
            out.append(doc)
        return out
    except Exception as e:
        logger.warning("Reranker %s failed; using un-reranked docs", model_name, exc_info=True)
        error = f"{type(e).__name__}: {e}"
        for doc in docs:
            doc.metadata = {**(doc.metadata or {}), "reranked": False, "rerank_error": error}
        return docs


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

    # Per-run BM25 cache: chunks are fetched from Qdrant once, not per question.
    bm25_cache: dict = {}

    for eval_item in eval_set:
        question = eval_item.get("question", "")
        query_id = eval_item.get("id", f"q_{len(results)}")

        start_time = time.monotonic()
        status = "ok"
        error_message = None

        try:
            # Build freshness filter for this eval item
            qdrant_filter = _build_freshness_filter(config.freshness_policy, eval_item)

            # Build retriever
            if config.retrieval_strategy == "vector_only":
                retriever = _build_vector_retriever(
                    config.embedding_model, config.top_k, qdrant_filter,
                    parsing_strategy=config.parsing_strategy,
                )
                lc_docs = retriever.invoke(question)

            elif config.retrieval_strategy == "bm25":
                bm25 = _build_bm25_retriever(
                    config.top_k, config.parsing_strategy, config.freshness_policy,
                    eval_item, bm25_cache,
                )
                if bm25:
                    lc_docs = _copy_docs(bm25.invoke(question))
                else:
                    lc_docs = []

            elif config.retrieval_strategy in ("hybrid", "hybrid_reranker"):
                from langchain.retrievers import EnsembleRetriever

                vector_ret = _build_vector_retriever(
                    config.embedding_model, config.top_k, qdrant_filter,
                    parsing_strategy=config.parsing_strategy,
                )
                bm25_ret = _build_bm25_retriever(
                    config.top_k, config.parsing_strategy, config.freshness_policy,
                    eval_item, bm25_cache,
                )

                if bm25_ret:
                    ensemble = EnsembleRetriever(
                        retrievers=[vector_ret, bm25_ret],
                        weights=[0.6, 0.4],
                    )
                    lc_docs = ensemble.invoke(question)
                else:
                    lc_docs = vector_ret.invoke(question)

                # BM25 docs are shared across questions of the run (cached
                # corpus); copy them so per-question metadata (e.g. reranker
                # scores) never leaks into later questions.
                lc_docs = _copy_docs(lc_docs)

                if config.retrieval_strategy == "hybrid_reranker" and lc_docs:
                    lc_docs = _rerank_docs(lc_docs, question, config.top_k)

            else:
                retriever = _build_vector_retriever(
                    config.embedding_model, config.top_k, qdrant_filter,
                    parsing_strategy=config.parsing_strategy,
                )
                lc_docs = retriever.invoke(question)

            # Filter out any docs with None page_content (LangChain validation issue)
            lc_docs = [d for d in lc_docs if getattr(d, 'page_content', None) is not None]

            # Convert LangChain docs to chunk dicts
            retrieved_chunks = []
            for doc in lc_docs:
                chunk = {"content": doc.page_content or ""}
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

            # Retry only on RateLimitError (3 attempts, 30→120s backoff).
            # _retry_on_rate_limit is defined at module level to avoid per-query import cost.
            @retry(
                retry=_retry_on_rate_limit,
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=30, max=120),
                reraise=True,
            )
            def _invoke_with_retry():
                return llm.invoke([HumanMessage(content=prompt)])

            response = _invoke_with_retry()
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
            # Record the failure instead of passing it off as an answer, so it
            # is excluded from scoring and diagnosis downstream.
            logger.exception("Query %s failed in pipeline run", query_id)
            status = "error"
            error_message = f"{type(e).__name__}: {e}"
            generated_answer = ""
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
            status=status,
            error_message=error_message,
        ))

    return results
