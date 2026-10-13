"""
Out-of-period chunks never reach the answer.

run_pipeline used to post-filter retrieved chunks by effective_to and, when
nothing passed, return every retrieved chunk. The retrievers already apply the
full date rule (chunk_passes_filters for BM25, _build_freshness_filter in the
Qdrant filter for the vector path), so that post-filter is gone. These tests
check that, for every retrieval strategy, a question whose corpus holds only
out-of-period chunks gets zero retrieved chunks rather than the stale ones.

No network: the vector retriever runs the Qdrant filter run_pipeline built
against an in-process qdrant-client (QdrantClient(":memory:")); BM25 runs over
a mocked fetch_all_chunks; the LLM is mocked.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever

from app.services.experiment import runner
from app.services.experiment.runner import PipelineConfig, run_pipeline


def _epoch(date: str) -> int:
    # Same conversion as indexer._dt_to_epoch
    return int(datetime.fromisoformat(date).timestamp())


# One chunk expired before the query date, one not yet effective on it.
CHUNKS = [
    {"content": "threshold v1 expired", "doc_id": "d", "chunk_type": "text",
     "effective_from": _epoch("2020-01-01"), "effective_to": _epoch("2023-12-31")},
    {"content": "threshold v3 future", "doc_id": "d", "chunk_type": "text",
     "effective_from": _epoch("2025-01-01"), "effective_to": None},
]

QUESTION = {"id": "q_gap", "question": "What is the threshold?", "valid_from": "2024-06-01"}

STRATEGIES = ["vector_only", "bm25", "hybrid", "hybrid_reranker"]


def _config(strategy, freshness):
    return PipelineConfig(
        retrieval_strategy=strategy,
        chunking_strategy="recursive",
        parsing_strategy="text_table",
        freshness_policy=freshness,
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        top_k=10,
    )


@pytest.fixture(scope="module")
def qdrant():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(":memory:")
    client.create_collection("chunks", vectors_config=VectorParams(size=1, distance=Distance.COSINE))
    client.upsert("chunks", points=[
        PointStruct(id=i, vector=[1.0], payload=dict(c)) for i, c in enumerate(CHUNKS)
    ])
    return client


class _QdrantFilterRetriever(BaseRetriever):
    """Stands in for the Qdrant vector retriever: returns the points that pass
    the filter run_pipeline built (a real BaseRetriever, so EnsembleRetriever
    accepts it)."""
    client: object
    qdrant_filter: object = None

    def _get_relevant_documents(self, query, *, run_manager=None):
        records, _ = self.client.scroll("chunks", scroll_filter=self.qdrant_filter, limit=100)
        return [
            Document(page_content=r.payload["content"],
                     metadata={k: v for k, v in r.payload.items() if k != "content"})
            for r in records
        ]


def _run(qdrant, config):
    def build_vector(embedding_model, top_k, qdrant_filter=None, parsing_strategy="text_table"):
        # Combined as the real _build_vector_retriever combines it.
        combined = runner._build_chunk_type_filter(parsing_strategy, qdrant_filter)
        return _QdrantFilterRetriever(client=qdrant, qdrant_filter=combined)

    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="answer", usage_metadata={"input_tokens": 1, "output_tokens": 1})
    rerank = MagicMock(side_effect=lambda docs, question, top_k: docs)
    with patch("app.services.ingestion.indexer.fetch_all_chunks",
               return_value=[dict(c) for c in CHUNKS]), \
         patch.object(runner, "_build_vector_retriever", side_effect=build_vector), \
         patch.object(runner, "_rerank_docs", rerank), \
         patch("langchain_openai.ChatOpenAI", return_value=llm):
        [result] = run_pipeline(config, [QUESTION])
    assert result.status == "ok", result.error_message
    return result


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("freshness", ["effective_date_filter", "version_aware"])
def test_only_out_of_period_chunks_yields_no_chunks(qdrant, strategy, freshness):
    result = _run(qdrant, _config(strategy, freshness))
    assert result.retrieved_chunks == []


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_same_corpus_unfiltered_retrieves_the_chunks(qdrant, strategy):
    # Control: without a date filter the same setup retrieves both chunks, so
    # the empty result above comes from the date rule, not from the mocks.
    result = _run(qdrant, _config(strategy, "none"))
    assert {c["content"] for c in result.retrieved_chunks} == {c["content"] for c in CHUNKS}
