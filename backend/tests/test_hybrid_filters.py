"""
BM25 / hybrid retrieval filter tests (Known Issue #4) and the spreadsheet
chunk-type label (Known Issue #6).

No network: fetch_all_chunks, the vector retriever and the LLM are mocked.
The Qdrant filter is checked against an in-process qdrant-client
(QdrantClient(":memory:")), so no Qdrant server is needed.
"""

import itertools
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever

from app.services.experiment import runner
from app.services.experiment.runner import PipelineConfig, chunk_passes_filters, run_pipeline


def _epoch(date: str) -> int:
    # Same conversion as indexer._dt_to_epoch
    return int(datetime.fromisoformat(date).timestamp())


# v1 of a policy, superseded on 2024-01-01 by v2. Every chunk shares the query
# term "threshold" so BM25 scores all of them.
_V1_FROM, _V1_TO, _V2_FROM = _epoch("2020-01-01"), _epoch("2023-12-31"), _epoch("2024-01-01")

CHUNKS = [
    {"content": "threshold v1 text expired", "chunk_type": "text", "doc_version": "v1",
     "effective_from": _V1_FROM, "effective_to": _V1_TO},
    {"content": "threshold v2 text current", "chunk_type": "text", "doc_version": "v2",
     "effective_from": _V2_FROM, "effective_to": None},
    {"content": "threshold v2 table current", "chunk_type": "table", "doc_version": "v2",
     "effective_from": _V2_FROM, "effective_to": None},
    {"content": "threshold v2 spreadsheet cell", "chunk_type": "spreadsheet_cell", "doc_version": "v2",
     "effective_from": _V2_FROM, "effective_to": None},
    {"content": "threshold v2 image caption", "chunk_type": "image", "doc_version": "v2",
     "effective_from": _V2_FROM, "effective_to": None},
]

QUESTION_2024 = {"id": "q_2024", "question": "What is the threshold?", "valid_from": "2024-06-01"}


def _config(strategy="bm25", parsing="text_table", freshness="effective_date_filter", top_k=10):
    return PipelineConfig(
        retrieval_strategy=strategy,
        chunking_strategy="recursive",
        parsing_strategy=parsing,
        freshness_policy=freshness,
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        top_k=top_k,
    )


class _StaticRetriever(BaseRetriever):
    """Stands in for the Qdrant vector retriever (a real BaseRetriever, so
    EnsembleRetriever accepts it)."""
    docs: list = []

    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content=d.page_content, metadata=dict(d.metadata)) for d in self.docs]


def _run(config, eval_set, chunks=CHUNKS, vector_docs=None):
    """Run the pipeline with Qdrant, the vector retriever and the LLM mocked."""
    fetch = MagicMock(return_value=[dict(c) for c in chunks])
    vector_ret = _StaticRetriever(docs=list(vector_docs or []))
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="answer", usage_metadata={"input_tokens": 1, "output_tokens": 1})
    with patch("app.services.ingestion.indexer.fetch_all_chunks", fetch), \
         patch.object(runner, "_build_vector_retriever", return_value=vector_ret), \
         patch("langchain_openai.ChatOpenAI", return_value=llm):
        results = run_pipeline(config, eval_set)
    assert all(r.status == "ok" for r in results), [r.error_message for r in results]
    return results, fetch


def _contents(result):
    return {c["content"] for c in result.retrieved_chunks}


# ─── BM25 path ──────────────────────────────────────────────────────────────

def test_bm25_excludes_expired_chunk_under_effective_date_filter():
    results, _ = _run(_config(freshness="effective_date_filter"), [QUESTION_2024])
    contents = _contents(results[0])
    assert "threshold v1 text expired" not in contents
    assert "threshold v2 text current" in contents


def test_bm25_expired_chunk_does_not_crowd_out_current_one():
    # top_k=1 and the question favours the v1 wording: unfiltered BM25 ranks
    # the expired chunk first, and the effective_to post-filter falls back to it.
    question = dict(QUESTION_2024, question="threshold v1 expired")
    results, _ = _run(_config(parsing="text_only", top_k=1), [question])
    assert _contents(results[0]) == {"threshold v2 text current"}


def test_bm25_excludes_not_yet_effective_chunk():
    # The effective_to post-filter cannot catch this: only effective_from fails.
    q_2022 = {"id": "q_2022", "question": "threshold v2 current", "valid_from": "2022-06-01"}
    results, _ = _run(_config(parsing="text_only"), [q_2022])
    assert _contents(results[0]) == {"threshold v1 text expired"}


def test_bm25_includes_expired_chunk_under_none():
    results, _ = _run(_config(freshness="none"), [QUESTION_2024])
    assert "threshold v1 text expired" in _contents(results[0])


def test_bm25_text_only_excludes_non_text_chunks():
    results, _ = _run(_config(parsing="text_only", freshness="none"), [QUESTION_2024])
    types = {c["chunk_type"] for c in results[0].retrieved_chunks}
    assert types == {"text"}


def test_bm25_spreadsheet_aware_includes_spreadsheet_cells():
    results, _ = _run(_config(parsing="spreadsheet_aware", freshness="none"), [QUESTION_2024])
    types = {c["chunk_type"] for c in results[0].retrieved_chunks}
    assert types == {"spreadsheet_cell", "text"}


def test_bm25_returns_no_docs_when_nothing_passes_filters():
    question_2019 = {"id": "q_2019", "question": "What is the threshold?", "valid_from": "2019-06-01"}
    results, _ = _run(_config(freshness="effective_date_filter"), [question_2019])
    assert results[0].retrieved_chunks == []


def test_hybrid_ensemble_gets_only_filtered_bm25_docs():
    captured = []

    class _CapturingEnsemble:
        def __init__(self, retrievers, weights):
            self.retrievers = retrievers
            captured.append(self)

        def invoke(self, question):
            return self.retrievers[1].invoke(question)

    vector_doc = Document(page_content="threshold vector doc",
                          metadata={"chunk_type": "text", "effective_from": _V2_FROM, "effective_to": None})
    with patch("langchain.retrievers.EnsembleRetriever", _CapturingEnsemble):
        results, _ = _run(_config("hybrid", parsing="text_only"), [QUESTION_2024], vector_docs=[vector_doc])

    assert len(captured) == 1
    bm25_docs = captured[0].retrievers[1].docs
    assert [d.page_content for d in bm25_docs] == ["threshold v2 text current"]
    assert _contents(results[0]) == {"threshold v2 text current"}


def test_fetch_all_chunks_called_once_per_run():
    eval_set = [dict(QUESTION_2024, id=f"q{i}", valid_from=f"202{i}-06-01") for i in range(5)]
    for strategy in ("bm25", "hybrid", "hybrid_reranker"):
        with patch.object(runner, "_rerank_docs", side_effect=lambda docs, q, k: docs):
            results, fetch = _run(_config(strategy), eval_set)
        assert len(results) == 5
        assert fetch.call_count == 1, strategy


def test_bm25_respects_per_question_date_within_one_run():
    q_2022 = {"id": "q_2022", "question": "What is the threshold?", "valid_from": "2022-06-01"}
    results, fetch = _run(_config(freshness="effective_date_filter"), [q_2022, QUESTION_2024])
    assert _contents(results[0]) == {"threshold v1 text expired"}
    assert "threshold v1 text expired" not in _contents(results[1])
    assert fetch.call_count == 1


def test_cached_bm25_docs_not_mutated_by_reranker():
    def _fake_rerank(docs, question, top_k):
        for d in docs:
            d.metadata = {**d.metadata, "reranked": True, "rerank_error": "boom"}
        return docs

    eval_set = [dict(QUESTION_2024, id="q1"), dict(QUESTION_2024, id="q2")]
    seen = []

    def _rerank_once_then_inspect(docs, question, top_k):
        seen.append([dict(d.metadata) for d in docs])
        return _fake_rerank(docs, question, top_k) if len(seen) == 1 else docs

    with patch.object(runner, "_rerank_docs", side_effect=_rerank_once_then_inspect):
        _run(_config("hybrid_reranker", freshness="none"), eval_set)

    assert len(seen) == 2
    assert all("rerank_error" not in m for m in seen[1])


# ─── Shared predicate vs. Qdrant filter ─────────────────────────────────────

PARSING_STRATEGIES = ["text_only", "text_table", "text_table_vision", "spreadsheet_aware"]
FRESHNESS_POLICIES = ["none", "recency_biased", "effective_date_filter", "version_aware"]


def _payload_grid():
    """Payloads covering every chunk type and effective_from/effective_to edge."""
    q = _epoch("2024-06-01")
    eff_from = [None, q - 1, q, q + 1]
    eff_to = [None, q - 1, q, q + 1, "missing"]
    types = ["text", "table", "image", "spreadsheet_cell", "spreadsheet"]
    payloads = []
    for ctype, f, t in itertools.product(types, eff_from, eff_to):
        p = {"chunk_type": ctype, "effective_from": f}
        if t != "missing":
            p["effective_to"] = t
        payloads.append(p)
    return payloads


@pytest.fixture(scope="module")
def qdrant_memory():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(":memory:")
    client.create_collection("grid", vectors_config=VectorParams(size=1, distance=Distance.COSINE))
    payloads = _payload_grid()
    client.upsert("grid", points=[
        PointStruct(id=i, vector=[1.0], payload=dict(p, idx=i)) for i, p in enumerate(payloads)
    ])
    return client, payloads


@pytest.mark.parametrize("parsing", PARSING_STRATEGIES)
@pytest.mark.parametrize("freshness", FRESHNESS_POLICIES)
@pytest.mark.parametrize("valid_from", ["2024-06-01", None, "not-a-date"])
def test_predicate_matches_qdrant_filter(qdrant_memory, parsing, freshness, valid_from):
    client, payloads = qdrant_memory
    eval_item = {"valid_from": valid_from}

    qdrant_filter = runner._build_chunk_type_filter(
        parsing, runner._build_freshness_filter(freshness, eval_item)
    )
    records, _ = client.scroll("grid", scroll_filter=qdrant_filter, limit=len(payloads) + 1)
    by_qdrant = {r.payload["idx"] for r in records}

    by_predicate = {
        i for i, p in enumerate(payloads)
        if chunk_passes_filters(p, parsing, freshness, eval_item)
    }
    assert by_predicate == by_qdrant


def test_spreadsheet_aware_filter_uses_parser_label():
    allowed = runner._allowed_chunk_types("spreadsheet_aware")
    assert "spreadsheet_cell" in allowed
    assert "spreadsheet" not in allowed
    assert runner._allowed_chunk_types("text_only") == ("text",)
    assert runner._allowed_chunk_types("text_table") == ("text", "table")
    assert runner._allowed_chunk_types("text_table_vision") == ("text", "table", "image")


def test_parser_emits_spreadsheet_cell_label(tmp_path):
    from app.services.ingestion.parser import parse_document

    csv = tmp_path / "t.csv"
    csv.write_text("item,limit\nlaptop,1500\n")
    types = {c["chunk_type"] for c in parse_document(csv, "spreadsheet_aware")}
    assert "spreadsheet_cell" in types
    assert types <= set(runner._allowed_chunk_types("spreadsheet_aware"))
