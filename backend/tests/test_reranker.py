"""
Reranker tests (hybrid_reranker retrieval strategy).

The cross-encoder is mocked: no model is downloaded in unit tests.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from langchain.schema import Document

from app.services.experiment import runner
from app.services.experiment.runner import PipelineConfig, run_pipeline


# Score each (question, passage) pair by a lookup on the passage text.
_SCORES = {"low": 0.1, "mid": 0.5, "high": 0.9, "top": 2.0}


class _FakeCrossEncoder:
    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, pairs, **kwargs):
        return [_SCORES[passage] for _, passage in pairs]


@pytest.fixture(autouse=True)
def _clear_reranker_cache():
    runner._get_cross_encoder.cache_clear()
    yield
    runner._get_cross_encoder.cache_clear()


def _docs():
    # Ensemble order: deliberately not sorted by cross-encoder score.
    return [
        Document(page_content="low", metadata={"chunk_type": "text"}),
        Document(page_content="high", metadata={"chunk_type": "text"}),
        Document(page_content="mid", metadata={"chunk_type": "table"}),
        Document(page_content="top", metadata={"chunk_type": "text"}),
    ]


def _config(strategy, top_k=2):
    return PipelineConfig(
        retrieval_strategy=strategy,
        chunking_strategy="recursive",
        parsing_strategy="text_table",
        freshness_policy="none",
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        top_k=top_k,
    )


def _run(config, eval_set):
    """Run the pipeline with retrieval and LLM mocked (BM25 unavailable -> vector docs only)."""
    vector_ret = MagicMock()
    vector_ret.invoke.side_effect = lambda q: _docs()
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="answer", usage_metadata={"input_tokens": 1, "output_tokens": 1})
    with patch.object(runner, "_build_vector_retriever", return_value=vector_ret), \
         patch.object(runner, "_build_bm25_retriever", return_value=None), \
         patch("langchain_openai.ChatOpenAI", return_value=llm):
        return run_pipeline(config, eval_set)


def _eval_set(n):
    return [{"id": f"q{i}", "question": f"question {i}"} for i in range(n)]


def test_rerank_docs_orders_by_score_and_keeps_top_k():
    with patch("sentence_transformers.CrossEncoder", _FakeCrossEncoder):
        out = runner._rerank_docs(_docs(), "q", top_k=3)

    assert [d.page_content for d in out] == ["top", "high", "mid"]
    assert all(d.metadata["reranked"] is True for d in out)
    assert [d.metadata["rerank_score"] for d in out] == [2.0, 0.9, 0.5]
    assert out[2].metadata["chunk_type"] == "table"  # existing metadata kept


def test_hybrid_reranker_reorders_retrieved_chunks():
    with patch("sentence_transformers.CrossEncoder", _FakeCrossEncoder):
        results = _run(_config("hybrid_reranker", top_k=2), _eval_set(1))

    chunks = results[0].retrieved_chunks
    assert [c["content"] for c in chunks] == ["top", "high"]
    assert all(c["reranked"] is True for c in chunks)
    assert "rerank_error" not in chunks[0]


def test_cross_encoder_loaded_once_across_queries():
    loader = MagicMock(side_effect=_FakeCrossEncoder)
    with patch("sentence_transformers.CrossEncoder", loader):
        results = _run(_config("hybrid_reranker"), _eval_set(5))

    assert len(results) == 5
    assert loader.call_count == 1
    loader.assert_called_once_with(runner.settings.RERANKER_MODEL)
    assert all(r.retrieved_chunks[0]["content"] == "top" for r in results)


def test_load_failure_marks_chunks_and_logs_warning(caplog):
    loader = MagicMock(side_effect=OSError("model download blocked"))
    with patch("sentence_transformers.CrossEncoder", loader), \
         caplog.at_level(logging.WARNING, logger=runner.__name__):
        results = _run(_config("hybrid_reranker", top_k=2), _eval_set(1))

    chunks = results[0].retrieved_chunks
    # Falls back to the un-reranked docs (same as before), but not silently.
    assert [c["content"] for c in chunks] == ["low", "high", "mid", "top"]
    assert all(c["reranked"] is False for c in chunks)
    assert all("model download blocked" in c["rerank_error"] for c in chunks)
    assert results[0].generated_answer == "answer"

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings and warnings[0].exc_info is not None
    assert "Reranker" in warnings[0].getMessage()


@pytest.mark.parametrize("strategy", ["hybrid", "vector_only"])
def test_non_reranker_configs_unaffected(strategy):
    loader = MagicMock(side_effect=_FakeCrossEncoder)
    with patch("sentence_transformers.CrossEncoder", loader):
        results = _run(_config(strategy, top_k=2), _eval_set(1))

    loader.assert_not_called()
    chunks = results[0].retrieved_chunks
    assert [c["content"] for c in chunks] == ["low", "high", "mid", "top"]
    assert all("reranked" not in c and "rerank_error" not in c for c in chunks)
