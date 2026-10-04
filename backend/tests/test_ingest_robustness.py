"""
Ingestion robustness (found in a live run with 4 Celery workers and a real Qdrant):

  - ensure_collection tolerates another worker creating the collection first
    (Qdrant answers 409 Conflict to the loser);
  - ingest_document replaces a document's chunks (delete, then upsert), so a
    retry after a partial upsert cannot leave duplicates;
  - a document is reported "failed" only when no retry is coming; while a
    retry is pending it stays "running" and the last error is recorded.

Plus the runner response parsing fixes:
  - usage_metadata=None no longer turns a good answer into an errored query;
  - list content is joined into the answer string;
  - the rate-limit retry predicate fallback is tenacity's retry_never.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.messages import AIMessage

from app.core.config import settings
from app.services.experiment.runner import PipelineConfig
from app.services.ingestion import indexer
from app.workers import tasks


# ─── ensure_collection ───────────────────────────────────────────────────────

def _collections(*names):
    return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in names])


class TestEnsureCollection:

    def test_conflict_from_concurrent_create_is_ignored(self):
        client = MagicMock()
        client.get_collections.side_effect = [_collections(), _collections(settings.QDRANT_COLLECTION)]
        client.create_collection.side_effect = RuntimeError("Unexpected Response: 409 (Conflict)")
        with patch.object(indexer, "_get_client", return_value=client):
            indexer.ensure_collection()  # must not raise
        client.create_collection.assert_called_once()

    def test_real_failure_still_raises(self):
        client = MagicMock()
        client.get_collections.side_effect = [_collections(), _collections()]
        client.create_collection.side_effect = RuntimeError("disk full")
        with patch.object(indexer, "_get_client", return_value=client):
            with pytest.raises(RuntimeError, match="disk full"):
                indexer.ensure_collection()

    def test_existing_collection_not_recreated(self):
        client = MagicMock()
        client.get_collections.return_value = _collections(settings.QDRANT_COLLECTION)
        with patch.object(indexer, "_get_client", return_value=client):
            indexer.ensure_collection()
        client.create_collection.assert_not_called()


# ─── ingest_document ─────────────────────────────────────────────────────────

RAW = [{"content": "A sentence about the policy. " * 20, "chunk_type": "text",
        "page": 1, "table_id": None, "metadata": {}}]


def _doc(tmp_path):
    f = tmp_path / "policy.pdf"
    f.write_bytes(b"%PDF")
    return SimpleNamespace(
        id="d1", storage_path=str(f), version_label="v1", filename="policy.pdf",
        effective_from=datetime(2024, 1, 1), effective_to=None,
        parse_status="pending", index_status="pending", doc_metadata={"strategy": "text_table"},
    )


def _ingest(doc, retries, upsert_side_effect=None):
    db = MagicMock()
    db.get.return_value = doc
    order = MagicMock()
    with patch("app.db.session.get_sync_db", return_value=db), \
         patch("app.services.ingestion.parser.parse_document", return_value=RAW), \
         patch("app.services.ingestion.indexer.ensure_collection"), \
         patch("app.services.ingestion.indexer.delete_doc_chunks", order.delete), \
         patch("app.services.ingestion.indexer.upsert_chunks", order.upsert), \
         patch.object(tasks.ingest_document, "retry", side_effect=RuntimeError("retry scheduled")):
        if upsert_side_effect is not None:
            order.upsert.side_effect = upsert_side_effect
        tasks.ingest_document.push_request(retries=retries)
        try:
            outcome = tasks.ingest_document.run(doc.id, "text_table")
        except Exception as exc:  # noqa: BLE001 - the test inspects what was raised
            outcome = exc
        finally:
            tasks.ingest_document.pop_request()
    return outcome, order


class TestIngestIdempotent:

    def test_existing_chunks_deleted_before_upsert(self, tmp_path):
        doc = _doc(tmp_path)
        out, order = _ingest(doc, retries=0)
        assert doc.parse_status == "done"
        names = [c[0] for c in order.mock_calls]
        assert names[0] == "delete" and "upsert" in names
        assert order.delete.call_args == call("d1")

    def test_retry_after_partial_upsert_deletes_again(self, tmp_path):
        doc = _doc(tmp_path)
        _ingest(doc, retries=0, upsert_side_effect=ConnectionError("qdrant dropped"))
        _, order = _ingest(doc, retries=1)  # the retry
        # The retry starts by deleting whatever the partial upsert left behind.
        assert [c[0] for c in order.mock_calls][0] == "delete"
        assert doc.parse_status == "done"


class TestIngestFailureStatus:

    def test_pending_retry_keeps_running_and_records_error(self, tmp_path):
        doc = _doc(tmp_path)
        out, _ = _ingest(doc, retries=0, upsert_side_effect=ConnectionError("qdrant dropped"))
        assert isinstance(out, RuntimeError) and "retry scheduled" in str(out)
        assert doc.parse_status == "running"
        assert doc.doc_metadata["last_error"] == "ConnectionError: qdrant dropped"
        assert doc.doc_metadata["attempts"] == 1

    def test_exhausted_retries_mark_failed_and_raise(self, tmp_path):
        doc = _doc(tmp_path)
        out, _ = _ingest(doc, retries=tasks.ingest_document.max_retries,
                         upsert_side_effect=ConnectionError("qdrant dropped"))
        assert isinstance(out, ConnectionError)  # no further retry scheduled
        assert doc.parse_status == "failed"
        assert doc.index_status == "failed"
        assert doc.doc_metadata["attempts"] == tasks.ingest_document.max_retries + 1


# ─── runner response parsing ─────────────────────────────────────────────────

EVAL = [{"id": "q1", "question": "What is the limit?", "ground_truth": "1500", "modality": "text"}]


def _config():
    return PipelineConfig(
        retrieval_strategy="vector_only", chunking_strategy="recursive",
        parsing_strategy="text_only", freshness_policy="none",
        embedding_model="text-embedding-3-small", llm_model="gpt-4o-mini", top_k=3,
    )


def _run_with_response(response):
    from app.services.experiment import runner
    retriever = MagicMock()
    retriever.invoke.return_value = []
    llm = MagicMock()
    llm.invoke.return_value = response
    with patch("langchain_openai.ChatOpenAI", return_value=llm), \
         patch.object(runner, "_build_vector_retriever", return_value=retriever):
        [result] = runner.run_pipeline(_config(), EVAL)
    return result


class TestRunnerResponseParsing:

    def test_missing_usage_metadata_keeps_the_answer(self):
        msg = AIMessage(content="The limit is 1500.",
                        response_metadata={"token_usage": {"prompt_tokens": 40, "completion_tokens": 6}})
        assert msg.usage_metadata is None
        r = _run_with_response(msg)
        assert r.status == "ok"
        assert r.generated_answer == "The limit is 1500."
        assert (r.input_tokens, r.output_tokens) == (40, 6)  # response_metadata fallback

    def test_list_content_joined_into_answer(self):
        msg = AIMessage(content=[{"type": "text", "text": "The limit "}, "is 1500."],
                        usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14})
        r = _run_with_response(msg)
        assert r.status == "ok"
        assert r.generated_answer == "The limit is 1500."

    def test_retry_fallback_is_retry_never(self):
        import importlib
        import builtins
        from tenacity import retry_never
        from app.services.experiment import runner

        real_import = builtins.__import__

        def no_openai(name, *a, **kw):
            if name == "openai":
                raise ImportError("no openai")
            return real_import(name, *a, **kw)

        try:
            with patch("builtins.__import__", side_effect=no_openai):
                importlib.reload(runner)
            assert runner._retry_on_rate_limit is retry_never
        finally:
            importlib.reload(runner)
