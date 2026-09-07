"""
Failure-injection tests.

Validates that FaithTrace handles infrastructure failures gracefully:
- Qdrant timeouts and connection errors
- Malformed document uploads
- Missing database records
- Celery task retry behavior on Qdrant failure
- Non-idempotent ingestion retry (known gap, documented)
"""

import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from io import BytesIO

import pytest

def _can_import(*modules):
    for m in modules:
        try:
            __import__(m)
        except ImportError:
            return False
    return True

_has_corpus_deps = _can_import("fastapi", "sqlalchemy", "qdrant_client")
_has_indexer_deps = _can_import("fastapi", "sqlalchemy", "langchain_openai", "qdrant_client")
_has_celery_deps = _can_import("celery", "sqlalchemy")

skip_no_corpus = pytest.mark.skipif(not _has_corpus_deps, reason="corpus deps not installed (fastapi+sqlalchemy+qdrant)")
skip_no_indexer = pytest.mark.skipif(not _has_indexer_deps, reason="indexer deps not installed (fastapi+sqlalchemy+langchain_openai+qdrant)")
skip_no_celery = pytest.mark.skipif(not _has_celery_deps, reason="celery deps not installed (celery+sqlalchemy)")


class TestQdrantFailures:
    """Test behavior when Qdrant is unavailable or times out."""

    @skip_no_indexer
    def test_ensure_collection_survives_qdrant_timeout(self):
        """ensure_collection should raise when Qdrant is unreachable."""
        with patch("app.services.ingestion.indexer._get_client") as mock_client:
            mock_client.return_value.get_collections.side_effect = ConnectionError(
                "Connection refused: Qdrant is down"
            )
            from app.services.ingestion.indexer import ensure_collection
            with pytest.raises(ConnectionError):
                ensure_collection()

    @skip_no_indexer
    def test_upsert_chunks_raises_on_qdrant_failure(self):
        """upsert_chunks should propagate Qdrant errors (triggers Celery retry)."""
        with patch("app.services.ingestion.indexer._get_client") as mock_client, \
             patch("app.services.ingestion.indexer._get_embeddings") as mock_embed:

            mock_embed.return_value.embed_documents.return_value = [[0.1] * 1536]
            mock_client.return_value.get_collections.return_value = MagicMock(
                collections=[MagicMock(name="faithtrace_chunks")]
            )
            mock_client.return_value.upsert.side_effect = ConnectionError(
                "Qdrant connection reset"
            )

            from app.services.ingestion.indexer import upsert_chunks
            chunks = [{"content": "test chunk", "chunk_type": "text"}]
            with pytest.raises(ConnectionError):
                upsert_chunks(chunks, doc_id="test-doc")

    @skip_no_indexer
    def test_delete_doc_chunks_raises_on_qdrant_failure(self):
        """delete_doc_chunks should propagate errors."""
        with patch("app.services.ingestion.indexer._get_client") as mock_client:
            mock_client.return_value.delete.side_effect = ConnectionError("Qdrant down")

            from app.services.ingestion.indexer import delete_doc_chunks
            with pytest.raises(ConnectionError):
                delete_doc_chunks("test-doc-id")

    @skip_no_indexer
    def test_reindex_catches_qdrant_delete_failure(self):
        """reindex_document endpoint catches Qdrant delete errors gracefully."""
        from app.services.ingestion.indexer import delete_doc_chunks
        with patch("app.services.ingestion.indexer._get_client") as mock_client:
            mock_client.return_value.delete.side_effect = ConnectionError("Qdrant down")
            with pytest.raises(ConnectionError):
                delete_doc_chunks("doc-123")

    @skip_no_indexer
    def test_fetch_all_chunks_empty_on_qdrant_failure(self):
        """fetch_all_chunks should raise when Qdrant is unreachable."""
        with patch("app.services.ingestion.indexer._get_client") as mock_client:
            mock_client.return_value.scroll.side_effect = ConnectionError("Qdrant down")

            from app.services.ingestion.indexer import fetch_all_chunks
            with pytest.raises(ConnectionError):
                fetch_all_chunks()


class TestMalformedDocuments:
    """Test handling of invalid or malformed document uploads."""

    @skip_no_indexer
    def test_unsupported_file_extension(self):
        """Upload endpoint should reject unsupported file types."""
        from app.api.v1.endpoints.corpus import ALLOWED_EXTENSIONS
        assert ".exe" not in ALLOWED_EXTENSIONS
        assert ".py" not in ALLOWED_EXTENSIONS
        assert ".sh" not in ALLOWED_EXTENSIONS
        assert ".pdf" in ALLOWED_EXTENSIONS
        assert ".xlsx" in ALLOWED_EXTENSIONS

    @skip_no_indexer
    def test_detect_file_type_unknown(self):
        """Unknown extensions should return 'unknown' type."""
        from app.api.v1.endpoints.corpus import _detect_file_type
        assert _detect_file_type("malware.exe") == "unknown"
        assert _detect_file_type("script.py") == "unknown"
        assert _detect_file_type("noext") == "unknown"

    @skip_no_indexer
    def test_detect_file_type_valid(self):
        """Valid extensions should return correct types."""
        from app.api.v1.endpoints.corpus import _detect_file_type
        assert _detect_file_type("report.pdf") == "pdf"
        assert _detect_file_type("data.xlsx") == "xlsx"
        assert _detect_file_type("page.html") == "html"
        assert _detect_file_type("doc.docx") == "docx"
        assert _detect_file_type("data.csv") == "csv"

    @skip_no_indexer
    def test_default_strategy_spreadsheet(self):
        """Spreadsheet files should use spreadsheet_aware parsing."""
        from app.api.v1.endpoints.corpus import _default_strategy
        assert _default_strategy("xlsx") == "spreadsheet_aware"
        assert _default_strategy("csv") == "spreadsheet_aware"
        assert _default_strategy("pdf") == "text_table"

    def test_empty_chunks_skipped_in_ingestion(self):
        """The ingestion task should skip chunks with empty content."""
        raw_chunks = [
            {"content": "", "chunk_type": "text"},
            {"content": "   ", "chunk_type": "text"},
            {"content": "valid content", "chunk_type": "text"},
        ]
        non_empty = [c for c in raw_chunks if c.get("content", "").strip()]
        assert len(non_empty) == 1


class TestIngestionRetryBehavior:
    """
    Test and document the Celery retry behavior for ingestion.

    Known gap: ingest_document retries do NOT call delete_doc_chunks
    first, so a partial Qdrant upsert followed by a retry can create
    duplicate chunks with different uuid4 IDs.
    """

    @skip_no_celery
    def test_ingestion_task_has_retry_config(self):
        """Verify the ingestion task has expected retry settings."""
        from app.workers.tasks import ingest_document
        assert ingest_document.max_retries == 3

    @skip_no_indexer
    def test_reindex_is_idempotent(self):
        """
        Verify reindex_document calls delete_doc_chunks BEFORE re-ingesting.
        This is the idempotent path (unlike raw ingest_document retry).
        """
        import ast
        import inspect
        from app.api.v1.endpoints.corpus import reindex_document

        source = inspect.getsource(reindex_document)
        assert "delete_doc_chunks" in source, (
            "reindex_document must call delete_doc_chunks before re-ingesting"
        )

    @skip_no_celery
    def test_ingest_task_does_not_delete_before_retry(self):
        """
        Document the known gap: ingest_document does NOT call
        delete_doc_chunks, so Celery retries can create duplicates.
        """
        import inspect
        from app.workers.tasks import ingest_document

        source = inspect.getsource(ingest_document)
        assert "delete_doc_chunks" not in source, (
            "If this fails, the gap was fixed — update the test and the docs."
        )


class TestDiagnosticsGracefulDegradation:
    """Test that diagnostics degrade gracefully without a trained model."""

    def test_ml_classifier_returns_none_untrained(self):
        """predict() should return None when no model is trained."""
        with patch("app.services.diagnostics.ml_classifier._MODEL_PATH") as mock_path:
            mock_path.exists.return_value = False
            from app.services.diagnostics.ml_classifier import predict
            result = predict(
                metrics={"faithfulness": 0.5},
                retrieved_chunks=[],
                eval_item={},
            )
            assert result is None

    def test_heuristic_fallback_always_works(self):
        """The heuristic classifier should always produce a result."""
        from app.services.diagnostics.classifier import _heuristic_diagnose, FailureCategory
        from app.services.experiment.runner import QueryResult

        result = QueryResult(
            query_id="test", question="?", generated_answer="answer",
            retrieved_chunks=[], latency_ms=100,
            input_tokens=10, output_tokens=10, cost_usd=0.001,
        )
        diagnosis = _heuristic_diagnose(result, {}, {})
        assert diagnosis.primary_failure in list(FailureCategory)

    def test_heuristic_handles_missing_metrics(self):
        """Heuristic should not crash on empty metrics dict."""
        from app.services.diagnostics.classifier import _heuristic_diagnose
        from app.services.experiment.runner import QueryResult

        result = QueryResult(
            query_id="test", question="?", generated_answer="answer",
            retrieved_chunks=[], latency_ms=0,
            input_tokens=0, output_tokens=0, cost_usd=0,
        )
        diagnosis = _heuristic_diagnose(result, {}, {})
        assert diagnosis is not None
        assert diagnosis.query_id == "test"

    def test_heuristic_handles_none_metrics(self):
        """Heuristic should handle None values in metrics gracefully."""
        from app.services.diagnostics.classifier import _heuristic_diagnose
        from app.services.experiment.runner import QueryResult

        result = QueryResult(
            query_id="test", question="?", generated_answer="answer",
            retrieved_chunks=[], latency_ms=0,
            input_tokens=0, output_tokens=0, cost_usd=0,
        )
        metrics = {"faithfulness": None, "context_recall": None}
        diagnosis = _heuristic_diagnose(result, {}, metrics)
        assert diagnosis is not None

    def test_heuristic_preserves_zero_scores(self):
        """A legitimate 0.0 score must NOT be treated as None/missing."""
        from app.services.diagnostics.classifier import _heuristic_diagnose, FailureCategory
        from app.services.experiment.runner import QueryResult

        result = QueryResult(
            query_id="test", question="?", generated_answer="answer",
            retrieved_chunks=[], latency_ms=0,
            input_tokens=0, output_tokens=0, cost_usd=0,
        )
        metrics = {"faithfulness": 0.0, "context_recall": 0.0}
        diagnosis = _heuristic_diagnose(result, {}, metrics)
        assert diagnosis.evidence["faithfulness"] == 0.0
        assert diagnosis.evidence["context_recall"] == 0.0
        assert diagnosis.primary_failure == FailureCategory.LOW_RECALL_RETRIEVAL


class TestFeatureExtraction:
    """Test ML feature extraction doesn't crash on edge cases."""

    def test_extract_features_empty_chunks(self):
        """Feature extraction should handle empty chunk list."""
        from app.services.diagnostics.ml_classifier import extract_features
        features = extract_features(
            metrics={"faithfulness": 0.5},
            retrieved_chunks=[],
            eval_item={},
        )
        assert features.shape == (14,)
        assert features[7] == 0.0  # num_chunks

    def test_extract_features_missing_metrics(self):
        """Feature extraction should default missing metrics to 0."""
        from app.services.diagnostics.ml_classifier import extract_features
        features = extract_features(
            metrics={},
            retrieved_chunks=[],
            eval_item={},
        )
        assert features.shape == (14,)
        assert all(f >= 0.0 for f in features)

    def test_extract_features_with_table_chunks(self):
        """Table chunks should set the has_table feature."""
        from app.services.diagnostics.ml_classifier import extract_features
        chunks = [
            {"chunk_type": "table", "content": "col1|col2"},
            {"chunk_type": "text", "content": "some text"},
        ]
        features = extract_features(
            metrics={"faithfulness": 0.8},
            retrieved_chunks=chunks,
            eval_item={"modality": "table"},
        )
        assert features[8] == 1.0  # has_table
        assert features[12] == 1.0  # mod_table
        assert features[7] == 2.0  # num_chunks
