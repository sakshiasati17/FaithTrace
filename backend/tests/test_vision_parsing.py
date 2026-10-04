"""
Vision parsing opt-in tests (Known Issue #6).

Covers:
  - strategy selection: enable_vision false/true, PDF vs non-PDF
  - upload/reindex: 400 without OPENAI_API_KEY, strategy in doc_metadata and response
  - parser vision path on a real corpus PDF with a mocked OpenAI client:
    image chunks with source gpt4o_vision, VISION_MAX_PAGES, per-page errors
  - ingest_document passes image chunks through and records vision stats
  - indexer keeps chunk fields readable by LangChain's Qdrant vector store
No network: the OpenAI client is always replaced.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints import corpus as corpus_ep
from app.core.config import settings
from app.db.models import Document
from app.db.session import get_db
from app.main import app
from app.services.ingestion import parser

CORPUS = Path(__file__).resolve().parents[2] / "corpus"
CHART_PDF = CORPUS / "cncf_ambassador_midyear_survey_2023.pdf"  # 18 pages, pie charts on 4-8

CHART_TEXT = "Pie chart: How long have you been an Ambassador? 1 year 40%, 2 years 35%, 3+ years 25%."
TABLE_TEXT = "| Initiative | Quarter |\n| --- | --- |\n| Mentoring | Q3 |"


class FakeVisionClient:
    """Stands in for openai.OpenAI; answers per page from a callable."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        page = len(self.calls) + 1  # pages are sent in order starting at page 1
        self.calls.append(kwargs)
        content = self.answer(page)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _answers(page):
    if page in (4, 5, 7, 8):
        return CHART_TEXT
    if page == 6:
        raise RuntimeError("simulated vision API error")
    if page == 17:
        return TABLE_TEXT
    return "PLAIN_TEXT_ONLY"


@pytest.fixture
def vision_client(monkeypatch):
    client = FakeVisionClient(_answers)
    monkeypatch.setattr(parser, "_make_vision_client", lambda api_key: client)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test")
    return client


# ─── Strategy selection ──────────────────────────────────────────────────────

class TestSelectStrategy:

    @pytest.mark.parametrize("file_type", ["pdf", "docx", "html", "xlsx", "csv"])
    def test_vision_off_is_default_strategy(self, file_type):
        assert corpus_ep._select_strategy(file_type) == corpus_ep._default_strategy(file_type)
        assert corpus_ep._select_strategy(file_type, False) == corpus_ep._default_strategy(file_type)

    def test_vision_on_pdf(self):
        assert corpus_ep._select_strategy("pdf", True) == "text_table_vision"

    @pytest.mark.parametrize("file_type,expected", [
        ("docx", "text_table"), ("html", "text_table"),
        ("xlsx", "spreadsheet_aware"), ("csv", "spreadsheet_aware"),
    ])
    def test_vision_on_non_pdf_ignored(self, file_type, expected):
        assert corpus_ep._select_strategy(file_type, True) == expected


# ─── Upload / reindex endpoints ──────────────────────────────────────────────

@pytest.fixture
def db():
    session = MagicMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.delete = AsyncMock()
    # created_at is filled by the DB on flush; do it on add for the response.
    session.add = MagicMock(side_effect=lambda d: setattr(d, "created_at", datetime(2026, 1, 1)))
    return session


@pytest.fixture
def client(db, tmp_path, monkeypatch):
    async def _override():
        yield db
    monkeypatch.setattr(settings, "OBJECT_STORAGE_PATH", str(tmp_path))
    monkeypatch.setattr(settings, "API_KEY", "")
    app.dependency_overrides[get_db] = _override
    with patch("app.workers.tasks.ingest_document.delay") as delay:
        tc = TestClient(app)
        tc.delay = delay
        yield tc
    app.dependency_overrides.pop(get_db, None)


def _upload(client, name, data=None):
    return client.post(
        "/api/v1/corpus/upload",
        files={"file": (name, b"%PDF-1.4 test", "application/octet-stream")},
        data=data or {},
    )


class TestUploadEndpoint:

    def test_default_upload_unchanged(self, client, db, monkeypatch):
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
        resp = _upload(client, "a.pdf")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["parsing_strategy"] == "text_table"
        assert body["doc_metadata"]["strategy"] == "text_table"
        assert body["doc_metadata"]["vision_requested"] is False
        assert client.delay.call_args.args[1] == "text_table"

    def test_vision_false_explicit(self, client, monkeypatch):
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test")
        resp = _upload(client, "a.pdf", {"enable_vision": "false"})
        assert resp.status_code == 200, resp.text
        assert client.delay.call_args.args[1] == "text_table"

    def test_vision_true_pdf(self, client, monkeypatch):
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test")
        resp = _upload(client, "charts.pdf", {"enable_vision": "true"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["parsing_strategy"] == "text_table_vision"
        assert body["doc_metadata"]["vision_requested"] is True
        assert "vision_ignored" not in body["doc_metadata"]
        assert client.delay.call_args.args[1] == "text_table_vision"

    def test_vision_true_non_pdf_ignored(self, client, monkeypatch):
        # Ignored even without a key: nothing would be sent to the vision model.
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
        resp = _upload(client, "data.csv", {"enable_vision": "true"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["parsing_strategy"] == "spreadsheet_aware"
        assert "only applies to PDFs" in body["doc_metadata"]["vision_ignored"]
        assert client.delay.call_args.args[1] == "spreadsheet_aware"

    def test_vision_without_key_is_400(self, client, db, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
        resp = _upload(client, "charts.pdf", {"enable_vision": "true"})
        assert resp.status_code == 400
        assert "OPENAI_API_KEY" in resp.json()["detail"]
        client.delay.assert_not_called()
        db.add.assert_not_called()
        assert list(tmp_path.iterdir()) == []  # nothing stored


def _doc_result(doc):
    res = MagicMock()
    res.scalar_one_or_none.return_value = doc
    return res


class TestReindexEndpoint:

    @pytest.fixture
    def stored_doc(self, tmp_path, db):
        f = tmp_path / "doc" / "charts.pdf"
        f.parent.mkdir()
        f.write_bytes(b"%PDF-1.4")
        doc = Document(
            id="d1", filename="charts.pdf", file_type="pdf", version_label="v1",
            storage_path=str(f), parse_status="done", index_status="done",
            created_at=datetime(2026, 1, 1),
            doc_metadata={"original_filename": "charts.pdf", "strategy": "text_table_vision",
                          "vision": {"pages_sent": 3}},
        )
        db.execute.return_value = _doc_result(doc)
        return doc

    def test_reindex_default_drops_vision(self, client, stored_doc):
        with patch("app.services.ingestion.indexer.delete_doc_chunks") as delete:
            resp = client.post("/api/v1/corpus/d1/reindex")
        assert resp.status_code == 200, resp.text
        delete.assert_called_once_with("d1")
        assert resp.json()["parsing_strategy"] == "text_table"
        assert "vision" not in stored_doc.doc_metadata  # stale stats removed
        assert client.delay.call_args.args == ("d1", "text_table")

    def test_reindex_with_vision(self, client, stored_doc, monkeypatch):
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test")
        with patch("app.services.ingestion.indexer.delete_doc_chunks"):
            resp = client.post("/api/v1/corpus/d1/reindex", data={"enable_vision": "true"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["parsing_strategy"] == "text_table_vision"
        assert client.delay.call_args.args == ("d1", "text_table_vision")

    def test_reindex_vision_without_key_is_400(self, client, stored_doc, monkeypatch):
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
        with patch("app.services.ingestion.indexer.delete_doc_chunks") as delete:
            resp = client.post("/api/v1/corpus/d1/reindex", data={"enable_vision": "true"})
        assert resp.status_code == 400
        delete.assert_not_called()  # old chunks kept
        client.delay.assert_not_called()


class TestDocumentResponse:

    def test_parsing_strategy_from_metadata(self):
        from app.schemas.corpus import DocumentResponse
        doc = Document(id="x", filename="f.pdf", file_type="pdf", version_label="v1",
                       storage_path="/x", parse_status="done", index_status="done",
                       created_at=datetime(2026, 1, 1), doc_metadata={"strategy": "text_table_vision"})
        assert DocumentResponse.model_validate(doc).parsing_strategy == "text_table_vision"

    def test_parsing_strategy_missing(self):
        from app.schemas.corpus import DocumentResponse
        doc = Document(id="x", filename="f.pdf", file_type="pdf", version_label="v1",
                       storage_path="/x", parse_status="done", index_status="done",
                       created_at=datetime(2026, 1, 1), doc_metadata={})
        assert DocumentResponse.model_validate(doc).parsing_strategy is None


# ─── Parser vision path (real PDF, mocked OpenAI) ────────────────────────────

class TestParserVision:

    def test_image_chunks_from_chart_pages(self, vision_client, monkeypatch, caplog):
        monkeypatch.setattr(settings, "VISION_MAX_PAGES", 20)
        report = {}
        with caplog.at_level(logging.WARNING, logger="app.services.ingestion.parser"):
            chunks = parser.parse_document(CHART_PDF, "text_table_vision", report=report)

        vision = [c for c in chunks if c["metadata"].get("source") == "gpt4o_vision"]
        assert [c["page"] for c in vision] == [4, 5, 7, 8, 17]
        assert all(c["chunk_type"] == "image" for c in vision)
        assert all(c["metadata"]["strategy"] == "text_table_vision" for c in vision)
        assert vision[0]["content"] == CHART_TEXT
        assert vision[-1]["metadata"]["vision_has_table"] is True
        assert vision[0]["metadata"]["vision_has_table"] is False

        # Text and pdfplumber tables are still there, same as text_table.
        baseline = parser.parse_document(CHART_PDF, "text_table")
        non_vision = [c for c in chunks if c["metadata"].get("source") != "gpt4o_vision"]
        strip = lambda cs: [(c["chunk_type"], c["page"], c["content"]) for c in cs]
        assert sorted(strip(non_vision)) == sorted(strip(baseline))

        # Every page was sent; the model comes from VISION_MODEL; images are PNG data URLs.
        assert len(vision_client.calls) == 18
        call = vision_client.calls[0]
        assert call["model"] == settings.VISION_MODEL
        image_part = call["messages"][0]["content"][1]
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

        # Page 6 failed: logged and recorded, other pages still parsed.
        assert report["vision"]["errors"] == [
            {"page": 6, "error": "RuntimeError: simulated vision API error"}
        ]
        assert "page 6" in caplog.text and "simulated vision API error" in caplog.text
        assert report["vision"]["pages_sent"] == 18
        assert report["vision"]["pages_skipped"] == 0
        assert report["vision"]["vision_chunks"] == 5
        assert report["vision"]["plain_text_pages"] == 12

    def test_max_pages_respected(self, vision_client, monkeypatch, caplog):
        monkeypatch.setattr(settings, "VISION_MAX_PAGES", 5)
        report = {}
        with caplog.at_level(logging.INFO, logger="app.services.ingestion.parser"):
            chunks = parser.parse_document(CHART_PDF, "text_table_vision", report=report)
        assert len(vision_client.calls) == 5
        vision_pages = [c["page"] for c in chunks if c["chunk_type"] == "image"]
        assert vision_pages == [4, 5]
        assert report["vision"]["pages_sent"] == 5
        assert report["vision"]["pages_skipped"] == 13
        assert "skipped 13" in caplog.text
        # Text of skipped pages is still extracted.
        assert {c["page"] for c in chunks if c["chunk_type"] == "text"} >= {17}

    def test_max_pages_zero_sends_nothing(self, vision_client, monkeypatch):
        monkeypatch.setattr(settings, "VISION_MAX_PAGES", 0)
        chunks = parser.parse_document(CHART_PDF, "text_table_vision")
        assert vision_client.calls == []
        assert not [c for c in chunks if c["chunk_type"] == "image"]

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
        monkeypatch.setattr(parser, "_make_vision_client",
                            lambda api_key: pytest.fail("client must not be created"))
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            parser.parse_document(CHART_PDF, "text_table_vision")

    def test_default_strategy_makes_no_vision_calls(self, monkeypatch):
        monkeypatch.setattr(parser, "_make_vision_client",
                            lambda api_key: pytest.fail("client must not be created"))
        chunks = parser.parse_document(CHART_PDF, "text_table")
        assert chunks and not [c for c in chunks if c["chunk_type"] == "image"]

    def test_classifier_sees_vision_chunk(self, vision_client):
        from app.services.diagnostics.classifier import _has_vision_chunk
        chunks = parser.parse_document(CHART_PDF, "text_table_vision")
        image = next(c for c in chunks if c["chunk_type"] == "image")
        # Retrieved chunks are flat (runner merges the stored payload into the chunk).
        retrieved = {"content": image["content"], "chunk_type": image["chunk_type"], **image["metadata"]}
        assert _has_vision_chunk(SimpleNamespace(retrieved_chunks=[retrieved]))


# ─── ingest_document ─────────────────────────────────────────────────────────

class TestIngestDocument:

    def _run(self, doc, raw_chunks, report_extra=None, retries=0):
        from app.workers.tasks import ingest_document

        def fake_parse(path, strategy, report=None):
            if report is not None and report_extra:
                report.update(report_extra)
            return raw_chunks

        db = MagicMock()
        db.get.return_value = doc
        with patch("app.db.session.get_sync_db", return_value=db), \
             patch("app.services.ingestion.parser.parse_document", side_effect=fake_parse) as parse, \
             patch("app.services.ingestion.indexer.ensure_collection"), \
             patch("app.services.ingestion.indexer.upsert_chunks") as upsert:
            ingest_document.push_request(retries=retries)
            try:
                result = ingest_document.run(doc.id, doc.doc_metadata["strategy"])
            finally:
                ingest_document.pop_request()
        return result, upsert, parse

    def _doc(self, tmp_path):
        f = tmp_path / "charts.pdf"
        f.write_bytes(b"%PDF")
        return SimpleNamespace(
            id="d1", storage_path=str(f), version_label="2023", filename="charts.pdf",
            effective_from=datetime(2023, 11, 29), effective_to=None,
            parse_status="pending", index_status="pending",
            doc_metadata={"strategy": "text_table_vision"},
        )

    def test_image_chunks_pass_through(self, tmp_path):
        doc = self._doc(tmp_path)
        image = {"content": CHART_TEXT, "chunk_type": "image", "page": 4, "table_id": "vision_4",
                 "metadata": {"source": "gpt4o_vision", "strategy": "text_table_vision"}}
        text = {"content": "Some page text", "chunk_type": "text", "page": 4, "table_id": None, "metadata": {}}
        stats = {"vision": {"pages_sent": 18, "errors": []}}

        result, upsert, _ = self._run(doc, [text, image], stats)

        # The short text is one chunk per default chunking strategy; the image once.
        assert result["chunks_indexed"] == 4
        assert result["chunks_by_strategy"] == {
            "fixed_size": 1, "recursive": 1, "structure_aware": 1, "atomic": 1,
        }
        indexed = upsert.call_args.args[0]
        img = next(c for c in indexed if c["chunk_type"] == "image")
        assert img["content"] == CHART_TEXT  # atomic, not re-chunked
        assert img["metadata"]["source"] == "gpt4o_vision"
        assert img["table_id"] == "vision_4" and img["doc_version"] == "2023"
        assert doc.parse_status == "done"
        assert doc.doc_metadata["vision"] == {"pages_sent": 18, "errors": []}
        assert doc.doc_metadata["strategy"] == "text_table_vision"
        # Cached so a Celery retry does not pay for vision again.
        cached = json.loads((tmp_path / "parsed_text_table_vision.json").read_text())
        assert cached["chunks"][1]["chunk_type"] == "image"

    def test_retry_reuses_cached_vision_parse(self, tmp_path):
        doc = self._doc(tmp_path)
        image = {"content": CHART_TEXT, "chunk_type": "image", "page": 4, "table_id": "vision_4",
                 "metadata": {"source": "gpt4o_vision"}}
        (tmp_path / "parsed_text_table_vision.json").write_text(
            json.dumps({"chunks": [image], "report": {"vision": {"pages_sent": 1}}}))

        result, upsert, parse = self._run(doc, [], retries=1)

        parse.assert_not_called()
        assert result["chunks_indexed"] == 1
        assert doc.doc_metadata["vision"] == {"pages_sent": 1}

    def test_first_attempt_ignores_stale_cache(self, tmp_path):
        doc = self._doc(tmp_path)
        (tmp_path / "parsed_text_table_vision.json").write_text(json.dumps({"chunks": [], "report": {}}))
        text = {"content": "fresh", "chunk_type": "text", "page": 1, "table_id": None, "metadata": {}}
        _, _, parse = self._run(doc, [text])
        parse.assert_called_once()


# ─── Indexer payload ─────────────────────────────────────────────────────────

class TestIndexerPayload:

    def _upsert(self, chunk):
        from app.services.ingestion import indexer
        client = MagicMock()
        emb = MagicMock()
        emb.embed_documents.return_value = [[0.0] * 3]
        with patch.object(indexer, "_get_client", return_value=client), \
             patch.object(indexer, "_get_embeddings", return_value=emb), \
             patch.object(indexer, "ensure_collection"):
            indexer.upsert_chunks([chunk], doc_id="d1")
        return client.upsert.call_args.kwargs["points"][0].payload

    def _image_chunk(self):
        return {"content": CHART_TEXT, "chunk_type": "image", "page": 4, "table_id": "vision_4",
                "doc_version": "2023", "effective_from": "2023-11-29T00:00:00", "effective_to": None,
                "filename": "charts.pdf", "metadata": {"source": "gpt4o_vision", "page_count": 18}}

    def test_image_chunk_payload(self):
        payload = self._upsert(self._image_chunk())
        assert payload["chunk_type"] == "image"  # top level, for Qdrant filters
        assert payload["source"] == "gpt4o_vision"
        nested = payload["metadata"]
        assert nested["chunk_type"] == "image" and nested["source"] == "gpt4o_vision"
        assert nested["doc_version"] == "2023" and nested["doc_id"] == "d1"
        assert "content" not in nested

    def test_langchain_vector_store_reads_chunk_fields(self):
        """Vector-retrieved docs keep chunk_type/source (LangChain reads payload['metadata'])."""
        from langchain_community.vectorstores import Qdrant as LCQdrant
        payload = self._upsert(self._image_chunk())
        point = SimpleNamespace(id="p1", payload=payload)
        doc = LCQdrant._document_from_scored_point(point, "c", "content", "metadata")
        assert doc.page_content == CHART_TEXT
        assert doc.metadata["chunk_type"] == "image"
        assert doc.metadata["source"] == "gpt4o_vision"
        assert doc.metadata["doc_version"] == "2023"

    def test_fetch_all_chunks_drops_nested_copy(self):
        from app.services.ingestion import indexer
        payload = self._upsert(self._image_chunk())
        client = MagicMock()
        client.scroll.return_value = ([SimpleNamespace(payload=payload)], None)
        with patch.object(indexer, "_get_client", return_value=client):
            rows = indexer.fetch_all_chunks()
        assert "metadata" not in rows[0]
        assert rows[0]["chunk_type"] == "image" and rows[0]["source"] == "gpt4o_vision"
