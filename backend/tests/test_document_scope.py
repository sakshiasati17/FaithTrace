"""
Experiment document scope tests (Known Issue #9).

An experiment may limit retrieval to some documents (document_ids); null keeps
the old behaviour (every document). The same rule drives the Qdrant filter and
the BM25 predicate (runner.chunk_passes_filters).

No network: Qdrant is an in-process QdrantClient(":memory:"); the vector
retriever, the chunk fetch and the LLM are mocked.
"""

import itertools
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain.schema import Document as LCDocument
from langchain_core.retrievers import BaseRetriever

from app.db.models import Experiment
from app.db.session import get_db
from app.main import app
from app.services.experiment import runner
from app.services.experiment.runner import PipelineConfig, chunk_passes_filters, run_pipeline


def _epoch(date: str) -> int:
    return int(datetime.fromisoformat(date).timestamp())


_FROM = _epoch("2024-01-01")

CHUNKS = [
    {"content": "threshold policy A text", "doc_id": "doc-a", "chunk_type": "text",
     "effective_from": _FROM, "effective_to": None},
    {"content": "threshold policy A table", "doc_id": "doc-a", "chunk_type": "table",
     "effective_from": _FROM, "effective_to": None},
    {"content": "threshold policy B text", "doc_id": "doc-b", "chunk_type": "text",
     "effective_from": _FROM, "effective_to": None},
    {"content": "threshold policy C text", "doc_id": "doc-c", "chunk_type": "text",
     "effective_from": _FROM, "effective_to": None},
]

QUESTION = {"id": "q1", "question": "What is the threshold?", "valid_from": "2024-06-01"}


def _config(strategy="bm25", parsing="text_table", freshness="none", top_k=10):
    return PipelineConfig(
        retrieval_strategy=strategy, chunking_strategy="recursive", parsing_strategy=parsing,
        freshness_policy=freshness, embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini", top_k=top_k,
    )


class _StaticRetriever(BaseRetriever):
    docs: list = []

    def _get_relevant_documents(self, query, *, run_manager=None):
        return [LCDocument(page_content=d.page_content, metadata=dict(d.metadata)) for d in self.docs]


def _run(config, document_ids=None, vector_docs=()):
    fetch = MagicMock(return_value=[dict(c) for c in CHUNKS])
    build_vector = MagicMock(return_value=_StaticRetriever(docs=list(vector_docs)))
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="answer", usage_metadata={"input_tokens": 1, "output_tokens": 1})
    with patch("app.services.ingestion.indexer.fetch_all_chunks", fetch), \
         patch.object(runner, "_build_vector_retriever", build_vector), \
         patch("langchain_openai.ChatOpenAI", return_value=llm):
        results = run_pipeline(config, [QUESTION], document_ids=document_ids)
    assert all(r.status == "ok" for r in results), [r.error_message for r in results]
    return results, fetch, build_vector


def _docs_of(result):
    return {c["doc_id"] for c in result.retrieved_chunks}


# ─── Predicate ──────────────────────────────────────────────────────────────

class TestPredicate:

    def test_excludes_chunks_of_other_documents(self):
        assert chunk_passes_filters(CHUNKS[0], "text_table", "none", QUESTION, ["doc-a"])
        assert not chunk_passes_filters(CHUNKS[2], "text_table", "none", QUESTION, ["doc-a"])

    def test_none_means_every_document(self):
        assert all(chunk_passes_filters(c, "text_table", "none", QUESTION) for c in CHUNKS)
        assert all(chunk_passes_filters(c, "text_table", "none", QUESTION, None) for c in CHUNKS)

    def test_chunk_without_doc_id_excluded_only_when_scoped(self):
        payload = {"chunk_type": "text"}
        assert chunk_passes_filters(payload, "text_table", "none", QUESTION, None)
        assert not chunk_passes_filters(payload, "text_table", "none", QUESTION, ["doc-a"])

    def test_document_scope_combines_with_other_filters(self):
        # doc-a's table chunk is in scope but excluded by text_only.
        assert not chunk_passes_filters(CHUNKS[1], "text_only", "none", QUESTION, ["doc-a"])


# ─── Vector path: Qdrant filter construction ────────────────────────────────

def _doc_conditions(qdrant_filter):
    return [c for c in (qdrant_filter.must or []) if getattr(c, "key", None) == "doc_id"]


class TestVectorFilter:

    def test_vector_retriever_gets_doc_id_filter(self):
        _, _, build_vector = _run(_config("vector_only"), document_ids=["doc-a", "doc-b"])
        qdrant_filter = build_vector.call_args.args[2]
        [cond] = _doc_conditions(qdrant_filter)
        assert list(cond.match.any) == ["doc-a", "doc-b"]

    def test_vector_retriever_filter_unchanged_without_scope(self):
        _, _, build_vector = _run(_config("vector_only"), document_ids=None)
        # Only the chunking-strategy condition remains: no doc_id restriction.
        qdrant_filter = build_vector.call_args.args[2]
        assert _doc_conditions(qdrant_filter) == []
        assert qdrant_filter.must == runner._build_chunking_filter("recursive").must

    def test_doc_filter_merges_with_freshness_and_chunk_type(self):
        f = runner._build_chunk_type_filter(
            "text_only",
            runner._build_document_filter(["doc-a"], runner._build_freshness_filter("effective_date_filter", QUESTION)),
        )
        keys = [getattr(c, "key", None) for c in f.must]
        assert "doc_id" in keys and "chunk_type" in keys and "effective_from" in keys

    def test_none_returns_existing_filter(self):
        existing = runner._build_freshness_filter("effective_date_filter", QUESTION)
        assert runner._build_document_filter(None, existing) is existing
        assert runner._build_document_filter(None) is None


# ─── BM25 path ──────────────────────────────────────────────────────────────

class TestBM25:

    def test_bm25_excludes_other_documents(self):
        results, fetch, _ = _run(_config("bm25"), document_ids=["doc-a"])
        assert _docs_of(results[0]) == {"doc-a"}
        # The fetch is scoped too (fewer chunks pulled from Qdrant).
        assert fetch.call_args.kwargs == {"doc_ids": ["doc-a"]}

    def test_bm25_predicate_applies_even_if_fetch_returns_everything(self):
        # fetch is mocked to ignore doc_ids: the predicate alone must exclude doc-b/c.
        results, _, _ = _run(_config("bm25"), document_ids=["doc-b"])
        assert _docs_of(results[0]) == {"doc-b"}

    def test_bm25_null_scope_keeps_every_document(self):
        results, fetch, _ = _run(_config("bm25"), document_ids=None)
        assert _docs_of(results[0]) == {"doc-a", "doc-b", "doc-c"}
        assert fetch.call_args.kwargs == {"doc_ids": None}

    def test_hybrid_bm25_side_is_scoped(self):
        captured = []

        class _CapturingEnsemble:
            def __init__(self, retrievers, weights):
                self.retrievers = retrievers
                captured.append(self)

            def invoke(self, question):
                return self.retrievers[1].invoke(question)

        with patch("langchain.retrievers.EnsembleRetriever", _CapturingEnsemble):
            results, _, build_vector = _run(_config("hybrid"), document_ids=["doc-c"])

        assert {d.metadata["doc_id"] for d in captured[0].retrievers[1].docs} == {"doc-c"}
        assert _doc_conditions(build_vector.call_args.args[2])
        assert _docs_of(results[0]) == {"doc-c"}


# ─── Predicate and Qdrant filter agree ──────────────────────────────────────

def _payload_grid():
    q = _epoch("2024-06-01")
    payloads = []
    for doc, ctype, f, t in itertools.product(
        ["doc-a", "doc-b", None], ["text", "table", "spreadsheet_cell"], [None, q - 1, q + 1], [None, q - 1, q + 1]
    ):
        p = {"chunk_type": ctype, "effective_from": f, "effective_to": t}
        if doc is not None:
            p["doc_id"] = doc
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


@pytest.mark.parametrize("document_ids", [None, ["doc-a"], ["doc-a", "doc-b"], ["doc-z"]])
@pytest.mark.parametrize("parsing", ["text_only", "text_table", "spreadsheet_aware"])
@pytest.mark.parametrize("freshness", ["none", "effective_date_filter"])
def test_predicate_matches_qdrant_filter(qdrant_memory, document_ids, parsing, freshness):
    client, payloads = qdrant_memory
    qdrant_filter = runner._build_chunk_type_filter(
        parsing,
        runner._build_document_filter(document_ids, runner._build_freshness_filter(freshness, QUESTION)),
    )
    records, _ = client.scroll("grid", scroll_filter=qdrant_filter, limit=len(payloads) + 1)
    by_qdrant = {r.payload["idx"] for r in records}
    by_predicate = {
        i for i, p in enumerate(payloads)
        if chunk_passes_filters(p, parsing, freshness, QUESTION, document_ids)
    }
    assert by_predicate == by_qdrant


# ─── API: create experiment with document_ids ───────────────────────────────

@pytest.fixture
def db():
    session = MagicMock()
    session.execute = AsyncMock()
    session.get = AsyncMock(return_value=None)
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    return session


@pytest.fixture
def client(db):
    async def _override():
        yield db
    app.dependency_overrides[get_db] = _override
    with patch("app.main.settings.API_KEY", ""):
        yield TestClient(app)
    app.dependency_overrides.pop(get_db, None)


def _wire_create(db, known_doc_ids):
    """Mock the queries create_experiment makes: the document id lookup, then the
    reload of the new experiment."""
    added = []
    db.add = MagicMock(side_effect=added.append)

    def execute(stmt):
        res = MagicMock()
        res.scalars.return_value.all.return_value = list(known_doc_ids)

        def reload():
            exp = next(o for o in added if isinstance(o, Experiment))
            exp.created_at = datetime.utcnow()
            return exp
        res.scalar_one.side_effect = reload
        return res

    db.execute.side_effect = execute
    return added


class TestCreateExperimentDocumentIds:

    def _post(self, client, **extra):
        with patch("app.workers.tasks.run_experiment.delay") as delay:
            resp = client.post("/api/v1/experiments/", json={"name": "scoped", **extra})
        return resp, delay

    def test_valid_ids_are_stored(self, client, db):
        added = _wire_create(db, ["doc-a", "doc-b"])
        resp, delay = self._post(client, document_ids=["doc-a", "doc-b", "doc-a"])

        assert resp.status_code == 201, resp.text
        assert resp.json()["document_ids"] == ["doc-a", "doc-b"]   # de-duplicated
        exp = next(o for o in added if isinstance(o, Experiment))
        assert exp.document_ids == ["doc-a", "doc-b"]
        delay.assert_called_once()

    def test_unknown_id_is_422(self, client, db):
        added = _wire_create(db, ["doc-a"])
        resp, delay = self._post(client, document_ids=["doc-a", "doc-nope"])

        assert resp.status_code == 422
        assert "doc-nope" in resp.json()["detail"]
        assert "doc-a" not in resp.json()["detail"]
        assert not added
        delay.assert_not_called()

    def test_empty_list_is_422(self, client, db):
        _wire_create(db, [])
        resp, delay = self._post(client, document_ids=[])
        assert resp.status_code == 422
        assert "null" in json.dumps(resp.json())
        delay.assert_not_called()

    @pytest.mark.parametrize("extra", [{}, {"document_ids": None}])
    def test_null_keeps_every_document(self, client, db, extra):
        added = _wire_create(db, [])
        resp, _ = self._post(client, **extra)

        assert resp.status_code == 201, resp.text
        assert resp.json()["document_ids"] is None
        exp = next(o for o in added if isinstance(o, Experiment))
        assert exp.document_ids is None
        # No document lookup when unscoped: the only query is the reload.
        assert db.execute.call_count == 1
