"""
Chunking strategy axis tests (Known Issue #7) and the PR #26 follow-ups.

Covers:
  - ingest_document chunks text with every INGEST_CHUNKING_STRATEGIES entry,
    tags each chunk with chunk_strategy, stores atomic chunks (table, image,
    spreadsheet_cell) once as "atomic", and records per-strategy counts
  - on real corpus files, fixed_size / recursive / structure_aware produce
    different chunk sets
  - retrieval selects only the run's strategy plus atomic chunks, in the BM25
    predicate and the Qdrant filter alike; untagged (pre-existing) text chunks
    count as recursive
  - a run whose chunking strategy is not indexed is failed with a reason and
    the experiment still finishes
  - _ALLOWED_CHUNK_TYPES: text_table excludes vision (image) chunks
  - vision chunk detection on flat retrieved chunks (classifier + ML feature)

No network: Qdrant is an in-process QdrantClient(":memory:"); OpenAI and
Celery are mocked.
"""

import functools
import itertools
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings
from app.services.experiment import runner
from app.services.experiment.runner import PipelineConfig, chunk_passes_filters, run_pipeline
from app.services.ingestion import chunker
from app.workers import tasks

CORPUS = Path(__file__).parent.parent.parent / "corpus"
CC_BY_4 = CORPUS / "cc_by_4.0_legalcode.html"
CNCF_PDF = CORPUS / "cncf_2022_survey_reference_model.pdf"

DEFAULT_STRATEGIES = ["fixed_size", "recursive", "structure_aware"]

LONG_TEXT = " ".join(
    f"Sentence {i} explains the procurement threshold rule in some detail." for i in range(40)
)


def _doc(tmp_path, strategy="text_table"):
    f = tmp_path / "policy.pdf"
    f.write_bytes(b"%PDF")
    return SimpleNamespace(
        id="d1", storage_path=str(f), version_label="v2", filename="policy.pdf",
        effective_from=datetime(2024, 1, 1), effective_to=None,
        parse_status="pending", index_status="pending",
        doc_metadata={"strategy": strategy},
    )


# ─── INGEST_CHUNKING_STRATEGIES ──────────────────────────────────────────────

class TestParseStrategies:

    def test_default_excludes_semantic(self):
        assert chunker.parse_strategies(settings.INGEST_CHUNKING_STRATEGIES) == DEFAULT_STRATEGIES

    def test_order_kept_and_duplicates_dropped(self):
        assert chunker.parse_strategies(" recursive, semantic ,recursive") == ["recursive", "semantic"]

    def test_unknown_names_are_logged_and_ignored(self, caplog):
        with caplog.at_level(logging.ERROR):
            assert chunker.parse_strategies("recursive,sentence") == ["recursive"]
        assert "sentence" in caplog.text

    def test_no_known_strategy_raises(self):
        with pytest.raises(ValueError):
            chunker.parse_strategies("bogus,")


# ─── ingest_document ─────────────────────────────────────────────────────────

class TestIngest:

    def _run(self, doc, raw_chunks, strategies="fixed_size,recursive,structure_aware"):
        db = MagicMock()
        db.get.return_value = doc
        with patch("app.db.session.get_sync_db", return_value=db), \
             patch("app.services.ingestion.parser.parse_document", return_value=raw_chunks), \
             patch("app.services.ingestion.indexer.ensure_collection"), \
             patch("app.services.ingestion.indexer.upsert_chunks") as upsert, \
             patch.object(settings, "INGEST_CHUNKING_STRATEGIES", strategies):
            tasks.ingest_document.push_request(retries=0)
            try:
                result = tasks.ingest_document.run(doc.id, doc.doc_metadata["strategy"])
            finally:
                tasks.ingest_document.pop_request()
        indexed = upsert.call_args.args[0] if upsert.called else []
        return result, indexed

    def _raw(self):
        return [
            {"content": LONG_TEXT, "chunk_type": "text", "page": 1, "table_id": None, "metadata": {}},
            {"content": "| item | limit |\n|---|---|\n| laptop | 1500 |", "chunk_type": "table",
             "page": 1, "table_id": "t1", "metadata": {}},
            {"content": "Pie chart: 40% yes", "chunk_type": "image", "page": 2, "table_id": "vision_2",
             "metadata": {"source": "gpt4o_vision"}},
            {"content": "laptop: 1500", "chunk_type": "spreadsheet_cell", "page": None,
             "table_id": None, "metadata": {}},
        ]

    def test_text_chunked_with_every_configured_strategy(self, tmp_path):
        doc = _doc(tmp_path)
        result, indexed = self._run(doc, self._raw())

        text = [c for c in indexed if c["chunk_type"] == "text"]
        assert {c["chunk_strategy"] for c in text} == set(DEFAULT_STRATEGIES)
        for strategy in DEFAULT_STRATEGIES:
            pieces = [c for c in text if c["chunk_strategy"] == strategy]
            assert len(pieces) > 1, strategy  # LONG_TEXT is split
            assert all(c["doc_version"] == "v2" and c["page"] == 1 for c in pieces)

    def test_atomic_chunks_stored_once(self, tmp_path):
        doc = _doc(tmp_path)
        _, indexed = self._run(doc, self._raw())
        atomic = [c for c in indexed if c["chunk_type"] != "text"]
        assert sorted(c["chunk_type"] for c in atomic) == ["image", "spreadsheet_cell", "table"]
        assert all(c["chunk_strategy"] == "atomic" for c in atomic)
        image = next(c for c in atomic if c["chunk_type"] == "image")
        assert image["content"] == "Pie chart: 40% yes" and image["metadata"]["source"] == "gpt4o_vision"

    def test_counts_per_strategy_recorded(self, tmp_path):
        doc = _doc(tmp_path)
        result, indexed = self._run(doc, self._raw())
        counts = doc.doc_metadata["chunks_by_strategy"]
        assert set(counts) == set(DEFAULT_STRATEGIES) | {"atomic"}
        assert counts["atomic"] == 3
        for strategy in DEFAULT_STRATEGIES:
            assert counts[strategy] == sum(1 for c in indexed if c["chunk_strategy"] == strategy)
        assert doc.doc_metadata["chunks_indexed"] == len(indexed) == sum(counts.values())
        assert doc.doc_metadata["chunking_strategies"] == DEFAULT_STRATEGIES
        assert result["chunks_by_strategy"] == counts
        assert doc.parse_status == "done" and doc.index_status == "done"

    def test_only_configured_strategies_indexed(self, tmp_path):
        doc = _doc(tmp_path)
        _, indexed = self._run(doc, self._raw(), strategies="recursive")
        assert {c["chunk_strategy"] for c in indexed} == {"recursive", "atomic"}
        assert doc.doc_metadata["chunks_by_strategy"].keys() == {"recursive", "atomic"}

    def test_semantic_when_configured(self, tmp_path):
        doc = _doc(tmp_path)
        fake = [{"content": "semantic piece", "start_index": 0, "end_index": 14, "metadata": {"strategy": "semantic"}}]
        with patch.object(chunker, "_semantic_chunk", return_value=fake) as semantic:
            _, indexed = self._run(doc, self._raw(), strategies="recursive,semantic")
        semantic.assert_called_once()
        assert [c["content"] for c in indexed if c["chunk_strategy"] == "semantic"] == ["semantic piece"]

    def test_bad_setting_fails_before_parsing(self, tmp_path):
        doc = _doc(tmp_path)
        db = MagicMock()
        db.get.return_value = doc
        with patch("app.db.session.get_sync_db", return_value=db), \
             patch("app.services.ingestion.parser.parse_document") as parse, \
             patch("app.services.ingestion.indexer.ensure_collection"), \
             patch.object(settings, "INGEST_CHUNKING_STRATEGIES", "nope"), \
             patch.object(tasks.ingest_document, "retry", side_effect=RuntimeError("retry")):
            with pytest.raises(RuntimeError):
                tasks.ingest_document.run(doc.id, "text_table")
        parse.assert_not_called()
        assert doc.parse_status == "failed"

    def test_indexer_payload_has_top_level_chunk_strategy(self):
        from app.services.ingestion import indexer
        client = MagicMock()
        emb = MagicMock()
        emb.embed_documents.return_value = [[0.0] * 3]
        chunk = {"content": "x", "chunk_type": "text", "chunk_strategy": "fixed_size",
                 "metadata": {"strategy": "fixed_size"}}
        with patch.object(indexer, "_get_client", return_value=client), \
             patch.object(indexer, "_get_embeddings", return_value=emb), \
             patch.object(indexer, "ensure_collection"):
            indexer.upsert_chunks([chunk], doc_id="d1")
        payload = client.upsert.call_args.kwargs["points"][0].payload
        assert payload["chunk_strategy"] == "fixed_size"
        assert payload["metadata"]["chunk_strategy"] == "fixed_size"  # vector-retrieved docs keep it


class TestStructureAwareUnits:

    def test_elements_joined_with_headings(self):
        raw = [
            {"content": "Section 1 – Definitions.", "chunk_type": "text", "page": None,
             "metadata": {"element_type": "Title"}},
            {"content": "a. Adapted Material means ...", "chunk_type": "text", "page": None,
             "metadata": {"element_type": "ListItem"}},
            {"content": "Section 2 – Scope.", "chunk_type": "text", "page": None,
             "metadata": {"element_type": "Title"}},
            {"content": "a. License grant ...", "chunk_type": "text", "page": None,
             "metadata": {"element_type": "ListItem"}},
        ]
        doc = SimpleNamespace(version_label="4.0", effective_from=None, effective_to=None, filename="f.html")
        chunks, counts = tasks.build_index_chunks(raw, doc, ["recursive", "structure_aware"])
        recursive = [c["content"] for c in chunks if c["chunk_strategy"] == "recursive"]
        structured = [c for c in chunks if c["chunk_strategy"] == "structure_aware"]
        assert recursive == [r["content"] for r in raw]  # one chunk per element
        assert [c["content"] for c in structured] == [
            "## Section 1 – Definitions.\n\na. Adapted Material means ...",
            "## Section 2 – Scope.\n\na. License grant ...",
        ]
        assert structured[0]["metadata"]["section_title"] == "## Section 1 – Definitions."
        assert "element_type" not in structured[0]["metadata"]

    def test_pages_and_atomic_chunks_break_blocks(self):
        raw = [
            {"content": "page one", "chunk_type": "text", "page": 1, "metadata": {}},
            {"content": "page two", "chunk_type": "text", "page": 2, "metadata": {}},
            {"content": "| t |", "chunk_type": "table", "page": 2, "metadata": {}},
            {"content": "after table", "chunk_type": "text", "page": 2, "metadata": {}},
        ]
        units = tasks._text_units(raw, "structure_aware")
        assert [(c, r["page"]) for c, r in units] == [("page one", 1), ("page two", 2), ("after table", 2)]

    def test_preamble_before_first_heading_kept(self):
        text = "Intro paragraph.\nSection 1 – Definitions.\nBody one."
        contents = [c["content"] for c in chunker.chunk(text, strategy="structure_aware")]
        assert contents[0] == "Intro paragraph."
        assert "Body one." in contents[1]

    def test_cross_reference_is_not_a_heading(self):
        text = "Granted under\nSection 2(a)(1)\nof this license."
        assert len(chunker.chunk(text, strategy="structure_aware")) == 1


# ─── Real corpus files ───────────────────────────────────────────────────────

def _strategy_sets(raw_chunks):
    doc = SimpleNamespace(version_label="v", effective_from=None, effective_to=None, filename="f")
    chunks, counts = tasks.build_index_chunks(raw_chunks, doc, DEFAULT_STRATEGIES)
    sets = {s: [c["content"] for c in chunks if c["chunk_strategy"] == s] for s in DEFAULT_STRATEGIES}
    return sets, counts


def _assert_pairwise_different(sets):
    for a, b in itertools.combinations(DEFAULT_STRATEGIES, 2):
        assert sets[a] and sets[b]
        assert set(sets[a]) != set(sets[b]), (a, b)


class TestRealCorpus:

    def test_html_strategies_differ(self):
        from app.services.ingestion.parser import parse_document
        sets, _ = _strategy_sets(parse_document(CC_BY_4, "text_table"))
        _assert_pairwise_different(sets)

    def test_html_basic_strip_strategies_differ(self):
        # The parser's fallback when unstructured is unavailable: one text block.
        from app.services.ingestion.parser import parse_document
        with patch("unstructured.partition.html.partition_html", side_effect=RuntimeError("off")):
            raw = parse_document(CC_BY_4, "text_table")
        assert len(raw) == 1
        sets, _ = _strategy_sets(raw)
        _assert_pairwise_different(sets)
        # structure_aware splits at the numbered sections and keeps the preamble.
        doc = SimpleNamespace(version_label="v", effective_from=None, effective_to=None, filename="f")
        chunks, _ = tasks.build_index_chunks(raw, doc, ["structure_aware"])
        titles = {c["metadata"].get("section_title") for c in chunks}
        assert "Section 4 – Sui Generis Database Rights." in titles
        assert any("Attribution 4.0 International" in c["content"] for c in chunks
                   if "section_title" not in c["metadata"])

    def test_pdf_strategies_differ(self):
        from app.services.ingestion.parser import parse_document
        raw = parse_document(CNCF_PDF, "text_table")
        sets, counts = _strategy_sets(raw)
        _assert_pairwise_different(sets)
        assert counts["atomic"] == sum(1 for r in raw if r["chunk_type"] != "text")


# ─── Retrieval filter: predicate ─────────────────────────────────────────────

class TestChunkingPredicate:

    def _passes(self, payload, strategy, parsing="text_table_vision"):
        return chunk_passes_filters(payload, parsing, "none", {}, None, strategy)

    def test_tagged_text_only_for_its_strategy(self):
        p = {"chunk_type": "text", "chunk_strategy": "fixed_size"}
        assert self._passes(p, "fixed_size")
        assert not self._passes(p, "recursive")
        assert not self._passes(p, "structure_aware")

    def test_atomic_passes_every_strategy(self):
        for ctype in ("table", "image"):
            p = {"chunk_type": ctype, "chunk_strategy": "atomic"}
            assert all(self._passes(p, s) for s in chunker.CHUNKING_STRATEGIES)

    def test_legacy_untagged_text_counts_as_recursive(self):
        for p in ({"chunk_type": "text"}, {"chunk_type": "text", "chunk_strategy": None}):
            assert self._passes(p, "recursive")
            assert not self._passes(p, "fixed_size")
            assert not self._passes(p, "semantic")

    def test_legacy_untagged_table_passes_every_strategy(self):
        p = {"chunk_type": "table"}
        assert all(self._passes(p, s) for s in chunker.CHUNKING_STRATEGIES)

    def test_no_chunking_strategy_means_no_restriction(self):
        p = {"chunk_type": "text", "chunk_strategy": "fixed_size"}
        assert chunk_passes_filters(p, "text_table", "none", {})


# ─── Retrieval filter: predicate == Qdrant filter ───────────────────────────

_TAGS = ["missing", None, "fixed_size", "recursive", "semantic", "structure_aware", "atomic"]
_TYPES = ["text", "table", "image", "spreadsheet_cell"]
_DOCS = ["doc-a", "doc-b"]


def _grid():
    payloads = []
    for tag, ctype, doc_id in itertools.product(_TAGS, _TYPES, _DOCS):
        p = {"chunk_type": ctype, "doc_id": doc_id, "effective_from": 0}
        if tag != "missing":
            p["chunk_strategy"] = tag
        payloads.append(p)
    return payloads


@pytest.fixture(scope="module")
def qdrant_grid():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(":memory:")
    client.create_collection("grid", vectors_config=VectorParams(size=1, distance=Distance.COSINE))
    payloads = _grid()
    client.upsert("grid", points=[
        PointStruct(id=i, vector=[1.0], payload=dict(p, idx=i)) for i, p in enumerate(payloads)
    ])
    return client, payloads


@pytest.mark.parametrize("chunking", list(chunker.CHUNKING_STRATEGIES))
@pytest.mark.parametrize("parsing", ["text_only", "text_table", "text_table_vision", "spreadsheet_aware"])
@pytest.mark.parametrize("document_ids", [None, ["doc-a"]])
def test_predicate_matches_qdrant_filter(qdrant_grid, chunking, parsing, document_ids):
    client, payloads = qdrant_grid
    eval_item = {"valid_from": "2024-06-01"}
    # Built as run_pipeline + _build_vector_retriever build it.
    qdrant_filter = runner._build_chunk_type_filter(
        parsing,
        runner._build_chunking_filter(
            chunking,
            runner._build_document_filter(
                document_ids, runner._build_freshness_filter("effective_date_filter", eval_item)
            ),
        ),
    )
    records, _ = client.scroll("grid", scroll_filter=qdrant_filter, limit=len(payloads) + 1)
    by_qdrant = {r.payload["idx"] for r in records}
    by_predicate = {
        i for i, p in enumerate(payloads)
        if chunk_passes_filters(p, parsing, "effective_date_filter", eval_item, document_ids, chunking)
    }
    assert by_predicate == by_qdrant
    # Text retrieved is only the run's strategy (or legacy text for recursive);
    # "atomic" text is not produced by ingest but passes like any atomic chunk.
    texts = {payloads[i].get("chunk_strategy", "missing") for i in by_qdrant
             if payloads[i]["chunk_type"] == "text"}
    expected = {chunking, "atomic"} | ({"missing", None} if chunking == "recursive" else set())
    assert texts <= expected
    assert chunking in texts


# ─── run_pipeline: both retrieval paths use the run's strategy ──────────────

CORPUS_CHUNKS = [
    {"content": "threshold fixed piece", "doc_id": "d", "chunk_type": "text", "chunk_strategy": "fixed_size"},
    {"content": "threshold recursive piece", "doc_id": "d", "chunk_type": "text", "chunk_strategy": "recursive"},
    {"content": "threshold structured piece", "doc_id": "d", "chunk_type": "text",
     "chunk_strategy": "structure_aware"},
    {"content": "threshold legacy piece", "doc_id": "d", "chunk_type": "text"},
    {"content": "threshold table", "doc_id": "d", "chunk_type": "table", "chunk_strategy": "atomic"},
    {"content": "threshold chart", "doc_id": "d", "chunk_type": "image", "chunk_strategy": "atomic"},
]


def _pipeline(chunking, retrieval="bm25", parsing="text_table"):
    config = PipelineConfig(
        retrieval_strategy=retrieval, chunking_strategy=chunking, parsing_strategy=parsing,
        freshness_policy="none", embedding_model="e", llm_model="gpt-4o-mini", top_k=10,
    )
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="a", usage_metadata={"input_tokens": 1, "output_tokens": 1})
    build_vector = MagicMock()
    build_vector.return_value.invoke.return_value = []
    with patch("app.services.ingestion.indexer.fetch_all_chunks",
               return_value=[dict(c) for c in CORPUS_CHUNKS]), \
         patch.object(runner, "_build_vector_retriever", build_vector), \
         patch("langchain_openai.ChatOpenAI", return_value=llm):
        [result] = run_pipeline(config, [{"id": "q", "question": "threshold"}])
    assert result.status == "ok", result.error_message
    return {c["content"] for c in result.retrieved_chunks}, build_vector


@pytest.mark.parametrize("chunking,text", [
    ("fixed_size", {"threshold fixed piece"}),
    ("recursive", {"threshold recursive piece", "threshold legacy piece"}),
    ("structure_aware", {"threshold structured piece"}),
])
def test_bm25_retrieves_only_run_strategy_and_atomic(chunking, text):
    contents, _ = _pipeline(chunking)
    assert contents == text | {"threshold table"}  # text_table: no image chunk


def test_vector_path_gets_chunking_filter(qdrant_grid):
    _, build_vector = _pipeline("structure_aware", retrieval="vector_only")
    qdrant_filter = build_vector.call_args.args[2]
    assert qdrant_filter.must == runner._build_chunking_filter("structure_aware").must


# ─── Not-indexed strategy ────────────────────────────────────────────────────

@pytest.fixture
def indexed_client():
    """In-memory Qdrant with the app collection: doc-a indexed with fixed_size and
    recursive (+ a table), doc-b legacy (untagged), doc-c only spreadsheet cells."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(":memory:")
    client.create_collection(settings.QDRANT_COLLECTION,
                             vectors_config=VectorParams(size=1, distance=Distance.COSINE))
    payloads = [
        {"doc_id": "doc-a", "chunk_type": "text", "chunk_strategy": "fixed_size"},
        {"doc_id": "doc-a", "chunk_type": "text", "chunk_strategy": "recursive"},
        {"doc_id": "doc-a", "chunk_type": "table", "chunk_strategy": "atomic"},
        {"doc_id": "doc-b", "chunk_type": "text"},
        {"doc_id": "doc-c", "chunk_type": "spreadsheet_cell", "chunk_strategy": "atomic"},
    ]
    client.upsert(settings.QDRANT_COLLECTION, points=[
        PointStruct(id=i, vector=[1.0], payload=p) for i, p in enumerate(payloads)
    ])
    return client


class TestNotIndexed:

    def _check(self, client, strategy, document_ids=None):
        return runner.chunking_strategy_not_indexed(strategy, document_ids, client=client)

    def test_missing_strategy_has_reason(self, indexed_client):
        reason = self._check(indexed_client, "semantic")
        assert reason == (
            "chunking strategy 'semantic' not indexed; set INGEST_CHUNKING_STRATEGIES and reindex"
        )

    def test_indexed_strategies_pass(self, indexed_client):
        assert self._check(indexed_client, "fixed_size") is None
        assert self._check(indexed_client, "recursive") is None

    def test_legacy_documents_count_as_recursive(self, indexed_client):
        assert self._check(indexed_client, "recursive", ["doc-b"]) is None
        assert self._check(indexed_client, "fixed_size", ["doc-b"]) is not None

    def test_scope_without_text_is_not_a_chunking_failure(self, indexed_client):
        assert self._check(indexed_client, "structure_aware", ["doc-c"]) is None

    def test_scope_is_respected(self, indexed_client):
        assert self._check(indexed_client, "structure_aware", ["doc-a"]) is not None


class TestRunExperimentNotIndexed:
    """A run whose strategy is not indexed is failed; the experiment still finishes."""

    @pytest.fixture
    def Session(self, tmp_path):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from app.db.models import Base

        engine = create_engine(f"sqlite:///{tmp_path / 'chunking.db'}")
        Base.metadata.create_all(engine)
        yield sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
        engine.dispose()

    def _seed(self, Session, strategies):
        import uuid
        from app.db.models import Experiment, Run

        base = {"retrieval_strategy": "vector_only", "parsing_strategy": "text_only",
                "freshness_policy": "none", "embedding_model": "e", "llm_model": "gpt-4o-mini", "top_k": 3}
        with Session() as db:
            exp = Experiment(id=str(uuid.uuid4()), name="e", description="", status="pending",
                             eval_set_path="eval_sets/faithtrace_v1.json")
            db.add(exp)
            runs = []
            for s in strategies:
                run = Run(id=str(uuid.uuid4()), experiment_id=exp.id,
                          config=dict(base, chunking_strategy=s), status="pending")
                db.add(run)
                runs.append(run.id)
            db.commit()
            return exp.id, runs

    def _ok_results(self, config, eval_set, **kw):
        from app.services.experiment.runner import QueryResult
        return [QueryResult(query_id=i["id"], question=i["question"], generated_answer="a",
                            retrieved_chunks=[{"content": "c", "chunk_type": "text"}], latency_ms=1.0,
                            input_tokens=1, output_tokens=1, cost_usd=0.0) for i in eval_set]

    def _call(self, Session, client, fn, *args, evaluate_queue=None, diagnose_queue=None):
        eval_set = [{"id": "q1", "question": "Q?", "ground_truth": "a", "modality": "text"}]
        metrics = {"faithfulness": 0.9, "context_recall": 0.9, "context_precision": 0.9,
                   "answer_correctness": 0.9, "answer_relevancy": 0.9}
        pipeline = MagicMock(side_effect=self._ok_results)
        with patch("app.db.session.get_sync_db", side_effect=lambda: Session()), \
             patch.object(tasks, "_load_eval_set", return_value=eval_set), \
             patch.object(tasks.evaluate_run, "delay",
                          side_effect=lambda run_id, *a: evaluate_queue.append(run_id)), \
             patch.object(tasks.diagnose_run, "delay",
                          side_effect=lambda run_id, *a: diagnose_queue.append(run_id)), \
             patch("app.services.evaluation.ragas_runner.run_ragas_evaluation",
                   side_effect=lambda results, es: [metrics] * len(results)), \
             patch("app.services.diagnostics.ml_classifier.predict", return_value=None), \
             patch("app.services.experiment.runner.run_pipeline", pipeline), \
             patch("app.services.experiment.runner.chunking_strategy_not_indexed",
                   functools.partial(runner.chunking_strategy_not_indexed, client=client)):
            return fn(*args), pipeline

    def test_not_indexed_run_failed_and_experiment_done(self, Session, indexed_client, caplog):
        from app.db.models import Experiment, Run

        exp_id, (ok_run, semantic_run) = self._seed(Session, ["recursive", "semantic"])
        evaluate_queue, diagnose_queue = [], []
        with caplog.at_level(logging.ERROR, logger="app.workers.tasks"):
            out, pipeline = self._call(Session, indexed_client, tasks.run_experiment.run, exp_id,
                                       evaluate_queue=evaluate_queue, diagnose_queue=diagnose_queue)

        assert pipeline.call_count == 1  # the semantic run issued no queries
        assert out["runs_not_indexed"] == {
            semantic_run: "chunking strategy 'semantic' not indexed; set INGEST_CHUNKING_STRATEGIES and reindex"
        }
        assert f"Run {semantic_run} marked failed: chunking strategy 'semantic' not indexed" in caplog.text
        with Session() as db:
            assert db.get(Run, semantic_run).status == "failed"
            assert db.get(Run, ok_run).status == "done"
            assert db.get(Experiment, exp_id).status == "evaluating"
        assert evaluate_queue == [ok_run]

        self._call(Session, indexed_client, tasks.evaluate_run.run, ok_run,
                   evaluate_queue=evaluate_queue, diagnose_queue=diagnose_queue)
        self._call(Session, indexed_client, tasks.diagnose_run.run, ok_run,
                   evaluate_queue=evaluate_queue, diagnose_queue=diagnose_queue)
        with Session() as db:
            assert db.get(Experiment, exp_id).status == "done"

    def test_all_runs_not_indexed_fails_experiment(self, Session, indexed_client):
        from app.db.models import Experiment

        exp_id, _ = self._seed(Session, ["semantic", "semantic"])
        out, pipeline = self._call(Session, indexed_client, tasks.run_experiment.run, exp_id,
                                   evaluate_queue=[], diagnose_queue=[])
        pipeline.assert_not_called()
        assert out["status"] == "failed"
        with Session() as db:
            assert db.get(Experiment, exp_id).status == "failed"


# ─── _ALLOWED_CHUNK_TYPES ────────────────────────────────────────────────────

def test_allowed_chunk_types_mapping():
    assert runner._allowed_chunk_types("text_only") == ("text",)
    assert runner._allowed_chunk_types("text_table") == ("text", "table")
    assert runner._allowed_chunk_types("text_table_vision") == ("text", "table", "image")
    assert runner._allowed_chunk_types("spreadsheet_aware") == ("spreadsheet_cell", "text")


def test_text_table_excludes_vision_chunks_vision_includes_them():
    image = {"chunk_type": "image", "chunk_strategy": "atomic"}
    assert not chunk_passes_filters(image, "text_table", "none", {})
    assert chunk_passes_filters(image, "text_table_vision", "none", {})
    contents, _ = _pipeline("fixed_size", parsing="text_table_vision")
    assert "threshold chart" in contents


# ─── Vision detection on flat retrieved chunks ──────────────────────────────

FLAT_VISION = {"content": "chart", "chunk_type": "text", "source": "gpt4o_vision"}
NESTED_VISION = {"content": "chart", "chunk_type": "text", "metadata": {"source": "gpt4o_vision"}}
IMAGE = {"content": "chart", "chunk_type": "image"}
PLAIN = {"content": "text", "chunk_type": "text", "source": "pdf"}


@pytest.mark.parametrize("chunk,expected", [
    (FLAT_VISION, True), (NESTED_VISION, True), (IMAGE, True), (PLAIN, False),
])
def test_classifier_vision_detection(chunk, expected):
    from app.services.diagnostics.classifier import _has_vision_chunk
    assert _has_vision_chunk(SimpleNamespace(retrieved_chunks=[chunk])) is expected


@pytest.mark.parametrize("chunk,expected", [
    (FLAT_VISION, 1.0), (NESTED_VISION, 1.0), (IMAGE, 1.0), (PLAIN, 0.0),
])
def test_ml_vision_feature(chunk, expected):
    from app.services.diagnostics.ml_classifier import extract_features
    features = extract_features({}, [chunk], {"modality": "chart"})
    assert features[10] == expected  # has_vision_chunk
