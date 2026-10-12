"""
Per-run query counts on the experiment detail endpoints.

A run can be "done" while some of its queries errored, were never scored, or
were never diagnosed. GET /experiments/{id} and GET /experiments/{id}/runs
report n_queries / n_errored / n_unscored / n_undiagnosed per run, from one
aggregate over query_results. Unknown counts are None, never 0.

The aggregate runs against a real SQLite database; the endpoint tests mock the
async session and feed it rows from that database.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.v1.endpoints.experiments import apply_query_counts, query_counts_stmt
from app.db.models import Base, Experiment, QueryResult, Run
from app.db.session import get_db
from app.main import app
from app.schemas.experiment_schemas import RunResponse

CONFIG = {
    "retrieval_strategy": "vector_only", "chunking_strategy": "recursive",
    "parsing_strategy": "text_only", "freshness_policy": "none",
}


def _qr(run_id, query_id, status="ok", evidence=None, failure="NO_FAILURE"):
    return QueryResult(
        run_id=run_id, query_id=query_id, question="q?", generated_answer="a",
        status=status, diagnosis_evidence=evidence if evidence is not None else {},
        failure_category=failure,
    )


def _scored(value=0.8):
    return {"scores": {"answer_correctness": value}}


@pytest.fixture
def session():
    # One shared connection: the TestClient calls the app from another thread.
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    now = datetime.utcnow()
    db.add(Experiment(id="e1", name="exp", status="done"))
    db.add_all([
        Run(id="r1", experiment_id="e1", config=CONFIG, status="done",
            evaluated_at=now, diagnosed_at=now),
        Run(id="r2", experiment_id="e1", config=CONFIG, status="pending"),
        Run(id="r3", experiment_id="e1", config=CONFIG, status="done",
            evaluated_at=now, diagnosed_at=None),
    ])
    db.add_all([
        # r1: 6 queries: 4 ok and scored (one undiagnosed), 1 errored,
        # 1 legacy NULL-status row whose score is a JSON null.
        _qr("r1", "q1", evidence=_scored()),
        _qr("r1", "q2", evidence=_scored(0.0)),  # a real 0.0 is scored
        _qr("r1", "q3", evidence=_scored(), failure=None),
        _qr("r1", "q4", evidence={"scores": {}}, failure=None),  # unscored + undiagnosed
        _qr("r1", "q5", status="error", evidence={}, failure=None),
        _qr("r1", "q6", status=None, evidence={"scores": {"answer_correctness": None}}),
        # r3: evaluated, not diagnosed.
        _qr("r3", "q1", evidence=_scored(), failure=None),
        _qr("r3", "q2", evidence={}, failure=None),
    ])
    db.commit()
    yield db
    db.close()


def _counts(session, run_ids):
    rows = session.execute(query_counts_stmt(run_ids)).all()
    runs = {r.id: r for r in session.query(Run).filter(Run.id.in_(run_ids))}
    responses = [
        RunResponse(id=rid, experiment_id="e1", config=CONFIG, status=runs[rid].status,
                    created_at=datetime.utcnow())
        for rid in run_ids
    ]
    apply_query_counts(
        responses, rows,
        evaluated={rid: runs[rid].evaluated_at is not None for rid in run_ids},
        diagnosed={rid: runs[rid].diagnosed_at is not None for rid in run_ids},
    )
    return {
        r.id: (r.n_queries, r.n_errored, r.n_unscored, r.n_undiagnosed) for r in responses
    }


def test_counts_errored_unscored_undiagnosed(session):
    counts = _counts(session, ["r1"])
    # 6 queries; 1 errored; q4 and q6 unscored (errored row not counted);
    # q3 and q4 undiagnosed (errored row not counted).
    assert counts["r1"] == (6, 1, 2, 2)


def test_run_without_queries_has_zero_queries_and_unknown_scoring(session):
    # Nothing ran yet: 0 queries is a fact; scoring/diagnosis haven't happened.
    assert _counts(session, ["r2"])["r2"] == (0, 0, None, None)


def test_undiagnosed_is_unknown_until_the_run_is_diagnosed(session):
    assert _counts(session, ["r3"])["r3"] == (2, 0, 1, None)


def test_default_counts_are_none():
    run = RunResponse(id="r", experiment_id="e", config={}, status="done",
                      created_at=datetime.utcnow())
    assert (run.n_queries, run.n_errored, run.n_unscored, run.n_undiagnosed) == (None,) * 4


def test_statement_compiles_for_postgres():
    # JSON null must read as SQL NULL on PostgreSQL too (->> / #>>, not ->).
    sql = str(query_counts_stmt(["r1"]).compile(dialect=postgresql.dialect()))
    assert "#>>" in sql or "->>" in sql


# ─── Endpoints ───────────────────────────────────────────────────────────────

@pytest.fixture
def client_for(session):
    """TestClient whose mocked async session serves ORM objects and aggregate rows from SQLite."""
    def make():
        db = MagicMock()

        def execute(stmt):
            res = MagicMock()
            rows = session.execute(stmt)
            if "count" in str(stmt).lower():
                res.all.return_value = rows.all()
            else:
                objs = rows.scalars().all()
                res.scalar_one_or_none.return_value = objs[0] if objs else None
                res.scalars.return_value.all.return_value = objs
            return res
        db.execute = AsyncMock(side_effect=execute)

        async def _override():
            yield db
        app.dependency_overrides[get_db] = _override
        return TestClient(app)

    with patch("app.main.settings.API_KEY", ""):
        yield make
    app.dependency_overrides.pop(get_db, None)


def _by_id(runs):
    return {
        r["id"]: (r["n_queries"], r["n_errored"], r["n_unscored"], r["n_undiagnosed"])
        for r in runs
    }


def test_experiment_detail_includes_counts(client_for):
    resp = client_for().get("/api/v1/experiments/e1")
    assert resp.status_code == 200
    assert _by_id(resp.json()["runs"]) == {
        "r1": (6, 1, 2, 2),
        "r2": (0, 0, None, None),
        "r3": (2, 0, 1, None),
    }


def test_list_runs_includes_counts(client_for):
    resp = client_for().get("/api/v1/experiments/e1/runs")
    assert resp.status_code == 200
    assert _by_id(resp.json())["r1"] == (6, 1, 2, 2)
