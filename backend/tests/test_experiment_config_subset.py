"""
Explicit config subsets for POST /api/v1/experiments/.

`configs` lets an experiment run a short, hand-picked list of pipeline configs
(e.g. 4 configs differing in one axis) instead of the 24-config MVP preset or
the full 256-config matrix. Without it, behaviour is unchanged.

No network or database: the DB session is mocked and the Celery task patched.
"""

import json
from dataclasses import asdict
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.db.models import Experiment, Run
from app.db.session import get_db
from app.main import app
from app.services.experiment.config_matrix import (
    DEFAULT_TOP_K,
    build_configs,
    build_matrix,
    build_mvp_matrix,
)


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


def _wire_create(db):
    """Record rows added by create_experiment and serve the experiment reload."""
    added = []
    db.add = MagicMock(side_effect=added.append)

    def execute(stmt):
        res = MagicMock()

        def reload():
            exp = next(o for o in added if isinstance(o, Experiment))
            exp.created_at = datetime.utcnow()
            return exp
        res.scalar_one.side_effect = reload
        return res

    db.execute.side_effect = execute
    return added


def _post(client, **extra):
    with patch("app.workers.tasks.run_experiment.delay") as delay:
        resp = client.post("/api/v1/experiments/", json={"name": "ablation", **extra})
    return resp, delay


def _runs(added):
    return [o for o in added if isinstance(o, Run)]


def _spec(retrieval="vector_only", chunking="recursive", parsing="text_only",
          freshness="none", **extra):
    return {
        "retrieval_strategy": retrieval,
        "chunking_strategy": chunking,
        "parsing_strategy": parsing,
        "freshness_policy": freshness,
        **extra,
    }


# One-axis ablation: only retrieval varies.
RETRIEVAL_ABLATION = [
    _spec(retrieval=r) for r in ["vector_only", "bm25", "hybrid", "hybrid_reranker"]
]


class TestExplicitConfigs:

    def test_four_configs_make_four_runs(self, client, db):
        added = _wire_create(db)
        resp, delay = _post(client, configs=RETRIEVAL_ABLATION)

        assert resp.status_code == 201, resp.text
        runs = _runs(added)
        assert len(runs) == 4
        for run, spec in zip(runs, RETRIEVAL_ABLATION):
            for key, value in spec.items():
                assert run.config[key] == value
            assert run.config["reranker_enabled"] == (spec["retrieval_strategy"] == "hybrid_reranker")
            assert run.config["embedding_model"] == "text-embedding-3-small"
            assert run.config["llm_model"] == "gpt-4o-mini"
            assert run.config["top_k"] == DEFAULT_TOP_K
            assert run.status == "pending"
        assert [r.config["reranker_enabled"] for r in runs] == [False, False, False, True]
        delay.assert_called_once()

    def test_llm_model_and_top_k_are_used(self, client, db):
        added = _wire_create(db)
        resp, _ = _post(client, configs=[_spec(llm_model="gpt-4o", top_k=12)])

        assert resp.status_code == 201, resp.text
        (run,) = _runs(added)
        assert run.config["llm_model"] == "gpt-4o"
        assert run.config["top_k"] == 12

    def test_run_config_matches_build_configs(self, client, db):
        added = _wire_create(db)
        resp, _ = _post(client, configs=RETRIEVAL_ABLATION)

        assert resp.status_code == 201, resp.text
        expected = [
            asdict(c) for c in build_configs(
                {**s, "llm_model": "gpt-4o-mini"} for s in RETRIEVAL_ABLATION
            )
        ]
        assert [r.config for r in _runs(added)] == expected

    @pytest.mark.parametrize("preset", ["mvp", "custom"])
    def test_configs_override_preset(self, client, db, preset):
        added = _wire_create(db)
        resp, _ = _post(client, config_preset=preset, configs=RETRIEVAL_ABLATION[:2])

        assert resp.status_code == 201, resp.text
        assert len(_runs(added)) == 2


class TestPresetsUnchanged:

    @pytest.mark.parametrize("extra", [{}, {"configs": None}, {"config_preset": "mvp"}])
    def test_omitted_configs_run_mvp_preset(self, client, db, extra):
        added = _wire_create(db)
        resp, _ = _post(client, **extra)

        assert resp.status_code == 201, resp.text
        runs = _runs(added)
        assert len(runs) == 24
        assert [r.config for r in runs] == [asdict(c) for c in build_mvp_matrix()]

    def test_non_mvp_preset_runs_full_matrix(self, client, db):
        added = _wire_create(db)
        resp, _ = _post(client, config_preset="custom")

        assert resp.status_code == 201, resp.text
        runs = _runs(added)
        assert len(runs) == 256
        assert [r.config for r in runs] == [asdict(c) for c in build_matrix()]


class TestValidation:

    def _assert_rejected(self, client, db, configs, *needles):
        added = _wire_create(db)
        resp, delay = _post(client, configs=configs)
        assert resp.status_code == 422, resp.text
        body = json.dumps(resp.json())
        for needle in needles:
            assert needle in body
        assert not added
        delay.assert_not_called()

    @pytest.mark.parametrize("field,bad", [
        ("retrieval_strategy", "dense_only"),
        ("chunking_strategy", "sentence"),
        ("parsing_strategy", "ocr"),
        ("freshness_policy", "latest_only"),
    ])
    def test_unknown_value_is_422_naming_it(self, client, db, field, bad):
        spec = _spec()
        spec[field] = bad
        self._assert_rejected(client, db, [spec], bad, field)

    def test_empty_list_is_422(self, client, db):
        self._assert_rejected(client, db, [])

    def test_33_configs_is_422(self, client, db):
        specs = [_spec(top_k=k) for k in range(1, 34)]
        self._assert_rejected(client, db, specs)

    def test_32_configs_is_accepted(self, client, db):
        added = _wire_create(db)
        resp, _ = _post(client, configs=[_spec(top_k=k) for k in range(1, 33)])
        assert resp.status_code == 201, resp.text
        assert len(_runs(added)) == 32

    def test_duplicates_are_422(self, client, db):
        self._assert_rejected(
            client, db, [_spec(), _spec(retrieval="bm25"), _spec()], "duplicates"
        )

    def test_default_and_explicit_defaults_are_duplicates(self, client, db):
        explicit = _spec(llm_model="gpt-4o-mini", top_k=DEFAULT_TOP_K)
        self._assert_rejected(client, db, [_spec(), explicit], "duplicates")

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_top_k_below_one_is_422(self, client, db, top_k):
        self._assert_rejected(client, db, [_spec(top_k=top_k)], "top_k")

    def test_missing_axis_is_422(self, client, db):
        spec = _spec()
        del spec["freshness_policy"]
        self._assert_rejected(client, db, [spec], "freshness_policy")

    def test_unknown_field_is_422(self, client, db):
        self._assert_rejected(client, db, [_spec(reranker_enabled=True)], "reranker_enabled")


class TestBuildConfigs:

    def test_build_matrix_unchanged(self):
        matrix = build_matrix()
        assert len(matrix) == 256
        assert all(c.reranker_enabled == (c.retrieval_strategy == "hybrid_reranker") for c in matrix)
        assert all(c.llm_model == "gpt-4o" and c.top_k == DEFAULT_TOP_K for c in matrix)

    def test_top_k_none_uses_default(self):
        (cfg,) = build_configs([{**_spec(), "llm_model": "m", "top_k": None}])
        assert cfg.top_k == DEFAULT_TOP_K
