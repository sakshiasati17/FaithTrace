"""
Tests for GET /api/v1/system/status.

Dependencies are mocked to fail or hang: the endpoint must still return 200
with ok:false per component, never a 500.
"""

import asyncio
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import __version__
from app.api.v1.endpoints import system
from app.main import app

COMPONENTS = {"database", "qdrant", "redis", "workers", "ml_classifier"}


@pytest.fixture
def client():
    with patch("app.main.settings.API_KEY", ""):
        yield TestClient(app)


def _boom(*_args, **_kwargs):
    raise ConnectionError("connection refused")


async def _async_boom():
    raise ConnectionError("connection refused")


class TestSystemStatus:

    def test_all_dependencies_failing_returns_ok_false_not_500(self, client):
        with patch.object(system, "_check_database", _async_boom), \
             patch.object(system, "_check_qdrant", _boom), \
             patch.object(system, "_check_redis", _boom), \
             patch.object(system, "_check_workers", _boom) as workers, \
             patch.object(system, "_check_ml_classifier", _boom):
            resp = client.get("/api/v1/system/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        assert body["version"] == __version__
        assert set(body["components"]) == COMPONENTS
        for name, comp in body["components"].items():
            assert comp["ok"] is False, name
            assert comp["detail"], name
        assert "ConnectionError" in body["components"]["database"]["detail"]
        assert "skipped" in body["components"]["workers"]["detail"]

    def test_hanging_dependency_times_out(self, client):
        def _hang():
            time.sleep(5)
            return {}

        with patch.object(system, "CHECK_TIMEOUT_S", 0.2), \
             patch.object(system, "_check_database", _async_boom), \
             patch.object(system, "_check_qdrant", _hang), \
             patch.object(system, "_check_redis", _boom), \
             patch.object(system, "_check_ml_classifier", lambda: {"trained": False}):
            start = time.perf_counter()
            resp = client.get("/api/v1/system/status")
            elapsed = time.perf_counter() - start

        assert resp.status_code == 200
        qdrant = resp.json()["components"]["qdrant"]
        assert qdrant["ok"] is False
        assert "timed out" in qdrant["detail"]
        assert elapsed < 3

    def test_all_healthy_reports_ok_true(self, client):
        async def _db_ok():
            return {"detail": "SELECT 1 succeeded"}

        with patch.object(system, "_check_database", _db_ok), \
             patch.object(system, "_check_qdrant", lambda: {"detail": "1 collection(s)"}), \
             patch.object(system, "_check_redis", lambda: {"detail": "PING succeeded"}), \
             patch.object(system, "_check_workers", lambda: {"detail": "1 worker(s) responding"}), \
             patch.object(system, "_check_ml_classifier", lambda: {"trained": False}):
            resp = client.get("/api/v1/system/status")

        body = resp.json()
        assert resp.status_code == 200
        assert body["ok"] is True
        assert all(c["ok"] for c in body["components"].values())

    def test_untrained_classifier_is_reported_not_failed(self):
        with patch("app.services.diagnostics.ml_classifier.is_trained", return_value=False):
            result = asyncio.run(system._run_check("ml_classifier", system._check_ml_classifier, 1.0))
        assert result["ok"] is True
        assert result["trained"] is False
        assert result["classifier_type"] == "heuristic"

    def test_health_endpoint_unchanged(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["service"] == "faithtrace-api"
