"""
Ragas evaluation runs with a bounded RunConfig.

ragas.evaluate() defaults to 16 concurrent judge calls, which bursts into
OpenAI rate limits. run_ragas_evaluation passes a RunConfig built from the
RAGAS_* settings; invalid settings fail instead of falling back.
"""

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import ValidationError
from ragas.run_config import RunConfig

from app.core.config import Settings, settings
from app.services.evaluation import ragas_runner
from app.services.experiment.runner import QueryResult

THROTTLE_SETTINGS = ["RAGAS_MAX_WORKERS", "RAGAS_MAX_RETRIES", "RAGAS_MAX_WAIT", "RAGAS_TIMEOUT"]


def _result(query_id):
    return QueryResult(
        query_id=query_id, question=f"question {query_id}", generated_answer="answer",
        retrieved_chunks=[{"content": "context"}], latency_ms=1.0,
        input_tokens=1, output_tokens=1, cost_usd=0.0,
    )


def _evaluate_mock():
    scores = {k: 0.5 for k in ragas_runner.METRIC_KEYS}
    ragas_result = MagicMock()
    ragas_result.to_pandas.return_value = pd.DataFrame([scores])
    return MagicMock(return_value=ragas_result)


def _run_and_capture_config():
    """Run run_ragas_evaluation with evaluate() patched; return the RunConfig it got."""
    evaluate = _evaluate_mock()
    with patch("ragas.evaluate", evaluate), \
         patch("langchain_openai.ChatOpenAI"), \
         patch("langchain_openai.OpenAIEmbeddings"):
        scores = ragas_runner.run_ragas_evaluation([_result("q1")], [{"id": "q1"}])
    assert scores == [{k: 0.5 for k in ragas_runner.METRIC_KEYS}]
    evaluate.assert_called_once()
    return evaluate.call_args.kwargs["run_config"]


class TestRunConfigPassedToEvaluate:

    def test_defaults(self):
        cfg = _run_and_capture_config()
        assert isinstance(cfg, RunConfig)
        assert cfg.max_workers == 4
        assert cfg.max_retries == 10
        assert cfg.max_wait == 60
        assert cfg.timeout == 180

    def test_overridden_settings_change_run_config(self):
        with patch.object(settings, "RAGAS_MAX_WORKERS", 2), \
             patch.object(settings, "RAGAS_MAX_RETRIES", 3), \
             patch.object(settings, "RAGAS_MAX_WAIT", 7), \
             patch.object(settings, "RAGAS_TIMEOUT", 30):
            cfg = _run_and_capture_config()
        assert (cfg.max_workers, cfg.max_retries, cfg.max_wait, cfg.timeout) == (2, 3, 7, 30)

    def test_effective_values_logged_once(self, caplog):
        with caplog.at_level(logging.INFO, logger=ragas_runner.logger.name):
            _run_and_capture_config()
        lines = [r.getMessage() for r in caplog.records if "max_workers=" in r.getMessage()]
        assert len(lines) == 1
        assert "max_workers=4 max_retries=10 max_wait=60s timeout=180s" in lines[0]

    def test_no_evaluate_call_for_empty_results(self):
        evaluate = _evaluate_mock()
        with patch("ragas.evaluate", evaluate):
            assert ragas_runner.run_ragas_evaluation([], []) == []
        evaluate.assert_not_called()


class TestSettingsValidation:

    @pytest.mark.parametrize("name", THROTTLE_SETTINGS)
    def test_env_override(self, name, monkeypatch):
        monkeypatch.setenv(name, "5")
        assert getattr(Settings(_env_file=None), name) == 5

    @pytest.mark.parametrize("name", THROTTLE_SETTINGS)
    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_non_positive_rejected(self, name, value, monkeypatch):
        monkeypatch.setenv(name, value)
        with pytest.raises(ValidationError, match=name):
            Settings(_env_file=None)

    @pytest.mark.parametrize("name", THROTTLE_SETTINGS)
    def test_non_integer_rejected(self, name, monkeypatch):
        monkeypatch.setenv(name, "many")
        with pytest.raises(ValidationError, match=name):
            Settings(_env_file=None)
