"""
Ragas evaluation runner.

Wraps Ragas metrics (faithfulness, context precision, context recall,
answer relevance, answer correctness) into a callable interface that
takes DriftLens QueryResult objects and returns per-query scores.
"""

from app.services.experiment.runner import QueryResult


def build_ragas_dataset(results: list[QueryResult], eval_set: list[dict]) -> dict:
    """
    Convert DriftLens QueryResult objects into the Ragas dataset format.

    Returns a dict ready to pass to ragas.evaluate().
    """
    raise NotImplementedError


def run_ragas_evaluation(results: list[QueryResult], eval_set: list[dict]) -> list[dict]:
    """
    Run all Ragas metrics on a completed pipeline run.

    Returns a list of per-query metric dicts.
    """
    raise NotImplementedError
