"""
Ragas evaluation runner.

Wraps Ragas metrics (faithfulness, context precision, context recall,
answer relevance, answer correctness) into a callable interface that
takes FaithTrace QueryResult objects and returns per-query scores.
"""

import math

from app.services.experiment.runner import QueryResult


def _sanitize_float(val) -> float:
    """Convert NaN/inf to 0.0 so Postgres JSON columns don't choke."""
    try:
        f = float(val or 0.0)
        return f if math.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def build_ragas_dataset(results: list[QueryResult], eval_set: list[dict]) -> dict:
    """
    Convert FaithTrace QueryResult objects into the Ragas dataset format.

    Returns a datasets.Dataset object ready to pass to ragas.evaluate().
    """
    from datasets import Dataset

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    eval_by_id = {item["id"]: item for item in eval_set if "id" in item}

    for result in results:
        eval_item = eval_by_id.get(result.query_id, {})
        ground_truth = eval_item.get("ground_truth", "")

        data["question"].append(result.question)
        data["answer"].append(result.generated_answer)
        data["contexts"].append(
            [c.get("content", "") for c in result.retrieved_chunks if c.get("content")]
        )
        data["ground_truth"].append(ground_truth)

    return Dataset.from_dict(data)


def run_ragas_evaluation(results: list[QueryResult], eval_set: list[dict]) -> list[dict]:
    """
    Run all Ragas metrics on a completed pipeline run.

    Returns a list of per-query metric dicts.
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        context_precision,
        context_recall,
        answer_relevancy,
        answer_correctness,
    )
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from app.core.config import settings

    if not results:
        return []

    dataset = build_ragas_dataset(results, eval_set)

    # Configure Ragas to use our OpenAI credentials
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # cheaper model for evaluation judging
        openai_api_key=settings.OPENAI_API_KEY,
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=settings.OPENAI_API_KEY,
    )

    try:
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                context_precision,
                context_recall,
                answer_relevancy,
                answer_correctness,
            ],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )

        # Convert to per-query list of dicts
        result_df = result.to_pandas()
        per_query_scores = result_df.to_dict(orient="records")

        # Ensure each score dict has all metric keys
        metric_keys = [
            "faithfulness", "context_precision", "context_recall",
            "answer_relevancy", "answer_correctness"
        ]
        cleaned = []
        for row in per_query_scores:
            cleaned.append({k: _sanitize_float(row.get(k, 0.0)) for k in metric_keys})
        return cleaned

    except Exception as e:
        # Re-raise rate limit errors so the Celery task can retry with backoff
        # instead of silently storing zero scores that look like real results.
        try:
            from openai import RateLimitError
            if isinstance(e, RateLimitError):
                raise
        except ImportError:
            pass

        # Return zero scores for all other evaluation failures (import errors,
        # dataset format issues, etc.) so the pipeline doesn't hard-crash.
        zero_scores = {
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "answer_relevancy": 0.0,
            "answer_correctness": 0.0,
        }
        return [zero_scores.copy() for _ in results]

