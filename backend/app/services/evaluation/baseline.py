"""
Baseline comparison module.

Defines a naive baseline RAG configuration (dense retrieval, fixed chunking,
text-only parsing, no freshness) and computes delta metrics between any run
and the baseline to quantify improvement.
"""

from dataclasses import dataclass, asdict
from app.services.experiment.runner import PipelineConfig


BASELINE_CONFIG = PipelineConfig(
    retrieval_strategy="vector_only",
    chunking_strategy="fixed_size",
    parsing_strategy="text_only",
    freshness_policy="none",
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    top_k=5,
    reranker_enabled=False,
)

COMPARISON_METRICS = [
    "faithfulness",
    "answer_correctness",
    "context_recall",
    "context_precision",
    "answer_relevance",
    "latency_p50_ms",
    "avg_cost_usd",
    "freshness_validity",
    "multimodal_grounding_rate",
]

LOWER_IS_BETTER = {"latency_p50_ms", "avg_cost_usd"}


@dataclass
class BaselineDelta:
    metric: str
    baseline_value: float
    comparison_value: float
    absolute_delta: float
    relative_delta_pct: float
    improved: bool


def is_baseline_config(config: dict) -> bool:
    """Check if a run config matches the baseline."""
    return (
        config.get("retrieval_strategy") == BASELINE_CONFIG.retrieval_strategy
        and config.get("chunking_strategy") == BASELINE_CONFIG.chunking_strategy
        and config.get("parsing_strategy") == BASELINE_CONFIG.parsing_strategy
        and config.get("freshness_policy") == BASELINE_CONFIG.freshness_policy
    )


def find_baseline_run(runs: list[dict]) -> dict | None:
    """Find the baseline run from a list of run dicts with config and metrics."""
    for run in runs:
        if is_baseline_config(run.get("config", {})):
            return run
    return None


def compute_deltas(
    baseline_metrics: dict,
    comparison_metrics: dict,
) -> list[BaselineDelta]:
    """
    Compute per-metric deltas between a baseline and a comparison run.

    Returns a list of BaselineDelta objects showing absolute and relative
    improvement for each metric.
    """
    deltas = []
    for metric in COMPARISON_METRICS:
        base_val = float(baseline_metrics.get(metric) or 0.0)
        comp_val = float(comparison_metrics.get(metric) or 0.0)
        abs_delta = comp_val - base_val

        if base_val != 0:
            rel_delta = (abs_delta / abs(base_val)) * 100
        else:
            rel_delta = 0.0 if comp_val == 0 else 100.0

        if metric in LOWER_IS_BETTER:
            improved = comp_val < base_val
        else:
            improved = comp_val > base_val

        deltas.append(BaselineDelta(
            metric=metric,
            baseline_value=base_val,
            comparison_value=comp_val,
            absolute_delta=abs_delta,
            relative_delta_pct=round(rel_delta, 2),
            improved=improved,
        ))

    return deltas


def compare_run_to_baseline(
    runs: list[dict],
    target_run_id: str | None = None,
) -> dict:
    """
    Compare one or all runs to the baseline.

    If target_run_id is provided, compare just that run.
    Otherwise, compare the best_overall run to baseline.

    Returns a summary dict with baseline config, comparison config,
    and per-metric deltas.
    """
    baseline_run = find_baseline_run(runs)

    if baseline_run is None:
        return {
            "error": "No baseline run found. Run an experiment that includes "
                     "the baseline config (vector_only, fixed_size, text_only, none).",
            "baseline_config": asdict(BASELINE_CONFIG),
        }

    baseline_metrics = baseline_run.get("metrics", {})

    valid_runs = [r for r in runs if r.get("metrics") and r != baseline_run]
    if not valid_runs:
        return {
            "error": "No non-baseline runs with metrics found for comparison.",
            "baseline_config": asdict(BASELINE_CONFIG),
        }

    if target_run_id:
        target = next((r for r in valid_runs if r.get("run_id") == target_run_id), None)
        if not target:
            return {"error": f"Run {target_run_id} not found or has no metrics."}
    else:
        def overall_score(r):
            m = r.get("metrics", {})
            return (
                0.4 * float(m.get("faithfulness") or 0)
                + 0.3 * float(m.get("answer_correctness") or 0)
                + 0.2 * float(m.get("context_recall") or 0)
            )
        target = max(valid_runs, key=overall_score)

    deltas = compute_deltas(baseline_metrics, target.get("metrics", {}))
    improvements = [d for d in deltas if d.improved]
    regressions = [d for d in deltas if not d.improved and d.absolute_delta != 0]

    return {
        "baseline": {
            "run_id": baseline_run.get("run_id"),
            "config": baseline_run.get("config"),
            "metrics": baseline_metrics,
        },
        "comparison": {
            "run_id": target.get("run_id"),
            "config": target.get("config"),
            "metrics": target.get("metrics"),
        },
        "deltas": [asdict(d) for d in deltas],
        "summary": {
            "improvements": len(improvements),
            "regressions": len(regressions),
            "top_improvement": asdict(max(deltas, key=lambda d: d.relative_delta_pct))
                if deltas else None,
        },
    }
