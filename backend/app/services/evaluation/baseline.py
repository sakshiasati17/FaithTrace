"""
Baseline comparison module.

Defines a naive baseline RAG configuration (dense retrieval, default recursive
chunking, text-only parsing, no freshness) and computes delta metrics between
any run and the baseline to quantify improvement.

The baseline must be a config that the default 24-config MVP preset
(config_matrix.build_mvp_matrix) contains, otherwise default experiments can
never be compared. The MVP preset only chunks recursive/structure_aware, so the
baseline uses recursive (the ingestion default). The full 256-config matrix
contains it too. Exactly one config per experiment matches the baseline.
"""

from dataclasses import dataclass, asdict
from app.services.experiment.runner import PipelineConfig


BASELINE_CONFIG = PipelineConfig(
    retrieval_strategy="vector_only",
    chunking_strategy="recursive",
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
    # None when the metric is missing for either run: no delta is computed.
    baseline_value: float | None
    comparison_value: float | None
    absolute_delta: float | None
    relative_delta_pct: float | None
    improved: bool | None


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
        base_raw = baseline_metrics.get(metric)
        comp_raw = comparison_metrics.get(metric)
        if base_raw is None or comp_raw is None:
            # Unknown is not 0: report the metric without a delta.
            deltas.append(BaselineDelta(
                metric=metric,
                baseline_value=None if base_raw is None else float(base_raw),
                comparison_value=None if comp_raw is None else float(comp_raw),
                absolute_delta=None,
                relative_delta_pct=None,
                improved=None,
            ))
            continue
        base_val = float(base_raw)
        comp_val = float(comp_raw)
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
                     "the baseline config (vector_only, recursive, text_only, none); "
                     "the default MVP preset and the full matrix both do.",
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
            parts = (m.get("faithfulness"), m.get("answer_correctness"), m.get("context_recall"))
            if any(v is None for v in parts):
                return None
            return 0.4 * float(parts[0]) + 0.3 * float(parts[1]) + 0.2 * float(parts[2])

        scored = [(overall_score(r), r) for r in valid_runs]
        scored = [(s, r) for s, r in scored if s is not None]
        if not scored:
            return {
                "error": "No non-baseline run has faithfulness, answer correctness and "
                         "context recall scores to pick a comparison run.",
                "baseline_config": asdict(BASELINE_CONFIG),
            }
        target = max(scored, key=lambda sr: sr[0])[1]

    deltas = compute_deltas(baseline_metrics, target.get("metrics", {}))
    compared = [d for d in deltas if d.improved is not None]
    improvements = [d for d in compared if d.improved]
    regressions = [d for d in compared if not d.improved and d.absolute_delta != 0]

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
            "top_improvement": asdict(max(compared, key=lambda d: d.relative_delta_pct))
                if compared else None,
            "not_compared": [d.metric for d in deltas if d.improved is None],
        },
    }
