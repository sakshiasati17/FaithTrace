"""
Recommendation engine.

Selects the best pipeline configuration per objective based on aggregated
experiment metrics and failure diagnostics.
"""

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    objective: str
    best_config: dict
    run_id: str
    # None when no run could be recommended for the objective.
    score: float | None
    rationale: str
    # "ok"; "no_eligible_runs" when no run has the metrics the objective
    # needs; "error" when scoring the objective raised.
    status: str = "ok"


class NoEligibleRuns(Exception):
    """No run has the metric(s) an objective is scored on."""


OBJECTIVES = [
    "best_overall",
    "lowest_cost",
    "best_latency",
    "best_faithfulness",
    "best_for_tables",
    "best_for_drift",
    "best_for_long_pdfs",
]

# Metrics each objective is scored on. A run missing any of them is not a
# candidate for that objective (unknown is neither 0 nor the worst value).
OBJECTIVE_METRICS = {
    "best_overall": ("faithfulness", "answer_correctness", "context_recall", "avg_cost_usd"),
    "lowest_cost": ("avg_cost_usd",),
    "best_latency": ("latency_p50_ms",),
    "best_faithfulness": ("faithfulness",),
    "best_for_tables": ("multimodal_grounding_rate",),
    "best_for_drift": ("freshness_validity",),
    "best_for_long_pdfs": ("context_recall",),
}


def recommend(
    runs: list[dict],  # list of {run_id, config, metrics, diagnostics}
    objectives: list[str] | None = None,
) -> list[Recommendation]:
    """
    For each objective, select and explain the best pipeline configuration.

    Args:
        runs: List of completed run dicts with run_id, config, metrics, and diagnostics
        objectives: Subset of OBJECTIVES to evaluate; defaults to all

    Returns:
        List of Recommendation objects, one per objective. An objective no
        run qualifies for (or that failed) is still returned, with
        ``status`` set and ``score`` None, rather than dropped.
    """
    if not runs:
        return []

    # Filter out runs without metrics
    valid_runs = [r for r in runs if r.get("metrics")]
    if not valid_runs:
        return []

    targets = objectives or OBJECTIVES
    results = []
    for obj in targets:
        try:
            rec = _recommend_for_objective(valid_runs, obj)
        except NoEligibleRuns as exc:
            logger.warning("No recommendation for %s: %s", obj, exc)
            rec = _empty(obj, "no_eligible_runs", str(exc))
        except Exception as exc:
            logger.exception("Recommendation for %s failed", obj)
            rec = _empty(obj, "error", f"Could not score this objective: {type(exc).__name__}: {exc}")
        results.append(rec)
    return results


def _empty(objective: str, status: str, rationale: str) -> Recommendation:
    return Recommendation(
        objective=objective, best_config={}, run_id="", score=None,
        rationale=rationale, status=status,
    )


def _safe_metric(run: dict, key: str) -> float | None:
    """A run's metric as a finite float, or None when missing/unscored."""
    m = run.get("metrics", {})
    if not isinstance(m, dict):
        return None
    value = m.get(key)
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _eligible(runs: list[dict], objective: str) -> list[dict]:
    """Runs that have every metric ``objective`` is scored on."""
    keys = OBJECTIVE_METRICS.get(objective, ())
    eligible = [r for r in runs if all(_safe_metric(r, k) is not None for k in keys)]
    skipped = len(runs) - len(eligible)
    if not eligible:
        raise NoEligibleRuns(
            f"none of {len(runs)} runs has {', '.join(keys)} scored"
        )
    if skipped:
        logger.info(
            "%s: %d of %d runs skipped (missing %s)",
            objective, skipped, len(runs), ", ".join(keys),
        )
    return eligible


def _recommend_for_objective(all_runs: list[dict], objective: str) -> Recommendation:
    # Metric values below are never None: runs come from _eligible().
    runs = _eligible(all_runs, objective)

    if objective == "best_overall":
        # Composite: 40% faithfulness + 30% answer_correctness + 20% context_recall + 10% cost savings
        def score(r):
            f = _safe_metric(r, "faithfulness")
            ac = _safe_metric(r, "answer_correctness")
            cr = _safe_metric(r, "context_recall")
            cost = _safe_metric(r, "avg_cost_usd")
            # Normalize cost (lower is better): 1 - (cost / max_cost)
            max_cost = max(_safe_metric(x, "avg_cost_usd") for x in runs) or 0.001
            cost_score = 1.0 - (cost / max_cost)
            return 0.4 * f + 0.3 * ac + 0.2 * cr + 0.1 * cost_score

        best = max(runs, key=score)
        s = score(best)
        f = _safe_metric(best, "faithfulness")
        rationale = (
            f"Selected as best overall: highest composite score ({s:.2f}) balancing "
            f"faithfulness ({f:.2f}), answer correctness, context recall, and cost efficiency."
        )

    elif objective == "lowest_cost":
        best = min(runs, key=lambda r: _safe_metric(r, "avg_cost_usd"))
        s = _safe_metric(best, "avg_cost_usd")
        rationale = f"Lowest average cost per query: ${s:.4f} USD."

    elif objective == "best_latency":
        best = min(runs, key=lambda r: _safe_metric(r, "latency_p50_ms"))
        s = _safe_metric(best, "latency_p50_ms")
        rationale = f"Fastest median response time: {s:.0f} ms at p50."

    elif objective == "best_faithfulness":
        best = max(runs, key=lambda r: _safe_metric(r, "faithfulness"))
        s = _safe_metric(best, "faithfulness")
        rationale = f"Highest faithfulness score: {s:.3f} — answers best grounded in retrieved context."

    elif objective == "best_for_tables":
        # Prefer parsing strategies that include table extraction
        table_runs = [
            r for r in runs
            if r.get("config", {}).get("parsing_strategy") in ("text_table", "spreadsheet_aware")
        ]
        candidate_runs = table_runs if table_runs else runs
        best = max(candidate_runs, key=lambda r: _safe_metric(r, "multimodal_grounding_rate"))
        s = _safe_metric(best, "multimodal_grounding_rate")
        rationale = (
            f"Best for table and spreadsheet queries: multimodal grounding rate {s:.3f}. "
            f"Uses {best.get('config', {}).get('parsing_strategy', 'N/A')} parsing strategy."
        )

    elif objective == "best_for_drift":
        # Prefer runs with freshness policy != "none"
        drift_runs = [
            r for r in runs
            if r.get("config", {}).get("freshness_policy", "none") != "none"
        ]
        candidate_runs = drift_runs if drift_runs else runs
        best = max(candidate_runs, key=lambda r: _safe_metric(r, "freshness_validity"))
        s = _safe_metric(best, "freshness_validity")
        policy = best.get("config", {}).get("freshness_policy", "N/A")
        rationale = (
            f"Best temporal drift handling: freshness validity {s:.3f} "
            f"using '{policy}' freshness policy."
        )

    elif objective == "best_for_long_pdfs":
        # Prefer text_table parsing + high context recall
        pdf_runs = [
            r for r in runs
            if r.get("config", {}).get("parsing_strategy") == "text_table"
        ]
        candidate_runs = pdf_runs if pdf_runs else runs
        best = max(candidate_runs, key=lambda r: _safe_metric(r, "context_recall"))
        s = _safe_metric(best, "context_recall")
        rationale = (
            f"Best for long PDFs: highest context recall {s:.3f} "
            f"with text+table parsing strategy."
        )

    else:
        best = runs[0]
        s = 0.0
        rationale = f"Default selection for unknown objective: {objective}"

    return Recommendation(
        objective=objective,
        best_config=best.get("config", {}),
        run_id=best.get("run_id", ""),
        score=s,
        rationale=rationale,
    )
