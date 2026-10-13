"""
Temporal failure diagnosis.

A temporal failure is reported only when wrong-period evidence dominates the
retrieved context (the top-ranked chunk, or more than half of the dated
chunks, is outside the question's validity period) and the answer was not
judged correct (answer_correctness below TEMPORAL_ANSWER_CORRECT, or
unscored). The direction names it: an expired version is STALE_ANSWER, a
version not yet in effect is WRONG_VERSION. Diagnostic accuracy stays strict
per label; a temporal-family accuracy counts the two as one category.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from app.services.diagnostics.classifier import (
    TEMPORAL_ANSWER_CORRECT,
    FailureCategory,
    compute_diagnostic_accuracy,
    compute_temporal_family_diagnostic_accuracy,
    diagnose,
)
from app.services.experiment.runner import QueryResult


@pytest.fixture(autouse=True)
def _heuristic_only():
    """Force the heuristic classifier so results don't depend on a trained model on disk."""
    with patch("app.services.diagnostics.ml_classifier.predict", return_value=None):
        yield


def _epoch(date: str) -> int:
    return int(datetime.fromisoformat(date).timestamp())


# Question dated 2010: CC 3.0 (2007-2013) is in period, 4.0 (2013-) is not yet valid.
OLD_Q = {"id": "q_old", "valid_from": "2010-01-01", "modality": "text"}
# Question dated 2015: 4.0 is in period, 3.0 has expired.
NEW_Q = {"id": "q_new", "valid_from": "2015-01-01", "modality": "text"}

V3 = {"content": "3.0 text", "chunk_type": "text",
      "effective_from": _epoch("2007-02-23"), "effective_to": _epoch("2013-11-24")}
V4 = {"content": "4.0 text", "chunk_type": "text",
      "effective_from": _epoch("2013-11-25"), "effective_to": None}
UNDATED = {"content": "survey text", "chunk_type": "text"}

GOOD = {"faithfulness": 0.9, "context_recall": 0.9, "context_precision": 0.9, "answer_correctness": 0.9}
BAD = {"faithfulness": 0.9, "context_recall": 0.9, "context_precision": 0.9, "answer_correctness": 0.2}


def _result(chunks, query_id="q") -> QueryResult:
    return QueryResult(
        query_id=query_id, question="q", generated_answer="a",
        retrieved_chunks=[dict(c) for c in chunks],
        latency_ms=1.0, input_tokens=1, output_tokens=1, cost_usd=0.0,
    )


class TestTemporalRule:

    def test_stray_wrong_period_chunk_with_correct_answer_is_not_a_failure(self):
        # Top chunks in period, one expired chunk at rank 3, answer correct.
        d = diagnose(_result([V4, V4, V3]), NEW_Q, GOOD)
        assert d.primary_failure == FailureCategory.NO_FAILURE
        assert d.evidence["temporal"]["expired_chunks"] == 1
        assert d.evidence["temporal"]["dominant"] is False
        assert "temporal_violation" not in d.evidence

    def test_dominant_wrong_period_but_correct_answer_is_not_temporal(self):
        d = diagnose(_result([V3, V4]), NEW_Q, GOOD)
        assert d.primary_failure not in (FailureCategory.STALE_ANSWER, FailureCategory.WRONG_VERSION)
        assert d.evidence["temporal"]["top_chunk_out_of_period"] is True

    def test_top_chunk_expired_and_wrong_answer_is_stale(self):
        d = diagnose(_result([V3, V4, V4]), NEW_Q, BAD)
        assert d.primary_failure == FailureCategory.STALE_ANSWER
        assert d.evidence["temporal_rule"] == "top_chunk_out_of_period"

    def test_top_chunk_not_yet_valid_and_wrong_answer_is_wrong_version(self):
        d = diagnose(_result([V4, V3, V3]), OLD_Q, BAD)
        assert d.primary_failure == FailureCategory.WRONG_VERSION
        assert d.evidence["temporal"]["out_of_period_direction"] == "not_yet_valid"

    def test_majority_out_of_period_fires_even_if_top_is_in_period(self):
        d = diagnose(_result([V4, V3, V3]), NEW_Q, BAD)
        assert d.primary_failure == FailureCategory.STALE_ANSWER
        assert d.evidence["temporal_rule"] == "majority_out_of_period"

    def test_half_out_of_period_with_top_in_period_does_not_fire(self):
        # 1 of 2 dated chunks out of period is not "more than half".
        d = diagnose(_result([V4, V3]), NEW_Q, BAD)
        assert d.primary_failure not in (FailureCategory.STALE_ANSWER, FailureCategory.WRONG_VERSION)

    def test_undated_chunks_do_not_count_toward_majority(self):
        d = diagnose(_result([UNDATED, V3, UNDATED]), NEW_Q, BAD)
        # The only dated chunk is expired: 1 of 1 dated -> dominant.
        assert d.primary_failure == FailureCategory.STALE_ANSWER
        assert d.evidence["temporal"]["dated_chunks"] == 1

    def test_unscored_answer_still_fires_and_says_so(self):
        d = diagnose(_result([V3]), NEW_Q, {})
        assert d.primary_failure == FailureCategory.STALE_ANSWER
        assert d.evidence["answer_not_scored"] is True

    def test_threshold_boundary(self):
        at = dict(BAD, answer_correctness=TEMPORAL_ANSWER_CORRECT)
        below = dict(BAD, answer_correctness=TEMPORAL_ANSWER_CORRECT - 0.01)
        assert diagnose(_result([V3]), NEW_Q, at).primary_failure != FailureCategory.STALE_ANSWER
        assert diagnose(_result([V3]), NEW_Q, below).primary_failure == FailureCategory.STALE_ANSWER

    def test_no_valid_from_means_no_temporal_rule(self):
        d = diagnose(_result([V3, V3]), {"id": "q", "modality": "text"}, BAD)
        assert d.primary_failure not in (FailureCategory.STALE_ANSWER, FailureCategory.WRONG_VERSION)
        assert "temporal" not in d.evidence

    def test_all_in_period_records_counts_without_failure(self):
        d = diagnose(_result([V4, V4]), NEW_Q, GOOD)
        assert d.primary_failure == FailureCategory.NO_FAILURE
        assert d.evidence["temporal"]["out_of_period_chunks"] == 0


class TestFamilyAccuracy:
    EVAL = [
        {"id": "a", "failure_type": "WRONG_VERSION"},
        {"id": "b", "failure_type": "WRONG_VERSION"},
        {"id": "c", "failure_type": "TABLE_RETRIEVAL_MISS"},
        {"id": "d"},  # unlabelled: not counted
    ]

    def test_strict_vs_family(self):
        preds = {"a": "STALE_ANSWER", "b": "WRONG_VERSION", "c": "NO_FAILURE", "d": "STALE_ANSWER"}
        assert compute_diagnostic_accuracy(preds, self.EVAL) == pytest.approx(1 / 3)
        assert compute_temporal_family_diagnostic_accuracy(preds, self.EVAL) == pytest.approx(2 / 3)

    def test_family_does_not_merge_other_categories(self):
        preds = {"c": "CHART_LAYOUT_BLINDNESS"}
        assert compute_temporal_family_diagnostic_accuracy(preds, self.EVAL) == 0.0

    def test_none_when_nothing_labelled(self):
        assert compute_temporal_family_diagnostic_accuracy({"d": "STALE_ANSWER"}, self.EVAL) is None
