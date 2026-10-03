"""Tests for the threshold-selection protocol in decision_model_eval.thresholds.

The operating threshold must be selected on the selection split, frozen, and
evaluated on the reporting split. These tests use synthetic arrays where the
selection-argmax and reporting-argmax thresholds deliberately differ, so any
regression that re-couples selection and reporting onto the same split changes
the reported numbers and fails.

The objective here is the Asimov discovery significance (Cowan et al. 2011,
Eq. 97), the objective these arrays were hand-computed for. Hand-computed
optima over the roc_curve threshold grid:
- selection split: argmax at threshold 0.4 (Z=0 at 0.9 since fp=0,
  ~5.72 at 0.6, ~6.13 at 0.4, ~5.16 at 0.1)
- reporting split: argmax at threshold 0.5 (~6.63), while the frozen
  selection threshold 0.4 gives only ~5.18 there (tp=10, fp=1.5)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from decision_model_eval.thresholds import (
    _report_at,
    _threshold_at_tpr,
    scan_thresholds,
    select_threshold,
    weighted_counts,
)


def asimov(tp: float, fp: float, pos_total: float, neg_total: float) -> float:
    """Asimov significance ``sqrt(2 * [(s+b) * ln(1 + s/b) - s])`` with s=tp, b=fp."""
    if fp <= 0 or tp <= 0:
        return 0.0
    return math.sqrt(2.0 * ((tp + fp) * math.log(1.0 + tp / fp) - tp))


# Selection split: objective-optimal threshold is 0.4
SCORES_SEL = np.array([0.1, 0.4, 0.6, 0.9])
Y_SEL = np.array([0, 1, 0, 1])
W_SEL = np.array([1.0, 1.0, 1.0, 10.0])

# Reporting split: its own argmax is 0.5, not 0.4
SCORES_REP = np.array([0.45, 0.5, 0.8, 0.95])
Y_REP = np.array([0, 1, 0, 1])
W_REP = np.array([1.0, 2.0, 0.5, 8.0])


def test_select_threshold_picks_objective_argmax() -> None:
    assert select_threshold(SCORES_SEL, Y_SEL, W_SEL, asimov) == pytest.approx(0.4)
    assert select_threshold(SCORES_REP, Y_REP, W_REP, asimov) == pytest.approx(0.5)


def test_frozen_row_uses_selection_threshold_evaluated_on_reporting() -> None:
    """The headline value must come from the selection threshold applied to reporting."""
    scan = scan_thresholds(
        SCORES_SEL, Y_SEL, W_SEL, SCORES_REP, Y_REP, W_REP,
        objective=asimov, baseline_tpr=0.5,
    )
    assert scan.frozen.threshold == pytest.approx(0.4)
    tp, fp = weighted_counts(SCORES_REP, Y_REP, W_REP, 0.4)
    assert tp == pytest.approx(10.0)
    assert fp == pytest.approx(1.5)
    assert scan.frozen.value == pytest.approx(asimov(tp, fp, 10.0, 1.5))
    # The regression this guards against: frozen must NOT be the reporting argmax
    assert scan.frozen.threshold != scan.oracle.threshold
    assert scan.frozen.value != pytest.approx(scan.oracle.value)


def test_oracle_row_is_reporting_argmax_upper_bound() -> None:
    scan = scan_thresholds(
        SCORES_SEL, Y_SEL, W_SEL, SCORES_REP, Y_REP, W_REP,
        objective=asimov, baseline_tpr=0.5,
    )
    assert scan.oracle.threshold == pytest.approx(0.5)
    tp, fp = weighted_counts(SCORES_REP, Y_REP, W_REP, 0.5)
    assert scan.oracle.value == pytest.approx(asimov(tp, fp, 10.0, 1.5))
    assert scan.oracle.value > scan.frozen.value


def test_baseline_matched_threshold_selected_on_selection_split() -> None:
    """The baseline-TPR-matched threshold is also chosen on selection, then frozen."""
    scan = scan_thresholds(
        SCORES_SEL, Y_SEL, W_SEL, SCORES_REP, Y_REP, W_REP,
        objective=asimov, baseline_tpr=0.5,
    )
    # On the selection split, TPR=0.5 (unweighted) is first reached at threshold 0.9
    assert scan.baseline_matched.threshold == pytest.approx(0.9)
    # Evaluated on reporting: only the 0.95 positive passes
    assert scan.baseline_matched.tpr == pytest.approx(0.5)
    assert scan.baseline_matched.tp == pytest.approx(8.0)
    assert scan.baseline_matched.fp == pytest.approx(0.0)


def test_report_at_unweighted_rates_weighted_counts() -> None:
    report = _report_at(SCORES_REP, Y_REP, W_REP, threshold=0.5, objective=asimov)
    # score >= 0.5: positives {0.5, 0.95}, negatives {0.8}
    assert report.tpr == pytest.approx(1.0)
    assert report.fpr == pytest.approx(0.5)
    assert report.tp == pytest.approx(10.0)
    assert report.fp == pytest.approx(0.5)


def test_tpr_match_reaches_points_inside_a_run_of_positives() -> None:
    """The threshold grid is every distinct score, not roc_curve's thinned default.

    Three positives in a row form a vertical run on the ROC curve; the default
    thinning drops the middle point (threshold 0.8, TPR 2/3), and the match
    would then land on 0.9 (TPR 1/3) instead.
    """
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    y = np.array([1, 1, 1, 0])
    assert _threshold_at_tpr(scores, y, target_tpr=2 / 3) == pytest.approx(0.8)


def test_objective_receives_split_totals() -> None:
    """Totals passed to the objective are the weighted totals of the split scored."""
    seen: list[tuple[float, float]] = []

    def recording(tp: float, fp: float, pos_total: float, neg_total: float) -> float:
        seen.append((pos_total, neg_total))
        return tp - fp

    select_threshold(SCORES_SEL, Y_SEL, W_SEL, recording)
    assert set(seen) == {(11.0, 2.0)}
    seen.clear()
    _report_at(SCORES_REP, Y_REP, W_REP, threshold=0.5, objective=recording)
    assert seen == [(10.0, 1.5)]


def test_weighted_counts_basic() -> None:
    """Weighted positives and negatives are summed correctly above threshold."""
    scores = np.array([0.1, 0.5, 0.8, 0.9])
    y = np.array([0, 1, 0, 1])
    w = np.array([2.0, 3.0, 4.0, 5.0])
    tp, fp = weighted_counts(scores, y, w, threshold=0.6)
    # score >= 0.6: indices 2 (negative, w=4) and 3 (positive, w=5)
    assert tp == pytest.approx(5.0)
    assert fp == pytest.approx(4.0)


def test_weighted_counts_threshold_is_inclusive() -> None:
    """An item with score exactly equal to threshold must pass (>= not >)."""
    scores = np.array([0.6, 0.3])
    y = np.array([1, 0])
    w = np.array([2.0, 1.0])
    tp, fp = weighted_counts(scores, y, w, threshold=0.6)
    assert tp == pytest.approx(2.0)
    assert fp == pytest.approx(0.0)


def test_weighted_counts_all_below_threshold() -> None:
    scores = np.array([0.1, 0.2])
    y = np.array([0, 1])
    w = np.array([1.0, 1.0])
    tp, fp = weighted_counts(scores, y, w, threshold=0.9)
    assert tp == pytest.approx(0.0)
    assert fp == pytest.approx(0.0)
