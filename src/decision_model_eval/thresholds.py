"""Working-point selection: choose a score threshold on one split, report it on another.

A decision threshold is a tuned hyperparameter. It is selected on a held-out
selection split, frozen, and evaluated on an untouched reporting split. Quoting
the value a threshold achieves on the split it was selected on is optimistically
biased, so that number is only ever reported as a labeled upper bound.

:func:`scan_thresholds` returns three rows (see :class:`ScanResults`):

- ``frozen``: selected on the selection split, evaluated on the reporting split.
  The headline number.
- ``oracle``: re-selected on the reporting split itself. A biased upper bound,
  shown only to display the size of the selection bias.
- ``baseline_matched``: the threshold whose selection-split TPR matches a
  non-model baseline's TPR, so model and baseline are compared at equal recall.

The quantity being optimized is a pluggable :data:`Objective`. Counts are
weighted (``w``); TPR and FPR are unweighted fractions.

Every distinct score is a candidate threshold (``roc_curve`` with
``drop_intermediate=False``). The default thinning drops points inside straight
runs of the ROC curve, which is harmless with tens of thousands of thresholds
but at a few hundred items moves a TPR match by a whole run, and the objective
is an arbitrary callable, so no thinned point can be assumed suboptimal.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_curve

Objective = Callable[[float, float, float, float], float]
"""``objective(tp, fp, pos_total, neg_total) -> value`` to maximize.

``tp`` and ``fp`` are the weighted positives and negatives passing the
threshold; ``pos_total`` and ``neg_total`` are the split's weighted totals, so
an objective can also charge for missed positives (``pos_total - tp``).
"""


def weighted_counts(
    scores: np.ndarray, y: np.ndarray, w: np.ndarray, threshold: float
) -> tuple[float, float]:
    """Weighted positives and negatives with ``score >= threshold``."""
    pass_mask = scores >= threshold
    tp = float(w[pass_mask & (y == 1)].sum())
    fp = float(w[pass_mask & (y == 0)].sum())
    return tp, fp


@dataclass(frozen=True)
class ThresholdReport:
    """Metrics for one operating point, evaluated on the reporting split.

    ``tpr``/``fpr`` are unweighted fractions; ``tp``/``fp`` are weighted counts
    and ``value`` the objective evaluated on them.
    """

    threshold: float
    value: float
    tpr: float
    fpr: float
    tp: float
    fp: float


@dataclass(frozen=True)
class ScanResults:
    """Operating-point rows produced by :func:`scan_thresholds`.

    - ``frozen``: objective-optimal threshold selected on the selection split,
      evaluated on the reporting split. The headline number.
    - ``oracle``: threshold re-selected on the reporting split itself.
      Optimistically biased upper bound (the argmax over many thresholds of a
      noisy curve rides upward fluctuations); reported only to display the size
      of the selection bias, never as a result.
    - ``baseline_matched``: threshold whose selection-split TPR matches the
      baseline's TPR, evaluated on the reporting split.
    """

    frozen: ThresholdReport
    oracle: ThresholdReport
    baseline_matched: ThresholdReport


def _totals(y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    return float(w[y == 1].sum()), float(w[y == 0].sum())


def _report_at(
    scores: np.ndarray, y: np.ndarray, w: np.ndarray, threshold: float, objective: Objective
) -> ThresholdReport:
    """Evaluate all operating-point metrics at a fixed, already-chosen threshold."""
    tp, fp = weighted_counts(scores, y, w, threshold)
    pass_mask = scores >= threshold
    pos_mask = y == 1
    neg_mask = y == 0
    tpr = float((pass_mask & pos_mask).sum() / pos_mask.sum()) if pos_mask.any() else 0.0
    fpr = float((pass_mask & neg_mask).sum() / neg_mask.sum()) if neg_mask.any() else 0.0
    return ThresholdReport(
        threshold=float(threshold), value=objective(tp, fp, *_totals(y, w)),
        tpr=tpr, fpr=fpr, tp=tp, fp=fp,
    )


def select_threshold(
    scores_sel: np.ndarray, y_sel: np.ndarray, w_sel: np.ndarray, objective: Objective
) -> float:
    """Objective-optimal score threshold, selected on a held-out selection split.

    Scans every ROC threshold of the selection split and returns the argmax of
    ``objective``. The threshold must then be frozen and evaluated on an
    untouched split.
    """
    pos_total, neg_total = _totals(y_sel, w_sel)
    _, _, score_thresholds = roc_curve(y_sel, scores_sel, drop_intermediate=False)
    values = np.array([
        objective(*weighted_counts(scores_sel, y_sel, w_sel, float(t)), pos_total, neg_total)
        for t in score_thresholds
    ])
    return float(score_thresholds[int(np.argmax(values))])


def _threshold_at_tpr(scores_sel: np.ndarray, y_sel: np.ndarray, target_tpr: float) -> float:
    """Score threshold whose selection-split TPR is closest to ``target_tpr``."""
    _, tpr_vals, score_thresholds = roc_curve(y_sel, scores_sel, drop_intermediate=False)
    return float(score_thresholds[int(np.argmin(np.abs(tpr_vals - target_tpr)))])


def scan_thresholds(
    scores_sel: np.ndarray, y_sel: np.ndarray, w_sel: np.ndarray,
    scores_rep: np.ndarray, y_rep: np.ndarray, w_rep: np.ndarray,
    objective: Objective, baseline_tpr: float,
) -> ScanResults:
    """Select operating thresholds on the selection split, report on the reporting split.

    ``scores_sel``/``y_sel``/``w_sel`` are used only to choose thresholds;
    every returned metric is evaluated on ``scores_rep``/``y_rep``/``w_rep``.
    See :class:`ScanResults` for the three rows.
    """
    thr_frozen = select_threshold(scores_sel, y_sel, w_sel, objective)
    thr_oracle = select_threshold(scores_rep, y_rep, w_rep, objective)
    thr_baseline = _threshold_at_tpr(scores_sel, y_sel, baseline_tpr)
    return ScanResults(
        frozen=_report_at(scores_rep, y_rep, w_rep, thr_frozen, objective),
        oracle=_report_at(scores_rep, y_rep, w_rep, thr_oracle, objective),
        baseline_matched=_report_at(scores_rep, y_rep, w_rep, thr_baseline, objective),
    )
