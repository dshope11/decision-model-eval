"""Calibration of stated probabilities: how far they sit from observed frequencies.

``p`` is an ``(items, options)`` array of stated probabilities and ``y`` the index of each
item's true option. A backend's probabilities are recorded as stated, so a row need not
sum to exactly 1; nothing here renormalizes them.

- ``reliability``: per bin, the mean stated probability of an event, the fraction of
  items where it happened, and the 1 sigma Clopper-Pearson interval on that fraction.
- ``ece``: the expected calibration error, the item-weighted mean of
  ``|observed - stated|`` over bins. ``top_label_ece`` asks it of the top option (the
  probability a working point acts on); ``classwise_ece`` averages it over options.
- ``brier``: the multi-class Brier score, a proper scoring rule that needs no bins.

Bins are ``BINS`` equal-width intervals ``[k/BINS, (k+1)/BINS)``; a probability of
exactly 1 falls in the last.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from decision_model_eval.stats import clopper_pearson

BINS = 10


@dataclass(frozen=True)
class Reliability:
    """One reliability curve; arrays hold the non-empty bins only, in order."""

    stated: np.ndarray  # mean stated probability in the bin
    observed: np.ndarray  # fraction of the bin's items where the event happened
    lo: np.ndarray  # Clopper-Pearson 1 sigma bounds on ``observed``
    hi: np.ndarray
    n: np.ndarray  # items in the bin


def reliability(p: np.ndarray, hit: np.ndarray, bins: int = BINS) -> Reliability:
    """Bin stated probabilities ``p`` of an event against whether it happened (``hit``)."""
    index = np.minimum((p * bins).astype(int), bins - 1)
    n = np.bincount(index, minlength=bins)
    k = np.bincount(index, weights=hit.astype(float), minlength=bins)
    total = np.bincount(index, weights=p, minlength=bins)
    full = n > 0
    lo, hi = clopper_pearson(k[full], n[full])
    return Reliability(stated=total[full] / n[full], observed=k[full] / n[full],
                       lo=lo, hi=hi, n=n[full])


def ece(p: np.ndarray, hit: np.ndarray, bins: int = BINS) -> float:
    curve = reliability(p, hit, bins)
    return float(np.sum(curve.n * np.abs(curve.observed - curve.stated)) / curve.n.sum())


def top_label(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The top option's stated probability per item, and whether it was the true one."""
    return p.max(axis=1), p.argmax(axis=1) == y


def top_label_ece(p: np.ndarray, y: np.ndarray, bins: int = BINS) -> float:
    return ece(*top_label(p, y), bins)


def classwise_ece(p: np.ndarray, y: np.ndarray, bins: int = BINS) -> float:
    return float(np.mean([ece(p[:, k], y == k, bins) for k in range(p.shape[1])]))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Mean over items of the squared distance to the one-hot truth (0 is perfect, 2 worst)."""
    truth = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - truth) ** 2, axis=1)))
