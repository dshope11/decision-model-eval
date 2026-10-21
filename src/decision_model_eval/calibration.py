"""Calibration of stated probabilities: how far they sit from observed frequencies.

``p`` is an ``(items, options)`` array of stated probabilities and ``y`` the index of each
item's true option. A backend's probabilities are recorded as stated, so a row need not
sum to exactly 1; nothing here renormalizes them.

- ``reliability``: per bin, the mean stated probability of an event, the fraction of
  items where it happened, and the 1 sigma Clopper-Pearson interval on that fraction.
- ``ece``: the expected calibration error, the item-weighted mean of
  ``|observed - stated|`` over bins. ``top_label_ece`` asks it of the top option (the
  probability a working point acts on); ``classwise_ece`` averages it over options.
- ``ece_floor``: the ECE a perfectly calibrated model stating the same probabilities would
  show from sampling noise alone. Binned ECE on a few hundred items sits above zero even
  then, so a measured ECE is read against this floor.
- ``brier``: the multi-class Brier score, a proper scoring rule that needs no bins.

Bins are ``BINS`` equal-width intervals ``[k/BINS, (k+1)/BINS)``; a probability of
exactly 1 falls in the last.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from decision_model_eval.stats import clopper_pearson

BINS = 10
FLOOR_DRAWS = 4000


@dataclass(frozen=True)
class Reliability:
    """One reliability curve; arrays hold the non-empty bins only, in order."""

    stated: np.ndarray  # mean stated probability in the bin
    observed: np.ndarray  # fraction of the bin's items where the event happened
    lo: np.ndarray  # Clopper-Pearson 1 sigma bounds on ``observed``
    hi: np.ndarray
    n: np.ndarray  # items in the bin


def _bin_sums(p: np.ndarray, hit: np.ndarray,
              bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per bin: the items, the events among them, and the sum of stated probabilities."""
    index = np.minimum((p * bins).astype(int), bins - 1)
    return (np.bincount(index, minlength=bins),
            np.bincount(index, weights=hit.astype(float), minlength=bins),
            np.bincount(index, weights=p, minlength=bins))


def reliability(p: np.ndarray, hit: np.ndarray, bins: int = BINS) -> Reliability:
    """Bin stated probabilities ``p`` of an event against whether it happened (``hit``)."""
    n, k, total = _bin_sums(p, hit, bins)
    full = n > 0
    lo, hi = clopper_pearson(k[full], n[full])
    return Reliability(stated=total[full] / n[full], observed=k[full] / n[full],
                       lo=lo, hi=hi, n=n[full])


def ece(p: np.ndarray, hit: np.ndarray, bins: int = BINS) -> float:
    n, k, total = _bin_sums(p, hit, bins)
    # A bin's n * |k / n - total / n| is |k - total|, and an empty bin adds nothing.
    return float(np.abs(k - total).sum() / n.sum())


def ece_floor(p: np.ndarray, seed: int, draws: int = FLOOR_DRAWS,
              bins: int = BINS) -> dict[str, float | int]:
    """The mean and 95th percentile of the ECE over ``draws`` simulated outcomes, each
    event happening with its stated probability ``p``: what sampling noise alone gives."""
    rng = np.random.default_rng(seed)
    values = np.array([ece(p, rng.random(p.shape) < p, bins) for _ in range(draws)])
    return {"mean": float(values.mean()), "p95": float(np.percentile(values, 95)),
            "draws": draws, "seed": seed}


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
