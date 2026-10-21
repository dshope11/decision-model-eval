"""Tests for decision_model_eval.calibration."""

from __future__ import annotations

import numpy as np
import pytest

from decision_model_eval.calibration import (
    brier,
    classwise_ece,
    ece,
    ece_floor,
    reliability,
    top_label,
    top_label_ece,
)


def test_reliability_bins_by_hand() -> None:
    p = np.array([0.05, 0.15, 0.12, 0.95, 1.0])
    hit = np.array([False, True, False, True, True])
    curve = reliability(p, hit)
    assert curve.n.tolist() == [1, 2, 2]  # bins 0, 1 and 9; 1.0 joins the last bin
    assert curve.stated == pytest.approx([0.05, 0.135, 0.975])
    assert curve.observed == pytest.approx([0.0, 0.5, 1.0])
    assert np.all(curve.lo <= curve.observed) and np.all(curve.observed <= curve.hi)


def test_ece_by_hand() -> None:
    p = np.array([0.05, 0.15, 0.12, 0.95, 1.0])
    hit = np.array([False, True, False, True, True])
    # (1 * 0.05 + 2 * 0.365 + 2 * 0.025) / 5
    assert ece(p, hit) == pytest.approx(0.166)


def test_perfectly_calibrated_bins_have_zero_ece() -> None:
    p = np.array([0.25] * 4 + [0.75] * 4)
    hit = np.array([True, False, False, False, True, True, True, False])
    assert ece(p, hit) == pytest.approx(0.0)


def test_ece_floor_of_one_bin_is_the_binomial_mean_deviation() -> None:
    p = np.full(200, 0.7)
    floor = ece_floor(p, seed=1, draws=500)
    # One bin: |k / 200 - 0.7| with k ~ Binomial(200, 0.7), mean about sd * sqrt(2 / pi)
    assert floor["mean"] == pytest.approx(np.sqrt(0.7 * 0.3 / 200 * 2 / np.pi), rel=0.1)
    assert floor["p95"] > floor["mean"]
    assert ece_floor(p, seed=1, draws=500) == floor


def test_certain_probabilities_have_no_floor() -> None:
    floor = ece_floor(np.array([0.0, 1.0, 1.0]), seed=0, draws=10)
    assert floor["mean"] == floor["p95"] == 0.0


def test_top_label_and_classwise() -> None:
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.5, 0.1, 0.4]])
    y = np.array([0, 2, 0])
    confidence, correct = top_label(p, y)
    assert confidence.tolist() == [0.7, 0.6, 0.5]
    assert correct.tolist() == [True, False, True]
    assert top_label_ece(p, y) == pytest.approx((0.3 + 0.6 + 0.5) / 3)
    by_class = [ece(p[:, k], y == k) for k in range(3)]
    assert classwise_ece(p, y) == pytest.approx(np.mean(by_class))


def test_brier() -> None:
    assert brier(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1])) == 0.0
    assert brier(np.array([[0.0, 1.0]]), np.array([0])) == 2.0
    assert brier(np.array([[0.5, 0.5]]), np.array([0])) == pytest.approx(0.5)
