"""Tests for decision_model_eval.stats."""

from __future__ import annotations

import numpy as np
import pytest

from decision_model_eval.stats import clopper_pearson


def test_clopper_pearson_k_zero_lo_is_zero() -> None:
    """Lower bound must be exactly 0 when k=0 (no successes observed)."""
    lo, hi = clopper_pearson(np.array([0]), np.array([10]))
    assert lo[0] == pytest.approx(0.0)
    assert 0.0 < hi[0] < 1.0


def test_clopper_pearson_k_equals_n_hi_is_one() -> None:
    """Upper bound must be exactly 1 when k=n (all trials succeeded)."""
    lo, hi = clopper_pearson(np.array([10]), np.array([10]))
    assert hi[0] == pytest.approx(1.0)
    assert 0.0 < lo[0] < 1.0


def test_clopper_pearson_central_interval_contains_true_rate() -> None:
    """For k/n = 0.5, the 1-sigma interval should contain 0.5 and be plausibly narrow."""
    lo, hi = clopper_pearson(np.array([50]), np.array([100]))
    assert lo[0] < 0.5 < hi[0]
    assert (hi[0] - lo[0]) < 0.15  # 68% CI on 100 trials is ~+/-5%
