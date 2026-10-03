"""Statistical helpers.

- ``clopper_pearson``: 1sigma Clopper-Pearson binomial confidence interval, used for
  the error bars on reliability diagrams and per-decision pass rates.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as _beta_dist


def clopper_pearson(
    k: np.ndarray, n: np.ndarray, alpha: float = 0.3173
) -> tuple[np.ndarray, np.ndarray]:
    """Clopper-Pearson 1sigma binomial confidence interval for k successes in n trials.

    alpha = 1 - 0.6827 ~= 0.3173; returns the (lo, hi) bounds bracketing the
    central 68.27% of the distribution (one Gaussian sigma).
    """
    lo = np.where(k == 0, 0.0, _beta_dist.ppf(alpha / 2, k, n - k + 1))
    hi = np.where(k == n, 1.0, _beta_dist.ppf(1 - alpha / 2, k + 1, n - k))
    return np.nan_to_num(lo, nan=0.0), np.nan_to_num(hi, nan=1.0)
