"""Spearman rank correlation.

Spearman's rho is Pearson's correlation computed on the *ranks* of the data. By
operating on ranks it captures any monotonic (not just linear) relationship and
is robust to outliers and non-Gaussian, skewed haemodynamic responses -- a
common, sensible alternative to Pearson for fNIRS functional connectivity.

Like Pearson it is symmetric and lives in ``[-1, 1]``. The same Student-t
approximation gives analytic two-sided p-values (valid for moderate-to-large
sample sizes, which fNIRS recordings comfortably satisfy).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from ...core.registry import register_estimator
from ...core.types import ConnectivityKind, Domain
from ...stats.significance import correlation_pvalues
from ..base import ConnectivityEstimator, EstimateOutput


@register_estimator(name="spearman")
class SpearmanCorrelation(ConnectivityEstimator):
    """Spearman rank correlation between channel time series.

    Parameters
    ----------
    compute_pvalues:
        Whether to return analytic two-sided p-values (default ``True``).
    """

    name = "spearman"
    kind = ConnectivityKind.FUNCTIONAL
    directed = False
    domain = Domain.TIME

    def __init__(self, *, compute_pvalues: bool = True, **params):
        super().__init__(compute_pvalues=compute_pvalues, **params)
        self.compute_pvalues = compute_pvalues

    def _estimate(self, x: np.ndarray) -> EstimateOutput:
        n_times = x.shape[1]

        # Rank each channel along time, then correlate the ranks.
        ranks = np.apply_along_axis(rankdata, 1, x)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.corrcoef(ranks)

        r = np.atleast_2d(r)
        np.fill_diagonal(r, 1.0)
        np.clip(r, -1.0, 1.0, out=r)

        pvalues = None
        if self.compute_pvalues:
            pvalues = correlation_pvalues(r, n_obs=n_times)

        return EstimateOutput(matrix=r, pvalues=pvalues)
