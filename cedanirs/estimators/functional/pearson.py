"""Pearson product-moment correlation.

The workhorse of fNIRS functional connectivity: the linear correlation between
every pair of channel time courses, yielding a symmetric matrix in ``[-1, 1]``.

The estimate is computed in one vectorised ``O(N^2 T)`` pass (via the
covariance), and analytic two-sided p-values are derived from the Student-t
distribution -- no slow pairwise Python loop. Channels with zero variance
(e.g. a flat-lined or rejected channel) yield ``NaN`` correlations rather than
crashing.
"""

from __future__ import annotations

import numpy as np

from ...core.registry import register_estimator
from ...core.types import ConnectivityKind, Domain
from ...stats.significance import correlation_pvalues
from ..base import ConnectivityEstimator, EstimateOutput


@register_estimator(name="pearson")
class PearsonCorrelation(ConnectivityEstimator):
    """Pearson correlation between channel time series.

    Parameters
    ----------
    compute_pvalues:
        Whether to return analytic two-sided p-values (default ``True``).
    """

    name = "pearson"
    kind = ConnectivityKind.FUNCTIONAL
    directed = False
    domain = Domain.TIME

    def __init__(self, *, compute_pvalues: bool = True, **params):
        super().__init__(compute_pvalues=compute_pvalues, **params)
        self.compute_pvalues = compute_pvalues

    def _estimate(self, x: np.ndarray) -> EstimateOutput:
        # x: (n_channels, n_times); variables are rows.
        n_times = x.shape[1]

        # np.corrcoef emits a RuntimeWarning for zero-variance rows and returns
        # NaN there, which is exactly the behaviour we want -- just silence it.
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.corrcoef(x)

        r = np.atleast_2d(r)
        # Guard the diagonal: a channel correlates perfectly with itself, even
        # if floating point or a degenerate single-channel case nudged it.
        np.fill_diagonal(r, 1.0)
        # Keep values numerically inside [-1, 1].
        np.clip(r, -1.0, 1.0, out=r)

        pvalues = None
        if self.compute_pvalues:
            pvalues = correlation_pvalues(r, n_obs=n_times)

        return EstimateOutput(matrix=r, pvalues=pvalues)
