"""Partial correlation.

Pearson correlation between two channels is easily inflated by a shared third
influence -- in fNIRS, global systemic physiology (heart rate, blood pressure,
scalp blood flow) drives many channels together and manufactures spurious
"connections". Partial correlation measures the linear association between two
channels *after regressing out every other channel*, isolating their direct
relationship.

It is computed from the precision matrix (the inverse of the correlation
matrix): ``pcorr_ij = -P_ij / sqrt(P_ii * P_jj)``. The result is symmetric and
in ``[-1, 1]``. Because each pair is conditioned on the remaining ``N - 2``
channels, the p-value uses ``dof = n_obs - N`` and therefore requires more time
samples than channels.
"""

from __future__ import annotations

import warnings

import numpy as np

from ...core.exceptions import DataError
from ...core.registry import register_estimator
from ...core.types import ConnectivityKind, Domain
from ...stats.significance import correlation_pvalues
from ..base import ConnectivityEstimator, EstimateOutput


@register_estimator(name="partial")
class PartialCorrelation(ConnectivityEstimator):
    """Partial correlation, controlling for all other channels.

    Parameters
    ----------
    compute_pvalues:
        Whether to return analytic two-sided p-values (default ``True``).
    shrinkage:
        Ridge/identity shrinkage in ``[0, 1]`` blended into the correlation
        matrix before inversion (``R' = (1 - s) R + s I``). A small value
        (e.g. ``0.01``) stabilises the inverse when channels are highly
        collinear or samples are limited. Default ``0.0`` (no shrinkage).
    """

    name = "partial"
    kind = ConnectivityKind.FUNCTIONAL
    directed = False
    domain = Domain.TIME

    def __init__(
        self, *, compute_pvalues: bool = True, shrinkage: float = 0.0, **params
    ):
        super().__init__(compute_pvalues=compute_pvalues, shrinkage=shrinkage, **params)
        self.compute_pvalues = compute_pvalues
        self.shrinkage = float(shrinkage)
        if not 0.0 <= self.shrinkage <= 1.0:
            raise DataError("shrinkage must be in [0, 1].")

    def _estimate(self, x: np.ndarray) -> EstimateOutput:
        n_ch, n_obs = x.shape

        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(x)
        corr = np.atleast_2d(corr)
        np.fill_diagonal(corr, 1.0)

        if not np.all(np.isfinite(corr)):
            raise DataError(
                "Partial correlation needs a finite correlation matrix; a "
                "channel is likely constant (zero variance). Remove flat "
                "channels first."
            )

        if self.shrinkage > 0.0:
            corr = (1.0 - self.shrinkage) * corr + self.shrinkage * np.eye(n_ch)

        # Precision matrix. Fall back to the pseudo-inverse if the correlation
        # matrix is singular (e.g. fewer time samples than channels, or perfect
        # collinearity), warning that the estimate is then unreliable.
        try:
            precision = np.linalg.inv(corr)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Correlation matrix is singular; using the pseudo-inverse. "
                "Partial correlations may be unreliable -- consider shrinkage= "
                "or more time samples.",
                RuntimeWarning,
                stacklevel=2,
            )
            precision = np.linalg.pinv(corr)

        d = np.sqrt(np.diag(precision))
        denom = np.outer(d, d)
        with np.errstate(divide="ignore", invalid="ignore"):
            pcorr = -precision / denom
        np.fill_diagonal(pcorr, 1.0)
        np.clip(pcorr, -1.0, 1.0, out=pcorr)

        pvalues = None
        if self.compute_pvalues:
            # Conditioning on the other N-2 channels costs N-2 degrees of
            # freedom: dof = (n_obs - 2) - (n_ch - 2) = n_obs - n_ch.
            dof = n_obs - n_ch
            if dof >= 1:
                pvalues = correlation_pvalues(pcorr, n_obs=n_obs, dof=dof)
            else:
                warnings.warn(
                    f"Too few samples for partial-correlation p-values "
                    f"(n_times={n_obs} <= n_channels={n_ch}); p-values omitted.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return EstimateOutput(matrix=pcorr, pvalues=pvalues)
