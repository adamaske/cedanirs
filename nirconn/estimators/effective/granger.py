"""Granger causality -- the first *effective* (directed) connectivity method.

Functional methods (correlation, coherence, phase locking) are symmetric: they
say two channels are related, not who drives whom. Granger causality asks a
directed question -- *does the past of channel j help predict channel i beyond
i's own past?* If adding j's history significantly reduces the prediction error
of i, then j Granger-causes i.

This is the pairwise, bivariate formulation (the standard choice for the wide,
sample-limited montages of fNIRS, where a full multivariate VAR over dozens of
channels is ill-posed). For each ordered pair ``(i, j)`` two autoregressive
models of i are fit by ordinary least squares:

* **restricted**   ``i_t = c + sum_k a_k i_{t-k} + e``
* **unrestricted** ``i_t = c + sum_k a_k i_{t-k} + sum_k b_k j_{t-k} + e``

over lags ``k = 1..order``. The reduction in residual sum of squares is tested
with an F-test; the matrix entry ``matrix[src, tgt]`` holds the **Granger
causality index** ``ln(RSS_restricted / RSS_unrestricted) >= 0`` for the
influence of ``src`` on ``tgt``, and ``pvalues[src, tgt]`` the F-test p-value.
This row=source, column=target convention matches the rest of nirconn (so
``edges()`` reads ``src -> tgt``); the matrix is directed and asymmetric.

The OLS / F-test is pure NumPy/SciPy (no extra dependency). The diagonal is
undefined (a channel cannot Granger-cause itself) and set to zero.
"""

from __future__ import annotations

import numpy as np

from ...core.exceptions import DataError
from ...core.registry import register_estimator
from ...core.types import ConnectivityKind, Domain
from ..base import ConnectivityEstimator, EstimateOutput


@register_estimator(name="granger")
class GrangerCausality(ConnectivityEstimator):
    """Pairwise bivariate Granger causality (directed).

    Parameters
    ----------
    order:
        Autoregressive model order (number of lags). Default ``1``.
    compute_pvalues:
        Whether to return the F-test p-values (default ``True``).
    """

    name = "granger"
    kind = ConnectivityKind.EFFECTIVE
    directed = True
    domain = Domain.TIME

    def __init__(self, *, order: int = 1, compute_pvalues: bool = True, **params):
        super().__init__(order=order, compute_pvalues=compute_pvalues, **params)
        self.order = int(order)
        self.compute_pvalues = compute_pvalues
        if self.order < 1:
            raise DataError("order must be >= 1.")

    def _estimate(self, x: np.ndarray) -> EstimateOutput:
        from scipy import stats

        n_ch, n_obs = x.shape
        p = self.order
        n_eff = n_obs - p  # usable rows after lagging

        # df2 of the F-test for the unrestricted model: n_eff observations minus
        # the unrestricted parameter count (intercept + 2p lagged regressors).
        df1 = p
        df2 = n_eff - (2 * p + 1)
        if df2 < 1:
            raise DataError(
                f"Too few samples for Granger causality at order {p}: need "
                f"n_times > 3*order + 1 = {3 * p + 1}, got {n_obs}."
            )

        # Build the lagged design once per channel. lags[c] is (n_eff, p) with
        # columns [c_{t-1}, ..., c_{t-p}]; target[c] is (n_eff,) = c_t.
        target = x[:, p:]  # (n_ch, n_eff)
        lags = np.empty((n_ch, n_eff, p))
        for k in range(1, p + 1):
            lags[:, :, k - 1] = x[:, p - k : n_obs - k]

        ones = np.ones((n_eff, 1))
        matrix = np.zeros((n_ch, n_ch))
        pvalues = np.ones((n_ch, n_ch)) if self.compute_pvalues else None

        for tgt in range(n_ch):
            y = target[tgt]
            restricted = np.hstack([ones, lags[tgt]])  # (n_eff, 1 + p)
            rss_r = _ols_rss(restricted, y)
            for src in range(n_ch):
                if src == tgt:
                    continue
                # Does src's past improve prediction of tgt? Store the influence
                # of src on tgt at matrix[src, tgt] (row=source, col=target).
                unrestricted = np.hstack([restricted, lags[src]])
                rss_u = _ols_rss(unrestricted, y)
                # Granger causality index (>= 0); guard degenerate fits.
                if rss_u <= 0 or not np.isfinite(rss_r) or not np.isfinite(rss_u):
                    continue
                matrix[src, tgt] = max(0.0, np.log(rss_r / rss_u))
                if pvalues is not None:
                    f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
                    f_stat = max(f_stat, 0.0)
                    pvalues[src, tgt] = float(stats.f.sf(f_stat, df1, df2))

        np.fill_diagonal(matrix, 0.0)
        if pvalues is not None:
            np.fill_diagonal(pvalues, np.nan)
        return EstimateOutput(matrix=matrix, pvalues=pvalues)


def _ols_rss(design: np.ndarray, y: np.ndarray) -> float:
    """Residual sum of squares of an OLS fit of ``y`` on ``design`` columns."""
    # lstsq is the numerically stable route (handles rank deficiency); when it
    # reports a full-rank residual we use it directly, else compute explicitly.
    coef, residuals, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if residuals.size and rank == design.shape[1]:
        return float(residuals[0])
    resid = y - design @ coef
    return float(resid @ resid)
