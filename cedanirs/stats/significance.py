"""Significance testing and multiple-comparison correction.

A whole-probe connectivity matrix produces ``N(N-1)/2`` simultaneous tests, so
correcting for multiplicity is not optional. The functions here are deliberately
dependency-free (NumPy/SciPy only) and operate on plain arrays, so they can be
reused by any estimator or applied to externally-computed matrices.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..core.exceptions import DataError


def correlation_pvalues(
    r: np.ndarray, n_obs: int, *, dof: int | None = None
) -> np.ndarray:
    """Two-sided p-values for a matrix of (possibly partial) correlations.

    Under H0 (zero correlation), ``t = r * sqrt(dof / (1 - r^2))`` follows a
    Student-t distribution. For a simple Pearson correlation ``dof = n - 2``
    (the default). For a *partial* correlation conditioned on ``k`` other
    variables, the caller passes ``dof = n - 2 - k``. The computation is fully
    vectorised over the matrix; the diagonal is set to ``0`` (a channel is
    trivially, significantly correlated with itself).

    Parameters
    ----------
    r:
        Correlation matrix (or any array of correlations) in ``[-1, 1]``.
    n_obs:
        Number of time samples the correlations were estimated from.
    dof:
        Degrees of freedom. Defaults to ``n_obs - 2``.
    """
    if dof is None:
        dof = n_obs - 2
    if dof < 1:
        raise DataError(
            f"Need at least 1 degree of freedom for a correlation p-value, "
            f"got dof={dof} (n_obs={n_obs})."
        )
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rc = np.clip(r, -1.0 + 1e-15, 1.0 - 1e-15)
        t = rc * np.sqrt(dof / (1.0 - rc**2))
        p = 2.0 * stats.t.sf(np.abs(t), dof)
    p = np.asarray(p, dtype=float)
    # NaN correlations (e.g. constant channels) -> NaN p-value.
    p[~np.isfinite(r)] = np.nan
    if p.ndim == 2 and p.shape[0] == p.shape[1]:
        np.fill_diagonal(p, 0.0)
    return p


def fdr_correction(
    pvalues: np.ndarray,
    alpha: float = 0.05,
    *,
    method: str = "bh",
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg (or Benjamini-Yekutieli) FDR correction.

    Operates on a 1-D array of p-values (extract the unique edges first).

    Parameters
    ----------
    method:
        ``"bh"`` for the standard Benjamini-Hochberg procedure (assumes
        independence/positive dependence), or ``"by"`` for the more
        conservative Benjamini-Yekutieli (valid under arbitrary dependence).

    Returns
    -------
    reject:
        Boolean array, ``True`` where H0 is rejected at the given ``alpha``.
    qvalues:
        Adjusted p-values (monotone, clipped to ``[0, 1]``).
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    finite = np.isfinite(p)
    m = int(finite.sum())
    reject = np.zeros(p.shape, dtype=bool)
    q = np.full(p.shape, np.nan, dtype=float)
    if m == 0:
        return reject, q

    pv = p[finite]
    order = np.argsort(pv)
    ranked = pv[order]
    ranks = np.arange(1, m + 1)

    if method == "bh":
        factor = m / ranks
    elif method == "by":
        c_m = np.sum(1.0 / ranks)
        factor = (m * c_m) / ranks
    else:
        raise DataError(f"Unknown FDR method {method!r}; use 'bh' or 'by'.")

    q_sorted = np.minimum.accumulate((ranked * factor)[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q_finite = np.empty(m, dtype=float)
    q_finite[order] = q_sorted
    q[finite] = q_finite
    reject[finite] = q_finite <= alpha
    return reject, q


def bonferroni_correction(
    pvalues: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Bonferroni correction over the finite p-values.

    Returns ``(reject, adjusted)`` where ``adjusted = min(1, p * m)`` and ``m``
    is the number of finite p-values.
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    finite = np.isfinite(p)
    m = int(finite.sum())
    adjusted = np.full(p.shape, np.nan, dtype=float)
    reject = np.zeros(p.shape, dtype=bool)
    if m == 0:
        return reject, adjusted
    adjusted[finite] = np.minimum(p[finite] * m, 1.0)
    reject[finite] = adjusted[finite] <= alpha
    return reject, adjusted


def correct_pvalues(
    pvalues: np.ndarray,
    alpha: float = 0.05,
    method: str | None = "fdr_bh",
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to a multiple-comparison procedure by name.

    ``method`` accepts ``"fdr_bh"``, ``"fdr_by"``, ``"bonferroni"`` or ``None``
    (no correction; just threshold the raw p-values at ``alpha``). Returns
    ``(reject, adjusted)``.
    """
    if method is None or method == "none":
        p = np.asarray(pvalues, dtype=float)
        return (np.where(np.isfinite(p), p <= alpha, False), p)
    if method in ("fdr_bh", "bh"):
        return fdr_correction(pvalues, alpha, method="bh")
    if method in ("fdr_by", "by"):
        return fdr_correction(pvalues, alpha, method="by")
    if method == "bonferroni":
        return bonferroni_correction(pvalues, alpha)
    raise DataError(
        f"Unknown correction {method!r}; use 'fdr_bh', 'fdr_by', "
        f"'bonferroni' or None."
    )
