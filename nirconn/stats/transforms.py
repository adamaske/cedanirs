"""Variance-stabilising transforms for correlation-valued connectivity.

A Pearson correlation's sampling variance depends on the underlying true
correlation, which makes raw *r* values unsafe to average. The Fisher *z*
transform (``arctanh``) maps ``r in (-1, 1)`` onto an approximately
normally-distributed, constant-variance scale; the standard recipe is to
*z*-transform, average/aggregate in *z*-space, then map back with ``tanh``.
"""

from __future__ import annotations

import numpy as np

# Largest |r| we transform before clipping, so arctanh(±1) = ±inf is avoided.
_CLIP = 1.0 - 1e-12


def fisher_z(r: np.ndarray | float) -> np.ndarray:
    """Fisher *z* transform, ``z = arctanh(r)``.

    Values are clipped to just inside ``(-1, 1)`` so that perfect correlations
    (notably the matrix diagonal of ``1.0``) yield a large finite number rather
    than ``inf``. NaNs propagate unchanged.
    """
    r = np.asarray(r, dtype=float)
    clipped = np.clip(r, -_CLIP, _CLIP)
    return np.arctanh(clipped)


def inverse_fisher_z(z: np.ndarray | float) -> np.ndarray:
    """Inverse Fisher transform, ``r = tanh(z)``."""
    return np.tanh(np.asarray(z, dtype=float))


def average_correlations(
    matrices: np.ndarray,
    *,
    axis: int = 0,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Average correlation matrices the statistically correct way.

    Transforms to *z*-space, takes the (optionally weighted) mean along
    ``axis``, then transforms back to *r*. ``matrices`` is typically a stack of
    shape ``(n_subjects, n, n)``.
    """
    import warnings

    z = fisher_z(np.asarray(matrices, dtype=float))
    with warnings.catch_warnings():
        # All-NaN slices (an edge no subject has) legitimately average to NaN.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if weights is None:
            z_mean = np.nanmean(z, axis=axis)
        else:
            weights = np.asarray(weights, dtype=float)
            shape = [1] * z.ndim
            shape[axis] = weights.size
            w = weights.reshape(shape)
            z_mean = np.nansum(z * w, axis=axis) / np.nansum(
                w * np.isfinite(z), axis=axis
            )
    return inverse_fisher_z(z_mean)
