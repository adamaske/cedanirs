"""Surrogate-data significance testing for connectivity estimators.

Some connectivity measures -- phase-locking value, coherence, wavelet coherence
-- have no tractable analytic null distribution. The standard alternative is a
**surrogate-data test**: generate many synthetic datasets that share the
single-channel properties of the data but in which the channels are mutually
independent, recompute the connectivity statistic on each, and read each edge's
p-value off the resulting empirical null.

The surrogate of choice for interdependence is the **Fourier phase-randomised
surrogate** (Theiler et al. 1992): take each channel's Fourier transform,
replace its phases with independent uniform random phases, and invert. This
preserves every channel's power spectrum (hence its autocorrelation) exactly
while destroying any *cross-channel* phase relationship -- precisely the null
"these two channels are uncoupled, given their individual spectra". Randomising
each channel independently is what breaks the coupling.

Connectivity magnitudes (PLV, coherence, ...) are non-negative and larger when
coupling is stronger, so the test is one-sided (``tail="greater"``): the
p-value is the fraction of surrogates whose statistic is at least as large as
the observed one.

Caveat: because phase randomisation preserves each channel's power spectrum
exactly, a (near-)deterministic periodic component survives in the surrogates.
Two channels dominated by the *same* pure tone therefore stay perfectly phase-
locked under the surrogate too, and the test loses power. This is not a concern
for broadband haemodynamic signals, but it makes the test inappropriate for
synthetic pure-sinusoid data.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fourier phase-randomised surrogate of a ``(channel, time)`` array.

    Each channel is transformed independently: its amplitude spectrum is kept
    and its phases replaced by i.i.d. uniform random phases (with the DC and, for
    even length, Nyquist components left real so the inverse transform is real).
    The result has the same shape, the same per-channel power spectrum, and no
    systematic cross-channel phase relationship.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    spec = np.fft.rfft(x, axis=-1)
    amp = np.abs(spec)

    # Independent random phases per channel and frequency bin.
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spec.shape)
    phases[..., 0] = 0.0  # DC stays real (preserves the mean)
    if n % 2 == 0:
        phases[..., -1] = 0.0  # Nyquist stays real for even-length signals

    surrogate = np.fft.irfft(amp * np.exp(1j * phases), n=n, axis=-1)
    return surrogate


def surrogate_pvalues(
    statistic: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    *,
    n_surrogates: int,
    rng: np.random.Generator | int | None = None,
    tail: str = "greater",
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical p-values for a connectivity statistic via phase surrogates.

    Parameters
    ----------
    statistic:
        Callable mapping a ``(channel, time)`` array to an ``N x N`` matrix of
        connectivity values (the same function used for the observed estimate).
    x:
        The observed ``(channel, time)`` data.
    n_surrogates:
        Number of phase-randomised surrogates. The smallest attainable p-value
        is ``1 / (n_surrogates + 1)``, so e.g. 200 surrogates resolve p down to
        ~0.005; use more for stringent thresholds.
    rng:
        A NumPy generator, an integer seed, or ``None``.
    tail:
        ``"greater"`` (default; appropriate for non-negative connectivity
        magnitudes), ``"less"``, or ``"two-sided"``.

    Returns
    -------
    (observed, pvalues):
        The observed statistic matrix and the aligned ``N x N`` p-value matrix.
        Edges that are ``NaN`` in the observed matrix get ``NaN`` p-values; the
        diagonal p-value is set to ``0`` to match the analytic convention.
    """
    if n_surrogates < 1:
        raise ValueError("n_surrogates must be >= 1.")
    if tail not in {"greater", "less", "two-sided"}:
        raise ValueError("tail must be 'greater', 'less' or 'two-sided'.")
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    observed = np.asarray(statistic(x), dtype=float)
    eps = 1e-12
    obs_finite = np.isfinite(observed)

    # Count, per edge, surrogates at least as extreme as the observed value.
    count = np.zeros_like(observed)
    n_valid = np.zeros_like(observed)
    for _ in range(n_surrogates):
        s = np.asarray(statistic(phase_randomize(x, rng)), dtype=float)
        m = obs_finite & np.isfinite(s)
        if tail == "greater":
            extreme = s >= observed - eps
        elif tail == "less":
            extreme = s <= observed + eps
        else:  # two-sided: compare absolute deviation from the surrogate mean
            extreme = np.abs(s) >= np.abs(observed) - eps
        count[m] += extreme[m]
        n_valid[m] += 1.0

    # Add-one estimator (counts the observed value itself), the unbiased,
    # never-zero convention of Davison & Hinkley / North et al.
    with np.errstate(invalid="ignore"):
        p = (1.0 + count) / (1.0 + n_valid)
    p[n_valid == 0] = np.nan
    p[~obs_finite] = np.nan
    if p.ndim == 2 and p.shape[0] == p.shape[1]:
        np.fill_diagonal(p, 0.0)
    return observed, p
