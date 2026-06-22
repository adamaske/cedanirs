"""Phase synchronization -- phase-locking value (PLV) and phase-lag index (PLI).

Correlation and coherence ask whether two channels' *amplitudes* co-vary. Phase
synchronization instead asks whether their *phases* stay locked, regardless of
amplitude -- a hallmark of genuine oscillatory coupling. The instantaneous phase
of each channel is taken from the analytic signal (Hilbert transform), and the
consistency of the phase difference across time is summarised in ``[0, 1]``.

Two related measures are provided via ``mode``:

* ``"plv"`` -- the **phase-locking value** (Lachaux 1999):
  ``PLV_ij = | mean_t exp(i (phi_i(t) - phi_j(t))) |``. ``1`` means the phase
  difference is perfectly constant; ``0`` means it is uniformly distributed.
* ``"pli"`` -- the **phase-lag index** (Stam 2007):
  ``PLI_ij = | mean_t sign(sin(phi_i(t) - phi_j(t))) |``. PLI ignores phase
  differences clustered around ``0``/``pi``, so it is insensitive to the
  zero-lag, instantaneous coupling that volume conduction (and, in fNIRS, shared
  systemic physiology) manufactures -- at the cost of discarding genuine
  zero-lag interactions.

The phase is only well defined for a reasonably narrow-band signal, so PLV/PLI
are normally computed on band-passed data. Pass ``fmin``/``fmax`` (with a known
``sfreq``) to band-pass inside the estimator; otherwise the input is assumed to
be already filtered to the band of interest.

Both measures are symmetric and undirected. There is no simple analytic null --
significance is properly assessed with phase-scrambled surrogates -- so no
p-values are returned.
"""

from __future__ import annotations

import numpy as np

from ...core.exceptions import DataError
from ...core.registry import register_estimator
from ...core.types import ConnectivityKind, Domain
from ..base import ConnectivityEstimator, EstimateOutput


@register_estimator(name="plv")
class PhaseLockingValue(ConnectivityEstimator):
    """Phase-locking value / phase-lag index between channel time series.

    Parameters
    ----------
    mode:
        ``"plv"`` (default) for the phase-locking value, or ``"pli"`` for the
        phase-lag index (robust to zero-lag/instantaneous coupling).
    fmin, fmax:
        Optional band-pass band (Hz). If given, the data is band-passed before
        the Hilbert transform, which requires a known sampling frequency
        (``sfreq``). If omitted, the input is assumed already band-limited.
    surrogates:
        If set to a positive integer, compute one-sided p-values from that many
        Fourier phase-randomised surrogates (the empirical null for phase
        synchronisation; there is no cheap analytic one). Default ``None`` --
        no p-values.
    surrogate_seed:
        Optional seed for reproducible surrogate p-values.
    """

    name = "plv"
    kind = ConnectivityKind.FUNCTIONAL
    directed = False
    domain = Domain.TIME

    def __init__(
        self,
        *,
        mode: str = "plv",
        fmin: float | None = None,
        fmax: float | None = None,
        surrogates: int | None = None,
        surrogate_seed: int | None = None,
        **params,
    ):
        super().__init__(
            mode=mode, fmin=fmin, fmax=fmax, surrogates=surrogates,
            surrogate_seed=surrogate_seed, **params,
        )
        self.mode = str(mode).lower()
        if self.mode not in {"plv", "pli"}:
            raise DataError(f"mode must be 'plv' or 'pli', got {mode!r}.")
        self.fmin = None if fmin is None else float(fmin)
        self.fmax = None if fmax is None else float(fmax)
        if (self.fmin is None) != (self.fmax is None):
            raise DataError("Provide both fmin and fmax, or neither.")
        if self.fmin is not None and self.fmin >= self.fmax:
            raise DataError(f"fmin ({fmin}) must be < fmax ({fmax}).")
        self.surrogates = surrogates
        self.surrogate_seed = surrogate_seed

    def _estimate(self, x: np.ndarray) -> EstimateOutput:
        if self.fmin is not None:
            fs = self._sfreq
            if not fs:
                raise DataError(
                    "Band-pass for PLV requires a sampling frequency (sfreq); "
                    "construct the NirsTimeSeries with sfreq=, or drop "
                    "fmin/fmax and pass pre-filtered data."
                )
            # Band-pass once; surrogates are then drawn from the band-limited
            # signal so the null stays in-band.
            x = _bandpass(x, fs, self.fmin, self.fmax)

        matrix = self._statistic(x)
        pvalues = self._surrogate_pvalues(x, self._statistic)
        return EstimateOutput(matrix=matrix, pvalues=pvalues)

    def _statistic(self, x: np.ndarray) -> np.ndarray:
        """The PLV/PLI matrix for a (band-limited) ``(channel, time)`` array."""
        from scipy.signal import hilbert

        n_ch, n_obs = x.shape

        # Instantaneous phase from the analytic signal (per channel, along time).
        x = x - x.mean(axis=1, keepdims=True)
        phase = np.angle(hilbert(x, axis=1))  # (n_ch, n_obs)

        if self.mode == "plv":
            # PLV_ij = |mean_t exp(i*(phi_i - phi_j))|. The matrix of mean phase
            # differences is U @ U.conj().T / T with U = exp(i*phi); no O(N^2 T)
            # temporary needed.
            u = np.exp(1j * phase)
            m = (u @ u.conj().T) / n_obs
            out = np.abs(m)
        else:  # pli
            out = np.empty((n_ch, n_ch), dtype=float)
            # sign(sin(dphi)) averaged over time; loop rows to avoid an
            # (N, N, T) temporary on full montages.
            for i in range(n_ch):
                dphi = phase[i][None, :] - phase  # (n_ch, n_obs)
                out[i] = np.abs(np.mean(np.sign(np.sin(dphi)), axis=1))

        np.clip(out, 0.0, 1.0, out=out)
        np.fill_diagonal(out, 1.0)
        return out


def _bandpass(x: np.ndarray, fs: float, fmin: float, fmax: float) -> np.ndarray:
    """Zero-phase Butterworth band-pass of a ``(channel, time)`` array."""
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * fs
    lo, hi = fmin / nyq, fmax / nyq
    if not 0 < lo < hi < 1:
        raise DataError(
            f"Band [{fmin}, {fmax}] Hz is invalid for sfreq={fs:g} Hz "
            f"(Nyquist {nyq:g} Hz)."
        )
    b, a = butter(4, [lo, hi], btype="band")
    return filtfilt(b, a, x, axis=1)
