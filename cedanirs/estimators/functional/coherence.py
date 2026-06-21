"""Magnitude-squared coherence (frequency domain).

Coherence measures the linear association between two channels *as a function of
frequency*, making it the natural functional-connectivity metric when the
coupling of interest lives in a specific band -- for resting-state fNIRS, the
low-frequency oscillations around 0.01-0.1 Hz, away from cardiac and
respiratory contamination.

For a pair of signals the magnitude-squared coherence is

    MSC(f) = |Pxy(f)|^2 / (Pxx(f) * Pyy(f))

estimated by Welch's method (averaging over overlapping, windowed segments),
then averaged across the requested band to give one value in ``[0, 1]`` per
channel pair. Coherence is symmetric and undirected.

This is the first estimator that needs a sampling rate, so it sets
``requires_sfreq = True``; :meth:`ConnectivityEstimator.estimate` enforces that
the input :class:`~cedanirs.core.timeseries.NirsTimeSeries` carries ``sfreq``.
"""

from __future__ import annotations

import warnings

import numpy as np

from ...core.exceptions import DataError
from ...core.registry import register_estimator
from ...core.types import ConnectivityKind, Domain
from ..base import ConnectivityEstimator, EstimateOutput


@register_estimator(name="coherence")
class Coherence(ConnectivityEstimator):
    """Band-averaged magnitude-squared coherence between channel time series.

    Parameters
    ----------
    fmin, fmax:
        Frequency band (Hz) to average coherence over. Defaults to the
        canonical resting-state fNIRS band ``0.01-0.1 Hz``.
    nperseg:
        Welch segment length in samples. Defaults to a value that yields several
        overlapping segments (coherence needs averaging over >= 2 segments to be
        meaningful) while remaining long enough to resolve ``fmin``.
    noverlap:
        Samples of overlap between segments. Defaults to ``nperseg // 2``.
    window:
        Window function name passed to :func:`scipy.signal.get_window`.
    """

    name = "coherence"
    kind = ConnectivityKind.FUNCTIONAL
    directed = False
    domain = Domain.FREQUENCY
    requires_sfreq = True

    def __init__(
        self,
        *,
        fmin: float = 0.01,
        fmax: float = 0.1,
        nperseg: int | None = None,
        noverlap: int | None = None,
        window: str = "hann",
        **params,
    ):
        super().__init__(
            fmin=fmin, fmax=fmax, nperseg=nperseg, noverlap=noverlap, window=window,
            **params,
        )
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.window = window
        if self.fmin >= self.fmax:
            raise DataError(f"fmin ({fmin}) must be < fmax ({fmax}).")

    def _estimate(self, x: np.ndarray) -> EstimateOutput:
        from scipy.signal import get_window

        fs = self._sfreq
        if not fs:
            raise DataError("Coherence requires a sampling frequency (sfreq).")

        n_ch, n_obs = x.shape

        # Choose a segment length giving several averaging segments by default.
        nperseg = self.nperseg or max(64, n_obs // 8)
        nperseg = int(min(nperseg, n_obs))
        noverlap = self.noverlap if self.noverlap is not None else nperseg // 2
        noverlap = int(min(max(noverlap, 0), nperseg - 1))
        step = nperseg - noverlap

        # Segment start indices.
        starts = list(range(0, n_obs - nperseg + 1, step))
        n_seg = len(starts)
        if n_seg < 2:
            warnings.warn(
                f"Only {n_seg} Welch segment(s) for coherence; estimates are "
                f"degenerate (coherence -> 1). Provide more samples or a smaller "
                f"nperseg.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Frequency grid and band mask.
        freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
        band = (freqs >= self.fmin) & (freqs <= self.fmax)
        if not band.any():
            raise DataError(
                f"No FFT frequencies fall in [{self.fmin}, {self.fmax}] Hz with "
                f"nperseg={nperseg} at {fs:g} Hz (resolution {freqs[1]:.4g} Hz). "
                f"A segment must span >= 1/fmin = {1.0 / self.fmin:g}s; increase "
                f"the recording length or nperseg."
            )

        win = get_window(self.window, nperseg)

        # Build the windowed, detrended segment spectra: S[seg, ch, band_freq].
        seg_spectra = np.empty((n_seg, n_ch, int(band.sum())), dtype=complex)
        for s_i, start in enumerate(starts):
            seg = x[:, start : start + nperseg]
            seg = seg - seg.mean(axis=1, keepdims=True)  # detrend (constant)
            spec = np.fft.rfft(seg * win, axis=1)
            seg_spectra[s_i] = spec[:, band]

        # Auto-spectra Pxx[ch, f] and cross-spectra Pxy[i, j, f], averaged
        # over segments. Scaling constants cancel in the coherence ratio.
        pxx = np.mean(np.abs(seg_spectra) ** 2, axis=0)  # (n_ch, n_freq)
        pxy = np.einsum(
            "sif,sjf->ijf", seg_spectra, seg_spectra.conj()
        ) / n_seg  # (n_ch, n_ch, n_freq)

        denom = pxx[:, None, :] * pxx[None, :, :]  # (n_ch, n_ch, n_freq)
        with np.errstate(divide="ignore", invalid="ignore"):
            msc = (np.abs(pxy) ** 2) / denom  # per-frequency MSC in [0, 1]

        coh = np.nanmean(msc, axis=2)  # average over the band
        coh = np.clip(np.real(coh), 0.0, 1.0)
        np.fill_diagonal(coh, 1.0)

        # No analytic p-values for band-averaged MSC in this first version.
        return EstimateOutput(matrix=coh, pvalues=None)
