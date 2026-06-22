"""Wavelet transform coherence (time-frequency).

Coherence (see :mod:`~nirconn.estimators.functional.coherence`) assumes the
coupling between two channels is stationary across the whole recording. Resting
fNIRS is not so tidy: low-frequency haemodynamic coupling waxes and wanes.
Wavelet transform coherence (WTC) localises coherence jointly in **time and
frequency**, then -- so it still yields a single connectivity matrix -- averages
the time-frequency coherence over the requested band and over time.

The implementation follows the standard Morlet formulation (Torrence & Compo
1998; Grinsted, Moore & Jevrejeva 2004):

1. Continuous wavelet transform :math:`W_x(s, t)` of each channel with a Morlet
   wavelet, computed in the Fourier domain.
2. Cross-wavelet spectrum :math:`W_{xy} = W_x W_y^*` and auto-spectra.
3. Scale-normalised smoothing :math:`S(\\cdot)` in both time (a per-scale
   Gaussian) and scale (a boxcar), giving

   .. math::
       R^2(s, t) = \\frac{|S(s^{-1} W_{xy})|^2}
                        {S(s^{-1}|W_x|^2)\\, S(s^{-1}|W_y|^2)} \\in [0, 1].

4. Reduction to one value per channel pair: average :math:`R^2` over the scales
   whose Fourier period lands in ``[fmin, fmax]`` and over time, excluding the
   cone of influence where edge effects dominate.

WTC is symmetric and undirected, and -- as for coherence -- there is no cheap
analytic null (significance is assessed by Monte-Carlo against red-noise
surrogates), so no p-values are returned. It is the heaviest estimator shipped:
cost grows with the number of scales and the square of the channel count.
"""

from __future__ import annotations

import warnings

import numpy as np

from ...core.exceptions import DataError
from ...core.registry import register_estimator
from ...core.types import ConnectivityKind, Domain
from ..base import ConnectivityEstimator, EstimateOutput

# Morlet central angular frequency. w0 = 6 makes the wavelet admissible and is
# the near-universal default; with it the Fourier period is ~1.03 * scale.
_W0 = 6.0
# Fourier-period / scale ratio for the Morlet wavelet (Torrence & Compo Table 1).
_FOURIER_FACTOR = (4.0 * np.pi) / (_W0 + np.sqrt(2.0 + _W0**2))
# Scale-smoothing decorrelation length for the Morlet wavelet (in units of dj).
_DJ0 = 0.6


@register_estimator(name="wavelet_coherence")
class WaveletCoherence(ConnectivityEstimator):
    """Band- and time-averaged Morlet wavelet coherence between channels.

    Parameters
    ----------
    fmin, fmax:
        Frequency band (Hz) to average coherence over. Defaults to the
        resting-state fNIRS band ``0.01-0.1 Hz``.
    dj:
        Scale resolution (smaller = more scales, finer frequency sampling,
        slower). Default ``0.25``.
    exclude_coi:
        Whether to mask the cone of influence (edge-affected coefficients)
        before time-averaging. Default ``True``.
    surrogates:
        If set to a positive integer, compute one-sided p-values from that many
        Fourier phase-randomised surrogates. Wavelet coherence is the heaviest
        estimator, so each surrogate re-runs the full CWT pipeline -- keep the
        count modest (e.g. 100-200) and expect it to dominate the runtime.
        Default ``None`` -- no p-values.
    surrogate_seed:
        Optional seed for reproducible surrogate p-values.
    """

    name = "wavelet_coherence"
    kind = ConnectivityKind.FUNCTIONAL
    directed = False
    domain = Domain.TIME_FREQUENCY
    requires_sfreq = True

    def __init__(
        self,
        *,
        fmin: float = 0.01,
        fmax: float = 0.1,
        dj: float = 0.25,
        exclude_coi: bool = True,
        surrogates: int | None = None,
        surrogate_seed: int | None = None,
        **params,
    ):
        super().__init__(
            fmin=fmin, fmax=fmax, dj=dj, exclude_coi=exclude_coi,
            surrogates=surrogates, surrogate_seed=surrogate_seed, **params,
        )
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.dj = float(dj)
        self.exclude_coi = bool(exclude_coi)
        self.surrogates = surrogates
        self.surrogate_seed = surrogate_seed
        if self.fmin >= self.fmax:
            raise DataError(f"fmin ({fmin}) must be < fmax ({fmax}).")
        if self.dj <= 0:
            raise DataError("dj must be > 0.")

    def _estimate(self, x: np.ndarray) -> EstimateOutput:
        matrix = self._statistic(x)
        pvalues = self._surrogate_pvalues(x, self._statistic)
        return EstimateOutput(matrix=matrix, pvalues=pvalues)

    def _statistic(self, x: np.ndarray) -> np.ndarray:
        fs = self._sfreq
        if not fs:
            raise DataError("Wavelet coherence requires a sampling frequency.")
        dt = 1.0 / fs
        n_ch, n_obs = x.shape

        scales, freqs = _scales_for_band(n_obs, dt, self.dj, self.fmin, self.fmax)
        if scales.size == 0:
            raise DataError(
                f"No wavelet scales fall in [{self.fmin}, {self.fmax}] Hz for a "
                f"{n_obs}-sample recording at {fs:g} Hz. The band must sit "
                f"between ~{1.0 / (n_obs * dt):.4g} Hz (record length) and "
                f"Nyquist; lengthen the recording or widen the band."
            )

        x = x - x.mean(axis=1, keepdims=True)

        # Continuous wavelet transform of every channel: W[ch] is (n_scale, n_obs).
        wavelets = [_cwt_morlet(x[c], dt, scales) for c in range(n_ch)]

        # Per-channel smoothed, scale-normalised auto-spectra S(s^-1 |W|^2).
        inv_s = (1.0 / scales)[:, None]
        smooth_auto = [_smooth(inv_s * np.abs(w) ** 2, dt, self.dj, scales)
                       for w in wavelets]

        # Cone-of-influence mask per (scale, time): True where reliable.
        valid = _coi_mask(scales, n_obs, dt) if self.exclude_coi else None

        out = np.ones((n_ch, n_ch), dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-masked nanmean
            for i in range(n_ch):
                for j in range(i + 1, n_ch):
                    cross = wavelets[i] * np.conj(wavelets[j])
                    s_cross = _smooth(inv_s * cross, dt, self.dj, scales)
                    rsq = (np.abs(s_cross) ** 2) / (smooth_auto[i] * smooth_auto[j])
                    rsq = np.clip(np.real(rsq), 0.0, 1.0)
                    if valid is not None:
                        rsq = np.where(valid, rsq, np.nan)
                    val = float(np.nanmean(rsq))
                    out[i, j] = out[j, i] = 0.0 if np.isnan(val) else val

        np.fill_diagonal(out, 1.0)
        return out


# --------------------------------------------------------------------------- #
# Morlet wavelet machinery (Torrence & Compo 1998; Grinsted 2004)
# --------------------------------------------------------------------------- #

def _scales_for_band(n_obs, dt, dj, fmin, fmax):
    """Log2-spaced Morlet scales, restricted to the [fmin, fmax] band."""
    s0 = 2.0 * dt  # smallest resolvable scale (~Nyquist)
    j_max = int(np.floor(np.log2(n_obs * dt / s0) / dj))
    if j_max < 0:
        return np.empty(0), np.empty(0)
    scales = s0 * 2.0 ** (np.arange(j_max + 1) * dj)
    freqs = 1.0 / (_FOURIER_FACTOR * scales)
    keep = (freqs >= fmin) & (freqs <= fmax)
    return scales[keep], freqs[keep]


def _cwt_morlet(sig, dt, scales):
    """Morlet continuous wavelet transform of a 1-D signal via the FFT.

    Returns a ``(n_scale, n_obs)`` complex array of wavelet coefficients.
    """
    n = sig.size
    sig_hat = np.fft.fft(sig)
    # Angular frequencies of the FFT bins.
    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)

    w = np.empty((scales.size, n), dtype=complex)
    norm_const = np.pi ** -0.25
    for k, s in enumerate(scales):
        # Morlet daughter wavelet in the Fourier domain (Torrence & Compo eq. 6),
        # nonzero only for positive frequencies; energy-normalised across scales.
        expnt = -0.5 * (s * omega - _W0) ** 2
        daughter = np.where(omega > 0, norm_const * np.exp(expnt), 0.0)
        daughter *= np.sqrt(2.0 * np.pi * s / dt)
        w[k] = np.fft.ifft(sig_hat * daughter)
    return w


def _smooth(field, dt, dj, scales):
    """Smooth a (scale, time) field in time (Gaussian) then scale (boxcar).

    This is the smoothing operator used in wavelet coherence (Torrence &
    Webster 1999; Grinsted 2004). ``field`` may be complex (cross-spectrum) or
    real (auto-spectrum); smoothing is linear so it applies element-wise.
    """
    from scipy.ndimage import uniform_filter1d

    n = field.shape[1]
    # --- time smoothing: per-scale normalised Gaussian, applied in Fourier ---
    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)
    field_hat = np.fft.fft(field, axis=1)
    # Filter exp(-s^2 omega^2 / 2) is the FT of a unit-area Gaussian, so the DC
    # (mean) is preserved and no separate normalisation is required.
    gauss = np.exp(-0.5 * (scales[:, None] ** 2) * (omega[None, :] ** 2))
    smoothed = np.fft.ifft(field_hat * gauss, axis=1)
    if not np.iscomplexobj(field):
        smoothed = smoothed.real

    # --- scale smoothing: boxcar of width dj0/dj scales ---
    win = max(1, int(round(_DJ0 / dj)))
    if win > 1:
        smoothed = uniform_filter1d(smoothed, size=win, axis=0, mode="nearest")
    return smoothed


def _coi_mask(scales, n_obs, dt):
    """Boolean ``(n_scale, n_obs)`` mask: True where outside the cone of influence.

    The Morlet e-folding time is ``sqrt(2) * s``; coefficients within that many
    seconds of either edge are contaminated by zero-padding and excluded.
    """
    t = np.arange(n_obs) * dt
    edge = np.minimum(t, (n_obs - 1) * dt - t)  # distance to nearest edge (s)
    efold = np.sqrt(2.0) * scales[:, None]
    return edge[None, :] >= efold
