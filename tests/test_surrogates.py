"""Tests for the phase-randomised surrogate-data machinery."""

import numpy as np
import pytest
from scipy.signal import butter, filtfilt

import nirconn as cn
from nirconn.stats.surrogates import phase_randomize, surrogate_pvalues


@pytest.fixture
def broadband_triplet():
    """A, B coupled via a shared band-limited broadband driver; C independent.

    Broadband (not pure-tone) so phase randomisation genuinely decouples the
    channels -- the regime where the surrogate test has power.
    """
    fs = 5.0
    n = 6000
    rng = np.random.default_rng(0)
    b, a = butter(4, [0.01 / (fs / 2), 0.1 / (fs / 2)], btype="band")
    bp = lambda z: filtfilt(b, a, z)
    shared = bp(rng.standard_normal(n))
    A = shared + bp(rng.standard_normal(n))
    B = shared + bp(rng.standard_normal(n))
    C = bp(rng.standard_normal(n))
    return np.vstack([A, B, C]), ["A", "B", "C"], fs


# --------------------------------------------------------------------------- #
# phase_randomize
# --------------------------------------------------------------------------- #

def test_phase_randomize_preserves_power_spectrum():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 1024))
    s = phase_randomize(x, rng)
    assert s.shape == x.shape
    assert np.isrealobj(s)
    # Amplitude spectrum is preserved channel-by-channel; phases are not.
    np.testing.assert_allclose(
        np.abs(np.fft.rfft(s, axis=1)), np.abs(np.fft.rfft(x, axis=1)), atol=1e-8
    )


def test_phase_randomize_breaks_cross_correlation():
    # Two identical channels: r = 1. Independent phase randomisation decouples.
    rng = np.random.default_rng(2)
    base = rng.standard_normal(4000)
    x = np.vstack([base, base])
    s = phase_randomize(x, rng)
    assert abs(np.corrcoef(s)[0, 1]) < 0.2


def test_phase_randomize_preserves_mean():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((3, 512)) + 5.0
    s = phase_randomize(x, rng)
    np.testing.assert_allclose(s.mean(axis=1), x.mean(axis=1), atol=1e-8)


# --------------------------------------------------------------------------- #
# surrogate_pvalues
# --------------------------------------------------------------------------- #

def test_surrogate_pvalues_detect_coupling(broadband_triplet):
    data, _, _ = broadband_triplet
    # Use Pearson |r| as a cheap statistic to exercise the driver directly.
    stat = lambda z: np.abs(np.corrcoef(z))
    obs, p = surrogate_pvalues(stat, data, n_surrogates=200, rng=0)
    assert obs.shape == p.shape == (3, 3)
    np.testing.assert_allclose(np.diag(p), 0.0)
    assert p[0, 1] < 0.05      # coupled A-B
    assert p[0, 2] > 0.2       # independent A-C


def test_surrogate_pvalue_floor():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 1000))
    stat = lambda z: np.abs(np.corrcoef(z))
    _, p = surrogate_pvalues(stat, x, n_surrogates=99, rng=1)
    iu = np.triu_indices(3, 1)
    assert np.nanmin(p[iu]) >= 1.0 / (99 + 1) - 1e-12


def test_surrogate_pvalues_reproducible():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, 800))
    stat = lambda z: np.abs(np.corrcoef(z))
    _, p1 = surrogate_pvalues(stat, x, n_surrogates=50, rng=42)
    _, p2 = surrogate_pvalues(stat, x, n_surrogates=50, rng=42)
    np.testing.assert_array_equal(p1, p2)


def test_surrogate_pvalues_nan_passthrough():
    # A constant channel yields NaN correlations -> NaN p-values, no crash.
    rng = np.random.default_rng(6)
    x = np.vstack([rng.standard_normal(500), np.ones(500), rng.standard_normal(500)])
    stat = lambda z: np.corrcoef(z)
    with np.errstate(invalid="ignore", divide="ignore"):
        _, p = surrogate_pvalues(stat, x, n_surrogates=30, rng=0)
    assert np.isnan(p[0, 1])   # involves the constant channel
    assert np.isfinite(p[0, 2])


def test_surrogate_pvalues_validates_args():
    x = np.random.default_rng(0).standard_normal((3, 200))
    with pytest.raises(ValueError):
        surrogate_pvalues(lambda z: np.corrcoef(z), x, n_surrogates=0)
    with pytest.raises(ValueError):
        surrogate_pvalues(lambda z: np.corrcoef(z), x, n_surrogates=10, tail="up")


# --------------------------------------------------------------------------- #
# estimator integration
# --------------------------------------------------------------------------- #

def test_plv_no_pvalues_without_surrogates():
    data = np.random.default_rng(0).standard_normal((3, 2000))
    m = cn.connectivity(data, method="plv").matrix
    assert m.pvalues is None


def test_plv_surrogate_pvalues(broadband_triplet):
    data, labels, fs = broadband_triplet
    m = cn.connectivity(data, method="plv", sfreq=fs, fmin=0.01, fmax=0.1,
                        surrogates=200, surrogate_seed=1, channels=labels).matrix
    assert m.pvalues is not None
    assert m.pvalues.shape == (3, 3)
    assert np.all((m.pvalues[np.triu_indices(3, 1)] >= 0))
    assert m.pvalues[0, 1] < 0.05     # coupled
    assert m.pvalues[0, 2] > 0.2      # independent


def test_coherence_surrogate_pvalues(broadband_triplet):
    data, labels, fs = broadband_triplet
    m = cn.connectivity(data, method="coherence", sfreq=fs, fmin=0.01, fmax=0.1,
                        surrogates=200, surrogate_seed=1, channels=labels).matrix
    assert m.pvalues is not None
    assert m.pvalues[0, 1] < 0.05
    assert m.pvalues[0, 2] > 0.2


def test_wavelet_coherence_surrogate_pvalues_shape():
    # Keep this cheap: small data + few surrogates (WTC surrogates are heavy).
    fs = 5.0
    n = 1500
    rng = np.random.default_rng(0)
    b, a = butter(4, [0.02 / (fs / 2), 0.2 / (fs / 2)], btype="band")
    bp = lambda z: filtfilt(b, a, z)
    shared = bp(rng.standard_normal(n))
    data = np.vstack([shared + bp(rng.standard_normal(n)),
                      shared + bp(rng.standard_normal(n)),
                      bp(rng.standard_normal(n))])
    m = cn.connectivity(data, method="wavelet_coherence", sfreq=fs, fmin=0.02,
                        fmax=0.2, surrogates=20, surrogate_seed=1).matrix
    assert m.pvalues is not None and m.pvalues.shape == (3, 3)
    np.testing.assert_allclose(np.diag(m.pvalues), 0.0)
    # Coupled pair should out-rank the independent pair.
    assert m.pvalues[0, 1] <= m.pvalues[0, 2]
