"""Tests for the Spearman, partial-correlation and coherence estimators."""

import numpy as np
import pytest
from scipy.stats import spearmanr

import nirconn as cn
from nirconn import DataError
from nirconn.core.types import ConnectivityKind, Domain


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

def test_all_methods_registered():
    names = {e["name"] for e in cn.list_estimators()}
    assert {
        "pearson", "spearman", "partial", "coherence",
        "plv", "wavelet_coherence", "granger",
    } <= names


@pytest.fixture
def band_limited_pair():
    """Two channels sharing a 0.05 Hz oscillation + noise, one independent."""
    fs = 5.0
    n = 6000
    t = np.arange(n) / fs
    rng = np.random.default_rng(42)
    shared = np.sin(2 * np.pi * 0.05 * t)
    a = shared + 0.5 * rng.standard_normal(n)
    # b is the shared oscillation delayed a few samples (constant phase lag)
    b = np.roll(shared, 4) + 0.5 * rng.standard_normal(n)
    c = rng.standard_normal(n)  # independent
    return np.vstack([a, b, c]), ["A", "B", "C"], fs


# --------------------------------------------------------------------------- #
# Spearman
# --------------------------------------------------------------------------- #

def test_spearman_matches_scipy(correlated_data):
    data, labels = correlated_data
    m = cn.connectivity(data, method="spearman", channels=labels).matrix
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            rho_ref, p_ref = spearmanr(data[i], data[j])
            assert m.values[i, j] == pytest.approx(rho_ref, abs=1e-9)
            assert m.pvalues[i, j] == pytest.approx(p_ref, abs=1e-6)


def test_spearman_symmetric_unit_diagonal(correlated_data):
    data, labels = correlated_data
    m = cn.connectivity(data, method="spearman", channels=labels).matrix
    np.testing.assert_allclose(m.values, m.values.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(m.values), 1.0)
    assert -1 <= np.nanmin(m.values) and np.nanmax(m.values) <= 1


def test_spearman_robust_to_outlier():
    rng = np.random.default_rng(3)
    base = rng.standard_normal(400)
    a = base.copy()
    b = base.copy()
    # A single huge outlier wrecks Pearson but barely moves Spearman.
    a[0] += 1000.0
    data = np.vstack([a, b])
    pear = cn.connectivity(data, method="pearson").matrix.values[0, 1]
    spear = cn.connectivity(data, method="spearman").matrix.values[0, 1]
    assert spear > 0.99
    assert spear > pear


# --------------------------------------------------------------------------- #
# Partial correlation
# --------------------------------------------------------------------------- #

def test_partial_removes_common_driver():
    # C is a shared driver of A and B; A and B have no direct link.
    rng = np.random.default_rng(7)
    n = 2000
    c = rng.standard_normal(n)
    a = c + 0.3 * rng.standard_normal(n)
    b = c + 0.3 * rng.standard_normal(n)
    data = np.vstack([a, b, c])
    labels = ["A", "B", "C"]

    pear = cn.connectivity(data, method="pearson", channels=labels).matrix
    part = cn.connectivity(data, method="partial", channels=labels).matrix

    i = {lab: k for k, lab in enumerate(labels)}
    # Pearson sees a strong spurious A-B link...
    assert pear.values[i["A"], i["B"]] > 0.6
    # ...partial correlation controlling for C collapses it toward zero.
    assert abs(part.values[i["A"], i["B"]]) < 0.15


def test_partial_symmetric_unit_diagonal(correlated_data):
    data, labels = correlated_data
    m = cn.connectivity(data, method="partial", channels=labels).matrix
    np.testing.assert_allclose(m.values, m.values.T, atol=1e-8)
    np.testing.assert_allclose(np.diag(m.values), 1.0)


def test_partial_pvalues_dof():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((5, 500))
    m = cn.connectivity(data, method="partial").matrix
    assert m.pvalues is not None
    assert np.all(np.isfinite(m.pvalues))


def test_partial_too_few_samples_omits_pvalues():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((10, 8))  # n_times < n_channels
    with pytest.warns(RuntimeWarning):
        m = cn.connectivity(data, method="partial", shrinkage=0.1).matrix
    assert m.pvalues is None


def test_partial_constant_channel_raises():
    data = np.vstack([np.random.default_rng(0).standard_normal(100), np.ones(100)])
    with pytest.raises(DataError):
        cn.connectivity(data, method="partial")


def test_partial_shrinkage_validation():
    with pytest.raises(DataError):
        cn.PartialCorrelation(shrinkage=2.0)


# --------------------------------------------------------------------------- #
# Coherence
# --------------------------------------------------------------------------- #

def test_coherence_requires_sfreq():
    data = np.random.default_rng(0).standard_normal((4, 2000))
    with pytest.raises(DataError):
        cn.connectivity(data, method="coherence")  # no sfreq


def test_coherence_is_metadata_frequency_domain():
    entry = next(e for e in cn.list_estimators() if e["name"] == "coherence")
    assert entry["domain"] == str(Domain.FREQUENCY)
    assert entry["kind"] == str(ConnectivityKind.FUNCTIONAL)
    assert cn.Coherence.requires_sfreq is True


def test_coherence_high_for_band_limited_shared_signal():
    # Two channels sharing a 0.05 Hz oscillation (inside the band) plus
    # independent noise should be highly coherent in-band.
    fs = 5.0
    n = 6000
    t = np.arange(n) / fs
    rng = np.random.default_rng(11)
    shared = np.sin(2 * np.pi * 0.05 * t)
    a = shared + 0.5 * rng.standard_normal(n)
    b = shared + 0.5 * rng.standard_normal(n)
    c = rng.standard_normal(n)  # independent
    data = np.vstack([a, b, c])

    # Average over a band straddling the shared 0.05 Hz oscillation.
    m = cn.connectivity(data, method="coherence", sfreq=fs, fmin=0.03, fmax=0.07).matrix
    vals = m.values
    assert np.all((vals >= 0) & (vals <= 1.0 + 1e-9))
    np.testing.assert_allclose(np.diag(vals), 1.0)
    np.testing.assert_allclose(vals, vals.T, atol=1e-12)
    # Shared in-band oscillation -> high coherence; independent -> lower.
    assert vals[0, 1] > 0.5
    assert vals[0, 1] > vals[0, 2]


def test_coherence_band_too_narrow_for_segment_raises():
    fs = 5.0
    data = np.random.default_rng(0).standard_normal((3, 4000))
    # An absurdly low band that no FFT bin can resolve at this nperseg.
    with pytest.raises(DataError):
        cn.connectivity(
            data, method="coherence", sfreq=fs, fmin=1e-6, fmax=2e-6, nperseg=64
        )


def test_coherence_no_pvalues():
    fs = 5.0
    data = np.random.default_rng(0).standard_normal((3, 3000))
    m = cn.connectivity(data, method="coherence", sfreq=fs).matrix
    assert m.pvalues is None


# --------------------------------------------------------------------------- #
# Phase-locking value / phase-lag index
# --------------------------------------------------------------------------- #

def test_plv_metadata():
    entry = next(e for e in cn.list_estimators() if e["name"] == "plv")
    assert entry["kind"] == str(ConnectivityKind.FUNCTIONAL)
    assert entry["directed"] is False


def test_plv_high_for_phase_locked_pair(band_limited_pair):
    data, labels, fs = band_limited_pair
    m = cn.connectivity(data, method="plv", sfreq=fs, fmin=0.03, fmax=0.07,
                        channels=labels).matrix
    vals = m.values
    assert np.all((vals >= 0) & (vals <= 1.0 + 1e-9))
    np.testing.assert_allclose(np.diag(vals), 1.0)
    np.testing.assert_allclose(vals, vals.T, atol=1e-12)
    assert m.pvalues is None
    # Constant-phase-lag pair locks; independent channel does not.
    assert vals[0, 1] > 0.5
    assert vals[0, 1] > vals[0, 2]


def test_pli_suppresses_zero_lag():
    # Two channels with identical phase (zero lag): PLV ~ 1, PLI ~ 0.
    fs = 5.0
    n = 6000
    t = np.arange(n) / fs
    rng = np.random.default_rng(7)
    shared = np.sin(2 * np.pi * 0.05 * t)
    a = shared + 0.3 * rng.standard_normal(n)
    b = shared + 0.3 * rng.standard_normal(n)  # same phase as a
    data = np.vstack([a, b])
    plv = cn.connectivity(data, method="plv", sfreq=fs, mode="plv",
                          fmin=0.03, fmax=0.07).matrix.values[0, 1]
    pli = cn.connectivity(data, method="plv", sfreq=fs, mode="pli",
                          fmin=0.03, fmax=0.07).matrix.values[0, 1]
    assert plv > 0.7
    assert pli < plv


def test_plv_band_requires_sfreq():
    data = np.random.default_rng(0).standard_normal((3, 2000))
    with pytest.raises(DataError):
        cn.connectivity(data, method="plv", fmin=0.03, fmax=0.07)  # no sfreq


def test_plv_works_without_band_on_prefiltered():
    # No fmin/fmax -> no sfreq needed; assumes pre-filtered input.
    fs = 5.0
    n = 4000
    t = np.arange(n) / fs
    shared = np.sin(2 * np.pi * 0.05 * t)
    data = np.vstack([shared, np.roll(shared, 3)])
    m = cn.connectivity(data, method="plv").matrix
    assert m.values[0, 1] > 0.9


def test_plv_mode_validation():
    with pytest.raises(DataError):
        cn.PhaseLockingValue(mode="bogus")


# --------------------------------------------------------------------------- #
# Wavelet coherence
# --------------------------------------------------------------------------- #

def test_wavelet_coherence_metadata():
    entry = next(e for e in cn.list_estimators() if e["name"] == "wavelet_coherence")
    assert entry["domain"] == str(Domain.TIME_FREQUENCY)
    assert entry["kind"] == str(ConnectivityKind.FUNCTIONAL)
    assert cn.WaveletCoherence.requires_sfreq is True


def test_wavelet_coherence_requires_sfreq():
    data = np.random.default_rng(0).standard_normal((3, 2000))
    with pytest.raises(DataError):
        cn.connectivity(data, method="wavelet_coherence")


def test_wavelet_coherence_high_for_shared_band(band_limited_pair):
    data, labels, fs = band_limited_pair
    m = cn.connectivity(data, method="wavelet_coherence", sfreq=fs,
                        fmin=0.03, fmax=0.08, channels=labels).matrix
    vals = m.values
    assert np.all((vals >= 0) & (vals <= 1.0 + 1e-9))
    np.testing.assert_allclose(np.diag(vals), 1.0)
    np.testing.assert_allclose(vals, vals.T, atol=1e-12)
    assert m.pvalues is None
    assert vals[0, 1] > vals[0, 2]


def test_wavelet_coherence_band_out_of_range_raises():
    fs = 5.0
    data = np.random.default_rng(0).standard_normal((3, 500))
    # Band far below what a 500-sample record can resolve.
    with pytest.raises(DataError):
        cn.connectivity(data, method="wavelet_coherence", sfreq=fs,
                        fmin=1e-5, fmax=2e-5)


# --------------------------------------------------------------------------- #
# Granger causality (effective)
# --------------------------------------------------------------------------- #

def test_granger_metadata():
    entry = next(e for e in cn.list_estimators() if e["name"] == "granger")
    assert entry["kind"] == str(ConnectivityKind.EFFECTIVE)
    assert entry["directed"] is True
    assert entry["domain"] == str(Domain.TIME)


def test_granger_recovers_direction():
    # x drives y at lag 1; x has no input from y.
    rng = np.random.default_rng(1)
    n = 3000
    x = np.zeros(n)
    y = np.zeros(n)
    ex, ey = rng.standard_normal(n), rng.standard_normal(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + ex[t]
        y[t] = 0.5 * y[t - 1] + 0.6 * x[t - 1] + ey[t]
    m = cn.connectivity(np.vstack([x, y]), method="granger", order=2,
                        channels=["x", "y"]).matrix
    assert m.directed
    np.testing.assert_allclose(np.diag(m.values), 0.0)
    # matrix[src, tgt]: x -> y strong & significant, y -> x negligible.
    assert m.values[0, 1] > 0.1
    assert m.pvalues[0, 1] < 1e-3
    assert m.values[1, 0] < m.values[0, 1]
    assert m.pvalues[1, 0] > 0.05


def test_granger_edges_label_direction():
    rng = np.random.default_rng(2)
    n = 2000
    x = np.zeros(n)
    y = np.zeros(n)
    ex, ey = rng.standard_normal(n), rng.standard_normal(n)
    for t in range(1, n):
        x[t] = 0.4 * x[t - 1] + ex[t]
        y[t] = 0.4 * y[t - 1] + 0.7 * x[t - 1] + ey[t]
    m = cn.connectivity(np.vstack([x, y]), method="granger",
                        channels=["x", "y"]).matrix
    edges = m.edges()
    # Both ordered pairs present; the strongest is x -> y.
    assert len(edges) == 2
    top = edges.sort_values("weight", ascending=False).iloc[0]
    assert (top["source"], top["target"]) == ("x", "y")


def test_granger_too_few_samples_raises():
    data = np.random.default_rng(0).standard_normal((3, 5))
    with pytest.raises(DataError):
        cn.connectivity(data, method="granger", order=3)


def test_granger_order_validation():
    with pytest.raises(DataError):
        cn.GrangerCausality(order=0)
