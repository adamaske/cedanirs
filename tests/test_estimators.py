"""Tests for the Spearman, partial-correlation and coherence estimators."""

import numpy as np
import pytest
from scipy.stats import spearmanr

import cedanirs as cn
from cedanirs import DataError
from cedanirs.core.types import ConnectivityKind, Domain


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

def test_all_methods_registered():
    names = {e["name"] for e in cn.list_estimators()}
    assert {"pearson", "spearman", "partial", "coherence"} <= names


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
