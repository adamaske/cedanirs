import numpy as np
import pytest
from scipy.stats import pearsonr

import cedanirs as cn
from cedanirs import ConnectivityResult
from cedanirs.core.types import ConnectivityKind


def test_known_correlation_structure(correlated_data, log):
    data, labels = correlated_data
    result = cn.connectivity(data, method="pearson", sfreq=10.0, channels=labels)
    m = result.matrix
    log.info("matrix:\n%s", np.round(m.values, 3))

    # A & B strongly positive, A & D strongly negative, C independent.
    assert m.data.sel(source="A", target="B") > 0.95
    assert m.data.sel(source="A", target="D") < -0.95
    assert abs(float(m.data.sel(source="A", target="C"))) < 0.2


def test_matrix_is_symmetric_with_unit_diagonal(correlated_data):
    data, labels = correlated_data
    m = cn.connectivity(data, channels=labels).matrix
    vals = m.values
    np.testing.assert_allclose(vals, vals.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(vals), 1.0)


def test_values_within_unit_range(correlated_data):
    data, labels = correlated_data
    m = cn.connectivity(data, channels=labels).matrix
    assert np.nanmax(m.values) <= 1.0
    assert np.nanmin(m.values) >= -1.0


def test_matches_scipy_pearsonr(correlated_data):
    data, labels = correlated_data
    m = cn.connectivity(data, channels=labels).matrix
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            r_ref, p_ref = pearsonr(data[i], data[j])
            assert m.values[i, j] == pytest.approx(r_ref, abs=1e-10)
            assert m.pvalues[i, j] == pytest.approx(p_ref, abs=1e-8)


def test_metadata(correlated_data):
    data, labels = correlated_data
    result = cn.connectivity(data, channels=labels)
    assert isinstance(result, ConnectivityResult)
    assert result.method == "pearson"
    assert result.kind == ConnectivityKind.FUNCTIONAL
    assert result.directed is False
    m = result.matrix
    assert m.labels == labels
    assert result.provenance["n_times"] == data.shape[1]
    assert result.provenance["n_channels"] == data.shape[0]


def test_disable_pvalues(correlated_data):
    data, labels = correlated_data
    m = cn.connectivity(data, channels=labels, compute_pvalues=False).matrix
    assert m.pvalues is None


def test_constant_channel_is_nan_not_crash():
    rng = np.random.default_rng(0)
    data = np.vstack([rng.standard_normal(100), np.ones(100), rng.standard_normal(100)])
    m = cn.connectivity(data).matrix
    # Off-diagonal correlations with the constant channel are NaN.
    assert np.isnan(m.values[0, 1])
    assert np.isnan(m.values[1, 2])
    # Diagonal is forced to 1.0 even for the constant channel.
    np.testing.assert_allclose(np.diag(m.values), 1.0)


def test_chromophore_cube_produces_two_matrices(chromophore_cube):
    cube, labels = chromophore_cube
    result = cn.connectivity(cube, chromophores=["HbO", "HbR"], channels=labels)
    assert result.chromophores == ["HbO", "HbR"]
    assert result["HbO"].chromophore == "HbO"
    with pytest.raises(Exception):
        _ = result.matrix  # ambiguous: more than one chromophore


def test_estimator_is_callable(correlated_data):
    data, labels = correlated_data
    est = cn.PearsonCorrelation()
    result = est(data)
    assert result.matrix.n == len(labels)
