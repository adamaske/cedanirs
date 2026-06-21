import numpy as np
import pytest
from scipy.stats import pearsonr

from cedanirs.stats import (
    average_correlations,
    bonferroni_correction,
    correct_pvalues,
    correlation_pvalues,
    fdr_correction,
    fisher_z,
    inverse_fisher_z,
)


def test_fisher_z_roundtrip():
    r = np.array([-0.9, -0.3, 0.0, 0.5, 0.99])
    np.testing.assert_allclose(inverse_fisher_z(fisher_z(r)), r, atol=1e-12)


def test_fisher_z_handles_unit_without_inf():
    z = fisher_z(np.array([1.0, -1.0]))
    assert np.all(np.isfinite(z))


def test_average_correlations_uses_z_space():
    # Averaging in z-space differs from naive r-space averaging for large r.
    mats = np.array([[[1.0, 0.9], [0.9, 1.0]], [[1.0, 0.5], [0.5, 1.0]]])
    avg = average_correlations(mats, axis=0)
    naive = mats.mean(axis=0)
    expected = inverse_fisher_z((fisher_z(0.9) + fisher_z(0.5)) / 2)
    assert avg[0, 1] == pytest.approx(expected)
    assert avg[0, 1] != pytest.approx(naive[0, 1])


def test_correlation_pvalues_match_scipy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 200))
    r = np.corrcoef(x)
    p = correlation_pvalues(r, n_obs=200)
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            _, p_ref = pearsonr(x[i], x[j])
            assert p[i, j] == pytest.approx(p_ref, abs=1e-8)
    np.testing.assert_allclose(np.diag(p), 0.0)


def test_correlation_pvalues_too_few_obs():
    from cedanirs import DataError

    with pytest.raises(DataError):
        correlation_pvalues(np.eye(2), n_obs=2)


def test_fdr_monotone_and_bounded():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=50)
    reject, q = fdr_correction(p, alpha=0.05)
    assert np.all((q >= 0) & (q <= 1))
    # A smaller p cannot get a larger q after sorting.
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_fdr_rejects_clear_signal():
    p = np.concatenate([np.full(5, 1e-8), np.full(45, 0.8)])
    reject, q = fdr_correction(p, alpha=0.05)
    assert reject[:5].all()
    assert not reject[5:].any()


def test_bonferroni():
    p = np.array([0.001, 0.02, 0.5])
    reject, adj = bonferroni_correction(p, alpha=0.05)
    np.testing.assert_allclose(adj, [0.003, 0.06, 1.0])
    assert reject.tolist() == [True, False, False]


def test_fdr_is_less_conservative_than_bonferroni():
    p = np.linspace(0.001, 0.05, 20)
    n_fdr = fdr_correction(p, 0.05)[0].sum()
    n_bonf = bonferroni_correction(p, 0.05)[0].sum()
    assert n_fdr >= n_bonf


def test_correct_pvalues_dispatch():
    p = np.array([0.001, 0.2, 0.04])
    assert correct_pvalues(p, method=None)[0].tolist() == [True, False, True]
    assert correct_pvalues(p, method="bonferroni")[0].shape == p.shape
    assert correct_pvalues(p, method="fdr_bh")[0].shape == p.shape


def test_nan_pvalues_are_ignored():
    p = np.array([0.001, np.nan, 0.04])
    reject, q = fdr_correction(p, 0.05)
    assert not reject[1]
    assert np.isnan(q[1])
