import numpy as np
import pytest

import nirconn as cn
from nirconn import ConnectivityMatrix, DataError
from nirconn.core.types import ConnectivityKind


@pytest.fixture
def matrix(correlated_data):
    data, labels = correlated_data
    return cn.connectivity(data, channels=labels).matrix


def test_to_dataframe(matrix):
    df = matrix.to_dataframe()
    assert list(df.index) == matrix.labels
    assert list(df.columns) == matrix.labels
    assert df.shape == (4, 4)


def test_edges_undirected_unique_pairs(matrix):
    edges = matrix.edges()
    # 4 channels -> 6 unique undirected pairs.
    assert len(edges) == 6
    assert set(edges.columns) >= {"source", "target", "weight"}


def test_threshold_by_value_zeros_weak_edges(matrix):
    t = matrix.threshold(0.5)
    off = t.values[~np.eye(4, dtype=bool)]
    nonzero = off[off != 0]
    assert np.all(np.abs(nonzero) >= 0.5)
    np.testing.assert_allclose(np.diag(t.values), 0.0)


def test_threshold_density_keeps_fraction(matrix):
    t = matrix.threshold(density=0.5)
    off_nonzero = np.count_nonzero(t.values[~np.eye(4, dtype=bool)])
    # 12 off-diagonal entries (directed view); roughly half retained.
    assert 0 < off_nonzero <= 12


def test_threshold_requires_exactly_one_criterion(matrix):
    with pytest.raises(DataError):
        matrix.threshold(0.5, percentile=90)
    with pytest.raises(DataError):
        matrix.threshold()


def test_binarize(matrix):
    t = matrix.threshold(0.5, binarize=True)
    vals = t.values[~np.eye(4, dtype=bool)]
    assert set(np.unique(vals)).issubset({0.0, 1.0})


def test_significant_mask_symmetric(matrix):
    mask = matrix.significant(0.05)
    assert mask.shape == (4, 4)
    np.testing.assert_array_equal(mask, mask.T)  # undirected -> symmetric
    # A-B and A-D are strong; A-C is not.
    labels = matrix.labels
    i = {lab: k for k, lab in enumerate(labels)}
    assert mask[i["A"], i["B"]]
    assert mask[i["A"], i["D"]]
    assert not mask[i["A"], i["C"]]


def test_significant_without_pvalues_raises(correlated_data):
    data, labels = correlated_data
    m = cn.connectivity(data, channels=labels, compute_pvalues=False).matrix
    from nirconn import NotFittedError

    with pytest.raises(NotFittedError):
        m.significant(0.05)


def test_fisher_z_matrix(matrix):
    z = matrix.fisher_z()
    assert np.all(np.isfinite(z.values))
    assert z.params.get("transform") == "fisher_z"


def test_summary(matrix):
    s = matrix.summary()
    assert s["method"] == "pearson"
    assert s["n_channels"] == 4
    assert s["n_edges"] == 6
    assert -1 <= s["mean"] <= 1


def test_shape_mismatch_raises():
    with pytest.raises(DataError):
        ConnectivityMatrix(np.zeros((3, 3)), ["a", "b"], method="x")


def test_result_indexing_and_dataset(chromophore_cube):
    cube, labels = chromophore_cube
    result = cn.connectivity(cube, chromophores=["HbO", "HbR"], channels=labels)
    assert isinstance(result["HbO"], ConnectivityMatrix)
    assert result["hbo"] is result["HbO"]  # coercion
    ds = result.to_dataset()
    assert set(ds.data_vars) == {"HbO", "HbR"}


def test_report_is_string(matrix, correlated_data):
    data, labels = correlated_data
    result = cn.connectivity(data, channels=labels, sfreq=10.0)
    text = result.report()
    assert isinstance(text, str)
    assert "pearson" in text
    assert "significant" in text
    # ASCII-safe: must encode on a Windows cp1252 console without error.
    text.encode("cp1252")
