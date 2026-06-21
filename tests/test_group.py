"""Tests for the group-level statistics, tables, reporting and poster."""

import numpy as np
import pytest
from scipy import stats as sst

import cedanirs as cn
from cedanirs import ConnectivityMatrix, DataError, GroupConnectivity, Study
from cedanirs.stats.transforms import fisher_z


# --------------------------------------------------------------------------- #
# helpers / fixtures
# --------------------------------------------------------------------------- #

def rand_matrix(rng, labels, *, center=0.0, scale=0.25, chromophore="HbO"):
    n = len(labels)
    m = rng.normal(center, scale, (n, n))
    m = (m + m.T) / 2.0
    np.clip(m, -0.95, 0.95, out=m)
    np.fill_diagonal(m, 1.0)
    return ConnectivityMatrix(m, labels, method="pearson", chromophore=chromophore)


@pytest.fixture
def labels():
    return [f"C{i}" for i in range(6)]


@pytest.fixture
def group_pos(rng, labels):
    mats = [rand_matrix(rng, labels, center=0.4) for _ in range(15)]
    return GroupConnectivity.from_matrices(mats, ids=[f"s{i}" for i in range(15)])


# --------------------------------------------------------------------------- #
# Study / GroupConnectivity data model
# --------------------------------------------------------------------------- #

def test_study_add_and_group(rng, labels):
    study = cn.Study(name="t")
    for i in range(5):
        m = rand_matrix(rng, labels)
        res = cn.ConnectivityResult({"HbO": m}, method="pearson")
        study.add(f"s{i}", res, group="g1", age=20 + i)
    assert study.subjects == [f"s{i}" for i in range(5)]
    assert "HbO" in study.chromophores
    g = study.group("HbO")
    assert g.n_subjects == 5
    assert g.labels == labels
    assert list(study.metadata["age"]) == [20, 21, 22, 23, 24]


def test_duplicate_subject_raises(rng, labels):
    study = cn.Study()
    res = cn.ConnectivityResult({"HbO": rand_matrix(rng, labels)}, method="pearson")
    study.add("s0", res)
    with pytest.raises(DataError):
        study.add("s0", res)


def test_from_matrices_outer_alignment(rng):
    a = rand_matrix(rng, ["A", "B", "C"])
    b = rand_matrix(rng, ["B", "C", "D"])
    # default min_coverage=1.0 keeps only channels present in ALL subjects.
    g = GroupConnectivity.from_matrices([a, b], ids=["s0", "s1"])
    assert set(g.labels) == {"B", "C"}
    # with min_coverage=0.5, the union is kept and gaps are NaN.
    g2 = GroupConnectivity.from_matrices([a, b], ids=["s0", "s1"], min_coverage=0.5)
    assert set(g2.labels) == {"A", "B", "C", "D"}
    assert np.isnan(g2.stack(fisher=False)).any()


# --------------------------------------------------------------------------- #
# one-sample
# --------------------------------------------------------------------------- #

def test_one_sample_matches_scipy(group_pos):
    res = group_pos.one_sample()
    z = group_pos.stack(fisher=True)
    t_ref, p_ref = sst.ttest_1samp(z, 0.0, axis=0)
    iu = np.triu_indices(group_pos.n, 1)
    np.testing.assert_allclose(res.tstat[iu], t_ref[iu], rtol=1e-9)
    np.testing.assert_allclose(res.pvalues[iu], p_ref[iu], rtol=1e-7)


def test_one_sample_recovers_positive_edges(group_pos):
    res = group_pos.one_sample()
    # center=0.4 -> almost every edge should be significantly > 0.
    assert res.summary()["n_significant"] >= 10
    # effect is reported in r-units in [-1, 1].
    off = res.effect[~np.eye(res.n, dtype=bool)]
    assert np.all(np.abs(off) <= 1.0)


def test_one_sample_correction_only_unique_edges(group_pos):
    res = group_pos.one_sample(correction="bonferroni")
    n = group_pos.n
    # symmetric mask, diagonal never significant
    np.testing.assert_array_equal(res.reject, res.reject.T)
    assert not np.any(np.diag(res.reject))


def test_one_sample_needs_two_subjects(rng, labels):
    g = GroupConnectivity.from_matrices([rand_matrix(rng, labels)], ids=["s0"])
    with pytest.raises(DataError):
        g.one_sample()


# --------------------------------------------------------------------------- #
# two-sample / paired
# --------------------------------------------------------------------------- #

def test_two_sample_welch_matches_scipy(rng, labels):
    a = GroupConnectivity.from_matrices(
        [rand_matrix(rng, labels, center=0.4) for _ in range(12)],
        ids=[f"a{i}" for i in range(12)],
    )
    b = GroupConnectivity.from_matrices(
        [rand_matrix(rng, labels, center=0.1) for _ in range(10)],
        ids=[f"b{i}" for i in range(10)],
    )
    res = a.two_sample(b, equal_var=False)
    za, zb = a.stack(fisher=True), b.stack(fisher=True)
    t_ref, p_ref = sst.ttest_ind(za, zb, axis=0, equal_var=False)
    iu = np.triu_indices(a.n, 1)
    np.testing.assert_allclose(res.tstat[iu], t_ref[iu], rtol=1e-9)
    np.testing.assert_allclose(res.pvalues[iu], p_ref[iu], rtol=1e-7)


def test_two_sample_pooled_matches_scipy(rng, labels):
    a = GroupConnectivity.from_matrices(
        [rand_matrix(rng, labels, center=0.3) for _ in range(11)],
        ids=[f"a{i}" for i in range(11)],
    )
    b = GroupConnectivity.from_matrices(
        [rand_matrix(rng, labels, center=0.0) for _ in range(11)],
        ids=[f"b{i}" for i in range(11)],
    )
    res = a.two_sample(b, equal_var=True)
    t_ref, p_ref = sst.ttest_ind(a.stack(fisher=True), b.stack(fisher=True), axis=0)
    iu = np.triu_indices(a.n, 1)
    np.testing.assert_allclose(res.tstat[iu], t_ref[iu], rtol=1e-9)


def test_paired_matches_scipy(rng, labels):
    ids = [f"s{i}" for i in range(10)]
    a = GroupConnectivity.from_matrices(
        [rand_matrix(rng, labels, center=0.3) for _ in ids], ids=ids
    )
    b = GroupConnectivity.from_matrices(
        [rand_matrix(rng, labels, center=0.1) for _ in ids], ids=ids
    )
    res = a.paired(b)
    t_ref, p_ref = sst.ttest_rel(a.stack(fisher=True), b.stack(fisher=True), axis=0)
    iu = np.triu_indices(a.n, 1)
    np.testing.assert_allclose(res.tstat[iu], t_ref[iu], rtol=1e-9)
    assert res.test == "paired_t"


# --------------------------------------------------------------------------- #
# regression
# --------------------------------------------------------------------------- #

def test_regression_recovers_covariate(rng):
    labels = ["A", "B", "C"]
    n_subj = 30
    x = rng.normal(size=n_subj)
    mats = []
    for s in range(n_subj):
        m = rand_matrix(rng, labels, center=0.0, scale=0.1)
        # make edge A-B (0,1) track the covariate strongly
        val = np.tanh(0.8 * x[s])
        m.values[0, 1] = m.values[1, 0] = np.clip(val, -0.95, 0.95)
        mats.append(ConnectivityMatrix(m.values, labels, method="pearson", chromophore="HbO"))
    g = GroupConnectivity.from_matrices(mats, ids=[f"s{i}" for i in range(n_subj)])
    res = g.regression(x, predictor=0)
    # the A-B edge should be the strongest / significant association
    assert res.reject[0, 1]
    # matches a simple correlation t for that edge
    z01 = g.stack(fisher=True)[:, 0, 1]
    r = np.corrcoef(x, z01)[0, 1]
    t_ref = r * np.sqrt((n_subj - 2) / (1 - r**2))
    assert res.tstat[0, 1] == pytest.approx(t_ref, rel=1e-6)


# --------------------------------------------------------------------------- #
# NBS
# --------------------------------------------------------------------------- #

def test_nbs_finds_planted_subnetwork(rng):
    labels = [f"C{i}" for i in range(6)]
    block = [(0, 1), (1, 2), (0, 2)]  # a connected triangle that will differ

    def make(center_block):
        m = rand_matrix(rng, labels, center=0.0, scale=0.1)
        for (i, j) in block:
            v = np.clip(rng.normal(center_block, 0.05), -0.95, 0.95)
            m.values[i, j] = m.values[j, i] = v
        return ConnectivityMatrix(m.values, labels, method="pearson", chromophore="HbO")

    a = GroupConnectivity.from_matrices([make(0.6) for _ in range(12)],
                                        ids=[f"a{i}" for i in range(12)])
    b = GroupConnectivity.from_matrices([make(0.0) for _ in range(12)],
                                        ids=[f"b{i}" for i in range(12)])
    res = a.nbs(b, threshold=2.5, n_permutations=300, seed=0)
    assert len(res.components) >= 1
    assert np.any(res.component_pvalues <= 0.05)
    tbl = res.table()
    assert "p_fwe" in tbl.columns


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #

def test_significant_edges_table_group(group_pos):
    res = group_pos.one_sample()
    df = res.table()
    for col in ("edge", "effect", "t", "df", "p", "q", "cohens_d", "significant"):
        assert col in df.columns
    # significant rows float to the top
    assert df["significant"].iloc[0]


def test_to_matrix_bridges_into_surface(group_pos):
    res = group_pos.one_sample()
    m = res.to_matrix()
    assert isinstance(m, ConnectivityMatrix)
    assert m.pvalues is not None
    # the whole ConnectivityMatrix surface now works on a group result
    assert m.to_graph(threshold=0.1) is not None


# --------------------------------------------------------------------------- #
# report + poster
# --------------------------------------------------------------------------- #

def test_result_report_is_json_serializable(rng, labels):
    import json

    res = cn.ConnectivityResult({"HbO": rand_matrix(rng, labels, center=0.3)},
                                method="pearson")
    rep = cn.report.result_report(res)
    assert set(rep) >= {"meta", "provenance", "params", "chromophores"}
    json.dumps(rep)  # must not raise


def test_summarize_group_ascii(group_pos):
    text = cn.report.summarize_group(group_pos.one_sample(), top_n=5)
    assert "Group connectivity report" in text
    text.encode("cp1252")  # ASCII-safe for Windows consoles


def test_build_poster_returns_figure(group_pos):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    res = group_pos.one_sample()
    fig = cn.build_poster(res, group_mean=group_pos.mean(), title="test")
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt

    plt.close(fig)
