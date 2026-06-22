"""Group-level connectivity statistics (edgewise, random-effects).

Every test here is a *second-level* (across-subjects) analysis run in Fisher-z
space, on the unique off-diagonal edges only, with multiple-comparison
correction applied across those edges and the result scattered back to an
N x N matrix. The cardinal rule from the design holds throughout: **aggregate
and test in z, present in r**.

Public functions
----------------
* :func:`one_sample` -- is each edge's group-mean FC different from a reference?
* :func:`two_sample` -- independent or paired contrast between two groups.
* :func:`regression` -- OLS of edge-z on a subject-level design.
* :func:`nbs` -- Network-Based Statistic (permutation cluster inference).

All return a :class:`~nirconn.stats._results.GroupStatResult` (or
:class:`~nirconn.stats._results.NBSResult`) which carries effect sizes, CIs,
corrected q-values, and a ``.to_matrix()`` bridge into the rest of nirconn.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats as _st

from ..core.exceptions import DataError
from .significance import correct_pvalues
from .transforms import fisher_z, inverse_fisher_z
from ._results import GroupStatResult, NBSResult

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from ..core.group import GroupConnectivity

SOURCE_DIM = "source"
TARGET_DIM = "target"


# --------------------------------------------------------------------------- #
# low-level helpers
# --------------------------------------------------------------------------- #

def _suppress():
    return np.errstate(divide="ignore", invalid="ignore")


def _moments(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-edge (mean, sample-sd ddof=1, n) over the subject axis, NaN-aware."""
    n = np.isfinite(stack).sum(axis=0).astype(float)
    with _suppress(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = np.nanmean(stack, axis=0)
        sd = np.nanstd(stack, axis=0, ddof=1)
    return mean, sd, n


def _two_sided_p(t: np.ndarray, dof: np.ndarray) -> np.ndarray:
    with _suppress():
        p = 2.0 * _st.t.sf(np.abs(t), dof)
    p = np.asarray(p, dtype=float)
    bad = ~np.isfinite(t) | ~(np.asarray(dof, dtype=float) >= 1)
    p[bad] = np.nan
    return p


def _tail_p(t: np.ndarray, dof: np.ndarray, tail: str) -> np.ndarray:
    if tail in ("two-sided", "two_sided", "both"):
        return _two_sided_p(t, dof)
    with _suppress():
        if tail in ("greater", "right"):
            p = _st.t.sf(t, dof)
        elif tail in ("less", "left"):
            p = _st.t.cdf(t, dof)
        else:
            raise DataError(f"Unknown tail {tail!r}; use two-sided|greater|less.")
    p = np.asarray(p, dtype=float)
    bad = ~np.isfinite(t) | ~(np.asarray(dof, dtype=float) >= 1)
    p[bad] = np.nan
    return p


def _tcrit(conf: float, dof: np.ndarray) -> np.ndarray:
    with _suppress():
        return _st.t.ppf(1.0 - (1.0 - conf) / 2.0, dof)


def _edge_index(n: int, directed: bool) -> tuple[np.ndarray, np.ndarray]:
    if directed:
        rows, cols = np.where(~np.eye(n, dtype=bool))
        return rows, cols
    return np.triu_indices(n, k=1)


def _correct(pvalues: np.ndarray, directed: bool, alpha: float, correction):
    """Correct over unique edges; scatter q/reject back to a symmetric N x N."""
    n = pvalues.shape[0]
    idx = _edge_index(n, directed)
    flat_p = pvalues[idx]
    reject_flat, q_flat = correct_pvalues(flat_p, alpha=alpha, method=correction)

    q = np.full((n, n), np.nan)
    reject = np.zeros((n, n), dtype=bool)
    q[idx] = q_flat
    reject[idx] = reject_flat
    if not directed:
        q = np.where(np.isnan(q), q.T, q)
        reject = reject | reject.T
    np.fill_diagonal(q, np.nan)
    np.fill_diagonal(reject, False)
    return q, reject


def _align(a: "GroupConnectivity", b: "GroupConnectivity"):
    """Reindex two groups onto a shared (union) channel-label set."""
    labels = list(a.labels) + [x for x in b.labels if x not in a.labels]
    da = a.data.reindex({SOURCE_DIM: labels, TARGET_DIM: labels})
    db = b.data.reindex({SOURCE_DIM: labels, TARGET_DIM: labels})
    return da, db, labels


def _hedges_g(d: np.ndarray, dof: np.ndarray) -> np.ndarray:
    with _suppress():
        j = 1.0 - 3.0 / (4.0 * dof - 1.0)
    return d * j


# --------------------------------------------------------------------------- #
# one-sample
# --------------------------------------------------------------------------- #

def one_sample(
    group: "GroupConnectivity",
    *,
    popmean: float = 0.0,
    tail: str = "two-sided",
    alpha: float = 0.05,
    correction: str | None = "fdr_bh",
    conf: float = 0.95,
) -> GroupStatResult:
    """One-sample edgewise t-test: is group-mean FC different from ``popmean``?

    ``popmean`` is given as a *correlation* (default 0) and converted to z
    internally. Reports the group-mean r, t (df = n-1), p, corrected q, Cohen's
    d / Hedges' g, and a confidence interval on the mean r.
    """
    z = group.stack(fisher=True)
    n_subj = z.shape[0]
    if n_subj < 2:
        raise DataError("One-sample test needs at least 2 subjects.")
    mu = fisher_z(popmean)

    mean, sd, n = _moments(z)
    with _suppress():
        se = sd / np.sqrt(n)
        t = (mean - mu) / se
    dof = n - 1.0
    p = _tail_p(t, dof, tail)

    with _suppress():
        d = (mean - mu) / sd  # one-sample Cohen's d
        g = _hedges_g(d, dof)
        tc = _tcrit(conf, dof)
        ci_lo_z = mean - tc * se
        ci_hi_z = mean + tc * se

    q, reject = _correct(p, group.directed, alpha, correction)
    effect = inverse_fisher_z(mean)  # group-mean r

    extras = {
        "mean_z": mean,
        "sd_z": sd,
        "n": n,
        "cohens_d": d,
        "hedges_g": g,
        "ci_low": inverse_fisher_z(ci_lo_z),
        "ci_high": inverse_fisher_z(ci_hi_z),
    }
    return GroupStatResult(
        labels=group.labels, chromophore=group.chromophore, method=group.method,
        directed=group.directed, test="one_sample_t", design="one-sample",
        effect=effect, tstat=t, dof=dof, pvalues=p, qvalues=q, reject=reject,
        extras=extras,
        params={"alpha": alpha, "correction": correction, "popmean": popmean,
                "tail": tail, "conf": conf, "n_subjects": n_subj},
    )


# --------------------------------------------------------------------------- #
# two-sample / paired
# --------------------------------------------------------------------------- #

def two_sample(
    a: "GroupConnectivity",
    b: "GroupConnectivity",
    *,
    paired: bool = False,
    equal_var: bool = False,
    alpha: float = 0.05,
    correction: str | None = "fdr_bh",
    conf: float = 0.95,
) -> GroupStatResult:
    """Two-sample (independent) or paired edgewise contrast ``a - b``.

    Independent test uses Welch's unequal-variance t by default
    (``equal_var=True`` for the pooled Student t). Paired test aligns subjects by
    id and runs a one-sample t on the within-subject z-differences.
    """
    da, db, labels = _align(a, b)
    n = len(labels)
    directed = a.directed

    if paired:
        subs = [s for s in a.subjects if s in set(b.subjects)]
        if len(subs) < 2:
            raise DataError("Paired test needs >= 2 subjects present in both groups.")
        za = fisher_z(da.sel(subject=subs).values)
        zb = fisher_z(db.sel(subject=subs).values)
        diff = za - zb
        mean, sd, nn = _moments(diff)
        with _suppress():
            se = sd / np.sqrt(nn)
            t = mean / se
        dof = nn - 1.0
        with _suppress():
            d = mean / sd
        test = "paired_t"
        design = "paired"
        mean_a = np.nanmean(za, axis=0)
        mean_b = np.nanmean(zb, axis=0)
        n_a = n_b = len(subs)
    else:
        za = fisher_z(da.values)
        zb = fisher_z(db.values)
        mA, sA, nA = _moments(za)
        mB, sB, nB = _moments(zb)
        with _suppress():
            vA = sA**2 / nA
            vB = sB**2 / nB
            if equal_var:
                sp2 = ((nA - 1) * sA**2 + (nB - 1) * sB**2) / (nA + nB - 2)
                se = np.sqrt(sp2 * (1.0 / nA + 1.0 / nB))
                dof = nA + nB - 2.0
                d = (mA - mB) / np.sqrt(sp2)
            else:
                se = np.sqrt(vA + vB)
                dof = (vA + vB) ** 2 / (vA**2 / (nA - 1) + vB**2 / (nB - 1))
                sp2 = ((nA - 1) * sA**2 + (nB - 1) * sB**2) / (nA + nB - 2)
                d = (mA - mB) / np.sqrt(sp2)
            t = (mA - mB) / se
            mean = mA - mB
        test = "welch_t" if not equal_var else "student_t"
        design = "two-sample"
        mean_a, mean_b = mA, mB
        n_a, n_b = int(np.nanmax(nA)), int(np.nanmax(nB))

    p = _two_sided_p(t, dof)
    g = _hedges_g(d, dof)
    with _suppress():
        tc = _tcrit(conf, dof)
        ci_lo = mean - tc * se
        ci_hi = mean + tc * se

    q, reject = _correct(p, directed, alpha, correction)
    effect = inverse_fisher_z(mean_a) - inverse_fisher_z(mean_b)  # mean r difference

    extras = {
        "mean_diff_z": mean,
        "cohens_d": d,
        "hedges_g": g,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "mean_a": inverse_fisher_z(mean_a),
        "mean_b": inverse_fisher_z(mean_b),
    }
    return GroupStatResult(
        labels=labels, chromophore=a.chromophore, method=a.method, directed=directed,
        test=test, design=design,
        effect=effect, tstat=t, dof=dof, pvalues=p, qvalues=q, reject=reject,
        extras=extras,
        params={"alpha": alpha, "correction": correction, "equal_var": equal_var,
                "conf": conf, "n_a": n_a, "n_b": n_b,
                "n_subjects": (n_a if paired else n_a + n_b)},
    )


# --------------------------------------------------------------------------- #
# regression
# --------------------------------------------------------------------------- #

def regression(
    group: "GroupConnectivity",
    design,
    *,
    predictor: "str | int" = -1,
    alpha: float = 0.05,
    correction: str | None = "fdr_bh",
) -> GroupStatResult:
    """Edgewise OLS of FC (z) on a subject-level ``design``.

    ``design`` is a ``(n_subjects, n_predictors)`` array or a pandas DataFrame
    (numeric columns; an intercept is added automatically). ``predictor`` selects
    the column to test (name for a DataFrame, index otherwise; default last).
    Reports, per edge, the partial correlation (effect), beta, t (df = n - p),
    p, corrected q, and R^2.
    """
    z = group.stack(fisher=True)
    n_subj = z.shape[0]
    n = group.n
    directed = group.directed

    X, pred_col, pred_name = _build_design(design, n_subj, predictor)
    p_reg = X.shape[1]
    if n_subj - p_reg < 1:
        raise DataError(
            f"Regression needs n_subjects ({n_subj}) > n_predictors ({p_reg})."
        )

    beta_m = np.zeros((n, n))
    t_m = np.full((n, n), np.nan)
    dof_m = np.full((n, n), np.nan)
    p_m = np.full((n, n), np.nan)
    partial_m = np.zeros((n, n))
    r2_m = np.full((n, n), np.nan)
    se_m = np.full((n, n), np.nan)

    idx = _edge_index(n, directed)
    for i, j in zip(*idx):
        y = z[:, i, j]
        ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if ok.sum() <= p_reg:
            continue
        Xe, ye = X[ok], y[ok]
        ne = ok.sum()
        beta, _, _, _ = np.linalg.lstsq(Xe, ye, rcond=None)
        resid = ye - Xe @ beta
        rss = float(resid @ resid)
        dfe = ne - p_reg
        if dfe < 1:
            continue
        sigma2 = rss / dfe
        try:
            xtx_inv = np.linalg.inv(Xe.T @ Xe)
        except np.linalg.LinAlgError:
            xtx_inv = np.linalg.pinv(Xe.T @ Xe)
        se = np.sqrt(np.maximum(sigma2 * np.diag(xtx_inv), 0.0))
        b = beta[pred_col]
        seb = se[pred_col]
        tval = b / seb if seb > 0 else np.nan
        tss = float(((ye - ye.mean()) ** 2).sum())
        r2 = 1.0 - rss / tss if tss > 0 else np.nan
        # partial correlation of predictor with y given other regressors
        partial = np.sign(tval) * np.sqrt(tval**2 / (tval**2 + dfe)) if np.isfinite(tval) else np.nan
        pval = float(_two_sided_p(np.array(float(tval)), np.array(float(dfe))))

        coords = [(i, j)] if directed else [(i, j), (j, i)]
        for r_, c_ in coords:
            beta_m[r_, c_] = b
            t_m[r_, c_] = tval
            dof_m[r_, c_] = dfe
            p_m[r_, c_] = pval
            partial_m[r_, c_] = partial
            r2_m[r_, c_] = r2
            se_m[r_, c_] = seb

    q, reject = _correct(p_m, directed, alpha, correction)
    extras = {"beta": beta_m, "se": se_m, "r_squared": r2_m, "partial_r": partial_m}
    return GroupStatResult(
        labels=group.labels, chromophore=group.chromophore, method=group.method,
        directed=directed, test="regression", design="regression",
        effect=partial_m, tstat=t_m, dof=dof_m, pvalues=p_m, qvalues=q, reject=reject,
        extras=extras,
        params={"alpha": alpha, "correction": correction, "predictor": pred_name,
                "n_predictors": p_reg, "n_subjects": n_subj},
    )


def _build_design(design, n_subj, predictor):
    """Return (X with intercept, tested-column-index-in-X, predictor-name)."""
    try:
        import pandas as pd

        is_df = isinstance(design, pd.DataFrame)
    except ImportError:  # pragma: no cover
        is_df = False

    if is_df:
        numeric = design.select_dtypes(include="number")
        if numeric.shape[1] == 0:
            raise DataError("Design DataFrame has no numeric columns.")
        cols = list(numeric.columns)
        if isinstance(predictor, str):
            if predictor not in cols:
                raise DataError(f"Predictor {predictor!r} not in design {cols}.")
            pred_name = predictor
            pred_pos = cols.index(predictor)
        else:
            pred_pos = predictor % len(cols)
            pred_name = cols[pred_pos]
        mat = numeric.to_numpy(dtype=float)
    else:
        mat = np.asarray(design, dtype=float)
        if mat.ndim == 1:
            mat = mat[:, None]
        pred_pos = predictor % mat.shape[1]
        pred_name = f"x{pred_pos}"

    if mat.shape[0] != n_subj:
        raise DataError(
            f"Design has {mat.shape[0]} rows for {n_subj} subjects."
        )
    X = np.column_stack([np.ones(n_subj), mat])  # intercept first
    return X, pred_pos + 1, pred_name


# --------------------------------------------------------------------------- #
# Network-Based Statistic
# --------------------------------------------------------------------------- #

def _one_sample_t(stack: np.ndarray, mu: float = 0.0) -> np.ndarray:
    mean, sd, n = _moments(stack)
    with _suppress():
        return (mean - mu) / (sd / np.sqrt(n))


def _welch_t(za: np.ndarray, zb: np.ndarray) -> np.ndarray:
    mA, sA, nA = _moments(za)
    mB, sB, nB = _moments(zb)
    with _suppress():
        return (mA - mB) / np.sqrt(sA**2 / nA + sB**2 / nB)


def _components(supra: np.ndarray) -> list[np.ndarray]:
    """Connected components (>= 1 edge) of a symmetric suprathreshold mask."""
    from scipy.sparse.csgraph import connected_components

    n_comp, lab = connected_components(supra, directed=False)
    out = []
    for c in range(n_comp):
        nodes = np.where(lab == c)[0]
        if nodes.size < 2:
            continue
        sub = supra[np.ix_(nodes, nodes)]
        n_edges = int(sub[np.triu_indices(nodes.size, 1)].sum())
        if n_edges > 0:
            out.append(nodes)
    return out


def _max_component_size(supra: np.ndarray) -> int:
    best = 0
    for nodes in _components(supra):
        sub = supra[np.ix_(nodes, nodes)]
        best = max(best, int(sub[np.triu_indices(nodes.size, 1)].sum()))
    return best


def nbs(
    a: "GroupConnectivity",
    b: "GroupConnectivity | None" = None,
    *,
    threshold: float = 3.0,
    n_permutations: int = 5000,
    paired: bool = False,
    tail: str = "both",
    alpha: float = 0.05,
    seed: int | None = None,
) -> NBSResult:
    """Network-Based Statistic (Zalesky 2010): FWE inference on sub-networks.

    Thresholds the per-edge t at ``threshold`` (a t-value), finds connected
    components in the suprathreshold graph, and assigns each component a
    family-wise-error p-value from the permutation null of the largest component
    size. ``b=None`` runs a one-sample design (sign-flip permutation); with ``b``
    it runs two-sample (label shuffle) or paired (sign-flip of differences).

    Significance is at the *component* level, not the individual edge.
    """
    rng = np.random.default_rng(seed)

    if b is None:
        z = a.stack(fisher=True)
        labels = a.labels
        directed = a.directed
        t_obs = _one_sample_t(z)
        mode = "one_sample"
    else:
        da, db, labels = _align(a, b)
        directed = a.directed
        if paired:
            subs = [s for s in a.subjects if s in set(b.subjects)]
            za = fisher_z(da.sel(subject=subs).values)
            zb = fisher_z(db.sel(subject=subs).values)
            diff = za - zb
            t_obs = _one_sample_t(diff)
            mode = "paired"
        else:
            za = fisher_z(da.values)
            zb = fisher_z(db.values)
            t_obs = _welch_t(za, zb)
            mode = "two_sample"

    n = len(labels)

    def supra_of(t: np.ndarray) -> np.ndarray:
        with _suppress():
            if tail in ("both", "two-sided"):
                m = np.abs(t) >= threshold
            elif tail in ("greater", "right"):
                m = t >= threshold
            elif tail in ("less", "left"):
                m = t <= -threshold
            else:
                raise DataError(f"Unknown tail {tail!r}.")
        m = np.where(np.isfinite(t), m, False)
        np.fill_diagonal(m, False)
        if not directed:
            m = m | m.T
        return m

    obs_supra = supra_of(t_obs)
    comps = _components(obs_supra)
    comp_sizes = []
    comp_node_sets = []
    for nodes in comps:
        sub = obs_supra[np.ix_(nodes, nodes)]
        size = int(sub[np.triu_indices(nodes.size, 1)].sum())
        comp_sizes.append(size)
        comp_node_sets.append({labels[k] for k in nodes})
    comp_sizes = np.array(comp_sizes, dtype=float)

    # permutation null of the maximum component size
    null_max = np.zeros(n_permutations)
    for pi in range(n_permutations):
        if mode == "two_sample":
            za_all = fisher_z(da.values)
            zb_all = fisher_z(db.values)
            pooled = np.concatenate([za_all, zb_all], axis=0)
            nA = za_all.shape[0]
            perm = rng.permutation(pooled.shape[0])
            t_perm = _welch_t(pooled[perm[:nA]], pooled[perm[nA:]])
        else:  # sign-flip one-sample / paired
            stack = z if b is None else diff
            signs = rng.choice([-1.0, 1.0], size=stack.shape[0])[:, None, None]
            t_perm = _one_sample_t(stack * signs)
        null_max[pi] = _max_component_size(supra_of(t_perm))

    if comp_sizes.size:
        comp_p = np.array([
            (1 + np.sum(null_max >= s)) / (n_permutations + 1) for s in comp_sizes
        ])
        order = np.argsort(-comp_sizes)
        comp_sizes = comp_sizes[order]
        comp_p = comp_p[order]
        comp_node_sets = [comp_node_sets[k] for k in order]
    else:
        comp_p = np.array([])

    return NBSResult(
        labels=labels, chromophore=a.chromophore, components=comp_node_sets,
        component_pvalues=comp_p, component_sizes=comp_sizes,
        component_stats=comp_sizes, edge_stat=t_obs, threshold=threshold,
        n_permutations=n_permutations, directed=directed,
        params={"alpha": alpha, "mode": mode, "tail": tail},
    )
