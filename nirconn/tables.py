"""Significant-findings and summary tables as tidy :class:`pandas.DataFrame` s.

These turn the rich result objects into the tabular output a scientist actually
puts in a paper or supplement: one row per edge / node / component, with the
statistics laid out in columns, sortable and exportable to CSV or Markdown
(``df.to_csv`` / ``df.to_markdown``) and embeddable in the poster figure.

The functions accept either a subject-level
:class:`~nirconn.core.result.ConnectivityMatrix` or a group-level
:class:`~nirconn.stats._results.GroupStatResult`, so the same call produces the
right table at either level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from .core.result import ConnectivityMatrix
    from .stats._results import GroupStatResult, NBSResult

__all__ = [
    "significant_edges_table",
    "nodal_summary_table",
    "global_graph_table",
    "group_summary_table",
    "nbs_components_table",
]


def _edge_iter(n: int, directed: bool):
    if directed:
        for i in range(n):
            for j in range(n):
                if i != j:
                    yield i, j
    else:
        for i in range(n):
            for j in range(i + 1, n):
                yield i, j


def significant_edges_table(
    result,
    *,
    alpha: float | None = None,
    correction: str | None = None,
    sort: str = "q",
    include_nonsignificant: bool = True,
    top_n: int | None = None,
) -> "pd.DataFrame":
    """Per-edge findings table for a group result or a single matrix.

    Columns (group): ``chromophore, source, target, edge, effect, t, df, p, q,
    ci_low, ci_high, cohens_d, n, significant, direction``. For a plain
    :class:`ConnectivityMatrix` the statistic columns hold the correlation and
    its analytic p/q.

    ``sort`` is ``"q"`` (corrected p, ascending), ``"effect"`` / ``"abs_effect"``
    (descending), or ``"p"``. Significant edges are always floated to the top.
    Pass ``top_n`` to keep only the leading rows.
    """
    import pandas as pd

    from .core.result import ConnectivityMatrix
    from .stats._results import GroupStatResult

    if isinstance(result, GroupStatResult):
        df = _group_edges_df(result, alpha=alpha, correction=correction)
    elif isinstance(result, ConnectivityMatrix):
        df = _matrix_edges_df(result, alpha=alpha, correction=correction)
    else:
        raise TypeError(
            f"Expected GroupStatResult or ConnectivityMatrix, got {type(result)!r}."
        )

    if not include_nonsignificant:
        df = df[df["significant"]]

    # Sort: significant first, then by the chosen key.
    ascending = sort in ("q", "p")
    key = {"abs_effect": "effect", "effect": "effect"}.get(sort, sort)
    if key == "effect":
        df = df.assign(_abs=df["effect"].abs())
        df = df.sort_values(
            ["significant", "_abs"], ascending=[False, False]
        ).drop(columns="_abs")
    elif key in df.columns:
        df = df.sort_values(["significant", key], ascending=[False, ascending])
    df = df.reset_index(drop=True)

    if top_n is not None:
        df = df.head(top_n)
    return df


def _group_edges_df(result: "GroupStatResult", *, alpha, correction) -> "pd.DataFrame":
    import pandas as pd

    if alpha is not None or correction is not None:
        from .stats.significance import correct_pvalues

        a = alpha if alpha is not None else result.params.get("alpha", 0.05)
        c = correction if correction is not None else result.params.get("correction")
        idx = (
            np.where(~np.eye(result.n, dtype=bool))
            if result.directed
            else np.triu_indices(result.n, 1)
        )
        reject_flat, q_flat = correct_pvalues(result.pvalues[idx], alpha=a, method=c)
        q = np.full((result.n, result.n), np.nan)
        reject = np.zeros((result.n, result.n), dtype=bool)
        q[idx] = q_flat
        reject[idx] = reject_flat
        if not result.directed:
            q = np.where(np.isnan(q), q.T, q)
            reject = reject | reject.T
    else:
        a = result.params.get("alpha", 0.05)
        q, reject = result.qvalues, result.reject

    labels = result.labels
    ex = result.extras
    dof = result.dof
    dof_arr = dof if isinstance(dof, np.ndarray) else None
    rows = []
    for i, j in _edge_iter(result.n, result.directed):
        if not np.isfinite(result.pvalues[i, j]):
            continue
        row = {
            "chromophore": result.chromophore,
            "source": labels[i],
            "target": labels[j],
            "edge": f"{labels[i]}{'->' if result.directed else '--'}{labels[j]}",
            "effect": float(result.effect[i, j]),
            "t": float(result.tstat[i, j]),
            "df": float(dof_arr[i, j]) if dof_arr is not None else float(dof),
            "p": float(result.pvalues[i, j]),
            "q": float(q[i, j]),
            "ci_low": float(ex["ci_low"][i, j]) if "ci_low" in ex else np.nan,
            "ci_high": float(ex["ci_high"][i, j]) if "ci_high" in ex else np.nan,
            "cohens_d": float(ex["cohens_d"][i, j]) if "cohens_d" in ex else np.nan,
            "n": int(ex["n"][i, j]) if "n" in ex else result.params.get("n_subjects", np.nan),
            "significant": bool(reject[i, j]),
            "direction": "+" if result.effect[i, j] >= 0 else "-",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _matrix_edges_df(matrix: "ConnectivityMatrix", *, alpha, correction) -> "pd.DataFrame":
    import pandas as pd

    from .stats.significance import correct_pvalues
    from .stats.transforms import fisher_z

    a = alpha if alpha is not None else 0.05
    c = correction if correction is not None else "fdr_bh"
    n = matrix.n
    vals = matrix.values
    p = matrix.pvalues
    idx = np.where(~np.eye(n, dtype=bool)) if matrix.directed else np.triu_indices(n, 1)

    q_full = np.full((n, n), np.nan)
    reject_full = np.zeros((n, n), dtype=bool)
    if p is not None:
        reject_flat, q_flat = correct_pvalues(p[idx], alpha=a, method=c)
        q_full[idx] = q_flat
        reject_full[idx] = reject_flat
        if not matrix.directed:
            q_full = np.where(np.isnan(q_full), q_full.T, q_full)
            reject_full = reject_full | reject_full.T

    labels = matrix.labels
    rows = []
    for i, j in _edge_iter(n, matrix.directed):
        w = vals[i, j]
        if not np.isfinite(w):
            continue
        rows.append({
            "chromophore": matrix.chromophore,
            "source": labels[i],
            "target": labels[j],
            "edge": f"{labels[i]}{'->' if matrix.directed else '--'}{labels[j]}",
            "effect": float(w),
            "fisher_z": float(fisher_z(w)),
            "t": np.nan,
            "df": np.nan,
            "p": float(p[i, j]) if p is not None else np.nan,
            "q": float(q_full[i, j]) if p is not None else np.nan,
            "significant": bool(reject_full[i, j]),
            "direction": "+" if w >= 0 else "-",
        })
    return pd.DataFrame(rows)


def nodal_summary_table(
    matrix: "ConnectivityMatrix", *, threshold: float | None = None
) -> "pd.DataFrame":
    """One row per node: strength, centralities, clustering, hub flag, rank."""
    import pandas as pd

    from .graph.metrics import nodal_metrics

    nm = nodal_metrics(matrix, threshold=threshold)
    labels = matrix.labels
    strength = nm["degree"]
    rows = []
    for node in labels:
        rows.append({
            "chromophore": matrix.chromophore,
            "node": node,
            "strength": float(strength.get(node, 0.0)),
            "degree_centrality": float(nm["degree_centrality"].get(node, 0.0)),
            "betweenness_centrality": float(nm["betweenness_centrality"].get(node, 0.0)),
            "clustering": float(nm["clustering"].get(node, 0.0)),
        })
    df = pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    if len(df):
        cutoff = df["strength"].quantile(0.9)
        df["is_hub"] = df["strength"] >= cutoff
    else:
        df["is_hub"] = []
    return df


def global_graph_table(
    matrix: "ConnectivityMatrix", *, threshold: float | None = None
) -> "pd.DataFrame":
    """One-row global graph-metric summary for a matrix."""
    import pandas as pd

    from .graph.metrics import graph_metrics

    gm = graph_metrics(matrix, threshold=threshold)
    keys = [
        "n_nodes", "n_edges", "density", "global_efficiency",
        "average_clustering", "transitivity", "modularity", "n_communities",
    ]
    row = {"chromophore": matrix.chromophore}
    row.update({k: gm.get(k) for k in keys})
    row["threshold"] = threshold
    return pd.DataFrame([row])


def group_summary_table(study_or_group) -> "pd.DataFrame":
    """Sample/coverage summary: n_subjects, n_channels, n_edges per cell."""
    import pandas as pd

    from .core.group import GroupConnectivity, Study

    rows = []
    if isinstance(study_or_group, GroupConnectivity):
        g = study_or_group
        n = g.n
        rows.append({
            "chromophore": g.chromophore, "method": g.method,
            "n_subjects": g.n_subjects, "n_channels": n,
            "n_edges": n * (n - 1) // (1 if g.directed else 2),
        })
    elif isinstance(study_or_group, Study):
        study = study_or_group
        for chrom in study.chromophores:
            try:
                g = study.group(chrom)
            except Exception:
                continue
            n = g.n
            rows.append({
                "chromophore": chrom, "method": g.method,
                "n_subjects": g.n_subjects, "n_channels": n,
                "n_edges": n * (n - 1) // (1 if g.directed else 2),
            })
    else:
        raise TypeError("Expected a Study or GroupConnectivity.")
    return pd.DataFrame(rows)


def nbs_components_table(nbs: "NBSResult") -> "pd.DataFrame":
    """One row per NBS component: size, FWE p, member nodes."""
    import pandas as pd

    rows = []
    for k, (comp, p, size) in enumerate(
        zip(nbs.components, nbs.component_pvalues, nbs.component_sizes), start=1
    ):
        nodes = sorted(comp)
        rows.append({
            "component": k,
            "chromophore": nbs.chromophore,
            "n_nodes": len(nodes),
            "n_edges": int(size),
            "component_statistic": float(size),
            "p_fwe": float(p),
            "n_permutations": nbs.n_permutations,
            "threshold": nbs.threshold,
            "significant": bool(p <= nbs.params.get("alpha", 0.05)),
            "nodes": ", ".join(nodes),
        })
    return pd.DataFrame(rows)
