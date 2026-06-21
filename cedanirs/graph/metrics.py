"""Graph construction and graph-theory metrics, backed by networkx."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.exceptions import DependencyError

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

    from ..core.result import ConnectivityMatrix


def _require_networkx():
    try:
        import networkx as nx  # noqa: F401

        return nx
    except ImportError as exc:  # pragma: no cover - exercised only without nx
        raise DependencyError("Graph analysis", "networkx") from exc


def _as_values_labels(matrix):
    from ..core.result import ConnectivityMatrix

    if isinstance(matrix, ConnectivityMatrix):
        return matrix.values, matrix.labels, matrix.directed
    arr = np.asarray(matrix, dtype=float)
    return arr, [str(i) for i in range(arr.shape[0])], not _is_symmetric(arr)


def _is_symmetric(arr: np.ndarray, tol: float = 1e-8) -> bool:
    return arr.shape[0] == arr.shape[1] and np.allclose(
        arr, arr.T, atol=tol, equal_nan=True
    )


def to_graph(
    matrix,
    *,
    threshold: float | None = None,
    absolute: bool = True,
    weight_attr: str = "weight",
) -> "nx.Graph":
    """Convert a connectivity matrix into a :mod:`networkx` graph.

    Self-loops (the diagonal) and non-finite / exactly-zero weights are dropped.
    A :class:`~networkx.DiGraph` is built for directed matrices, otherwise a
    :class:`~networkx.Graph`.

    Parameters
    ----------
    threshold:
        Drop edges whose (absolute, by default) weight is below this value.
    weight_attr:
        Name of the edge attribute the weight is stored under.
    """
    nx = _require_networkx()
    values, labels, directed = _as_values_labels(matrix)
    n = values.shape[0]

    g = nx.DiGraph() if directed else nx.Graph()
    g.add_nodes_from(labels)

    for i in range(n):
        jrange = range(n) if directed else range(i + 1, n)
        for j in jrange:
            if i == j:
                continue
            w = values[i, j]
            if not np.isfinite(w) or w == 0.0:
                continue
            mag = abs(w) if absolute else w
            if threshold is not None and mag < threshold:
                continue
            g.add_edge(labels[i], labels[j], **{weight_attr: float(w)})
    return g


def graph_metrics(
    matrix,
    *,
    threshold: float | None = None,
    absolute: bool = True,
    use_abs_weights: bool = True,
) -> dict:
    """Compute a standard battery of global graph-theory metrics.

    Returns a dict with global metrics (``density``, ``global_efficiency``,
    ``average_clustering``, ``transitivity``, ``modularity``,
    ``n_nodes``, ``n_edges``) plus the nodal metrics from :func:`nodal_metrics`
    under ``"nodal"``.

    ``use_abs_weights`` takes the absolute value of edge weights first, which is
    the usual choice for correlation networks where a strong anti-correlation is
    still a strong connection.
    """
    nx = _require_networkx()
    g = to_graph(matrix, threshold=threshold, absolute=absolute)

    if use_abs_weights:
        for _, _, d in g.edges(data=True):
            d["abs_weight"] = abs(d.get("weight", 0.0))
        weight = "abs_weight"
    else:
        weight = "weight"

    directed = g.is_directed()
    out: dict = {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "density": nx.density(g),
    }

    # global_efficiency / clustering are defined for undirected graphs.
    ug = g.to_undirected() if directed else g
    try:
        out["global_efficiency"] = nx.global_efficiency(ug)
    except Exception:  # pragma: no cover - tiny/empty graphs
        out["global_efficiency"] = float("nan")
    out["average_clustering"] = nx.average_clustering(ug, weight=weight)
    out["transitivity"] = nx.transitivity(ug)

    try:
        communities = nx.community.greedy_modularity_communities(ug, weight=weight)
        out["modularity"] = nx.community.modularity(ug, communities, weight=weight)
        out["n_communities"] = len(communities)
    except Exception:  # pragma: no cover - degenerate graphs
        out["modularity"] = float("nan")
        out["n_communities"] = 0

    out["nodal"] = nodal_metrics(matrix, threshold=threshold, absolute=absolute,
                                 use_abs_weights=use_abs_weights)
    return out


def nodal_metrics(
    matrix,
    *,
    threshold: float | None = None,
    absolute: bool = True,
    use_abs_weights: bool = True,
) -> dict:
    """Per-node centrality metrics, keyed by metric then node label.

    Includes ``degree`` (strength), ``degree_centrality``,
    ``betweenness_centrality`` and ``clustering``. These locate the hubs that
    are critical for network communication.
    """
    nx = _require_networkx()
    g = to_graph(matrix, threshold=threshold, absolute=absolute)

    if use_abs_weights:
        for _, _, d in g.edges(data=True):
            d["abs_weight"] = abs(d.get("weight", 0.0))
        weight = "abs_weight"
    else:
        weight = "weight"

    ug = g.to_undirected() if g.is_directed() else g
    strength = dict(g.degree(weight=weight))
    return {
        "degree": strength,
        "degree_centrality": nx.degree_centrality(g),
        "betweenness_centrality": nx.betweenness_centrality(g, weight=weight),
        "clustering": nx.clustering(ug, weight=weight),
    }
