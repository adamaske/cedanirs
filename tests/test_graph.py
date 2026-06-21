import networkx as nx
import numpy as np
import pytest

import cedanirs as cn
from cedanirs.graph import graph_metrics, nodal_metrics, to_graph


@pytest.fixture
def matrix(correlated_data):
    data, labels = correlated_data
    return cn.connectivity(data, channels=labels).matrix


def test_to_graph_basic(matrix):
    g = to_graph(matrix, threshold=0.5)
    assert isinstance(g, nx.Graph)
    assert not g.is_directed()
    assert set(g.nodes) == set(matrix.labels)
    # No self-loops.
    assert nx.number_of_selfloops(g) == 0
    # A-B and A-D survive a 0.5 threshold; C is isolated.
    assert g.has_edge("A", "B")
    assert g.has_edge("A", "D")
    assert g.degree("C") == 0


def test_to_graph_directed_for_asymmetric():
    vals = np.array([[1.0, 0.8, 0.0], [0.1, 1.0, 0.6], [0.0, 0.0, 1.0]])
    m = cn.ConnectivityMatrix(vals, ["a", "b", "c"], method="x", directed=True)
    g = to_graph(m)
    assert g.is_directed()
    assert g.has_edge("a", "b")
    assert not g.has_edge("b", "a") or g["a"]["b"]["weight"] != g["b"]["a"]["weight"]


def test_graph_metrics_keys(matrix):
    gm = graph_metrics(matrix, threshold=0.5)
    for key in (
        "n_nodes",
        "n_edges",
        "density",
        "global_efficiency",
        "average_clustering",
        "transitivity",
        "modularity",
        "nodal",
    ):
        assert key in gm
    assert gm["n_nodes"] == 4


def test_nodal_metrics(matrix):
    nm = nodal_metrics(matrix, threshold=0.5)
    assert set(nm) == {
        "degree",
        "degree_centrality",
        "betweenness_centrality",
        "clustering",
    }
    assert set(nm["degree"]) == set(matrix.labels)


def test_to_graph_from_plain_array():
    vals = np.array([[1.0, 0.9], [0.9, 1.0]])
    g = to_graph(vals, threshold=0.5)
    assert g.number_of_edges() == 1
