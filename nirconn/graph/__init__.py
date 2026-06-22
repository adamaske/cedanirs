"""Graph-theoretic analysis of connectivity matrices.

Treating channels/regions as nodes and connections as weighted edges lets us
describe the *topology* of an fNIRS network: its efficiency, segregation, and
which nodes act as hubs. This subpackage wraps :mod:`networkx` (a standard,
well-tested graph library -- no need to reimplement graph algorithms) behind a
small, connectivity-aware API.

Shipped:

* :func:`~nirconn.graph.metrics.to_graph` -- matrix → :mod:`networkx` graph.
* :func:`~nirconn.graph.metrics.graph_metrics` -- global + nodal metrics.
"""

from __future__ import annotations

from .metrics import graph_metrics, nodal_metrics, to_graph

__all__ = ["to_graph", "graph_metrics", "nodal_metrics"]
