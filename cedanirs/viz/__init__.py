"""Visualisation of connectivity results.

Matplotlib is imported lazily (inside the plotting functions) so that importing
cedanirs never forces a backend or a windowing system to load -- important for
headless servers and for the eventual use of this package as a compute backend
behind a GUI.

Shipped:

* :func:`~cedanirs.viz.matrix.plot_matrix` -- a labelled heatmap of a
  connectivity matrix.

Planned: circular/chord connectogram, network/node-link layout, glass-brain
projection onto montage coordinates.
"""

from __future__ import annotations

from .matrix import plot_matrix
from .poster import build_poster

__all__ = ["plot_matrix", "build_poster"]
