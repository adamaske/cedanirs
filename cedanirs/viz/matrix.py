"""Heatmap rendering of a connectivity matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes

    from ..core.result import ConnectivityMatrix


def plot_matrix(
    matrix,
    *,
    ax: "Axes | None" = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    labels: bool = True,
    colorbar: bool = True,
    title: str | None = None,
    mask: np.ndarray | None = None,
    annotate: bool = False,
    grid: bool = True,
) -> "Axes":
    """Render a connectivity matrix as a labelled heatmap.

    Parameters
    ----------
    matrix:
        A :class:`~cedanirs.core.result.ConnectivityMatrix`, a NumPy array, or
        anything array-like.
    ax:
        Existing axes to draw on; a new figure/axes is created if omitted.
    cmap:
        Colormap. Defaults to a diverging ``"RdBu_r"`` for correlation-like
        matrices (sensible for the ``[-1, 1]`` range), centred at zero.
    vmin, vmax:
        Colour limits. Default to a symmetric range around zero inferred from
        the data (or ``[-1, 1]`` for correlations).
    labels:
        Draw channel tick labels (auto-hidden when there are too many).
    mask:
        Optional boolean array; ``True`` entries are hidden (e.g. show only the
        significant edges via ``matrix.significant()``).
    annotate:
        Write each cell's value into the heatmap (only sensible for small N).

    Returns
    -------
    matplotlib.axes.Axes
        The axes the heatmap was drawn on.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    from ..core.result import ConnectivityMatrix

    is_cm = isinstance(matrix, ConnectivityMatrix)
    data = np.asarray(matrix.values if is_cm else matrix, dtype=float)
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {data.shape}.")
    n = data.shape[0]

    tick_labels = matrix.labels if is_cm else [str(i) for i in range(n)]
    looks_like_corr = np.nanmin(data) >= -1.0001 and np.nanmax(data) <= 1.0001

    if cmap is None:
        cmap = "RdBu_r"
    if vmin is None and vmax is None:
        if looks_like_corr:
            vmin, vmax = -1.0, 1.0
        else:
            amax = float(np.nanmax(np.abs(data))) or 1.0
            vmin, vmax = -amax, amax

    display = data.copy()
    if mask is not None:
        display = np.where(np.asarray(mask, dtype=bool), np.nan, display)

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, n * 0.35), max(4, n * 0.35)))

    norm = None
    if vmin is not None and vmax is not None and vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax.imshow(
        display,
        cmap=cmap,
        norm=norm,
        vmin=None if norm else vmin,
        vmax=None if norm else vmax,
        interpolation="nearest",
        aspect="equal",
    )

    # Tick labels: hide automatically when too dense to be readable.
    if labels and n <= 40:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
        ax.set_yticklabels(tick_labels, fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel("target" if is_cm and matrix.directed else "channel")
    ax.set_ylabel("source" if is_cm and matrix.directed else "channel")

    if grid and n <= 40:
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", length=0)

    if annotate and n <= 20:
        for i in range(n):
            for j in range(n):
                v = display[i, j]
                if np.isfinite(v):
                    ax.text(
                        j, i, f"{v:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if abs(v) < 0.6 else "white",
                    )

    if title is None and is_cm:
        chrom = f" — {matrix.chromophore}" if matrix.chromophore else ""
        title = f"{matrix.method}{chrom}"
    if title:
        ax.set_title(title)

    if colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return ax
