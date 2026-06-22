"""Poster-style multi-panel figure summarising a whole connectivity analysis.

:func:`build_poster` assembles one figure that reads top-to-bottom as the
pipeline: metadata banner -> connectivity matrix -> significance -> edge-weight
distribution -> global graph metrics -> hub nodes -> circular connectogram ->
table of top findings. It works on a subject-level
:class:`~nirconn.core.result.ConnectivityMatrix`/``ConnectivityResult`` or a
group-level :class:`~nirconn.stats._results.GroupStatResult`.

matplotlib is imported lazily and panels degrade gracefully (missing p-values,
missing networkx, tiny networks) so the function never hard-fails on partial
input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

__all__ = ["build_poster"]


def _resolve(result, group_mean):
    """Normalise input into (display_matrix, stat_or_None, primary_label)."""
    from ..core.result import ConnectivityMatrix, ConnectivityResult
    from ..stats._results import GroupStatResult

    if isinstance(result, GroupStatResult):
        matrix = group_mean if group_mean is not None else result.to_matrix()
        return matrix, result, f"{result.method}:{result.test}"
    if isinstance(result, ConnectivityResult):
        m = result["HbO"] if "HbO" in result.chromophores else next(iter(result.values()))
        return m, None, result.method
    if isinstance(result, ConnectivityMatrix):
        return result, None, result.method
    raise TypeError(
        f"build_poster expects a ConnectivityMatrix, ConnectivityResult or "
        f"GroupStatResult, got {type(result)!r}."
    )


def build_poster(
    result,
    *,
    group_mean=None,
    alpha: float = 0.05,
    correction: str | None = "fdr_bh",
    threshold: float = 0.5,
    top_n: int = 8,
    title: str | None = None,
    figsize: tuple[float, float] = (15, 19),
) -> "Figure":
    """Render the full-pipeline poster figure and return the matplotlib Figure.

    Parameters
    ----------
    result:
        A ``ConnectivityMatrix``, ``ConnectivityResult`` or ``GroupStatResult``.
    group_mean:
        Optional group-mean ``ConnectivityMatrix`` to display instead of a stat
        result's effect matrix.
    alpha, correction:
        Significance level and multiple-comparison method for the masks/tables.
    threshold:
        Absolute weight threshold for the graph-theory panels.
    top_n:
        Rows in the embedded findings table / chords in the connectogram.
    """
    import matplotlib.pyplot as plt

    matrix, stat, primary = _resolve(result, group_mean)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        5, 2,
        height_ratios=[0.32, 1.25, 0.95, 1.15, 0.95],
        hspace=0.45, wspace=0.22,
        left=0.06, right=0.96, top=0.95, bottom=0.04,
    )

    _panel_banner(fig.add_subplot(gs[0, :]), matrix, stat, alpha, correction, title, primary)
    _panel_matrix(fig.add_subplot(gs[1, 0]), matrix)
    _panel_significance(fig.add_subplot(gs[1, 1]), matrix, stat, alpha, correction)
    _panel_distribution(fig.add_subplot(gs[2, 0]), matrix, threshold)
    _panel_global_graph(fig.add_subplot(gs[2, 1]), matrix, threshold)
    _panel_hubs(fig.add_subplot(gs[3, 0]), matrix, threshold)
    _panel_connectogram(fig.add_subplot(gs[3, 1]), matrix, stat, alpha, correction, top_n)
    _panel_table(fig.add_subplot(gs[4, :]), result, matrix, stat, alpha, correction, top_n)

    return fig


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #

def _sig_mask(matrix, stat, alpha, correction):
    """Return a significance mask (True = significant) or None."""
    if stat is not None:
        return stat.significant(alpha)
    if matrix.pvalues is not None:
        try:
            return matrix.significant(alpha, correction=correction)
        except Exception:
            return None
    return None


def _panel_banner(ax, matrix, stat, alpha, correction, title, primary):
    from .._version import __version__

    ax.axis("off")
    ax.axhspan(0.78, 1.0, color="#2c3e50", transform=ax.transAxes)
    head = title or (
        f"fNIRS connectivity poster — {primary}"
        + (f" ({stat.design})" if stat is not None else "")
    )
    ax.text(0.5, 0.89, head, transform=ax.transAxes, ha="center", va="center",
            fontsize=16, color="white", fontweight="bold")

    bits = [f"method: {matrix.method}", f"chromophore: {matrix.chromophore}",
            f"channels: {matrix.n}"]
    if stat is not None:
        s = stat.summary()
        bits += [f"subjects: {s.get('n_subjects')}", f"test: {stat.test}",
                 f"significant: {s['n_significant']}/{s['n_edges_tested']}"]
    bits += [f"alpha: {alpha}", f"correction: {correction}", f"nirconn {__version__}"]
    ax.text(0.5, 0.35, "   |   ".join(str(b) for b in bits), transform=ax.transAxes,
            ha="center", va="center", fontsize=9.5, color="#222")


def _panel_matrix(ax, matrix):
    from .matrix import plot_matrix

    plot_matrix(matrix, ax=ax, colorbar=True,
                title=f"Connectivity ({matrix.chromophore})")


def _panel_significance(ax, matrix, stat, alpha, correction):
    from .matrix import plot_matrix

    mask = _sig_mask(matrix, stat, alpha, correction)
    if mask is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "No p-values available\n(significance panel skipped)",
                ha="center", va="center", fontsize=11, color="#888",
                transform=ax.transAxes)
        return
    plot_matrix(matrix, ax=ax, mask=~mask, colorbar=True,
                title=f"Significant edges (a={alpha}, {correction})")


def _panel_distribution(ax, matrix, threshold):
    n = matrix.n
    off = matrix.values[~np.eye(n, dtype=bool)]
    off = off[np.isfinite(off)]
    if off.size == 0:
        ax.axis("off")
        return
    ax.hist(off, bins=40, color="#5b8def", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(float(np.mean(off)), color="#2c3e50", lw=1.5, ls="--",
               label=f"mean={np.mean(off):.2f}")
    ax.axvline(threshold, color="#c0392b", lw=1.2, ls=":", label=f"thr={threshold}")
    ax.axvline(-threshold, color="#c0392b", lw=1.2, ls=":")
    ax.set_title("Edge-weight distribution")
    ax.set_xlabel("edge weight")
    ax.set_ylabel("count")
    ax.legend(fontsize=8, frameon=False)


def _panel_global_graph(ax, matrix, threshold):
    try:
        from ..graph.metrics import graph_metrics

        gm = graph_metrics(matrix, threshold=threshold)
    except Exception as exc:  # networkx missing or degenerate
        ax.axis("off")
        ax.text(0.5, 0.5, f"Graph metrics unavailable\n({exc})", ha="center",
                va="center", fontsize=10, color="#888", transform=ax.transAxes)
        return
    keys = ["density", "global_efficiency", "average_clustering", "transitivity",
            "modularity"]
    vals = [gm.get(k, np.nan) for k in keys]
    y = np.arange(len(keys))
    ax.barh(y, vals, color="#16a085", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([k.replace("_", " ") for k in keys], fontsize=9)
    ax.invert_yaxis()
    for yi, v in zip(y, vals):
        if np.isfinite(v):
            ax.text(v, yi, f" {v:.3f}", va="center", fontsize=8)
    ax.set_title(
        f"Global graph metrics  (|r|>={threshold}; "
        f"{gm.get('n_edges', 0)} edges, {gm.get('n_communities', 0)} comm.)"
    )
    ax.set_xlim(0, max(1.0, max([v for v in vals if np.isfinite(v)], default=1.0) * 1.15))


def _panel_hubs(ax, matrix, threshold, top_k: int = 10):
    try:
        from ..graph.metrics import nodal_metrics

        nm = nodal_metrics(matrix, threshold=threshold)
    except Exception as exc:
        ax.axis("off")
        ax.text(0.5, 0.5, f"Nodal metrics unavailable\n({exc})", ha="center",
                va="center", fontsize=10, color="#888", transform=ax.transAxes)
        return
    strength = nm["degree"]
    items = sorted(strength.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    if not items:
        ax.axis("off")
        ax.text(0.5, 0.5, "No supra-threshold nodes", ha="center", va="center",
                fontsize=10, color="#888", transform=ax.transAxes)
        return
    names = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    y = np.arange(len(names))
    ax.barh(y, vals, color="#e67e22", alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title(f"Hub nodes by strength (top {len(items)})")
    ax.set_xlabel("weighted degree")


def _panel_connectogram(ax, matrix, stat, alpha, correction, top_n):
    """Circular connectogram of the strongest (or significant) edges."""
    labels = matrix.labels
    n = matrix.n
    vals = matrix.values
    ax.set_aspect("equal")
    ax.axis("off")

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs, ys = np.cos(angles), np.sin(angles)

    # Choose edges: significant if available, else top-|weight|.
    mask = _sig_mask(matrix, stat, alpha, correction)
    iu = np.triu_indices(n, 1)
    weights = vals[iu]
    pairs = list(zip(iu[0], iu[1], weights))
    if mask is not None and mask[iu].any():
        pairs = [(i, j, w) for i, j, w in pairs if mask[i, j] and np.isfinite(w)]
        pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    else:
        pairs = [p for p in pairs if np.isfinite(p[2])]
        pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    pairs = pairs[: max(top_n * 3, top_n)]

    wmax = max((abs(w) for _, _, w in pairs), default=1.0) or 1.0
    for i, j, w in pairs:
        color = "#c0392b" if w >= 0 else "#2c6fbb"
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=color,
                alpha=min(0.9, 0.2 + 0.8 * abs(w) / wmax),
                lw=0.5 + 2.5 * abs(w) / wmax, zorder=1)
    ax.scatter(xs, ys, s=26, color="#34495e", zorder=2)

    if n <= 40:
        for k, (x, y) in enumerate(zip(xs, ys)):
            rot = np.degrees(angles[k])
            ha = "left" if x >= 0 else "right"
            if x < 0:
                rot += 180
            ax.text(1.08 * x, 1.08 * y, labels[k], fontsize=6.5, rotation=rot,
                    rotation_mode="anchor", ha=ha, va="center")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    kind = "significant" if (mask is not None and mask[iu].any()) else "strongest"
    ax.set_title(f"Connectogram ({kind} edges)")


def _panel_table(ax, result, matrix, stat, alpha, correction, top_n):
    from ..tables import significant_edges_table

    ax.axis("off")
    src = stat if stat is not None else matrix
    try:
        df = significant_edges_table(
            src, alpha=alpha, correction=correction, sort="q" if stat else "abs_effect",
            include_nonsignificant=True, top_n=top_n,
        )
    except Exception as exc:  # pragma: no cover
        ax.text(0.5, 0.5, f"Findings table unavailable ({exc})", ha="center",
                va="center", transform=ax.transAxes)
        return
    if df.empty:
        ax.text(0.5, 0.5, "No edges to report", ha="center", va="center",
                transform=ax.transAxes)
        return

    # Choose compact columns that exist.
    prefer = ["edge", "effect", "t", "p", "q", "cohens_d", "significant", "direction"]
    cols = [c for c in prefer if c in df.columns]
    disp = df[cols].copy()
    for c in disp.columns:
        if disp[c].dtype.kind == "f":
            disp[c] = disp[c].map(lambda v: "" if v != v else f"{v:.3g}")
        else:
            disp[c] = disp[c].astype(str)

    tbl = ax.table(cellText=disp.values, colLabels=disp.columns, loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)
    # header styling + zebra + bold significant rows
    sig_col = list(df.columns).index("significant") if "significant" in df.columns else None
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            row = df.iloc[r - 1]
            is_sig = bool(row["significant"]) if "significant" in df.columns else False
            cell.set_facecolor("#eaf2fb" if is_sig else ("#f7f7f7" if r % 2 else "white"))
    ax.set_title(f"Top findings (n={min(top_n, len(df))})", fontsize=11, pad=8)
