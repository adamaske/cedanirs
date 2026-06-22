"""Human-readable reporting of connectivity results.

The first implementation produces a plain-text report suitable for logs, the
console, or a methods section. The structure (a sequence of sections rendered
from the result's metadata and summaries) is deliberately format-agnostic so
that richer renderers -- Markdown, HTML via Jinja, a PDF figure panel -- can be
added later without changing the data the report is built from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..core.result import ConnectivityMatrix, ConnectivityResult

__all__ = ["summarize_result", "result_report", "summarize_group", "summarize_study"]


def __getattr__(name):
    # Lazily expose the group-report functions without importing pandas eagerly.
    if name in ("summarize_group", "summarize_study"):
        from . import group

        return getattr(group, name)
    raise AttributeError(name)


def summarize_result(
    result: "ConnectivityResult",
    *,
    alpha: float = 0.05,
    correction: str | None = "fdr_bh",
    top_n: int = 5,
) -> str:
    """Render a connectivity result as a formatted text report.

    Reports, per chromophore: matrix dimensions, descriptive statistics over the
    edges, the number of significant connections (after multiple-comparison
    ``correction``) and the strongest connections.
    """
    lines: list[str] = []
    prov = result.provenance
    lines.append("=" * 64)
    lines.append(f"  Connectivity report - method: {result.method}")
    lines.append("=" * 64)
    lines.append(f"  kind            : {result.kind}")
    lines.append(f"  directed        : {result.directed}")
    if prov:
        lines.append(
            f"  data            : {prov.get('n_channels', '?')} channels × "
            f"{prov.get('n_times', '?')} samples"
            + (f" @ {prov['sfreq']:g} Hz" if prov.get("sfreq") else "")
        )
        if prov.get("name"):
            lines.append(f"  recording       : {prov['name']}")
    if result.params:
        params = ", ".join(f"{k}={v}" for k, v in result.params.items())
        lines.append(f"  parameters      : {params}")
    lines.append("")

    for chrom, m in result.items():
        s = m.summary()
        lines.append(f"  [{chrom}]")
        lines.append(
            f"    edges         : {s['n_edges']}  "
            f"(mean={s['mean']:.3f}, sd={s['std']:.3f}, "
            f"min={s['min']:.3f}, max={s['max']:.3f})"
        )

        if m.pvalues is not None:
            mask = m.significant(alpha, correction=correction)
            import numpy as np

            n_sig = int(np.sum(mask) // (1 if m.directed else 2))
            label = correction or "uncorrected"
            lines.append(
                f"    significant   : {n_sig} edges at alpha={alpha} ({label})"
            )

        edges = m.edges()
        if not edges.empty:
            edges = edges.reindex(
                edges["weight"].abs().sort_values(ascending=False).index
            )
            lines.append(f"    top {min(top_n, len(edges))} connections :")
            for _, row in edges.head(top_n).iterrows():
                arrow = "->" if m.directed else "--"
                extra = (
                    f"  (p={row['pvalue']:.3g})"
                    if "pvalue" in row and row["pvalue"] == row["pvalue"]
                    else ""
                )
                lines.append(
                    f"        {row['source']:>10} {arrow} {row['target']:<10} "
                    f"r={row['weight']:+.3f}{extra}"
                )
        lines.append("")

    lines.append("=" * 64)
    return "\n".join(lines)


def _py(x):
    """Cast numpy scalars to plain Python for JSON-serialisability."""
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def result_report(
    result: "ConnectivityResult",
    *,
    alpha: float = 0.05,
    correction: str | None = "fdr_bh",
    threshold: float = 0.5,
    top_n: int = 10,
    include_graph: bool = True,
) -> dict:
    """Build a thorough, JSON-serialisable nested report of a result.

    The single structured object behind the human report, the tables and the
    poster: ``meta`` / ``provenance`` / ``params`` plus a ``chromophores`` map
    with, per chromophore, descriptive ``summary``, edge-weight ``distribution``
    percentiles, ``significance`` counts, optional ``graph`` metrics, the ranked
    ``hubs``, and the top significant ``edges`` (as records).
    """
    out: dict = {
        "meta": {
            "method": result.method,
            "kind": str(result.kind),
            "directed": result.directed,
            "alpha": alpha,
            "correction": correction,
            "threshold": threshold,
        },
        "provenance": {k: _py(v) for k, v in result.provenance.items()},
        "params": {k: _py(v) for k, v in result.params.items()},
        "chromophores": {},
    }

    for chrom, m in result.items():
        n = m.n
        off = m.values[~np.eye(n, dtype=bool)]
        finite = off[np.isfinite(off)]
        pct = (
            {p: float(np.percentile(finite, p)) for p in (5, 25, 50, 75, 95)}
            if finite.size
            else {}
        )
        entry: dict = {
            "summary": {k: _py(v) for k, v in m.summary().items()},
            "distribution": {
                "percentiles": pct,
                "frac_negative": float(np.mean(finite < 0)) if finite.size else float("nan"),
            },
        }
        if m.pvalues is not None:
            mask = m.significant(alpha, correction=correction)
            n_sig = int(np.sum(mask) // (1 if m.directed else 2))
            entry["significance"] = {
                "alpha": alpha, "correction": correction,
                "n_edges": int(finite.size // (1 if m.directed else 2)),
                "n_significant": n_sig,
            }
        if include_graph:
            try:
                from ..graph.metrics import graph_metrics

                gm = graph_metrics(m, threshold=threshold)
                entry["graph"] = {
                    k: _py(v) for k, v in gm.items() if k != "nodal"
                }
                strength = gm.get("nodal", {}).get("degree", {})
                entry["hubs"] = [
                    {"node": k, "strength": float(v)}
                    for k, v in sorted(strength.items(), key=lambda kv: kv[1],
                                       reverse=True)[:top_n]
                ]
            except Exception:  # networkx missing / degenerate graph
                entry["graph"] = None

        from ..tables import significant_edges_table

        df = significant_edges_table(
            m, alpha=alpha, correction=correction, sort="abs_effect",
            include_nonsignificant=True, top_n=top_n,
        )
        entry["edges"] = df.to_dict(orient="records")
        out["chromophores"][chrom] = entry

    return out
