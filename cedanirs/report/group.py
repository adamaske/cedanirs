"""Human-readable reports for group-level analyses.

* :func:`summarize_group` -- a sectioned text/markdown report for one
  :class:`~cedanirs.stats._results.GroupStatResult`.
* :func:`summarize_study` -- an overview of a whole
  :class:`~cedanirs.core.group.Study` (sample sizes, coverage, and a one-sample
  screen per chromophore).

All output is ASCII-safe so it prints cleanly on any console.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..core.group import Study
    from ..stats._results import GroupStatResult

__all__ = ["summarize_group", "summarize_study"]

_BAR = "=" * 68


def summarize_group(
    stat: "GroupStatResult",
    *,
    alpha: float | None = None,
    correction: str | None = None,
    top_n: int = 20,
    fmt: str = "text",
) -> str:
    """Render a group-stat result as a sectioned report string."""
    s = stat.summary()
    a = alpha if alpha is not None else s.get("alpha", 0.05)
    corr = correction if correction is not None else s.get("correction")

    lines: list[str] = []
    lines.append(_BAR)
    lines.append(f"  Group connectivity report - {stat.test} ({stat.design})")
    lines.append(_BAR)
    lines.append(f"  chromophore     : {stat.chromophore}")
    lines.append(f"  estimator       : {stat.method}")
    lines.append(f"  subjects        : {s.get('n_subjects')}")
    for k in ("popmean", "equal_var", "predictor", "tail", "conf"):
        if k in stat.params:
            lines.append(f"  {k:<15} : {stat.params[k]}")
    lines.append("")

    lines.append("  SIGNIFICANCE")
    lines.append(f"    edges tested  : {s['n_edges_tested']}")
    lines.append(f"    significant   : {s['n_significant']} at alpha={a} ({corr})")
    lines.append(f"    min q-value   : {s['min_q']:.3g}")
    lines.append("")

    df = stat.table(alpha=a, correction=corr, sort="q",
                    include_nonsignificant=False, top_n=top_n)
    lines.append(f"  TOP FINDINGS (up to {top_n})")
    if df.empty:
        lines.append("    (no edges survived correction)")
    else:
        has_d = "cohens_d" in df.columns
        header = f"    {'edge':<22} {'effect':>8} {'t':>8} {'p':>10} {'q':>10}"
        if has_d:
            header += f" {'d':>7}"
        lines.append(header)
        for _, row in df.iterrows():
            line = (
                f"    {row['edge']:<22} {row['effect']:>+8.3f} "
                f"{row['t']:>8.2f} {row['p']:>10.2e} {row['q']:>10.2e}"
            )
            if has_d:
                line += f" {row['cohens_d']:>7.2f}"
            lines.append(line)
    lines.append("")
    lines.append(_BAR)
    text = "\n".join(lines)

    if fmt == "markdown":
        return _to_markdown(text)
    return text


def summarize_study(study: "Study", *, alpha: float = 0.05,
                    correction: str | None = "fdr_bh") -> str:
    """Overview report for a Study: sample, coverage, per-chromophore screen."""
    info = study.summary()
    lines: list[str] = []
    lines.append(_BAR)
    lines.append(f"  Study report - {info['name'] or '(unnamed)'}")
    lines.append(_BAR)
    lines.append(f"  subjects        : {info['n_subjects']}")
    lines.append(f"  records         : {info['n_records']}")
    lines.append(f"  conditions      : {info['conditions'] or '-'}")
    lines.append(f"  groups          : {info['groups'] or '-'}")
    lines.append(f"  chromophores    : {info['chromophores']}")
    lines.append("")

    for chrom in info["chromophores"]:
        try:
            g = study.group(chrom)
        except Exception as exc:
            lines.append(f"  [{chrom}] could not assemble group: {exc}")
            continue
        lines.append(f"  [{chrom}]  n_subjects={g.n_subjects}, channels={g.n}")
        try:
            stat = g.one_sample(alpha=alpha, correction=correction)
            ss = stat.summary()
            lines.append(
                f"      one-sample FC>0: {ss['n_significant']}/{ss['n_edges_tested']} "
                f"edges at alpha={alpha} ({correction})"
            )
        except Exception as exc:  # pragma: no cover
            lines.append(f"      one-sample failed: {exc}")
    lines.append("")
    lines.append(_BAR)
    return "\n".join(lines)


def _to_markdown(text: str) -> str:
    # Minimal: wrap the fixed-width body in a code block to preserve alignment.
    return "```\n" + text + "\n```"
