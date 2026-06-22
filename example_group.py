"""Group-level connectivity example: many recordings -> statistics -> poster.

This demonstrates the nirconn *group* workflow end to end. To keep it runnable
from a single SNIRF file, it splits one recording into several equal time windows
and treats each window as a "run" (subject), then asks the canonical group
question: **which connections are reliably present across runs?** (a one-sample
edgewise test on the Fisher-z connectivity, FDR-corrected).

The same API works for real multi-subject studies -- just ``study.add(subject_id,
result, group=..., **covariates)`` one ConnectivityResult per subject and use
``study.group(chromophore).two_sample(...)`` / ``.regression(...)`` / ``.nbs(...)``.

Outputs (in ``--output``, default ./example_group_output):
    * group_poster.png        -- the whole pipeline at a glance
    * significant_edges.csv    -- the table of significant connections
    * nodal_metrics.csv        -- per-node graph-theory summary
    * report.txt               -- the thorough text report

Usage
-----
::

    python example_group.py [path/to/file.snirf] [--windows K] [--method M] [--output DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import nirconn as cn

# Reuse the verified SNIRF loader and path resolution from the single-subject example.
from example import load_snirf, resolve_path


def study_from_windows(
    ts: cn.NirsTimeSeries, *, n_windows: int, method: str
) -> cn.Study:
    """Split one recording into ``n_windows`` runs and assemble a Study."""
    cube = ts.data.values  # (chromophore, channel, time)
    n_time = cube.shape[-1]
    win = n_time // n_windows
    if win < 20:
        raise SystemExit(
            f"Recording too short to split into {n_windows} windows "
            f"({n_time} samples). Use fewer --windows."
        )

    study = cn.Study(name=f"{ts.name or 'recording'} ({n_windows} runs)")
    for w in range(n_windows):
        seg = cube[..., w * win : (w + 1) * win]
        result = cn.connectivity(
            seg,
            method=method,
            sfreq=ts.sfreq,
            channels=ts.channels,
            chromophores=ts.chromophores,
        )
        study.add(f"run{w + 1}", result, group="all")
    return study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Group-level fNIRS connectivity statistics + poster.",
    )
    parser.add_argument("path", nargs="?", default=None, help="A .snirf file.")
    parser.add_argument("--windows", type=int, default=6,
                        help="Number of time windows / pseudo-runs (default 6).")
    parser.add_argument("--method", default="pearson",
                        choices=[e["name"] for e in cn.list_estimators()],
                        help="Connectivity estimator (default pearson).")
    parser.add_argument("--chromophore", default="HbO", help="Chromophore to test.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", default="example_group_output")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    path = resolve_path(args.path)
    if path is None or not Path(path).exists():
        print("No SNIRF file found. Pass one: python example_group.py file.snirf")
        return 0

    print(f"Loading SNIRF: {path}")
    ts = load_snirf(path)
    print(f"  -> {ts.n_channels} channels x {ts.n_times} samples @ {ts.sfreq:g} Hz")

    print(f"\nBuilding study: {args.windows} runs x {args.method} connectivity ...")
    study = study_from_windows(ts, n_windows=args.windows, method=args.method)
    print(" ", study)

    group = study.group(args.chromophore)
    print(f"\nGroup ({args.chromophore}): {group.n_subjects} runs, {group.n} channels")

    # One-sample edgewise test: which edges are reliably non-zero across runs?
    stat = group.one_sample(tail="two-sided", alpha=args.alpha, correction="fdr_bh")
    print(stat)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Thorough text report.
    report = cn.report.summarize_group(stat, top_n=15)
    print("\n" + report)
    (out / "report.txt").write_text(report, encoding="utf-8")

    # Significant-findings tables -> CSV.
    edges = stat.table(include_nonsignificant=False)
    edges.to_csv(out / "significant_edges.csv", index=False)
    print(f"\nSignificant edges: {len(edges)}  (-> significant_edges.csv)")

    nodal = cn.tables.nodal_summary_table(stat.to_matrix(), threshold=0.3)
    nodal.to_csv(out / "nodal_metrics.csv", index=False)

    # Poster: the whole pipeline at a glance.
    fig = cn.build_poster(
        stat,
        group_mean=group.mean(),
        alpha=args.alpha,
        title=f"{study.name} - {args.method} {args.chromophore} (one-sample FC)",
    )
    poster_path = out / "group_poster.png"
    fig.savefig(poster_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved:\n  {poster_path.resolve()}")
    print(f"  {(out / 'significant_edges.csv').resolve()}")
    print(f"  {(out / 'nodal_metrics.csv').resolve()}")
    print(f"  {(out / 'report.txt').resolve()}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
