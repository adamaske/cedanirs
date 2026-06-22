"""Effective-connectivity (Granger causality) group analysis -- Sivert study.

Where the Pearson/wavelet runs ask "do these two channels co-vary?", Granger
asks "does the *past* of channel A predict the *future* of channel B, beyond
B's own past?" -- a directional (effective) connectivity measure. The result is
an N x N asymmetric matrix where entry [src, tgt] is the Granger causality
index (GCI) of src -> tgt (the log-ratio of prediction errors with vs without
src's history as a covariate).

Design notes
------------
* Pairwise bivariate Granger, order=2 (two lags), on 0.01-0.1 Hz band-passed
  HbO/HbR traces (already applied by read_timeseries).
* Within-subject run averaging: **arithmetic mean** of GCI matrices (not
  Fisher-z, since GCI >= 0 and is not a correlation). The diagonal is set to 0
  (a channel does not Granger-cause itself).
* Group test: one-sample t-test via the existing group pipeline. GCI values for
  fNIRS are empirically << 1, so Fisher-z ~ identity in that range and the test
  is approximately equivalent to a direct t-test on raw GCI. The effect column
  in the output tables gives the group-mean GCI.
* All N*(N-1) directed edges are tested (not just the upper triangle).
* Poster threshold: 0.05 (GCI scale 0--0.5, not 0--1 like WTC / -1--1 like r).
* Output: output_granger/

Run:  python run_granger.py     (or SIVERT_ROOT=... python run_granger.py)
"""

from __future__ import annotations

import itertools
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

import nirconn as cn
from nirconn.core.result import ConnectivityMatrix
from nirconn.core.types import ConnectivityKind

ROOT       = Path(os.environ.get("SIVERT_ROOT", Path(__file__).parent))
RESULTS    = ROOT / "results"
OUT        = Path(__file__).parent / "output_granger"
CONDITIONS = ["handwriting", "ipad", "keyboard", "remarkable"]
BAND       = (0.01, 0.1)
ORDER      = 2                 # Granger model order (lags)
ALPHA      = 0.05
MIN_SECONDS   = 60.0
GROUP_COVERAGE = 0.5
POSTER_THRESHOLD = 0.05       # GCI threshold for graph panels (GCI not in [-1,1])
SUBJECT_RE = re.compile(r"sub\s*0*(\d+)", re.IGNORECASE)

load_log: list[str] = []


def _gci_mean(matrices: list[ConnectivityMatrix]) -> ConnectivityMatrix:
    """Arithmetic mean of GCI matrices, union of channels, diagonal=0.

    Unlike correlation/WTC, GCI is not bounded to [-1, 1], so Fisher-z is not
    appropriate. We stack via GroupConnectivity (which handles the union of
    channel labels and NaN-fills missing channels), take the arithmetic nanmean,
    then zero the diagonal.
    """
    if len(matrices) == 1:
        return matrices[0]
    gc = cn.GroupConnectivity.from_matrices(
        matrices,
        ids=[f"run{i}" for i in range(len(matrices))],
        min_coverage=0.0,
    )
    raw = gc.stack(fisher=False)        # (n_runs, n_ch, n_ch)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_vals = np.nanmean(raw, axis=0)  # (n_ch, n_ch)
    mean_vals = np.nan_to_num(mean_vals, nan=0.0)
    np.fill_diagonal(mean_vals, 0.0)    # GCI diagonal is 0, not 1
    return ConnectivityMatrix(
        mean_vals,
        gc.labels,
        method="granger",
        kind=ConnectivityKind.EFFECTIVE,
        directed=True,
        chromophore=matrices[0].chromophore,
        params={"order": ORDER, "n_runs": len(matrices)},
    )


def subject_condition_result(files: list[Path]):
    runs: dict[str, list[ConnectivityMatrix]] = defaultdict(list)
    for f in files:
        ts  = cn.read_timeseries(str(f), bandpass=BAND)
        dur = (ts.n_times / ts.sfreq) if ts.sfreq else 0.0
        # Granger needs at least 3*order+1 samples after lagging; the 60 s
        # floor already guarantees >> that, so use the same cut-off.
        used = dur >= MIN_SECONDS
        load_log.append(
            f"{f.parent.name:<12} {f.name:<48} {ts.n_channels:>2}ch "
            f"{ts.n_times:>5}smp {ts.sfreq:>6.2f}Hz {dur:>6.1f}s "
            f"{'USED' if used else 'SKIP(short)'}"
        )
        if not used:
            continue
        res = cn.connectivity(ts, method="granger", order=ORDER)
        for c in res.chromophores:
            runs[c].append(res[c])
    if not runs:
        return None
    mats = {c: _gci_mean(mlist) for c, mlist in runs.items()}
    return cn.ConnectivityResult(mats, method="granger",
                                 kind=ConnectivityKind.EFFECTIVE,
                                 directed=True)


def _group_mean_for_poster(g: cn.GroupConnectivity) -> ConnectivityMatrix:
    """Arithmetic-mean GCI matrix suitable for poster visualisation."""
    raw = g.stack(fisher=False)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_vals = np.nanmean(raw, axis=0)
    mean_vals = np.nan_to_num(mean_vals, nan=0.0)
    np.fill_diagonal(mean_vals, 0.0)
    return ConnectivityMatrix(
        mean_vals,
        g.labels,
        method="granger-groupmean",
        kind=ConnectivityKind.EFFECTIVE,
        directed=True,
        chromophore=g.chromophore,
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "per_condition").mkdir(exist_ok=True)
    (OUT / "contrasts").mkdir(exist_ok=True)

    study = cn.Study(name="Sivert writing modalities -- Granger causality")
    for cond in CONDITIONS:
        by_sub: dict[str, list[Path]] = defaultdict(list)
        for f in sorted((RESULTS / cond).glob("*.snirf")):
            m = SUBJECT_RE.search(f.name)
            if m:
                by_sub[f"sub{m.group(1)}"].append(f)
        for sid, files in sorted(by_sub.items()):
            print(f"  [{cond}] {sid} ({len(files)} files)...", flush=True)
            res = subject_condition_result(files)
            if res is not None:
                study.add(sid, res, condition=cond, group="all")

    print(study)
    (OUT / "loading_log.txt").write_text("\n".join(load_log), encoding="utf-8")

    all_labels = [set(r["HbO"].labels) for r in study._results.values()]
    union  = sorted(set().union(*all_labels))
    common = sorted(set.intersection(*all_labels))

    summary = [
        "=" * 72,
        "  Sivert writing-modality connectivity  --  Granger causality (effective)",
        "=" * 72,
        f"  subjects   : {sorted(study.subjects)}",
        f"  conditions : {CONDITIONS}",
        f"  method     : granger  order={ORDER}  band={BAND[0]}-{BAND[1]} Hz",
        f"  alpha={ALPHA} (FDR-BH)   min length={MIN_SECONDS:g}s",
        f"  within-subject run averaging: arithmetic mean GCI (diagonal=0)",
        "",
        "  MONTAGE COVERAGE",
        f"    full montage (union)        : {len(union)} channels",
        f"    common to every subject     : {len(common)} channels",
        f"    group tests use channels in >= {GROUP_COVERAGE:.0%} of subjects",
        "    NOTE: directed analysis -- all N*(N-1) ordered pairs are tested.",
        "",
        "  PER-CONDITION group directed connectome (one-sample HbO, GCI > 0)",
    ]

    for cond in CONDITIONS:
        g    = study.group("HbO", condition=cond, min_coverage=GROUP_COVERAGE)
        # one_sample tests GCI > 0 (tail="greater"); Fisher-z ~= identity for
        # small GCI values, so this is essentially a t-test on raw GCI.
        stat = g.one_sample(alpha=ALPHA, correction="fdr_bh", tail="greater")
        s    = stat.summary()
        n_directed = stat.n * (stat.n - 1)  # total directed edges
        summary.append(
            f"    {cond:<12} n={g.n_subjects} subj, {g.n} ch, "
            f"{s['n_significant']}/{n_directed} directed edges sig"
        )
        stat.table(include_nonsignificant=False).to_csv(
            OUT / "per_condition" / f"{cond}_significant_edges.csv", index=False
        )
        (OUT / "per_condition" / f"{cond}_report.txt").write_text(
            cn.report.summarize_group(stat, top_n=20), encoding="utf-8"
        )
        gm = _group_mean_for_poster(g)
        fig = cn.build_poster(
            stat, group_mean=gm, alpha=ALPHA,
            threshold=POSTER_THRESHOLD,
            title=f"Sivert - {cond} -- Granger GCI (HbO, directed)"
        )
        fig.savefig(OUT / "per_condition" / f"{cond}_poster.png",
                    dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)

    summary += ["", "  CONDITION CONTRASTS (paired HbO, GCI difference)"]
    groups = {
        c: study.group("HbO", condition=c, min_coverage=GROUP_COVERAGE)
        for c in CONDITIONS
    }
    for a, b in itertools.combinations(CONDITIONS, 2):
        stat = groups[a].paired(groups[b], alpha=ALPHA, correction="fdr_bh")
        s    = stat.summary()
        # For directed, count all finite off-diagonal p-values.
        p_all = stat.pvalues.ravel()
        praw  = p_all[np.isfinite(p_all)]
        n_unc = int(np.sum(praw < ALPHA))
        n_directed = stat.n * (stat.n - 1)
        summary.append(
            f"    {a:>11} vs {b:<11} | FDR-sig={s['n_significant']}  "
            f"uncorrected p<{ALPHA}: {n_unc}/{n_directed}  (min q={s['min_q']:.2f})"
        )
        stat.table(include_nonsignificant=True, sort="q").to_csv(
            OUT / "contrasts" / f"{a}_vs_{b}_edges.csv", index=False
        )

    head_a = groups["handwriting"]
    head_b = groups["keyboard"]
    head   = head_a.paired(head_b, alpha=ALPHA)
    gm_hw  = _group_mean_for_poster(head_a)
    fig = cn.build_poster(
        head, group_mean=gm_hw, alpha=ALPHA,
        threshold=POSTER_THRESHOLD,
        title="Sivert - handwriting vs keyboard -- Granger GCI (HbO, directed)"
    )
    fig.savefig(OUT / "contrasts" / "handwriting_vs_keyboard_poster.png",
                dpi=150, bbox_inches="tight")

    summary += ["", "=" * 72]
    (OUT / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
    print(f"\nResults written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
