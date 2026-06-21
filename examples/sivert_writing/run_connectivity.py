"""Functional-connectivity group analysis of the Sivert writing-modality study.

Uses cedanirs' native SNIRF reader (h5py) — the files are already HbO/HbR, so no
MNE/Beer-Lambert is needed. Channels are aligned across subjects on the FULL
montage (union of source-detector pairs); group tests are NaN-aware where a
recording lacks a channel.

Design: 6 subjects x 4 conditions (handwriting, ipad, keyboard, remarkable),
multiple runs each.

Pipeline (HbO primary):
  1. Native read of each run + 0.01-0.1 Hz band-pass; skip runs shorter than
     MIN_SECONDS (too short for resting-band connectivity).
  2. Pearson connectivity per run; Fisher-z average a subject's runs (union of
     channels) -> one matrix per (subject, condition).
  3. Group analyses: per-condition one-sample (FC > 0) and every pairwise paired
     contrast, FDR-BH, on channels present in >= GROUP_COVERAGE of subjects.
  4. Write posters, tables and reports to ./connectivity/.

Run:  python run_connectivity.py     (or SIVERT_ROOT=... python run_connectivity.py)
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

import cedanirs as cn

ROOT = Path(os.environ.get("SIVERT_ROOT", Path(__file__).parent))
RESULTS = ROOT / "results"
OUT = Path(__file__).parent / "output"
CONDITIONS = ["handwriting", "ipad", "keyboard", "remarkable"]
BAND = (0.01, 0.1)
ALPHA = 0.05
MIN_SECONDS = 60.0          # skip runs too short for 0.01-0.1 Hz connectivity
GROUP_COVERAGE = 0.5        # keep channels present in >= this fraction of subjects
SUBJECT_RE = re.compile(r"sub\s*0*(\d+)", re.IGNORECASE)

load_log: list[str] = []


def subject_condition_result(files: list[Path]):
    runs = defaultdict(list)
    for f in files:
        ts = cn.read_timeseries(str(f), bandpass=BAND)
        dur = (ts.n_times / ts.sfreq) if ts.sfreq else 0.0
        used = dur >= MIN_SECONDS
        load_log.append(
            f"{f.parent.name:<12} {f.name:<48} {ts.n_channels:>2}ch "
            f"{ts.n_times:>5}smp {ts.sfreq:>6.2f}Hz {dur:>6.1f}s "
            f"{'USED' if used else 'SKIP(short)'}"
        )
        if not used:
            continue
        res = cn.connectivity(ts, method="pearson")
        for c in res.chromophores:
            runs[c].append(res[c])
    if not runs:
        return None
    mats = {}
    for c, mlist in runs.items():
        if len(mlist) == 1:
            mats[c] = mlist[0]
        else:
            # union of channels across runs (NaN where a run lacks one), z-mean
            gc = cn.GroupConnectivity.from_matrices(
                mlist, ids=[f"run{i}" for i in range(len(mlist))], min_coverage=0.0
            )
            mats[c] = gc.mean()
    return cn.ConnectivityResult(mats, method="pearson")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "per_condition").mkdir(exist_ok=True)
    (OUT / "contrasts").mkdir(exist_ok=True)

    study = cn.Study(name="Sivert writing modalities")
    for cond in CONDITIONS:
        by_sub = defaultdict(list)
        for f in sorted((RESULTS / cond).glob("*.snirf")):
            m = SUBJECT_RE.search(f.name)
            if m:
                by_sub[f"sub{m.group(1)}"].append(f)
        for sid, files in sorted(by_sub.items()):
            res = subject_condition_result(files)
            if res is not None:
                study.add(sid, res, condition=cond, group="all")

    print(study)
    (OUT / "loading_log.txt").write_text("\n".join(load_log), encoding="utf-8")

    # montage coverage across the full study
    all_labels = [set(r["HbO"].labels) for r in study._results.values()]
    union = sorted(set().union(*all_labels))
    common = sorted(set.intersection(*all_labels))

    summary = [
        "=" * 72, "  Sivert writing-modality functional connectivity (native reader)",
        "=" * 72,
        f"  subjects   : {sorted(study.subjects)}",
        f"  conditions : {CONDITIONS}",
        f"  bandpass   : {BAND[0]}-{BAND[1]} Hz   alpha={ALPHA} (FDR-BH)",
        f"  min length : {MIN_SECONDS:g}s per run",
        "",
        "  MONTAGE COVERAGE (source-detector pairs)",
        f"    full montage (union over all recordings) : {len(union)} channels",
        f"    common to every subject (intersection)   : {len(common)} channels",
        f"    group tests use channels in >= {GROUP_COVERAGE:.0%} of subjects",
        "    NOTE: recordings differ in sfreq (5-12 Hz), length and which channels",
        "    survived QC, so per-edge subject counts vary (see tables' 'n' column).",
        "",
        "  PER-CONDITION group connectome (one-sample HbO, FC > 0)",
    ]
    for cond in CONDITIONS:
        g = study.group("HbO", condition=cond, min_coverage=GROUP_COVERAGE)
        stat = g.one_sample(alpha=ALPHA, correction="fdr_bh")
        s = stat.summary()
        summary.append(f"    {cond:<12} n={g.n_subjects} subj, {g.n} ch, "
                       f"{s['n_significant']}/{s['n_edges_tested']} edges sig")
        stat.table(include_nonsignificant=False).to_csv(
            OUT / "per_condition" / f"{cond}_significant_edges.csv", index=False)
        (OUT / "per_condition" / f"{cond}_report.txt").write_text(
            cn.report.summarize_group(stat, top_n=20), encoding="utf-8")
        fig = cn.build_poster(stat, group_mean=g.mean(), alpha=ALPHA,
                              title=f"Sivert - {cond} (one-sample HbO, FC>0)")
        fig.savefig(OUT / "per_condition" / f"{cond}_poster.png",
                    dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)

    summary += ["", "  CONDITION CONTRASTS (paired HbO)"]
    groups = {c: study.group("HbO", condition=c, min_coverage=GROUP_COVERAGE)
              for c in CONDITIONS}
    for a, b in itertools.combinations(CONDITIONS, 2):
        stat = groups[a].paired(groups[b], alpha=ALPHA, correction="fdr_bh")
        s = stat.summary()
        iu = np.triu_indices(stat.n, 1)
        praw = stat.pvalues[iu]
        praw = praw[np.isfinite(praw)]
        n_unc = int(np.sum(praw < ALPHA))
        summary.append(
            f"    {a:>11} vs {b:<11} | FDR-sig={s['n_significant']}  "
            f"uncorrected p<{ALPHA}: {n_unc}/{praw.size}  (min q={s['min_q']:.2f})")
        stat.table(include_nonsignificant=True, sort="q").to_csv(
            OUT / "contrasts" / f"{a}_vs_{b}_edges.csv", index=False)

    head = groups["handwriting"].paired(groups["keyboard"], alpha=ALPHA)
    fig = cn.build_poster(head, group_mean=groups["handwriting"].mean(), alpha=ALPHA,
                          title="Sivert - handwriting vs keyboard (paired HbO)")
    fig.savefig(OUT / "contrasts" / "handwriting_vs_keyboard_poster.png",
                dpi=150, bbox_inches="tight")

    summary += ["", "=" * 72]
    (OUT / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
    print(f"\nResults written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
