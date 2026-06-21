"""Functional-connectivity group analysis of the Sivert writing-modality study.

Design: 6 subjects x 4 writing conditions (handwriting, ipad, keyboard,
remarkable), multiple runs per subject/condition. Files are already-processed
HbO/HbR SNIRF.

Pipeline (per chromophore, HbO primary):
  1. Load every run (MNE), keep long channels, band-pass 0.01-0.1 Hz.
  2. Pearson connectivity per run; average the runs of one subject/condition in
     Fisher-z space -> one matrix per (subject, condition).
  3. Group analyses with cedanirs:
       * per condition: one-sample edgewise test (which connections are reliably
         present across subjects) -> poster + significant-edges table + report
       * every condition pair: paired (within-subject) contrast -> edge table
  4. Write everything to ./connectivity/.

Run:  python run_connectivity.py
"""

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

import cedanirs as cn

import os

# The SNIRF data is not in the repo. Point SIVERT_ROOT at the directory holding
# results/<condition>/*.snirf; outputs are written to ./output next to this file.
ROOT = Path(os.environ.get("SIVERT_ROOT", Path(__file__).parent))
RESULTS = ROOT / "results"
OUT = Path(__file__).parent / "output"
CONDITIONS = ["handwriting", "ipad", "keyboard", "remarkable"]
BAND = (0.01, 0.1)
ALPHA = 0.05
SUBJECT_RE = re.compile(r"sub\s*0*(\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# loading (self-contained MNE loader)
# --------------------------------------------------------------------------- #

def load_snirf(path: str) -> cn.NirsTimeSeries:
    import mne
    from mne.preprocessing.nirs import beer_lambert_law, optical_density

    raw = mne.io.read_raw_snirf(path, preload=True, verbose="error")
    types = set(raw.get_channel_types())
    if types <= {"hbo", "hbr"}:
        hb = raw
    elif "fnirs_od" in types:
        hb = beer_lambert_law(raw, ppf=6.0)
    elif "fnirs_cw_amplitude" in types:
        hb = beer_lambert_law(optical_density(raw), ppf=6.0)
    else:
        raise ValueError(f"Unsupported channel types: {types}")

    try:
        from mne_nirs.channels import get_long_channels

        hb = get_long_channels(hb)
    except Exception:
        pass

    hb.filter(*BAND, l_trans_bandwidth=0.005, h_trans_bandwidth=0.02, verbose="error")
    picks_hbo = mne.pick_types(hb.info, fnirs="hbo")
    picks_hbr = mne.pick_types(hb.info, fnirs="hbr")
    base = lambda i: hb.ch_names[i].rsplit(" ", 1)[0]
    labels = [base(i) for i in picks_hbo]
    if labels != [base(i) for i in picks_hbr]:
        raise ValueError("HbO/HbR channel order mismatch")
    data = hb.get_data()
    cube = np.stack([data[picks_hbo], data[picks_hbr]])
    return cn.NirsTimeSeries(cube, sfreq=float(hb.info["sfreq"]),
                             channels=labels, chromophores=["HbO", "HbR"])


def subject_of(path: Path) -> str | None:
    m = SUBJECT_RE.search(path.name)
    return f"sub{m.group(1)}" if m else None


def subject_condition_result(files: list[Path]) -> cn.ConnectivityResult:
    """Average a subject's runs (z-space) into one ConnectivityResult."""
    runs = defaultdict(list)
    for f in files:
        res = cn.connectivity(load_snirf(str(f)), method="pearson")
        for c in res.chromophores:
            runs[c].append(res[c])
    mats = {}
    for c, mlist in runs.items():
        if len(mlist) == 1:
            mats[c] = mlist[0]
        else:
            gc = cn.GroupConnectivity.from_matrices(
                mlist, ids=[f"run{i}" for i in range(len(mlist))]
            )
            mats[c] = gc.mean()  # Fisher-z mean across runs
    return cn.ConnectivityResult(mats, method="pearson")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "per_condition").mkdir(exist_ok=True)
    (OUT / "contrasts").mkdir(exist_ok=True)

    study = cn.Study(name="Sivert writing modalities")
    log: list[str] = []
    for cond in CONDITIONS:
        files = sorted((RESULTS / cond).glob("*.snirf"))
        by_sub = defaultdict(list)
        for f in files:
            sid = subject_of(f)
            if sid:
                by_sub[sid].append(f)
        for sid, runs in sorted(by_sub.items()):
            res = subject_condition_result(runs)
            study.add(sid, res, condition=cond, group="all")
            log.append(f"{cond:<12} {sid:<6} {len(runs)} run(s), "
                       f"{res['HbO'].n} channels")
            print(log[-1])

    print("\n", study)

    # Channel coverage: subjects have heterogeneous montages, so group tests
    # run on the channels common to all subjects.
    common = set.intersection(*[set(r["HbO"].labels) for r in study._results.values()])
    summary = ["=" * 70, "  Sivert writing-modality functional connectivity",
               "=" * 70, f"  subjects   : {sorted(study.subjects)}",
               f"  conditions : {CONDITIONS}",
               f"  bandpass   : {BAND[0]}-{BAND[1]} Hz   alpha={ALPHA} (FDR-BH)", "",
               "  CHANNEL COVERAGE",
               f"    common channels across all 6 subjects: {len(common)} "
               f"(montages differ; see loading_log.txt for per-subject counts)",
               f"    common set: {', '.join(sorted(common))}", ""]

    # ---- per-condition one-sample group connectomes (HbO) ----
    summary.append("  PER-CONDITION group connectome (one-sample HbO, FC > 0)")
    for cond in CONDITIONS:
        g = study.group("HbO", condition=cond)
        stat = g.one_sample(alpha=ALPHA, correction="fdr_bh")
        s = stat.summary()
        summary.append(f"    {cond:<12} n={g.n_subjects} subj, {g.n} ch, "
                       f"{s['n_significant']}/{s['n_edges_tested']} edges sig")
        stat.table(include_nonsignificant=False).to_csv(
            OUT / "per_condition" / f"{cond}_significant_edges.csv", index=False)
        (OUT / "per_condition" / f"{cond}_report.txt").write_text(
            cn.report.summarize_group(stat, top_n=20), encoding="utf-8")
        fig = cn.build_poster(stat, group_mean=g.mean(), alpha=ALPHA,
                              title=f"Sivert — {cond} (one-sample HbO, FC>0)")
        fig.savefig(OUT / "per_condition" / f"{cond}_poster.png",
                    dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
    summary.append("")

    # ---- pairwise within-subject condition contrasts (paired, HbO) ----
    summary.append("  CONDITION CONTRASTS (paired HbO)")
    groups = {c: study.group("HbO", condition=c) for c in CONDITIONS}
    for a, b in itertools.combinations(CONDITIONS, 2):
        stat = groups[a].paired(groups[b], alpha=ALPHA, correction="fdr_bh")
        s = stat.summary()
        iu = np.triu_indices(stat.n, 1)
        praw = stat.pvalues[iu]
        praw = praw[np.isfinite(praw)]
        n_unc = int(np.sum(praw < ALPHA))
        min_q = s["min_q"]
        summary.append(
            f"    {a:>11} vs {b:<11} n={s['n_subjects']} | FDR-sig={s['n_significant']}"
            f"  uncorrected p<{ALPHA}: {n_unc}/{praw.size}  (min q={min_q:.2f})"
        )
        # save the full edge table (sorted by q) so sub-threshold trends are visible
        stat.table(include_nonsignificant=True, sort="q").to_csv(
            OUT / "contrasts" / f"{a}_vs_{b}_edges.csv", index=False)

    # headline contrast poster: handwriting vs keyboard (analog vs digital)
    head = groups["handwriting"].paired(groups["keyboard"], alpha=ALPHA)
    fig = cn.build_poster(head, group_mean=groups["handwriting"].mean(), alpha=ALPHA,
                          title="Sivert — handwriting vs keyboard (paired HbO)")
    fig.savefig(OUT / "contrasts" / "handwriting_vs_keyboard_poster.png",
                dpi=150, bbox_inches="tight")
    summary += ["", "=" * 70]

    (OUT / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    (OUT / "loading_log.txt").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(summary))
    print(f"\nResults written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
