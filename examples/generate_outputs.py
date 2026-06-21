"""Generate the committed example outputs in ``examples/output/``.

Runs the full cedanirs pipeline on a deterministic **synthetic two-group study**
so the artifacts are fully reproducible from the repo (no private SNIRF needed):

* 16 "control" + 16 "patient" subjects, 14 channels in two communities.
* Patients carry an extra shared drive within the second community, planting a
  connected sub-network difference for the two-sample / NBS tests to find.

Outputs (regenerate with ``python examples/generate_outputs.py``):
    output/group_poster.png        -- whole-pipeline poster (README image)
    output/subject_matrix.png      -- one subject's HbO connectivity heatmap
    output/significant_edges.csv   -- two-sample significant-findings table
    output/nodal_metrics.csv       -- per-node graph-theory summary
    output/nbs_components.csv       -- NBS significant sub-networks
    output/group_report.txt        -- thorough text report
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

import cedanirs as cn

OUT = Path(__file__).parent / "output"
N_CHANNELS = 14
N_TIME = 300
N_PER_GROUP = 16
COMMUNITY_B = range(7, 14)  # second community: channels 7..13
LABELS = [f"Ch{i + 1:02d}" for i in range(N_CHANNELS)]


def make_subject(rng: np.random.Generator, *, extra_drive: float):
    """Simulate one subject's HbO recording and return its connectivity result.

    Two communities share latent signals; ``extra_drive`` adds a further shared
    component within community B (non-zero only for the patient group), which
    raises within-B connectivity.
    """
    comm_a = rng.standard_normal(N_TIME)
    comm_b = rng.standard_normal(N_TIME)
    extra_b = rng.standard_normal(N_TIME)
    chans = []
    for i in range(N_CHANNELS):
        if i < 7:
            sig = 0.7 * comm_a + 0.3 * rng.standard_normal(N_TIME)
        else:
            sig = (
                0.6 * comm_b
                + extra_drive * extra_b
                + 0.3 * rng.standard_normal(N_TIME)
            )
        chans.append(sig)
    x = np.vstack(chans)
    return cn.connectivity(x, method="pearson", channels=LABELS, chromophore="HbO")


def build_study() -> cn.Study:
    rng = np.random.default_rng(42)
    study = cn.Study(name="synthetic patients vs controls")
    for s in range(N_PER_GROUP):
        study.add(f"ctrl{s:02d}", make_subject(rng, extra_drive=0.0),
                  group="control", age=20 + s)
    for s in range(N_PER_GROUP):
        study.add(f"pat{s:02d}", make_subject(rng, extra_drive=0.6),
                  group="patient", age=22 + s)
    return study


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    study = build_study()
    print(study)

    controls = study.group("HbO", group="control")
    patients = study.group("HbO", group="patient")

    # Two-sample contrast: patient - control.
    stat = patients.two_sample(controls, equal_var=False)
    print(stat)

    # Single-subject heatmap.
    one = next(iter(study._results.values()))["HbO"]
    ax = one.plot(title="Single subject — HbO connectivity")
    ax.figure.savefig(OUT / "subject_matrix.png", dpi=150, bbox_inches="tight")

    # Poster (headline figure for the README): contrast effect + significance,
    # over the patient group-mean connectivity.
    fig = cn.build_poster(
        stat,
        group_mean=patients.mean(),
        alpha=0.05,
        title="Synthetic study — patient vs control (HbO, Welch t)",
    )
    fig.savefig(OUT / "group_poster.png", dpi=150, bbox_inches="tight")

    # Significant-findings tables -> CSV.
    stat.table(include_nonsignificant=False).to_csv(
        OUT / "significant_edges.csv", index=False
    )
    cn.tables.nodal_summary_table(patients.mean(), threshold=0.3).to_csv(
        OUT / "nodal_metrics.csv", index=False
    )

    # Network-Based Statistic.
    nbs = patients.nbs(controls, threshold=3.0, n_permutations=2000, seed=0)
    print(nbs)
    nbs.table().to_csv(OUT / "nbs_components.csv", index=False)

    # Thorough text report.
    report = cn.report.summarize_group(stat, top_n=15)
    (OUT / "group_report.txt").write_text(report, encoding="utf-8")

    print(f"\nWrote outputs to {OUT.resolve()}:")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:<24} {p.stat().st_size:>8} bytes")


if __name__ == "__main__":
    main()
