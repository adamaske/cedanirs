"""End-to-end example: SNIRF -> haemoglobin -> connectivity -> figures.

This script demonstrates a complete functional-connectivity workflow with
:mod:`nirconn`:

1. Load a SNIRF file with MNE, detecting whether it holds raw intensity,
   optical density, or already-converted haemoglobin, and convert it to HbO/HbR
   concentration as needed (Beer-Lambert), keeping only long channels.
2. Band-pass filter for resting-state functional connectivity (0.01-0.1 Hz).
3. Wrap the HbO/HbR cube in a :class:`nirconn.NirsTimeSeries`.
4. Estimate connectivity (Pearson by default; ``--method`` selects any
   registered estimator), print a text report, save heatmap figures, and print
   graph-theory metrics.

Usage
-----
::

    python example.py [path/to/file.snirf] [--method NAME] [--show] [--output DIR]

``--method`` accepts any registered estimator (pearson, spearman, partial,
coherence, ...).

With no path, the script looks for ``./data/*.snirf`` and then a local sample
directory; if nothing is found it prints instructions and exits cleanly.
Figures are written to ``--output`` (default ``./example_output``); pass
``--show`` to also open them interactively (omitted by default so the script
runs headless).
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

import nirconn as cn

# Local-machine convenience fallback: a known sample-data directory on the
# author's machine. Harmless (and silently skipped) anywhere else.
_SAMPLE_DIR = Path("C:/Users/adama/dev/NIRWizard/examples/example_snirf_data")
_SAMPLE_FILE = _SAMPLE_DIR / "raw.snirf"


def load_snirf(
    path: str | Path,
    *,
    ppf: float = 6.0,
    l_freq: float = 0.01,
    h_freq: float = 0.1,
    long_only: bool = True,
) -> cn.NirsTimeSeries:
    """Load a SNIRF file and return a HbO/HbR :class:`~nirconn.NirsTimeSeries`.

    The file is read with MNE and routed through the right preprocessing steps
    depending on what stage the data is already at:

    * raw continuous-wave amplitude -> optical density -> Beer-Lambert -> Hb,
    * optical density -> Beer-Lambert -> Hb,
    * already haemoglobin -> used as-is (no double conversion).

    Long (non short-separation) channels are kept when a 3-D optode montage is
    available, then the data is band-pass filtered for resting-state functional
    connectivity. HbO and HbR are extracted with a single shared, ordered list
    of source-detector labels and stacked into a ``(chromophore, channel, time)``
    cube.

    Parameters
    ----------
    path:
        Path to the ``.snirf`` file.
    ppf:
        Partial pathlength factor passed to :func:`beer_lambert_law`. Only
        rescales concentration amplitudes; it does not change the correlation
        structure used for connectivity.
    l_freq, h_freq:
        Band-pass edges in Hz (defaults give the canonical 0.01-0.1 Hz
        resting-state band).
    long_only:
        Drop short-separation channels when a montage is present.

    Returns
    -------
    nirconn.NirsTimeSeries
        A 3-D cube with chromophores ``["HbO", "HbR"]``.
    """
    try:
        import mne
        from mne.preprocessing.nirs import beer_lambert_law, optical_density
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "This example needs MNE to read SNIRF files. Install the optional "
            'I/O dependencies with:\n\n    pip install "nirconn[io]"\n\n'
            "(this pulls in mne and mne-nirs)."
        ) from exc

    path = str(path)
    raw = mne.io.read_raw_snirf(path, preload=True, verbose="error")
    types = set(raw.get_channel_types())

    # Branch on the actual channel types, never the filename: some files are
    # mislabelled, and beer_lambert_law raises if run on already-Hb data.
    if types <= {"hbo", "hbr"}:
        hb = raw  # already haemoglobin concentration
    elif "fnirs_od" in types:
        hb = beer_lambert_law(raw, ppf=ppf)  # optical density -> Hb
    elif "fnirs_cw_amplitude" in types:
        hb = beer_lambert_law(optical_density(raw), ppf=ppf)  # raw -> OD -> Hb
    else:
        raise ValueError(f"Unsupported fNIRS channel types: {types}")

    # Keep only long channels when a 3-D montage exists; degrade gracefully
    # (keep everything) when it does not.
    if long_only:
        try:
            from mne_nirs.channels import get_long_channels

            hb = get_long_channels(hb)  # default 15 mm < dist < 45 mm
        except Exception:
            pass

    # Band-pass for resting-state FC. Explicit transition bandwidths avoid
    # MNE auto-widening the narrow lower edge below 0 Hz (and silence warnings).
    hb.filter(
        l_freq,
        h_freq,
        l_trans_bandwidth=0.005,
        h_trans_bandwidth=0.02,
        verbose="error",
    )

    # Split into HbO / HbR with a single shared, ordered label list. MNE keeps
    # source-detector ordering across both types, so the base names line up.
    picks_hbo = mne.pick_types(hb.info, fnirs="hbo")
    picks_hbr = mne.pick_types(hb.info, fnirs="hbr")

    def _base(idx: int) -> str:
        # "S1_D1 hbo" -> "S1_D1"; rsplit is robust to extra spaces.
        return hb.ch_names[idx].rsplit(" ", 1)[0]

    labels_hbo = [_base(i) for i in picks_hbo]
    labels_hbr = [_base(i) for i in picks_hbr]
    if labels_hbo != labels_hbr:
        raise ValueError(
            "HbO/HbR channel order mismatch -- the two arrays would be "
            "misaligned. Check the SNIRF montage."
        )
    if not labels_hbo:
        raise ValueError("No HbO/HbR channels found after preprocessing.")

    data = hb.get_data()
    hbo = data[picks_hbo]
    hbr = data[picks_hbr]
    sfreq = float(hb.info["sfreq"])

    # cube layout is strictly (chromophore, channel, time).
    cube = np.stack([hbo, hbr])
    return cn.NirsTimeSeries(
        cube,
        sfreq=sfreq,
        channels=labels_hbo,
        chromophores=["HbO", "HbR"],
        name=Path(path).stem,
    )


def resolve_path(cli_path: str | None) -> Path | None:
    """Resolve the SNIRF file to use, or ``None`` if nothing is found.

    Search order: an explicit CLI argument, then ``./data/*.snirf``, then a
    local-machine sample directory.
    """
    if cli_path:
        return Path(cli_path)

    local = sorted(glob.glob("data/*.snirf"))
    if local:
        return Path(local[0])

    # Local-machine convenience fallback (author's sample data).
    if _SAMPLE_FILE.exists():
        return _SAMPLE_FILE
    if _SAMPLE_DIR.is_dir():
        found = sorted(_SAMPLE_DIR.glob("*.snirf"))
        if found:
            return found[0]

    return None


def make_figures(
    result: cn.ConnectivityResult,
    output_dir: Path,
    *,
    alpha: float = 0.05,
) -> list[Path]:
    """Render and save the connectivity figures, returning the written paths."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    chromophores = result.chromophores
    hbo = result["HbO"] if "HbO" in chromophores else result[chromophores[0]]
    method = result.method

    # fig1: HbO and HbR heatmaps side by side.
    fig1, axes = plt.subplots(1, len(chromophores), figsize=(6 * len(chromophores), 5))
    if len(chromophores) == 1:
        axes = [axes]
    for ax, chromo in zip(axes, chromophores):
        result[chromo].plot(ax=ax, title=f"{method} -- {chromo}")
    fig1.suptitle(f"fNIRS functional connectivity ({method})", fontsize=14)
    fig1.tight_layout()
    fig1_path = output_dir / "connectivity_hbo_hbr.png"
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    written.append(fig1_path)

    # fig2: HbO matrix showing ONLY FDR-significant edges. significant() marks
    # True where an edge IS significant; plot_matrix hides True cells, so invert.
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    try:
        sig = hbo.significant(alpha, correction="fdr_bh")
        hbo.plot(
            ax=ax2,
            mask=~sig,
            title=f"HbO -- significant edges (FDR, p < {alpha})",
        )
    except cn.NotFittedError:
        # p-values unavailable (e.g. coherence); fall back to the full matrix.
        hbo.plot(ax=ax2, title="HbO -- connectivity (no p-values)")
    fig2.tight_layout()
    fig2_path = output_dir / "connectivity_hbo_significant.png"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    written.append(fig2_path)

    return written


def print_graph_metrics(matrix: cn.ConnectivityMatrix, *, threshold: float) -> None:
    """Print the headline graph-theory metrics for one connectivity matrix."""
    metrics = cn.graph.graph_metrics(matrix, threshold=threshold)
    print(f"\nGraph-theory metrics (HbO, |r| >= {threshold}):")
    for key in (
        "n_nodes",
        "n_edges",
        "density",
        "global_efficiency",
        "average_clustering",
        "transitivity",
        "modularity",
        "n_communities",
    ):
        value = metrics.get(key)
        if isinstance(value, float):
            print(f"  {key:<20} {value:.4f}")
        else:
            print(f"  {key:<20} {value}")

    # Highlight the strongest hub by weighted degree (strength).
    strength = metrics.get("nodal", {}).get("degree", {})
    if strength:
        hub, value = max(strength.items(), key=lambda kv: kv[1])
        print(f"  {'top hub (strength)':<20} {hub} ({value:.3f})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute and visualise fNIRS connectivity from a SNIRF file.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to a .snirf file (default: search ./data then sample data).",
    )
    parser.add_argument(
        "--method",
        default="pearson",
        choices=[e["name"] for e in cn.list_estimators()],
        help="Connectivity estimator to use (default: pearson).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively (headless save-only by default).",
    )
    parser.add_argument(
        "--output",
        default="example_output",
        help="Directory for the saved PNG figures (default: ./example_output).",
    )
    args = parser.parse_args(argv)

    path = resolve_path(args.path)
    if path is None:
        print(
            "No SNIRF file found.\n\n"
            "Provide one explicitly:\n"
            "    python example.py path/to/file.snirf\n\n"
            "or drop a .snirf file into ./data/ and re-run.",
        )
        return 0
    if not Path(path).exists():
        print(f"File not found: {path}")
        return 0

    print(f"Loading SNIRF: {path}")
    ts = load_snirf(path)
    print(
        f"  -> {ts.n_channels} channels x {ts.n_times} samples "
        f"@ {ts.sfreq:g} Hz, chromophores {ts.chromophores}"
    )

    print(f"\nEstimating {args.method} connectivity ...")
    result = cn.connectivity(ts, method=args.method)
    print(result.report())

    # graph metrics on HbO at a sensible correlation threshold.
    hbo = result["HbO"] if "HbO" in result.chromophores else result[result.chromophores[0]]
    print_graph_metrics(hbo, threshold=0.5)

    output_dir = Path(args.output)
    written = make_figures(result, output_dir)
    print("\nSaved figures:")
    for fig_path in written:
        print(f"  {fig_path.resolve()}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
