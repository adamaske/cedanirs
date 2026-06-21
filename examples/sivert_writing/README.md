# Case study: writing-modality functional connectivity

A real-data application of `cedanirs` to an fNIRS study of four writing input
methods. It exercises the full group pipeline: per-subject run aggregation →
group connectomes → within-subject condition contrasts → posters, tables and
reports.

> The raw SNIRF recordings are **not** included (human-subject data). This folder
> holds the analysis script and its committed outputs. To reproduce, point the
> script at your data with `SIVERT_ROOT=/path/to/data python run_connectivity.py`
> (expects `results/<condition>/*.snirf`).

## Design

- **6 subjects × 4 conditions**: handwriting, iPad, keyboard, reMarkable.
- Multiple runs per subject/condition (24 subject–condition cells, 102 runs).
- Already-processed HbO/HbR SNIRF, 49 long channels @ ~12.2 Hz, ~285 s each.

## Pipeline

1. Load each run (MNE), keep long channels, band-pass **0.01–0.1 Hz**.
2. Pearson connectivity per run; average a subject's runs in **Fisher-z** space →
   one matrix per (subject, condition).
3. Group analyses (`cedanirs`):
   - **per condition** — one-sample edgewise test (FC > 0), FDR-BH;
   - **every condition pair** — paired (within-subject) contrast, FDR-BH.

Subjects' montages differ, so group tests run on the **15 channels common to all
6 subjects** (105 edges). See [`output/summary.txt`](output/summary.txt) and
[`output/loading_log.txt`](output/loading_log.txt).

## Results

**Each condition has a strong, reliable group connectome** (one-sample HbO, FC > 0,
FDR `q < 0.05`):

| condition   | significant edges |
|-------------|-------------------|
| handwriting | 69 / 105 |
| iPad        | 95 / 105 |
| keyboard    | 63 / 105 |
| reMarkable  | 97 / 105 |

**No connectivity differences between writing modalities.** Every pairwise paired
contrast yields **0** FDR-significant edges, and the *uncorrected* counts (0–7 of
105, at or below the ~5 expected by chance) confirm this is a genuine null rather
than low power — at this sample size, band and channel set, resting-band HbO
connectivity does not distinguish the four input methods.

![Handwriting group connectome](output/per_condition/handwriting_poster.png)

## Outputs

- `output/per_condition/<cond>_poster.png` — full-pipeline poster per condition
- `output/per_condition/<cond>_significant_edges.csv` — significant edges + stats
- `output/per_condition/<cond>_report.txt` — text report
- `output/contrasts/<a>_vs_<b>_edges.csv` — every paired contrast (sorted by q)
- `output/contrasts/handwriting_vs_keyboard_poster.png` — headline contrast
- `output/summary.txt`, `output/loading_log.txt`

## Caveats

Small sample (n = 6); 15-channel common montage limits spatial coverage; HbO
only in the summary tables (HbR available by re-running). Treat the per-condition
connectomes as descriptive and the null contrasts as "no detectable difference at
this power", not proof of equivalence.
