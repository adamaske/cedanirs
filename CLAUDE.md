# nirconn — project guide for Claude

## What this project is

**nirconn** (`import nirconn as cn`) is a Python package for functional and
effective connectivity analysis of fNIRS data. It covers the full pipeline:
preprocessed haemoglobin time series → connectivity estimation → group-level
statistics → graph metrics → visualisation and reporting.

The package lives at `C:\Users\adama\dev\cedanirs\` (the *folder* kept the old
name; the *package* is `nirconn/`).

---

## Install

```bash
pip install -e .          # core (numpy, scipy, xarray, pandas, matplotlib, networkx)
pip install -e .[io]      # + h5py for the native SNIRF reader
pip install -e .[dev]     # + pytest, ruff
```

The editable install is mandatory — do not install from PyPI.

---

## Running tests

```bash
python -m pytest          # full suite (~127 tests, ~10-25 s)
python -m pytest tests/test_estimators.py   # estimators only
```

All 127 tests must pass before committing. The three RuntimeWarning lines from
`test_group.py` (scipy precision loss on near-identical data) are expected and
harmless.

---

## Package layout

```
nirconn/
  api.py                  connectivity() — the one-liner entry point
  _version.py             __version__ = "0.1.0"
  core/
    types.py              Chromophore, ConnectivityKind, Domain enums
    exceptions.py         CedanirsError hierarchy (DataError, DependencyError, ...)
    timeseries.py         NirsTimeSeries — (chromophore, channel, time) xarray cube
    result.py             ConnectivityMatrix / ConnectivityResult
    group.py              Study + GroupConnectivity — group data model
    registry.py           @register_estimator / get / create / list
  estimators/
    base.py               ConnectivityEstimator ABC; _surrogate_pvalues helper
    functional/           pearson, spearman, partial, coherence, plv, wavelet_coherence
    effective/            granger (directed, F-test p-values)
  stats/
    significance.py       correlation_pvalues, fdr_correction, bonferroni_correction
    transforms.py         fisher_z, inverse_fisher_z, average_correlations
    surrogates.py         phase_randomize, surrogate_pvalues (Theiler 1992)
    group.py              one_sample, two_sample, paired, regression, nbs
    _results.py           GroupStatResult, NBSResult
  graph/                  networkx graph + metrics (graph_metrics, etc.)
  io/                     native h5py SNIRF reader: read_snirf, read_timeseries
  tables.py               significant_edges_table, nodal_summary_table, ...
  viz/
    matrix.py             ConnectivityMatrix heatmap
    poster.py             build_poster() — 8-panel figure
  report/                 summarize_result, result_report() JSON, summarize_group
  preprocessing/          Preprocessor (cedalion backend, lazy)
tests/                    pytest suite — one file per module
examples/
  generate_outputs.py     deterministic synthetic two-group outputs → examples/output/
  sivert_writing/         real-data case study (6 subj × 4 conditions)
    run_connectivity.py   Pearson → output/
    run_wavelet.py        wavelet coherence → output_wavelet/
    run_granger.py        Granger causality → output_granger/
docs/
  OUTPUT_FORMAT.md        machine-readable output contract (CSV/JSON schemas)
```

---

## Key design rules

### Adding a new estimator

1. Create `nirconn/estimators/functional/<name>.py` (or `effective/`).
2. Subclass `ConnectivityEstimator`, set `name`, `kind`, `directed`, `domain`.
3. Implement `_estimate(self, x: np.ndarray) -> EstimateOutput` on a 2-D
   `(channel, time)` array. Return `EstimateOutput(matrix, pvalues)`.
4. Decorate with `@register_estimator(name="...")`.
5. Import in the package's `functional/__init__.py` (or `effective/`).
6. Export from `nirconn/__init__.py`.
7. Add tests in `tests/test_estimators.py`.

For estimators without analytic p-values (PLV, coherence, WTC), split the math
into a `_statistic(self, x)` method and call `self._surrogate_pvalues(x,
self._statistic)` to support the `surrogates=` opt-in.

### Statistics cardinal rule

- **Aggregate and test in Fisher-z space; report in r-units.**
- `average_correlations()` = `inverse_fisher_z(nanmean(fisher_z(matrices)))`.
- `one_sample` / `two_sample` / `paired` / `regression` all work in z-space.
- GCI (Granger) values are not correlations — use `fisher=False` averaging and
  arithmetic nanmean; see `run_granger.py` for the pattern.

### Group pipeline

```python
study = cn.Study(name="...")
study.add(subject_id, result, condition=..., group=...)

g = study.group("HbO", condition="rest", min_coverage=0.5)
stat = g.one_sample(alpha=0.05, correction="fdr_bh")

stat.table()            # pandas DataFrame — significant edges
cn.build_poster(stat, group_mean=g.mean())
```

### Directed matrices (Granger)

- `matrix[src, tgt]` = influence of `src` on `tgt` (row = source, col = target).
- `edges()` lists all N×(N-1) ordered pairs.
- Diagonal is 0 (not 1) for GCI — set explicitly after any averaging.
- Poster threshold must be scaled to GCI magnitude (~0.05, not 0.5).

---

## SNIRF loading

```python
ts = cn.read_timeseries("sub01.snirf", bandpass=(0.01, 0.1))
# Returns NirsTimeSeries with HbO/HbR chromophores, sfreq set.
# Native h5py reader — no MNE needed for already-processed files.
```

The Sivert dataset lives at `~/nirs/sivertdata/relabeled/results/<condition>/`.
Pass `SIVERT_ROOT` env var to the analysis scripts to point at it.

---

## Surrogate significance

Coherence, PLV and wavelet coherence support empirical p-values:

```python
m = cn.connectivity(ts, method="plv", sfreq=10.0,
                    fmin=0.01, fmax=0.1,
                    surrogates=500, surrogate_seed=0).matrix
```

Surrogates are Fourier phase-randomised (Theiler 1992): per-channel independent
random phases, preserving each channel's power spectrum. The null is one-sided
("greater"). Note: pure-sinusoid test data defeats the surrogate (the power
spectrum is a single spike, so surrogates stay phase-locked). Use broadband data.

---

## Output format

`docs/OUTPUT_FORMAT.md` is the stable parsing contract. Key points:
- Edge ID: `src--tgt` (undirected) or `src->tgt` (directed).
- `p` = raw p-value; `q` = FDR-corrected; `significant` = bool.
- NaN → empty field in CSV; booleans as `True`/`False`.
- Parse by column name, not position.
- The `result_report()` JSON is the canonical machine-readable single-subject summary.
