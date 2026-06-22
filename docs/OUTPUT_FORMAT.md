# nirconn output format

This document specifies the formats nirconn produces so other programs can parse
a complete analysis. It is a **stable contract**: column names, JSON keys and
conventions below are what downstream tools should rely on.

There are four kinds of output:

| Kind | Format | Machine-readable | Produced by |
|------|--------|------------------|-------------|
| Edge / node / component **tables** | CSV (or Markdown) | ✅ | `cn.tables.*`, `GroupStatResult.table()`, `ConnectivityMatrix.edges()` |
| Complete **result report** | JSON (from a nested dict) | ✅ | `cn.report.result_report()` |
| Connectivity **matrices** | CSV (square) or netCDF/xarray | ✅ | `ConnectivityMatrix.to_dataframe()` / `.to_dataset()` |
| **Text report** / **poster** | `.txt` / `.png` | ❌ (human-facing) | `cn.report.summarize_*`, `cn.build_poster()` |

All tables are `pandas.DataFrame`s; write them with `df.to_csv(path, index=False)`
(the convention used throughout). Everything numeric is IEEE float64 unless noted.

---

## 1. Conventions (apply to every output)

- **Channel labels**: source–detector pairs as `S<src>_D<det>` (1-based, e.g.
  `S1_D1`), matching the SNIRF probe indexing.
- **Chromophore**: one of `HbO`, `HbR`, `HbT`, `OD`, `raw`, `unknown` (string).
- **Edge identifier** (`edge` column): `"<source>--<target>"` for undirected
  (functional) results, `"<source>-><target>"` for directed (effective) results.
- **Undirected vs directed**: functional methods (pearson, spearman, partial,
  coherence, plv, wavelet_coherence) are symmetric — tables list each unordered
  pair **once** (upper triangle). Directed/effective methods (e.g. granger,
  where `matrix[src, tgt]` is the influence of `src` on `tgt`) list every
  ordered pair.
- **Statistics are computed in Fisher-z space and reported in r-units.** `effect`
  is in correlation/native units; group tests internally use `z = arctanh(r)`.
- **p vs q**: `p` is the raw per-edge p-value; `q` is the multiple-comparison
  adjusted value (Benjamini–Hochberg FDR by default; also `fdr_by`, `bonferroni`,
  or none). `significant` is the post-correction decision at `alpha`.
- **Missing values**: in CSV, a missing/NaN number is written as an **empty
  field**. Booleans are `True` / `False`. A `NaN` correlation means the edge could
  not be estimated (e.g. a constant channel, or no subject had both channels).
- **Diagonal / self-edges are never included** in edge tables.
- **Encoding**: UTF-8. Text reports are ASCII-safe.

---

## 2. Edge tables

Produced by `cn.tables.significant_edges_table(obj, ...)` and the convenience
`GroupStatResult.table()` / `ConnectivityMatrix`-based calls. One row per unique
edge. Sorted with significant edges first, then by the chosen key (`q` ascending
by default). `include_nonsignificant=False` keeps only significant rows; `top_n`
truncates.

The column set depends on whether the source is a **group statistic** or a
**single matrix**.

### 2a. Group-level edge table (`GroupStatResult.table()`)

| Column | Type | Description |
|--------|------|-------------|
| `chromophore` | string | Chromophore the test was run on. |
| `source` | string | Source-detector label of node *i*. |
| `target` | string | Source-detector label of node *j*. |
| `edge` | string | `source--target` (or `->` if directed). |
| `effect` | float | Effect in r-units: group-mean *r* (one-sample), mean *r*-difference (two-sample/paired), or partial correlation (regression). |
| `t` | float | Test statistic (t). |
| `df` | float | Degrees of freedom (may be fractional for Welch). |
| `p` | float | Two-sided raw p-value. |
| `q` | float | Corrected p-value (q) at the table's correction method. |
| `ci_low` | float | Lower confidence bound (r-units for one-sample; z-units for contrasts). |
| `ci_high` | float | Upper confidence bound. |
| `cohens_d` | float | Effect size (Cohen's d). |
| `n` | int | Subjects contributing to this edge (varies with coverage). |
| `significant` | bool | `q <= alpha` after correction. |
| `direction` | string | `+` or `-` (sign of `effect`). |

### 2b. Subject-level / matrix edge table (a single `ConnectivityMatrix`)

| Column | Type | Description |
|--------|------|-------------|
| `chromophore` | string | Chromophore. |
| `source`, `target` | string | Node labels. |
| `edge` | string | `source--target` / `source->target`. |
| `effect` | float | The connectivity weight (e.g. Pearson *r*, coherence). |
| `fisher_z` | float | `arctanh(effect)` (variance-stabilised). |
| `t` | float | Empty (not defined for a raw matrix). |
| `df` | float | Empty. |
| `p` | float | Per-edge p-value: analytic (pearson/spearman/partial) or, when `surrogates=` was requested, an empirical phase-surrogate p-value (plv/coherence/wavelet_coherence). Empty if the estimator computed none. |
| `q` | float | FDR-corrected p-value (empty if no p). |
| `significant` | bool | `q <= alpha`. |
| `direction` | string | `+` / `-`. |

> The bare matrix has no `t`/`df`/`ci`/`cohens_d`/`n`; the group table has no
> `fisher_z`. Parse by **column name**, not position.

---

## 3. Node and global graph tables

### 3a. Nodal metrics — `cn.tables.nodal_summary_table(matrix, threshold=...)`

One row per node. Sorted by `strength` descending.

| Column | Type | Description |
|--------|------|-------------|
| `chromophore` | string | Chromophore. |
| `node` | string | Channel label. |
| `strength` | float | Weighted degree (sum of \|edge weights\|). |
| `degree_centrality` | float | networkx degree centrality. |
| `betweenness_centrality` | float | networkx betweenness centrality. |
| `clustering` | float | Weighted clustering coefficient. |
| `rank` | int | 1-based rank by strength. |
| `is_hub` | bool | Strength in the top decile. |

### 3b. Global metrics — `cn.tables.global_graph_table(matrix, threshold=...)`

One row (per chromophore).

| Column | Type | Description |
|--------|------|-------------|
| `chromophore` | string | Chromophore. |
| `n_nodes` | int | Nodes in the thresholded graph. |
| `n_edges` | int | Surviving edges. |
| `density` | float | Graph density. |
| `global_efficiency` | float | Global efficiency. |
| `average_clustering` | float | Mean clustering coefficient. |
| `transitivity` | float | Transitivity. |
| `modularity` | float | Modularity of the greedy community partition. |
| `n_communities` | int | Number of communities. |
| `threshold` | float | Absolute weight threshold applied (empty if none). |

---

## 4. NBS component table — `NBSResult.table()`

One row per connected sub-network found by the Network-Based Statistic.

| Column | Type | Description |
|--------|------|-------------|
| `component` | int | 1-based component id (sorted largest first). |
| `chromophore` | string | Chromophore. |
| `n_nodes` | int | Nodes in the component. |
| `n_edges` | int | Suprathreshold edges (the component "size"). |
| `component_statistic` | float | Component size statistic used for inference. |
| `p_fwe` | float | Family-wise-error corrected p-value (permutation). |
| `n_permutations` | int | Permutations used. |
| `threshold` | float | Primary edge t-threshold. |
| `significant` | bool | `p_fwe <= alpha`. |
| `nodes` | string | Comma-separated member node labels. |

> NBS significance is at the **component** level — individual edges within a
> significant component are not themselves declared significant.

---

## 5. Group summary table — `cn.tables.group_summary_table(study_or_group)`

One row per chromophore (Study) or one row (GroupConnectivity): `chromophore`,
`method`, `n_subjects`, `n_channels`, `n_edges`.

---

## 6. Complete result report (JSON) — `cn.report.result_report(result)`

The **canonical machine-readable summary** of a single (subject-level)
`ConnectivityResult`. Returns a nested, JSON-serialisable dict; dump with
`json.dump(cn.report.result_report(result), f)`.

```jsonc
{
  "meta": {
    "method": "pearson",        // estimator name
    "kind": "functional",       // "functional" | "effective"
    "directed": false,
    "alpha": 0.05,
    "correction": "fdr_bh",     // or "fdr_by" | "bonferroni" | null
    "threshold": 0.5            // graph threshold used for the graph block
  },
  "provenance": {               // from the time series, may contain nulls
    "n_channels": 49,
    "n_times": 3480,
    "sfreq": 12.21,
    "name": "sub1"
  },
  "params": { /* estimator parameters, e.g. {"compute_pvalues": true} */ },
  "chromophores": {
    "HbO": {
      "summary": {
        "method": "pearson", "kind": "functional", "directed": false,
        "chromophore": "HbO", "n_channels": 49, "n_edges": 1176,
        "mean": 0.41, "std": 0.22, "min": -0.30, "max": 0.98,
        "n_significant_p<.05": 880        // present only if p-values exist
      },
      "distribution": {
        "percentiles": { "5": -0.05, "25": 0.25, "50": 0.42, "75": 0.58, "95": 0.80 },
        "frac_negative": 0.06
      },
      "significance": {                   // present only if p-values exist
        "alpha": 0.05, "correction": "fdr_bh",
        "n_edges": 1176, "n_significant": 880
      },
      "graph": {                          // null if networkx unavailable
        "n_nodes": 49, "n_edges": 540, "density": 0.45,
        "global_efficiency": 0.71, "average_clustering": 0.52,
        "transitivity": 0.68, "modularity": 0.27, "n_communities": 3
      },
      "hubs": [ { "node": "S5_D3", "strength": 26.4 }, /* ...top_n... */ ],
      "edges": [ /* list of edge records, schema = section 2b */ ]
    }
    // ... one entry per chromophore ...
  }
}
```

Notes:
- `significance`, `graph`, `hubs` keys are **omitted/null** when not applicable
  (no p-values; networkx missing). Always check for presence.
- `edges` records follow the **single-matrix** edge schema (section 2b).
- All numbers are plain JSON floats/ints (numpy types are cast).

There is no single built-in JSON dump for a *group* `GroupStatResult`; use its
`.table()` (CSV, section 2a) plus `.summary()` (a flat dict: `test`, `design`,
`chromophore`, `method`, `n_channels`, `n_edges_tested`, `n_significant`, `alpha`,
`correction`, `min_q`, `n_subjects`).

---

## 7. Connectivity matrices

A matrix is not auto-serialised; export it explicitly:

- **Square CSV** (labelled rows/cols): `matrix.to_dataframe().to_csv(path)`. The
  first column/row are channel labels; cell `[i, j]` is the connectivity weight;
  the diagonal is 1.0; unestimated cells are empty (NaN).
- **xarray / netCDF** (keeps `source`/`target` coords, all chromophores):
  `result.to_dataset().to_netcdf(path)`.
- **Raw array**: `matrix.values` (N×N float64), labels in `matrix.labels`.

`p`-values, when present, live in `matrix.pvalues` (N×N, same ordering); they are
not in the square CSV — use the edge table (section 2) for per-edge p/q.

---

## 8. Text reports & posters (human-facing, not parse targets)

- `cn.report.summarize_result()` / `summarize_group()` / `Study.report()` →
  sectioned plain-text (`.txt`). Layout is for humans; **do not parse it** —
  every number in it is available structured via sections 2–6.
- `cn.build_poster(...)` → a multi-panel `matplotlib` figure saved as `.png`
  (metadata, matrices, significance, edge distribution, graph metrics, hubs,
  connectogram, findings table). Image only.

---

## 9. Reference: analysis directory layout

The example scripts (`examples/*/`) write a conventional tree. This layout is
defined by the *script*, not the library, but it is a useful pattern for a
"complete analysis" a parser would walk:

```
output/
  summary.txt                          # human-readable run overview
  loading_log.txt                      # per-recording sfreq / length / channels
  per_condition/
    <cond>_poster.png                  # poster (image)
    <cond>_significant_edges.csv       # group edge table (section 2a)
    <cond>_report.txt                  # text report
  contrasts/
    <a>_vs_<b>_edges.csv               # group edge table (section 2a)
    <headline>_poster.png
```

For a programmatic, language-agnostic consumer the recommended targets are the
**CSV edge/node/component tables** (sections 2–5) and the **`result_report()`
JSON** (section 6); treat `.txt`/`.png` as presentation artifacts.

---

## 10. Versioning

This format ships with nirconn `__version__` (see `nirconn._version`). Columns
and JSON keys here are additive-stable within a major version; new optional
columns/keys may be appended. Parse by name and tolerate unknown extra fields.
