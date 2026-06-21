"""Rich result objects returned by every estimator.

* :class:`ConnectivityMatrix` -- one N×N matrix (a single chromophore) plus its
  optional p-values and the metadata describing how it was produced. It knows
  how to threshold itself, become a :mod:`networkx` graph, test significance,
  tabulate, and plot.
* :class:`ConnectivityResult` -- the top-level object an estimator returns: a
  mapping from chromophore to :class:`ConnectivityMatrix` plus provenance. For
  the common single-chromophore case, ``result.matrix`` gives the one matrix.

The heavy/optional machinery (matplotlib, networkx, statistics) is imported
lazily inside methods so that ``import cedanirs`` stays cheap and side-effect
free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Mapping, Sequence

import numpy as np
import xarray as xr

from .exceptions import DataError, NotFittedError
from .types import ConnectivityKind

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx
    import pandas as pd
    from matplotlib.axes import Axes

SOURCE_DIM = "source"
TARGET_DIM = "target"


class ConnectivityMatrix:
    """A single N×N connectivity matrix with metadata and behaviour.

    The matrix is stored as an :class:`xarray.DataArray` with ``source`` and
    ``target`` dims both labelled by channel name, so ``m.data.sel(source="S1_D1",
    target="S2_D2")`` reads naturally and tabulation is free.

    Parameters
    ----------
    values:
        N×N array of connectivity strengths.
    labels:
        Channel labels (length N).
    method:
        Identifier of the estimator that produced this (e.g. ``"pearson"``).
    kind:
        :class:`~cedanirs.core.types.ConnectivityKind`.
    directed:
        Whether the matrix is asymmetric (``source -> target``).
    chromophore:
        The chromophore this matrix was computed on.
    pvalues:
        Optional N×N array of p-values aligned with ``values``.
    params:
        The estimator parameters, kept for provenance.
    """

    __slots__ = ("_da", "_p", "method", "kind", "directed", "chromophore", "params")

    def __init__(
        self,
        values: np.ndarray,
        labels: Sequence[str],
        *,
        method: str,
        kind: ConnectivityKind = ConnectivityKind.FUNCTIONAL,
        directed: bool = False,
        chromophore: str | None = None,
        pvalues: np.ndarray | None = None,
        params: Mapping | None = None,
    ):
        values = np.asarray(values, dtype=float)
        n = len(labels)
        if values.shape != (n, n):
            raise DataError(
                f"Matrix shape {values.shape} does not match {n} labels."
            )
        labels = [str(x) for x in labels]
        self._da = xr.DataArray(
            values,
            dims=(SOURCE_DIM, TARGET_DIM),
            coords={SOURCE_DIM: labels, TARGET_DIM: labels},
            name=method,
        )
        if pvalues is not None:
            pvalues = np.asarray(pvalues, dtype=float)
            if pvalues.shape != (n, n):
                raise DataError(
                    f"pvalues shape {pvalues.shape} does not match {n} labels."
                )
        self._p = pvalues
        self.method = method
        self.kind = ConnectivityKind(kind)
        self.directed = bool(directed)
        self.chromophore = chromophore
        self.params = dict(params or {})

    # -- basic access --------------------------------------------------------

    @property
    def data(self) -> xr.DataArray:
        """The matrix as a labelled :class:`xarray.DataArray`."""
        return self._da

    @property
    def values(self) -> np.ndarray:
        """The matrix as a plain NumPy array."""
        return np.asarray(self._da.values)

    @property
    def labels(self) -> list[str]:
        return [str(x) for x in self._da.coords[SOURCE_DIM].values]

    @property
    def n(self) -> int:
        return int(self._da.sizes[SOURCE_DIM])

    @property
    def pvalues(self) -> np.ndarray | None:
        """The p-value matrix, or ``None`` if the estimator did not compute it."""
        return self._p

    def to_dataframe(self) -> "pd.DataFrame":
        """The matrix as a labelled, square :class:`pandas.DataFrame`."""
        import pandas as pd

        return pd.DataFrame(self.values, index=self.labels, columns=self.labels)

    def edges(self, *, include_diagonal: bool = False) -> "pd.DataFrame":
        """A long-form table of edges: ``source, target, weight[, pvalue]``.

        For undirected matrices only the upper triangle is returned (each pair
        once); for directed matrices every ordered pair is returned.
        """
        import pandas as pd

        vals = self.values
        n = self.n
        rows = []
        for i in range(n):
            for j in range(n):
                if i == j and not include_diagonal:
                    continue
                if not self.directed and j < i:
                    continue
                row = {
                    "source": self.labels[i],
                    "target": self.labels[j],
                    "weight": vals[i, j],
                }
                if self._p is not None:
                    row["pvalue"] = self._p[i, j]
                rows.append(row)
        return pd.DataFrame(rows)

    # -- transforms ----------------------------------------------------------

    def fisher_z(self) -> "ConnectivityMatrix":
        """Return a copy with values Fisher *z*-transformed (``arctanh``).

        Only meaningful for correlation-valued matrices in ``[-1, 1]``; used to
        stabilise variance before averaging across runs/subjects.
        """
        from ..stats.transforms import fisher_z

        out = self._clone(fisher_z(self.values))
        out.params = {**self.params, "transform": "fisher_z"}
        return out

    def threshold(
        self,
        value: float | None = None,
        *,
        percentile: float | None = None,
        density: float | None = None,
        absolute: bool = True,
        binarize: bool = False,
    ) -> "ConnectivityMatrix":
        """Return a copy with weak edges zeroed out.

        Exactly one of ``value``, ``percentile`` or ``density`` selects the
        cut-off:

        * ``value`` -- keep edges with strength above this absolute threshold.
        * ``percentile`` -- keep the top ``(100 - percentile)`` percent of edges.
        * ``density`` -- keep the strongest fraction (0-1) of possible edges.

        ``absolute`` (default) compares ``|weight|`` so strong negative
        correlations survive. ``binarize`` replaces survivors with 1.0.
        """
        chosen = [x is not None for x in (value, percentile, density)]
        if sum(chosen) != 1:
            raise DataError(
                "Specify exactly one of value=, percentile= or density=."
            )

        vals = self.values.copy()
        comp = np.abs(vals) if absolute else vals

        # Off-diagonal pool used to derive percentile/density cut-offs.
        mask_off = ~np.eye(self.n, dtype=bool)
        pool = comp[mask_off]

        if value is not None:
            cutoff = float(value)
        elif percentile is not None:
            cutoff = float(np.nanpercentile(pool, percentile))
        else:  # density
            if not 0.0 < density <= 1.0:
                raise DataError("density must be in (0, 1].")
            keep_n = max(1, int(round(density * pool.size)))
            order = np.sort(pool[~np.isnan(pool)])[::-1]
            cutoff = float(order[min(keep_n, order.size) - 1])

        keep = comp >= cutoff
        out_vals = np.where(keep, (1.0 if binarize else vals), 0.0)
        np.fill_diagonal(out_vals, 0.0)
        out = self._clone(out_vals)
        out.params = {**self.params, "thresholded": True}
        return out

    # -- statistics ----------------------------------------------------------

    def significant(
        self,
        alpha: float = 0.05,
        *,
        correction: str | None = "fdr_bh",
    ) -> np.ndarray:
        """Boolean mask of edges significant at level ``alpha``.

        Multiple-comparison ``correction`` is applied across the unique
        off-diagonal edges (upper triangle for undirected matrices, all ordered
        pairs for directed). Supported: ``"fdr_bh"``, ``"bonferroni"`` and
        ``None`` (uncorrected).
        """
        if self._p is None:
            raise NotFittedError(
                f"The {self.method!r} matrix has no p-values to test."
            )
        from ..stats.significance import correct_pvalues

        n = self.n
        if self.directed:
            idx = np.where(~np.eye(n, dtype=bool))
        else:
            idx = np.triu_indices(n, k=1)

        flat_p = self._p[idx]
        reject, _ = correct_pvalues(flat_p, alpha=alpha, method=correction)

        mask = np.zeros((n, n), dtype=bool)
        mask[idx] = reject
        if not self.directed:
            mask = mask | mask.T
        return mask

    # -- graph ---------------------------------------------------------------

    def to_graph(self, *, threshold: float | None = None, absolute: bool = True) -> "nx.Graph":
        """Build a :mod:`networkx` graph (``DiGraph`` when directed).

        Edge weights are the connectivity strengths. Self-loops are dropped.
        An optional ``threshold`` prunes weak edges first.
        """
        from ..graph.metrics import to_graph

        return to_graph(self, threshold=threshold, absolute=absolute)

    # -- visualisation -------------------------------------------------------

    def plot(self, **kwargs) -> "Axes":
        """Render the matrix as a heatmap (see :func:`cedanirs.viz.plot_matrix`)."""
        from ..viz.matrix import plot_matrix

        return plot_matrix(self, **kwargs)

    # -- summaries -----------------------------------------------------------

    def summary(self) -> dict:
        """A small dict of descriptive statistics over the off-diagonal edges."""
        n = self.n
        off = self.values[~np.eye(n, dtype=bool)]
        finite = off[np.isfinite(off)]
        out = {
            "method": self.method,
            "kind": str(self.kind),
            "directed": self.directed,
            "chromophore": self.chromophore,
            "n_channels": n,
            "n_edges": int(finite.size if self.directed else finite.size // 2),
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "std": float(np.std(finite)) if finite.size else float("nan"),
            "min": float(np.min(finite)) if finite.size else float("nan"),
            "max": float(np.max(finite)) if finite.size else float("nan"),
        }
        if self._p is not None:
            out["n_significant_p<.05"] = int(np.sum(self.significant(0.05)) // (1 if self.directed else 2))
        return out

    # -- internals -----------------------------------------------------------

    def _clone(self, values: np.ndarray) -> "ConnectivityMatrix":
        return ConnectivityMatrix(
            values,
            self.labels,
            method=self.method,
            kind=self.kind,
            directed=self.directed,
            chromophore=self.chromophore,
            pvalues=self._p,
            params=self.params,
        )

    def __array__(self, dtype=None):  # pragma: no cover - numpy protocol
        return np.asarray(self.values, dtype=dtype)

    def __repr__(self) -> str:
        chrom = f", {self.chromophore}" if self.chromophore else ""
        arrow = "directed" if self.directed else "symmetric"
        return (
            f"ConnectivityMatrix({self.method}{chrom}, {self.n}x{self.n}, "
            f"{arrow})"
        )


class ConnectivityResult:
    """The object returned by an estimator: one matrix per chromophore.

    Behaves like an ordered mapping ``{chromophore: ConnectivityMatrix}`` and,
    in the common single-chromophore case, exposes that one matrix directly via
    :attr:`matrix`.
    """

    __slots__ = ("_matrices", "method", "kind", "directed", "params", "provenance")

    def __init__(
        self,
        matrices: Mapping[str, ConnectivityMatrix],
        *,
        method: str,
        kind: ConnectivityKind = ConnectivityKind.FUNCTIONAL,
        directed: bool = False,
        params: Mapping | None = None,
        provenance: Mapping | None = None,
    ):
        if not matrices:
            raise DataError("ConnectivityResult requires at least one matrix.")
        self._matrices: dict[str, ConnectivityMatrix] = dict(matrices)
        self.method = method
        self.kind = ConnectivityKind(kind)
        self.directed = bool(directed)
        self.params = dict(params or {})
        self.provenance = dict(provenance or {})

    @property
    def chromophores(self) -> list[str]:
        return list(self._matrices.keys())

    @property
    def matrix(self) -> ConnectivityMatrix:
        """The single matrix, when there is exactly one chromophore."""
        if len(self._matrices) != 1:
            raise DataError(
                f"Result holds {len(self._matrices)} matrices "
                f"({self.chromophores}); index by chromophore instead, "
                f"e.g. result['HbO']."
            )
        return next(iter(self._matrices.values()))

    @property
    def matrices(self) -> dict[str, ConnectivityMatrix]:
        return dict(self._matrices)

    def __getitem__(self, chromophore: str) -> ConnectivityMatrix:
        from .types import Chromophore

        key = str(Chromophore.coerce(chromophore))
        if key in self._matrices:
            return self._matrices[key]
        if chromophore in self._matrices:
            return self._matrices[chromophore]
        raise EstimatorKeyError(chromophore, self.chromophores)

    def __iter__(self) -> Iterator[str]:
        return iter(self._matrices)

    def __len__(self) -> int:
        return len(self._matrices)

    def items(self):
        return self._matrices.items()

    def values(self):
        return self._matrices.values()

    def keys(self):
        return self._matrices.keys()

    def to_dataset(self) -> xr.Dataset:
        """Stack every chromophore matrix into one :class:`xarray.Dataset`."""
        return xr.Dataset({c: m.data for c, m in self._matrices.items()})

    def summary(self) -> dict:
        """Per-chromophore summary dicts, keyed by chromophore."""
        return {c: m.summary() for c, m in self._matrices.items()}

    def report(self, **kwargs) -> str:
        """A formatted, human-readable text report (see :mod:`cedanirs.report`)."""
        from ..report import summarize_result

        return summarize_result(self, **kwargs)

    def plot(self, **kwargs):
        """Plot every chromophore matrix; returns a single Axes or a list."""
        from ..viz.matrix import plot_matrix

        if len(self._matrices) == 1:
            return self.matrix.plot(**kwargs)
        return [
            plot_matrix(m, title=f"{self.method} — {c}", **kwargs)
            for c, m in self._matrices.items()
        ]

    def __repr__(self) -> str:
        return (
            f"ConnectivityResult({self.method}, "
            f"chromophores={self.chromophores}, "
            f"{'directed' if self.directed else 'symmetric'})"
        )


class EstimatorKeyError(DataError, KeyError):
    """Raised when indexing a result by a chromophore it does not contain."""

    def __init__(self, key: str, available: list[str]):
        super().__init__(
            f"No matrix for chromophore {key!r}; available: {available}."
        )
