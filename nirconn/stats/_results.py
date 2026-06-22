"""Result objects for group-level statistics.

Kept separate from :mod:`nirconn.stats.group` (which is pure math) so the
computation and the rich, plottable/tabulatable result containers stay
decoupled. A :class:`GroupStatResult` holds the full per-edge inferential
output of one group test; :class:`NBSResult` holds the components of a
Network-Based Statistic run.

The key design move is :meth:`GroupStatResult.to_matrix`, which returns an
ordinary :class:`~nirconn.core.result.ConnectivityMatrix` (effect as values,
p-values attached) so the *entire* existing surface -- ``.plot()``,
``.significant()``, ``.to_graph()``, ``graph_metrics``, the table builders --
works on group results with no extra code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import numpy as np

from ..core.result import ConnectivityMatrix
from ..core.types import ConnectivityKind

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from matplotlib.axes import Axes


class GroupStatResult:
    """Per-edge results of a group-level connectivity test.

    Attributes
    ----------
    test:
        Test name, e.g. ``"one_sample_t"``, ``"welch_t"``, ``"paired_t"``,
        ``"regression"``.
    effect:
        N x N effect in interpretable units (group-mean r for one-sample,
        mean r-difference for contrasts, standardized beta for regression).
    tstat, dof, pvalues, qvalues, reject:
        N x N statistic, degrees of freedom, raw and corrected p-values, and the
        post-correction rejection mask.
    extras:
        Dict of additional N x N arrays (``cohens_d``, ``hedges_g``, ``ci_low``,
        ``ci_high``, ``mean_z``, ``n``, and for two-sample ``mean_a``/``mean_b``).
    """

    __slots__ = (
        "labels", "chromophore", "method", "directed", "test", "design",
        "effect", "tstat", "dof", "pvalues", "qvalues", "reject",
        "extras", "params",
    )

    def __init__(
        self,
        *,
        labels,
        chromophore,
        method,
        directed,
        test,
        design,
        effect,
        tstat,
        dof,
        pvalues,
        qvalues,
        reject,
        extras: Mapping[str, np.ndarray] | None = None,
        params: Mapping | None = None,
    ):
        self.labels = [str(x) for x in labels]
        self.chromophore = chromophore
        self.method = method
        self.directed = bool(directed)
        self.test = test
        self.design = design
        self.effect = np.asarray(effect, dtype=float)
        self.tstat = np.asarray(tstat, dtype=float)
        self.dof = dof
        self.pvalues = np.asarray(pvalues, dtype=float)
        self.qvalues = np.asarray(qvalues, dtype=float)
        self.reject = np.asarray(reject, dtype=bool)
        self.extras = {k: np.asarray(v, dtype=float) for k, v in (extras or {}).items()}
        self.params = dict(params or {})

    @property
    def n(self) -> int:
        return len(self.labels)

    def significant(self, alpha: float | None = None) -> np.ndarray:
        """Boolean N x N mask of edges surviving correction.

        With no ``alpha`` returns the mask computed at analysis time; otherwise
        re-thresholds the stored corrected q-values.
        """
        if alpha is None:
            return self.reject.copy()
        mask = np.zeros((self.n, self.n), dtype=bool)
        finite = np.isfinite(self.qvalues)
        mask[finite] = self.qvalues[finite] <= alpha
        np.fill_diagonal(mask, False)
        return mask

    def to_matrix(self, value: str = "effect") -> ConnectivityMatrix:
        """Return a :class:`ConnectivityMatrix` of ``value`` with p-values attached.

        ``value`` is ``"effect"`` (default), ``"tstat"``, or any key in
        :attr:`extras` (e.g. ``"cohens_d"``).
        """
        if value == "effect":
            vals = self.effect
        elif value == "tstat":
            vals = self.tstat
        elif value in self.extras:
            vals = self.extras[value]
        else:
            raise KeyError(f"Unknown value {value!r}; have effect, tstat, {list(self.extras)}.")
        vals = vals.copy()
        np.fill_diagonal(vals, 1.0 if value == "effect" else 0.0)
        return ConnectivityMatrix(
            vals,
            self.labels,
            method=f"{self.method}:{self.test}",
            kind=ConnectivityKind.FUNCTIONAL,
            directed=self.directed,
            chromophore=self.chromophore,
            pvalues=self.pvalues,
            params={**self.params, "test": self.test, "value": value},
        )

    def table(self, **kwargs) -> "pd.DataFrame":
        """Tidy DataFrame of edges with statistics (see tables.significant_edges_table)."""
        from ..tables import significant_edges_table

        return significant_edges_table(self, **kwargs)

    def summary(self) -> dict:
        n = self.n
        off = ~np.eye(n, dtype=bool)
        n_tests = int(np.isfinite(self.pvalues[off]).sum() // (1 if self.directed else 2))
        n_sig = int(self.reject[off].sum() // (1 if self.directed else 2))
        finite_q = self.qvalues[off][np.isfinite(self.qvalues[off])]
        return {
            "test": self.test,
            "design": self.design,
            "chromophore": self.chromophore,
            "method": self.method,
            "n_channels": n,
            "n_edges_tested": n_tests,
            "n_significant": n_sig,
            "alpha": self.params.get("alpha"),
            "correction": self.params.get("correction"),
            "min_q": float(np.min(finite_q)) if finite_q.size else float("nan"),
            "n_subjects": self.params.get("n_subjects"),
        }

    def plot(self, **kwargs):
        """Plot the effect matrix, masking non-significant edges by default."""
        show_all = kwargs.pop("show_all", False)
        m = self.to_matrix()
        if not show_all and np.isfinite(self.pvalues).any():
            kwargs.setdefault("mask", ~self.significant())
        return m.plot(**kwargs)

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"GroupStatResult({self.test}, {self.chromophore}, "
            f"{s['n_significant']}/{s['n_edges_tested']} edges sig, "
            f"alpha={s['alpha']}, {s['correction']})"
        )


class NBSResult:
    """Network-Based Statistic result: significant connected sub-networks.

    Significance is assigned to whole *components*, not individual edges -- you
    may not claim any single edge inside a significant component is itself
    significant (that is the price NBS pays for its sensitivity to distributed
    effects).
    """

    __slots__ = (
        "labels", "chromophore", "components", "component_pvalues",
        "component_sizes", "component_stats", "edge_stat", "threshold",
        "n_permutations", "directed", "params",
    )

    def __init__(
        self,
        *,
        labels,
        chromophore,
        components: list[set],
        component_pvalues: np.ndarray,
        component_sizes: np.ndarray,
        component_stats: np.ndarray,
        edge_stat: np.ndarray,
        threshold: float,
        n_permutations: int,
        directed: bool = False,
        params: Mapping | None = None,
    ):
        self.labels = [str(x) for x in labels]
        self.chromophore = chromophore
        self.components = components
        self.component_pvalues = np.asarray(component_pvalues, dtype=float)
        self.component_sizes = np.asarray(component_sizes, dtype=float)
        self.component_stats = np.asarray(component_stats, dtype=float)
        self.edge_stat = np.asarray(edge_stat, dtype=float)
        self.threshold = float(threshold)
        self.n_permutations = int(n_permutations)
        self.directed = bool(directed)
        self.params = dict(params or {})

    @property
    def n(self) -> int:
        return len(self.labels)

    def significant(self, alpha: float = 0.05) -> np.ndarray:
        """Edge mask of all edges belonging to a component with p_FWE <= alpha."""
        idx = {lab: k for k, lab in enumerate(self.labels)}
        mask = np.zeros((self.n, self.n), dtype=bool)
        for comp, p in zip(self.components, self.component_pvalues):
            if p <= alpha:
                nodes = [idx[x] for x in comp]
                supra = np.abs(self.edge_stat) >= self.threshold
                for a in nodes:
                    for b in nodes:
                        if a != b and supra[a, b]:
                            mask[a, b] = True
        if not self.directed:
            mask = mask | mask.T
        return mask

    def table(self) -> "pd.DataFrame":
        from ..tables import nbs_components_table

        return nbs_components_table(self)

    def summary(self) -> dict:
        sig = int(np.sum(self.component_pvalues <= self.params.get("alpha", 0.05)))
        return {
            "n_components": len(self.components),
            "n_significant": sig,
            "threshold": self.threshold,
            "n_permutations": self.n_permutations,
            "alpha": self.params.get("alpha"),
            "largest_component_size": (
                float(np.max(self.component_sizes)) if self.component_sizes.size else 0.0
            ),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"NBSResult({s['n_significant']}/{s['n_components']} components sig, "
            f"threshold={self.threshold}, B={self.n_permutations})"
        )
