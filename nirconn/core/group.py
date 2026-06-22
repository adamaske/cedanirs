"""Group-level data model: :class:`Study` and :class:`GroupConnectivity`.

Connectivity science is mostly *group* science: a study has many subjects (often
under several conditions or split into groups), each with a per-chromophore
connectivity matrix, and the scientific claims ("this connection is reliably
present", "patients differ from controls", "connectivity tracks symptom score")
are statements about the population, not one recording.

Two label-driven containers make that tractable:

* :class:`Study` -- the user-facing collection. It holds each subject's
  :class:`~nirconn.core.result.ConnectivityResult` plus a single metadata table
  (group/condition labels and covariates). Metadata lives here and nowhere else.
* :class:`GroupConnectivity` -- an analysis-ready stack for *one* chromophore and
  *one* method/condition cell: an :class:`xarray.DataArray` of dims
  ``(subject, source, target)``. Stacking is done with an outer join on the
  channel-label coordinates, so alignment across subjects is automatic and a
  subject missing a channel becomes ``NaN`` (never silently mis-indexed).

The statistical tests themselves live in :mod:`nirconn.stats.group`; the
methods here delegate to them (imported lazily to avoid a core->stats import
cycle).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Mapping, Sequence

import numpy as np

from .exceptions import DataError
from .result import ConnectivityMatrix, ConnectivityResult
from .types import Chromophore

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    import xarray as xr
    from matplotlib.axes import Axes

    from ..stats._results import GroupStatResult, NBSResult

SUBJECT_DIM = "subject"
SOURCE_DIM = "source"
TARGET_DIM = "target"


class GroupConnectivity:
    """A stack of per-subject connectivity matrices for one chromophore/method.

    Backed by an :class:`xarray.DataArray` of dims ``(subject, source, target)``
    whose ``source``/``target`` coordinates are channel labels. Build one from a
    :class:`Study` (the usual path) or directly from a list of matrices via
    :meth:`from_matrices`.
    """

    __slots__ = ("_da", "method", "chromophore", "directed", "_covariates", "params")

    def __init__(
        self,
        data: "xr.DataArray",
        *,
        method: str,
        chromophore: str,
        directed: bool = False,
        covariates: "pd.DataFrame | None" = None,
        params: Mapping | None = None,
    ):
        if SUBJECT_DIM not in data.dims:
            raise DataError(f"GroupConnectivity data needs a {SUBJECT_DIM!r} dim.")
        self._da = data
        self.method = method
        self.chromophore = chromophore
        self.directed = bool(directed)
        self._covariates = covariates
        self.params = dict(params or {})

    # -- construction --------------------------------------------------------

    @classmethod
    def from_matrices(
        cls,
        matrices: Sequence[ConnectivityMatrix],
        *,
        ids: Sequence[str] | None = None,
        covariates: "pd.DataFrame | None" = None,
        join: str = "outer",
        min_coverage: float = 1.0,
    ) -> "GroupConnectivity":
        """Stack per-subject :class:`ConnectivityMatrix` objects into one cube.

        Channels are aligned by label (``join="outer"`` keeps the union and
        NaN-fills gaps; ``"inner"`` keeps the intersection). ``min_coverage`` then
        drops channels present in fewer than that fraction of subjects.
        """
        import pandas as pd
        import xarray as xr

        if not matrices:
            raise DataError("Need at least one matrix to build a group.")
        if ids is None:
            ids = [f"sub{i}" for i in range(len(matrices))]
        if len(ids) != len(matrices):
            raise DataError(f"Got {len(ids)} ids for {len(matrices)} matrices.")

        method = matrices[0].method
        chromophore = matrices[0].chromophore or str(Chromophore.UNKNOWN)
        directed = matrices[0].directed

        arrays = [m.data for m in matrices]
        index = pd.Index(list(ids), name=SUBJECT_DIM)
        stacked = xr.concat(arrays, dim=index, join=join)

        obj = cls(
            stacked,
            method=method,
            chromophore=chromophore,
            directed=directed,
            covariates=covariates,
        )
        if min_coverage < 1.0 or join == "outer":
            obj = obj._drop_low_coverage_channels(min_coverage)
        return obj

    def _drop_low_coverage_channels(self, min_coverage: float) -> "GroupConnectivity":
        # A channel is "present" for a subject if its row is not all-NaN.
        stack = self.stack(fisher=False)
        present = np.isfinite(stack).any(axis=2)  # (subject, channel)
        frac = present.mean(axis=0)  # (channel,)
        keep = frac >= min_coverage
        if keep.all():
            return self
        labels = np.asarray(self.labels)
        keep_labels = list(labels[keep])
        sub = self._da.sel({SOURCE_DIM: keep_labels, TARGET_DIM: keep_labels})
        return GroupConnectivity(
            sub,
            method=self.method,
            chromophore=self.chromophore,
            directed=self.directed,
            covariates=self._covariates,
            params={**self.params, "min_coverage": min_coverage},
        )

    # -- properties ----------------------------------------------------------

    @property
    def data(self) -> "xr.DataArray":
        return self._da

    @property
    def labels(self) -> list[str]:
        return [str(x) for x in self._da.coords[SOURCE_DIM].values]

    @property
    def subjects(self) -> list[str]:
        return [str(x) for x in self._da.coords[SUBJECT_DIM].values]

    @property
    def n_subjects(self) -> int:
        return int(self._da.sizes[SUBJECT_DIM])

    @property
    def n(self) -> int:
        return int(self._da.sizes[SOURCE_DIM])

    @property
    def covariates(self) -> "pd.DataFrame | None":
        return self._covariates

    @property
    def coverage(self) -> np.ndarray:
        """Per-edge count of subjects with a finite value (N x N)."""
        return np.isfinite(self.stack(fisher=False)).sum(axis=0)

    # -- data access ---------------------------------------------------------

    def stack(self, *, fisher: bool = True) -> np.ndarray:
        """The cube as ``(n_subjects, n, n)``; Fisher-z transformed if ``fisher``."""
        arr = np.asarray(self._da.values, dtype=float)
        if fisher:
            from ..stats.transforms import fisher_z

            arr = fisher_z(arr)
        return arr

    def mean(self, *, fisher: bool = True) -> ConnectivityMatrix:
        """Group-mean matrix (averaged in z-space when ``fisher``)."""
        from ..stats.transforms import average_correlations

        raw = self.stack(fisher=False)
        if fisher:
            mean_vals = average_correlations(raw, axis=0)
        else:
            mean_vals = np.nanmean(raw, axis=0)
        np.fill_diagonal(mean_vals, 1.0)
        return ConnectivityMatrix(
            mean_vals,
            self.labels,
            method=f"{self.method}-groupmean",
            directed=self.directed,
            chromophore=self.chromophore,
            params={**self.params, "n_subjects": self.n_subjects},
        )

    # -- statistics (delegated to stats.group) -------------------------------

    def one_sample(self, **kwargs) -> "GroupStatResult":
        """One-sample edgewise test (is group FC != popmean). See stats.group."""
        from ..stats.group import one_sample

        return one_sample(self, **kwargs)

    def two_sample(self, other: "GroupConnectivity", **kwargs) -> "GroupStatResult":
        """Two-sample / paired edgewise contrast vs ``other``. See stats.group."""
        from ..stats.group import two_sample

        return two_sample(self, other, **kwargs)

    def paired(self, other: "GroupConnectivity", **kwargs) -> "GroupStatResult":
        """Paired (within-subject) edgewise contrast vs ``other``."""
        from ..stats.group import two_sample

        return two_sample(self, other, paired=True, **kwargs)

    def regression(self, design, **kwargs) -> "GroupStatResult":
        """Edgewise OLS regression of FC (z) on a design. See stats.group."""
        from ..stats.group import regression

        return regression(self, design, **kwargs)

    def nbs(self, other: "GroupConnectivity | None" = None, **kwargs) -> "NBSResult":
        """Network-Based Statistic (permutation cluster test). See stats.group."""
        from ..stats.group import nbs

        return nbs(self, other, **kwargs)

    def plot(self, **kwargs) -> "Axes":
        """Plot the group-mean matrix."""
        return self.mean().plot(**kwargs)

    def __repr__(self) -> str:
        return (
            f"GroupConnectivity({self.method}, {self.chromophore}, "
            f"n_subjects={self.n_subjects}, {self.n}x{self.n})"
        )


class Study:
    """A collection of subjects' connectivity results plus their metadata.

    Add each subject's :class:`~nirconn.core.result.ConnectivityResult` with
    optional ``group``/``condition`` labels and covariates, then slice an
    analysis-ready :class:`GroupConnectivity` with :meth:`group`.

    Example
    -------
    ::

        study = nirconn.Study(name="rest-vs-task")
        for sid, rec in recordings.items():
            res = nirconn.connectivity(rec, method="pearson")
            study.add(sid, res, group="patient", age=values[sid])

        grp = study.group("HbO", group="patient")
        stat = grp.one_sample()          # which edges are reliably > 0
        print(stat.table())
    """

    __slots__ = ("name", "_results", "_meta_rows")

    def __init__(self, *, name: str | None = None):
        self.name = name
        # keyed by (subject_id, condition) so repeated-measures are first-class.
        self._results: dict[tuple, ConnectivityResult] = {}
        self._meta_rows: dict[tuple, dict] = {}

    def add(
        self,
        subject_id: str,
        result: ConnectivityResult,
        *,
        condition: str | None = None,
        group: str | None = None,
        **covariates,
    ) -> "Study":
        """Add one subject's result (chainable)."""
        if not isinstance(result, ConnectivityResult):
            raise DataError("result must be a ConnectivityResult.")
        key = (str(subject_id), condition)
        if key in self._results:
            raise DataError(
                f"Subject {subject_id!r} (condition={condition!r}) already added."
            )
        self._results[key] = result
        self._meta_rows[key] = {
            "subject": str(subject_id),
            "condition": condition,
            "group": group,
            **covariates,
        }
        return self

    def add_many(
        self,
        results: Mapping[str, ConnectivityResult],
        *,
        metadata: "pd.DataFrame | None" = None,
    ) -> "Study":
        """Add several subjects at once, optionally with a metadata frame."""
        meta = {}
        if metadata is not None:
            meta = {str(idx): row.to_dict() for idx, row in metadata.iterrows()}
        for sid, res in results.items():
            self.add(sid, res, **meta.get(str(sid), {}))
        return self

    # -- properties ----------------------------------------------------------

    @property
    def subjects(self) -> list[str]:
        seen: list[str] = []
        for sid, _ in self._results:
            if sid not in seen:
                seen.append(sid)
        return seen

    @property
    def conditions(self) -> list[str]:
        conds = {cond for _, cond in self._results if cond is not None}
        return sorted(conds)

    @property
    def chromophores(self) -> list[str]:
        chroms: list[str] = []
        for res in self._results.values():
            for c in res.chromophores:
                if c not in chroms:
                    chroms.append(c)
        return chroms

    @property
    def metadata(self) -> "pd.DataFrame":
        import pandas as pd

        return pd.DataFrame(list(self._meta_rows.values()))

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self) -> Iterator[tuple]:
        return iter(self._results)

    # -- slicing -------------------------------------------------------------

    def group(
        self,
        chromophore: str,
        *,
        condition: str | None = None,
        group: str | None = None,
        method: str | None = None,
        min_coverage: float = 1.0,
    ) -> GroupConnectivity:
        """Build a :class:`GroupConnectivity` for one chromophore.

        Filters subjects by ``condition`` and/or ``group``. Covariates from the
        metadata table are carried along in subject order for regression.
        """
        chrom = str(Chromophore.coerce(chromophore))
        matrices: list[ConnectivityMatrix] = []
        ids: list[str] = []
        meta_rows: list[dict] = []

        for key, res in self._results.items():
            sid, cond = key
            if condition is not None and cond != condition:
                continue
            meta = self._meta_rows[key]
            if group is not None and meta.get("group") != group:
                continue
            # find the matrix for this chromophore
            mat = self._select_matrix(res, chrom)
            if mat is None:
                continue
            if method is not None and mat.method != method:
                continue
            matrices.append(mat)
            ids.append(sid)
            meta_rows.append(meta)

        if not matrices:
            raise DataError(
                f"No subjects matched chromophore={chromophore!r}, "
                f"condition={condition!r}, group={group!r}."
            )

        import pandas as pd

        covariates = pd.DataFrame(meta_rows)
        gc = GroupConnectivity.from_matrices(
            matrices, ids=ids, covariates=covariates, min_coverage=min_coverage
        )
        return gc

    @staticmethod
    def _select_matrix(res: ConnectivityResult, chrom: str) -> ConnectivityMatrix | None:
        if chrom in res.chromophores:
            return res[chrom]
        # fall back to the single matrix if there is exactly one
        if len(res) == 1:
            return next(iter(res.values()))
        return None

    # -- reporting -----------------------------------------------------------

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_subjects": len(self.subjects),
            "n_records": len(self._results),
            "conditions": self.conditions,
            "chromophores": self.chromophores,
            "groups": sorted(
                {m.get("group") for m in self._meta_rows.values() if m.get("group")}
            ),
        }

    def report(self, **kwargs) -> str:
        from ..report.group import summarize_study

        return summarize_study(self, **kwargs)

    def __repr__(self) -> str:
        return (
            f"Study({self.name!r}, subjects={len(self.subjects)}, "
            f"records={len(self._results)}, chromophores={self.chromophores})"
        )
