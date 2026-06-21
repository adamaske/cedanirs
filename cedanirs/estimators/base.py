"""The estimator contract.

Every connectivity method is a subclass of :class:`ConnectivityEstimator`. The
base class does all the bookkeeping that is identical across methods:

* coercing arbitrary input into a :class:`~cedanirs.core.timeseries.NirsTimeSeries`,
* iterating over chromophores (so a method author never writes that loop),
* validating shapes,
* wrapping raw NumPy output in labelled :class:`~cedanirs.core.result.ConnectivityMatrix`
  / :class:`~cedanirs.core.result.ConnectivityResult` objects with full provenance.

A concrete method therefore implements just one thing: :meth:`_estimate`, the
N×N math on a single 2-D ``(channel, time)`` array.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, NamedTuple

import numpy as np

from ..core.exceptions import DataError
from ..core.result import ConnectivityMatrix, ConnectivityResult
from ..core.timeseries import NirsTimeSeries
from ..core.types import ConnectivityKind, Domain


class EstimateOutput(NamedTuple):
    """What :meth:`ConnectivityEstimator._estimate` returns.

    ``matrix`` is the N×N connectivity array; ``pvalues`` is an optional aligned
    N×N array of p-values (``None`` when the method has no analytic test).
    """

    matrix: np.ndarray
    pvalues: np.ndarray | None = None


class ConnectivityEstimator(ABC):
    """Abstract base class for all connectivity estimators.

    Subclasses set the class-level metadata (:attr:`name`, :attr:`kind`,
    :attr:`directed`, :attr:`domain`) and implement :meth:`_estimate`. Estimator
    parameters are accepted as keyword arguments in ``__init__`` and recorded on
    ``self.params`` for provenance.
    """

    #: Registry key; set by ``@register_estimator`` if omitted.
    name: ClassVar[str] = ""
    #: Functional (symmetric) or effective (directed).
    kind: ClassVar[ConnectivityKind] = ConnectivityKind.FUNCTIONAL
    #: Whether the produced matrix is asymmetric.
    directed: ClassVar[bool] = False
    #: Analysis domain (time / frequency / time-frequency).
    domain: ClassVar[Domain] = Domain.TIME
    #: Whether this method needs a known sampling frequency.
    requires_sfreq: ClassVar[bool] = False

    def __init__(self, **params):
        self.params = params
        # Acquisition context, populated by estimate() for each run. Frequency-
        # domain estimators read self._sfreq inside _estimate.
        self._sfreq: float | None = None
        self._n_times: int | None = None

    # -- the one method subclasses must implement ----------------------------

    @abstractmethod
    def _estimate(self, x: np.ndarray) -> EstimateOutput:
        """Compute connectivity for one chromophore.

        Parameters
        ----------
        x:
            A 2-D ``(n_channels, n_times)`` array (rows are channels/variables,
            columns are time samples/observations).

        Returns
        -------
        EstimateOutput
            The N×N matrix and an optional N×N p-value matrix.
        """

    # -- public entry point --------------------------------------------------

    def estimate(self, data) -> ConnectivityResult:
        """Run the estimator over ``data``, returning a rich result.

        ``data`` may be a :class:`NirsTimeSeries`, a NumPy array, or an
        :class:`xarray.DataArray`. Chromophore iteration is handled here.
        """
        ts = NirsTimeSeries.coerce(data)

        if self.requires_sfreq and not ts.sfreq:
            raise DataError(
                f"The {self.name!r} estimator needs a sampling frequency; "
                f"construct the NirsTimeSeries with sfreq=..."
            )

        # Make acquisition context available to _estimate (frequency-domain
        # methods read self._sfreq). Set once per estimate() call.
        self._sfreq = ts.sfreq
        self._n_times = ts.n_times

        matrices: dict[str, ConnectivityMatrix] = {}
        for chrom in ts.chromophores:
            sub = ts.select(chrom)
            x = sub.to_numpy()
            if x.ndim != 2:
                raise DataError(
                    f"Expected a 2-D (channel, time) slice, got {x.shape}."
                )
            out = self._estimate(np.ascontiguousarray(x, dtype=float))
            self._check_output(out, ts.n_channels)
            matrices[chrom] = ConnectivityMatrix(
                out.matrix,
                sub.channels,
                method=self.name,
                kind=self.kind,
                directed=self.directed,
                chromophore=chrom,
                pvalues=out.pvalues,
                params=self.params,
            )

        return ConnectivityResult(
            matrices,
            method=self.name,
            kind=self.kind,
            directed=self.directed,
            params=self.params,
            provenance={
                "n_channels": ts.n_channels,
                "n_times": ts.n_times,
                "sfreq": ts.sfreq,
                "name": ts.name,
            },
        )

    # convenience: estimator instances are callable
    def __call__(self, data) -> ConnectivityResult:
        return self.estimate(data)

    # -- helpers -------------------------------------------------------------

    def _check_output(self, out: EstimateOutput, n: int) -> None:
        if out.matrix.shape != (n, n):
            raise DataError(
                f"{type(self).__name__}._estimate returned a "
                f"{out.matrix.shape} matrix for {n} channels."
            )

    def __repr__(self) -> str:
        ps = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({ps})"
