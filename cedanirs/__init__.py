"""cedanirs — functional & effective connectivity analysis for fNIRS.

A modern, extensible toolkit for estimating, testing, visualising and reporting
brain connectivity from functional near-infrared spectroscopy data.

Quick start
-----------
::

    import numpy as np
    import cedanirs as cn

    # data: (n_channels, n_times), already preprocessed
    result = cn.connectivity(data, method="pearson", sfreq=10.0,
                             channels=["S1_D1", "S1_D2", "S2_D1"])

    m = result.matrix            # a ConnectivityMatrix
    m.plot()                     # heatmap
    sig = m.significant(0.05)    # FDR-corrected significance mask
    print(result.report())       # text report

    cn.list_estimators()         # discover available methods

The public surface is intentionally small; everything below
:mod:`cedanirs.core`, :mod:`cedanirs.estimators`, :mod:`cedanirs.stats`,
:mod:`cedanirs.graph`, :mod:`cedanirs.viz` and :mod:`cedanirs.report` is
available for advanced use.
"""

from __future__ import annotations

from ._version import __version__

# Core data model
from .core.exceptions import (
    CedanirsError,
    DataError,
    DependencyError,
    EstimatorNotFoundError,
    NotFittedError,
)
from .core.registry import (
    create_estimator,
    get_estimator,
    list_estimators,
    register_estimator,
)
from .core.group import GroupConnectivity, Study
from .core.result import ConnectivityMatrix, ConnectivityResult
from .core.timeseries import NirsTimeSeries
from .core.types import Chromophore, ConnectivityKind, Domain

# Estimators (importing this registers the shipped methods)
from .estimators.base import ConnectivityEstimator, EstimateOutput
from .estimators.functional.coherence import Coherence
from .estimators.functional.partial import PartialCorrelation
from .estimators.functional.pearson import PearsonCorrelation
from .estimators.functional.spearman import SpearmanCorrelation

# Top-level convenience API
from .api import connectivity

# Subpackages exposed as namespaces for advanced use
from . import graph, io, report, stats, tables, viz
from .io import read_snirf, read_timeseries
from .preprocessing import Preprocessor
from .viz.poster import build_poster

__all__ = [
    "__version__",
    # API
    "connectivity",
    # data model
    "NirsTimeSeries",
    "ConnectivityMatrix",
    "ConnectivityResult",
    "Study",
    "GroupConnectivity",
    "build_poster",
    "read_snirf",
    "read_timeseries",
    "io",
    "Chromophore",
    "ConnectivityKind",
    "Domain",
    # registry
    "register_estimator",
    "get_estimator",
    "create_estimator",
    "list_estimators",
    # estimators
    "ConnectivityEstimator",
    "EstimateOutput",
    "PearsonCorrelation",
    "SpearmanCorrelation",
    "PartialCorrelation",
    "Coherence",
    # preprocessing
    "Preprocessor",
    # exceptions
    "CedanirsError",
    "DataError",
    "DependencyError",
    "EstimatorNotFoundError",
    "NotFittedError",
    # subpackages
    "stats",
    "graph",
    "viz",
    "report",
    "tables",
]
