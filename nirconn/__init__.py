"""nirconn — functional & effective connectivity analysis for fNIRS.

A modern, extensible toolkit for estimating, testing, visualising and reporting
brain connectivity from functional near-infrared spectroscopy data.

Quick start
-----------
::

    import numpy as np
    import nirconn as cn

    # data: (n_channels, n_times), already preprocessed
    result = cn.connectivity(data, method="pearson", sfreq=10.0,
                             channels=["S1_D1", "S1_D2", "S2_D1"])

    m = result.matrix            # a ConnectivityMatrix
    m.plot()                     # heatmap
    sig = m.significant(0.05)    # FDR-corrected significance mask
    print(result.report())       # text report

    cn.list_estimators()         # discover available methods

The public surface is intentionally small; everything below
:mod:`nirconn.core`, :mod:`nirconn.estimators`, :mod:`nirconn.stats`,
:mod:`nirconn.graph`, :mod:`nirconn.viz` and :mod:`nirconn.report` is
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
from .estimators.effective.granger import GrangerCausality
from .estimators.functional.coherence import Coherence
from .estimators.functional.partial import PartialCorrelation
from .estimators.functional.pearson import PearsonCorrelation
from .estimators.functional.plv import PhaseLockingValue
from .estimators.functional.spearman import SpearmanCorrelation
from .estimators.functional.wavelet_coherence import WaveletCoherence

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
    "PhaseLockingValue",
    "WaveletCoherence",
    "GrangerCausality",
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
