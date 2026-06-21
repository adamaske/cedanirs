"""Core data model and machinery for cedanirs.

This subpackage holds the framework-agnostic building blocks that the rest of
the library is composed from:

* :mod:`~cedanirs.core.types` -- enumerations describing chromophores and the
  kind of connectivity an estimator produces.
* :mod:`~cedanirs.core.exceptions` -- the exception hierarchy.
* :mod:`~cedanirs.core.timeseries` -- the :class:`NirsTimeSeries` container.
* :mod:`~cedanirs.core.result` -- :class:`ConnectivityMatrix` /
  :class:`ConnectivityResult`, the rich result objects.
* :mod:`~cedanirs.core.registry` -- the estimator plugin registry.
"""

from .exceptions import (
    CedanirsError,
    DataError,
    DependencyError,
    EstimatorNotFoundError,
    NotFittedError,
)
from .registry import (
    create_estimator,
    get_estimator,
    list_estimators,
    register_estimator,
)
from .group import GroupConnectivity, Study
from .result import ConnectivityMatrix, ConnectivityResult
from .timeseries import NirsTimeSeries
from .types import Chromophore, ConnectivityKind, Domain

__all__ = [
    "CedanirsError",
    "DataError",
    "DependencyError",
    "EstimatorNotFoundError",
    "NotFittedError",
    "Chromophore",
    "ConnectivityKind",
    "Domain",
    "NirsTimeSeries",
    "ConnectivityMatrix",
    "ConnectivityResult",
    "Study",
    "GroupConnectivity",
    "register_estimator",
    "get_estimator",
    "create_estimator",
    "list_estimators",
]
