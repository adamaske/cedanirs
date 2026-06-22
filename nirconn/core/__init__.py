"""Core data model and machinery for nirconn.

This subpackage holds the framework-agnostic building blocks that the rest of
the library is composed from:

* :mod:`~nirconn.core.types` -- enumerations describing chromophores and the
  kind of connectivity an estimator produces.
* :mod:`~nirconn.core.exceptions` -- the exception hierarchy.
* :mod:`~nirconn.core.timeseries` -- the :class:`NirsTimeSeries` container.
* :mod:`~nirconn.core.result` -- :class:`ConnectivityMatrix` /
  :class:`ConnectivityResult`, the rich result objects.
* :mod:`~nirconn.core.registry` -- the estimator plugin registry.
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
