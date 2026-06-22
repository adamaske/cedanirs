"""Connectivity estimators.

The :class:`~nirconn.estimators.base.ConnectivityEstimator` base class defines
the contract; concrete methods live under :mod:`~nirconn.estimators.functional`
(undirected statistical dependence) and
:mod:`~nirconn.estimators.effective` (directed causal influence).

Importing this package imports every shipped estimator module, which is what
populates the registry via the ``@register_estimator`` decorators.
"""

from __future__ import annotations

from .base import ConnectivityEstimator, EstimateOutput

# Importing the functional and effective packages triggers every shipped
# method's registration side-effect (via its ``@register_estimator`` decorator).
from . import effective as _effective  # noqa: F401
from . import functional as _functional  # noqa: F401

__all__ = ["ConnectivityEstimator", "EstimateOutput"]
