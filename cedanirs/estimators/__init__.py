"""Connectivity estimators.

The :class:`~cedanirs.estimators.base.ConnectivityEstimator` base class defines
the contract; concrete methods live under :mod:`~cedanirs.estimators.functional`
(undirected statistical dependence) and
:mod:`~cedanirs.estimators.effective` (directed causal influence).

Importing this package imports every shipped estimator module, which is what
populates the registry via the ``@register_estimator`` decorators.
"""

from __future__ import annotations

from .base import ConnectivityEstimator, EstimateOutput

# Importing the functional package triggers every shipped method's
# registration side-effect (via its ``@register_estimator`` decorator).
from . import functional as _functional  # noqa: F401

__all__ = ["ConnectivityEstimator", "EstimateOutput"]
