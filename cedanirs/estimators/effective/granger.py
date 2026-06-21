"""Granger causality -- *extension template* for effective connectivity.

This module is intentionally **not registered**: it is the reference example for
how a directed estimator slots into cedanirs. Everything except the numerical
core is already wired up. To turn it on:

1. Implement the body of :meth:`GrangerCausality._estimate`.
2. Uncomment the ``@register_estimator`` decorator.

That is the entire contract -- the base class handles chromophore iteration,
labelling, p-values plumbing and result assembly, exactly as it does for the
Pearson estimator.

Granger causality tests whether the past of channel *j* improves the prediction
of channel *i* beyond *i*'s own past (a bivariate vector-autoregressive model
comparison). It is directed, so the resulting matrix is asymmetric:
``matrix[i, j]`` is the influence of *j* on *i*.

A robust implementation typically delegates the VAR fitting / F-test to
``statsmodels`` (``statsmodels.tsa.stattools.grangercausalitytests`` or a fitted
``VAR`` model). That dependency should be declared in the ``effective`` extra
and imported lazily here, raising
:class:`~cedanirs.core.exceptions.DependencyError` if missing.
"""

from __future__ import annotations

import numpy as np

# from ...core.registry import register_estimator   # <- uncomment to enable
from ...core.types import ConnectivityKind, Domain
from ..base import ConnectivityEstimator, EstimateOutput


# @register_estimator(name="granger")               # <- uncomment to enable
class GrangerCausality(ConnectivityEstimator):
    """Pairwise Granger causality (directed). **Not yet implemented.**

    Parameters
    ----------
    order:
        Maximum lag (model order) of the autoregressive fit.
    """

    name = "granger"
    kind = ConnectivityKind.EFFECTIVE
    directed = True
    domain = Domain.TIME

    def __init__(self, *, order: int = 1, **params):
        super().__init__(order=order, **params)
        self.order = order

    def _estimate(self, x: np.ndarray) -> EstimateOutput:
        # Pattern for implementers:
        #   from ...core.exceptions import DependencyError
        #   try:
        #       from statsmodels.tsa.stattools import grangercausalitytests
        #   except ImportError as exc:
        #       raise DependencyError("Granger causality", "statsmodels",
        #                             extra="effective") from exc
        #   n = x.shape[0]
        #   matrix = np.zeros((n, n)); pvalues = np.ones((n, n))
        #   for i in range(n):
        #       for j in range(n):
        #           if i == j: continue
        #           # test whether x[j] Granger-causes x[i]
        #           ...
        #   return EstimateOutput(matrix=matrix, pvalues=pvalues)
        raise NotImplementedError(
            "GrangerCausality is a roadmap skeleton. Implement _estimate and "
            "enable the @register_estimator decorator to activate it."
        )
