"""Functional connectivity estimators (undirected statistical dependence).

These methods quantify how strongly two channels' time courses co-vary, without
making any claim about direction or causality. The matrices they produce are
symmetric.

Shipped:

* :class:`~cedanirs.estimators.functional.pearson.PearsonCorrelation` -- linear
  Pearson product-moment correlation.
* :class:`~cedanirs.estimators.functional.spearman.SpearmanCorrelation` -- rank
  correlation (monotonic, outlier-robust).
* :class:`~cedanirs.estimators.functional.partial.PartialCorrelation` -- direct
  association controlling for all other channels.
* :class:`~cedanirs.estimators.functional.coherence.Coherence` -- band-averaged
  magnitude-squared coherence (frequency domain).

Planned (see the project roadmap): wavelet coherence, phase-locking value,
mutual information.
"""

from __future__ import annotations

from .coherence import Coherence
from .partial import PartialCorrelation
from .pearson import PearsonCorrelation
from .spearman import SpearmanCorrelation

__all__ = [
    "PearsonCorrelation",
    "SpearmanCorrelation",
    "PartialCorrelation",
    "Coherence",
]
