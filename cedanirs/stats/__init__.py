"""Statistical helpers for connectivity analysis.

Two concerns live here, kept deliberately separate from the estimators so they
can be reused by any method and tested in isolation:

* :mod:`~cedanirs.stats.transforms` -- variance-stabilising transforms, chiefly
  the Fisher *z* transform used before averaging correlations.
* :mod:`~cedanirs.stats.significance` -- analytic p-values and multiple-
  comparison correction (Benjamini-Hochberg FDR, Bonferroni).
"""

from __future__ import annotations

from ._results import GroupStatResult, NBSResult
from .group import nbs, one_sample, regression, two_sample
from .significance import (
    bonferroni_correction,
    correct_pvalues,
    correlation_pvalues,
    fdr_correction,
)
from .transforms import average_correlations, fisher_z, inverse_fisher_z

__all__ = [
    # transforms
    "fisher_z",
    "inverse_fisher_z",
    "average_correlations",
    # significance
    "correlation_pvalues",
    "fdr_correction",
    "bonferroni_correction",
    "correct_pvalues",
    # group-level tests
    "one_sample",
    "two_sample",
    "regression",
    "nbs",
    "GroupStatResult",
    "NBSResult",
]
