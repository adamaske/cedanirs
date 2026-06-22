"""Effective connectivity estimators (directed causal influence).

Where functional connectivity asks *"do these two channels co-vary?"*,
effective connectivity asks *"does activity in channel A help predict the
future of channel B?"* -- a directed, asymmetric relationship. Methods here
produce non-symmetric matrices (``directed = True``).

Shipped:

* :class:`~nirconn.estimators.effective.granger.GrangerCausality` -- pairwise
  bivariate Granger causality (F-test on a lagged autoregressive model).

Planned: transfer entropy, dynamic causal modelling (DCM), directed phase
transfer entropy.
"""

from __future__ import annotations

from .granger import GrangerCausality

__all__ = ["GrangerCausality"]
