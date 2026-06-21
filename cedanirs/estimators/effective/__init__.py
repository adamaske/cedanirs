"""Effective connectivity estimators (directed causal influence).

Where functional connectivity asks *"do these two channels co-vary?"*,
effective connectivity asks *"does activity in channel A help predict the
future of channel B?"* -- a directed, asymmetric relationship. Methods here
produce non-symmetric matrices (``directed = True``).

No effective method is registered yet. :mod:`~cedanirs.estimators.effective.granger`
contains a fully-worked *skeleton* showing exactly how to add one; flip on its
``@register_estimator`` decorator and implement ``_estimate`` to make it live.

Planned: Granger causality, transfer entropy, dynamic causal modelling (DCM),
directed phase transfer entropy.
"""

from __future__ import annotations

__all__: list[str] = []
