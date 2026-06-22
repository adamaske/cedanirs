"""Enumerations describing the fNIRS connectivity domain.

These are intentionally small, ``str``-based enums so they serialise cleanly
(to JSON, to xarray attrs, to a report) while still being type-safe in Python.
"""

from __future__ import annotations

from enum import Enum


class Chromophore(str, Enum):
    """A hemodynamic signal type carried by an fNIRS channel.

    fNIRS measures changes in light attenuation that are converted, via the
    modified Beer-Lambert law, into concentration changes of oxygenated and
    deoxygenated hemoglobin. Connectivity is almost always computed *within* a
    single chromophore, so this label is attached to every time series and
    every resulting matrix.
    """

    HBO = "HbO"
    """Oxygenated hemoglobin (Δ[HbO])."""
    HBR = "HbR"
    """Deoxygenated hemoglobin (Δ[HbR])."""
    HBT = "HbT"
    """Total hemoglobin (HbO + HbR)."""
    OD = "OD"
    """Optical density (pre-Beer-Lambert)."""
    RAW = "raw"
    """Raw light intensity."""
    UNKNOWN = "unknown"
    """Chromophore not specified."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    @classmethod
    def coerce(cls, value: "Chromophore | str | None") -> "Chromophore":
        """Best-effort conversion from a string/enum to a :class:`Chromophore`.

        Matching is case-insensitive on both the member name and value, so
        ``"hbo"``, ``"HbO"`` and ``Chromophore.HBO`` all resolve identically.
        Unrecognised strings become :attr:`UNKNOWN` rather than raising, since a
        free-text channel label should never crash an analysis.
        """
        if value is None:
            return cls.UNKNOWN
        if isinstance(value, cls):
            return value
        text = str(value).strip()
        for member in cls:
            if text.lower() == member.value.lower() or text.lower() == member.name.lower():
                return member
        return cls.UNKNOWN


class ConnectivityKind(str, Enum):
    """Whether an estimator measures statistical dependence or causal influence.

    * :attr:`FUNCTIONAL` -- undirected statistical dependency between time
      series (e.g. Pearson/Spearman correlation, coherence). Produces a
      symmetric matrix.
    * :attr:`EFFECTIVE` -- directed causal influence one region exerts over
      another (e.g. Granger causality, dynamic causal modelling). Produces an
      asymmetric matrix.
    """

    FUNCTIONAL = "functional"
    EFFECTIVE = "effective"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class Domain(str, Enum):
    """The analysis domain an estimator operates in."""

    TIME = "time"
    FREQUENCY = "frequency"
    TIME_FREQUENCY = "time-frequency"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value
