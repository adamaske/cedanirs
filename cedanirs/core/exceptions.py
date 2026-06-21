"""Exception hierarchy for cedanirs.

Every error raised by the library inherits from :class:`CedanirsError`, so
callers can catch the whole family with a single ``except``. Where it adds
ergonomic value, leaf exceptions also inherit from the closest builtin
(``KeyError``, ``ImportError``, ...) so existing ``except`` clauses keep working.
"""

from __future__ import annotations


class CedanirsError(Exception):
    """Base class for all cedanirs-specific errors."""


class DataError(CedanirsError, ValueError):
    """The input data is malformed, the wrong shape, or otherwise unusable."""


class EstimatorNotFoundError(CedanirsError, KeyError):
    """No estimator is registered under the requested name."""

    def __init__(self, name: str, available: list[str] | None = None):
        self.name = name
        self.available = available or []
        msg = f"No connectivity estimator registered as {name!r}."
        if self.available:
            msg += f" Available: {', '.join(sorted(self.available))}."
        super().__init__(msg)

    def __str__(self) -> str:
        # KeyError.__str__ would wrap the message in quotes; override it.
        return self.args[0]


class NotFittedError(CedanirsError):
    """An operation needs results that have not been computed yet."""


class DependencyError(CedanirsError, ImportError):
    """An optional dependency required for this feature is not installed."""

    def __init__(self, feature: str, package: str, extra: str | None = None):
        self.feature = feature
        self.package = package
        self.extra = extra
        hint = f"pip install {package}"
        if extra:
            hint = f"pip install 'cedanirs[{extra}]'  (or: {hint})"
        super().__init__(
            f"{feature} requires the optional dependency {package!r}, which is "
            f"not installed. Install it with: {hint}"
        )
