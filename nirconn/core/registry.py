"""The estimator plugin registry.

Adding a new connectivity method to nirconn is deliberately a two-line affair:
subclass :class:`~nirconn.estimators.base.ConnectivityEstimator` and decorate
it with :func:`register_estimator`. From that point the method is reachable by
name through :func:`nirconn.connectivity`, appears in
:func:`list_estimators`, and (eventually) is offered by any UI that reads the
catalog.

The registry stores *classes*, not instances, so each call constructs a fresh,
independently-configured estimator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Type, TypeVar

from .exceptions import EstimatorNotFoundError

if TYPE_CHECKING:  # pragma: no cover
    from ..estimators.base import ConnectivityEstimator

_REGISTRY: dict[str, Type["ConnectivityEstimator"]] = {}

E = TypeVar("E", bound="ConnectivityEstimator")


def register_estimator(
    cls: Type[E] | None = None,
    *,
    name: str | None = None,
    override: bool = False,
) -> Callable[[Type[E]], Type[E]] | Type[E]:
    """Class decorator that registers a connectivity estimator.

    Usable bare or with arguments::

        @register_estimator
        class PearsonCorrelation(ConnectivityEstimator): ...

        @register_estimator(name="pearson")
        class PearsonCorrelation(ConnectivityEstimator): ...

    The registry key defaults to the class attribute ``name``. Re-registering an
    existing key raises unless ``override=True``.
    """

    def _decorate(klass: Type[E]) -> Type[E]:
        key = name or getattr(klass, "name", "") or klass.__name__.lower()
        if not key:
            raise ValueError(f"{klass.__name__} needs a non-empty 'name'.")
        if key in _REGISTRY and not override and _REGISTRY[key] is not klass:
            raise ValueError(
                f"An estimator named {key!r} is already registered "
                f"({_REGISTRY[key].__name__}). Pass override=True to replace it."
            )
        klass.name = key
        _REGISTRY[key] = klass
        return klass

    if cls is not None:  # used as @register_estimator
        return _decorate(cls)
    return _decorate  # used as @register_estimator(...)


def get_estimator(name: str) -> Type["ConnectivityEstimator"]:
    """Return the estimator *class* registered under ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise EstimatorNotFoundError(name, list(_REGISTRY)) from None


def create_estimator(name: str, **params) -> "ConnectivityEstimator":
    """Instantiate the estimator registered under ``name`` with ``params``."""
    return get_estimator(name)(**params)


def list_estimators() -> list[dict]:
    """Catalog of registered estimators as plain dicts.

    Each entry has ``name``, ``kind``, ``directed``, ``domain`` and
    ``description`` -- enough to populate a UI menu or a docs table without
    importing the classes.
    """
    catalog = []
    for key, klass in sorted(_REGISTRY.items()):
        catalog.append(
            {
                "name": key,
                "kind": str(getattr(klass, "kind", "")),
                "directed": bool(getattr(klass, "directed", False)),
                "domain": str(getattr(klass, "domain", "")),
                "description": (klass.__doc__ or "").strip().split("\n")[0],
            }
        )
    return catalog


def is_registered(name: str) -> bool:
    return name in _REGISTRY
