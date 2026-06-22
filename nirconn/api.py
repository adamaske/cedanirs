"""The top-level convenience API.

Most users only need one function. :func:`connectivity` looks up an estimator by
name, configures it with any keyword arguments, and runs it -- the one-liner
entry point to the whole library::

    import nirconn as cn
    result = cn.connectivity(data, method="pearson", sfreq=10.0)
    result.matrix.plot()
    print(result.report())
"""

from __future__ import annotations

from .core.registry import create_estimator
from .core.result import ConnectivityResult
from .core.timeseries import NirsTimeSeries


def connectivity(
    data,
    method: str = "pearson",
    *,
    sfreq: float | None = None,
    channels=None,
    chromophores=None,
    chromophore=None,
    **params,
) -> ConnectivityResult:
    """Estimate connectivity from fNIRS time-series data.

    Parameters
    ----------
    data:
        A :class:`~nirconn.core.timeseries.NirsTimeSeries`, a NumPy array
        (2-D ``(channel, time)`` or 3-D ``(chromophore, channel, time)``), or an
        :class:`xarray.DataArray`.
    method:
        Registered estimator name (see
        :func:`~nirconn.core.registry.list_estimators`). Defaults to
        ``"pearson"``.
    sfreq, channels, chromophores, chromophore:
        Forwarded to :class:`NirsTimeSeries` when ``data`` is a raw array;
        ignored when ``data`` is already a :class:`NirsTimeSeries`.
    **params:
        Estimator-specific parameters (e.g. ``compute_pvalues=False``).

    Returns
    -------
    ConnectivityResult
        One :class:`~nirconn.core.result.ConnectivityMatrix` per chromophore.
    """
    if not isinstance(data, NirsTimeSeries):
        data = NirsTimeSeries(
            data,
            sfreq=sfreq,
            channels=channels,
            chromophores=chromophores,
            chromophore=chromophore,
        )
    estimator = create_estimator(method, **params)
    return estimator.estimate(data)
